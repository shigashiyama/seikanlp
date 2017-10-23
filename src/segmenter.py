import sys
import logging
import argparse
import copy
import os
import pickle
from datetime import datetime
from collections import Counter

#os.environ["CHAINER_TYPE_CHECK"] = "0"
import numpy as np
import chainer
from chainer import serializers
from chainer import cuda

import data
import lattice
import models
import embedding.read_embedding as emb
from tools import conlleval


LOG_DIR = 'log'
MODEL_DIR = 'models/main'
OUT_DIR = 'out'


def batch_generator(instances, label_seqs, label_seqs2=None, batchsize=100, shuffle=True, xp=np):
    len_data = len(instances)
    perm = np.random.permutation(len_data) if shuffle else range(0, len_data)

    for i in range(0, len_data, batchsize):
        i_max = min(i + batchsize, len_data)
        xs = [xp.asarray(instances[perm[i]], dtype=np.int32) for i in range(i, i_max)]
        ts = [xp.asarray(label_seqs[perm[i]], dtype=np.int32) for i in range(i, i_max)]

        if label_seqs2:
            ts2 = [label_seqs2[perm[i]] for i in range(i, i_max)]
            yield xs, ts, ts2
        else:
            yield xs, ts

    raise StopIteration


def decode(tagger, instances, batchsize=100, decode_type='tag', stream=sys.stderr, xp=np):
    tagger.predictor.train = False
    tagger.compute_fscore = False

    len_data = len(instances)
    for i in range(0, len_data, batchsize):
        i_max = min(i + batchsize, len_data)
        xs = [xp.asarray(instances[i], dtype=np.int32) for i in range(i, i_max)]
        decode_batch(xs, tagger, decode_type=decode_type, stream=stream)


def decode_batch(xs, tagger, decode_type='tag', stream=sys.stderr):
    ys = tagger.predictor.decode(xs)

    for x, y in zip(xs, ys):
        x_str = [tagger.indices.get_token(int(xi)) for xi in x]
        y_str = [tagger.indices.get_label(int(yi)) for yi in y]

        if decode_type == 'tag':
            res = ['{}/{}'.format(xi_str, yi_str) for xi_str, yi_str in zip(x_str, y_str)]
            res = ' '.join(res)

        elif decode_type == 'seg':
            res = ['{}{}'.format(xi_str, ' ' if (yi_str == 'E' or yi_str == 'S') else '') for xi_str, yi_str in zip(x_str, y_str)]
            res = ''.join(res).rstrip()

        elif decode_type == 'seg_tag':
            res = ['{}{}'.format(
                xi_str, 
                ('/'+yi_str[2:]+' ') if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
            ) for xi_str, yi_str in zip(x_str, y_str)]
            res = ''.join(res).rstrip()

        else:
            print('Error: Invalid decode type', file=stream)
            sys.exit()

        print(res, file=stream)


def evaluate(tagger, instances, slab_seqs, plab_seqs=None, batchsize=100, xp=np, stream=sys.stderr):
    eval_pos = plab_seqs != None

    evaluator = tagger.copy()
    evaluator.predictor.train = False # dropout does nothing
    evaluator.compute_fscore = True

    count = 0
    num_tokens = 0
    total_loss = 0
    total_ecounts_seg = conlleval.EvalCounts()
    total_ecounts_pos = conlleval.EvalCounts()

    for batch in batch_generator(instances, slab_seqs, plab_seqs, batchsize=batchsize, shuffle=False, xp=xp):
        xs = batch[0]
        ts_seg = batch[1]
        ts_pos = batch[2] if len(batch) == 3 else None
        with chainer.no_backprop_mode():
            if eval_pos:
                loss, ecounts_seg, ecounts_pos = evaluator(xs, ts_seg, ts_pos, train=True)
            else:
                loss, ecounts_seg = evaluator(xs, ts_seg, train=False)
                ecounts_pos = None

        num_tokens += sum([len(x) for x in xs])
        count += len(xs)
        total_loss += loss.data
        ave_loss = total_loss / num_tokens
        total_ecounts_seg = conlleval.merge_counts(total_ecounts_seg, ecounts_seg)
        if eval_pos:
            total_ecounts_pos = conlleval.merge_counts(total_ecounts_pos, ecounts_pos)

        seg_c = total_ecounts_seg
        seg_acc = conlleval.calculate_accuracy(seg_c.correct_tags, seg_c.token_counter)
        seg_overall = conlleval.calculate_metrics(
            seg_c.correct_chunk, seg_c.found_guessed, seg_c.found_correct)

        if eval_pos:
            pos_c = total_ecounts_pos
            pos_acc = conlleval.calculate_accuracy(pos_c.correct_tags, pos_c.token_counter)
            pos_overall = conlleval.calculate_metrics(
                pos_c.correct_chunk, pos_c.found_guessed, pos_c.found_correct)
        else:
            pos_c = None
            pos_acc = None
            pos_overall = None

    print_results(
        total_loss, ave_loss, count, seg_c, seg_acc, seg_overall, pos_acc, pos_overall, stream=stream)
    stat = 'n_sen: %d, n_token: %d, n_chunk: %d, n_chunk_p: %d' % (
        count, seg_c.token_counter, seg_c.found_correct, seg_c.found_guessed)
    if eval_pos:
        res = '%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f' % (
            (100.*seg_acc), 
            (100.*seg_overall.prec), (100.*seg_overall.rec), (100.*seg_overall.fscore), 
            total_loss, ave_loss)
    else:
        res = '%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f' % (
            (100.*seg_acc), (100.*seg_overall.prec), (100.*seg_overall.rec), (100.*seg_overall.fscore), 
            total_loss, ave_loss)
        
    return stat, res


def print_results(total_loss, ave_loss, count, c, acc, overall, acc2=None, overall2=None, 
                  stream=sys.stderr):
    print('total loss: %.4f' % total_loss, file=stream)
    print('ave loss: %.5f'% ave_loss, file=stream)
    print('#sen, #token, #chunk, #chunk_pred: {} {} {} {}'.format(
        count, c.token_counter, c.found_correct, c.found_guessed),
          file=stream)
    if acc2:
        print('TP, FP, FN: %d %d %d | %d %d %d' % (
            overall.tp, overall.fp, overall.fn, overall2.tp, overall2.fp, overall2.fn),
              file=stream)
    else:
        print('TP, FP, FN: %d %d %d' % (overall.tp, overall.fp, overall.fn),
              file=stream)
    if acc2:
        print('A, P, R, F | A:%6.2f %6.2f %6.2f %6.2f | A:%6.2f %6.2f %6.2f %6.2f' % 
              (100.*acc, 100.*overall.prec, 100.*overall.rec, 100.*overall.fscore,
               100.*acc2, 100.*overall2.prec, 100.*overall2.rec, 100.*overall2.fscore),
              file=stream)
    else:
        print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' % 
              (100.*acc, 100.*overall.prec, 100.*overall.rec, 100.*overall.fscore),
              file=stream)
    print('', file=stream)


def parse_arguments():
    # used letters: bcedfgilmnopqrstvx

    parser = argparse.ArgumentParser()

    # mandatory options
    parser.add_argument('--execute_mode', '-x', required=True,
                        help='Choose a mode from among \'train\', \'eval\', \'decode\' and \'interactive\'')
    
    # gpu options
    parser.add_argument('--gpu', '-g', type=int, default=-1, help=
                        'GPU device ID (Use CPU if unspecify any values or specify a negative value)')
    parser.add_argument('--cudnn', dest='use_cudnn', action='store_true',
                        help='Use cuDNN')

    # training parameters
    parser.add_argument('--epoch_begin', type=int, default=1, 
                        help='Conduct training from i-th epoch (Default: 1)')
    parser.add_argument('--epoch_end', '-e', type=int, default=5,
                        help='Conduct training up to i-th epoch (Default: 5)')
    parser.add_argument('--iterations', '-i', type=int, default=10000, 
                        help='The number of iterations over mini-batch.'
                        + ' Trained model is evaluated and saved at each multiple of the number'
                        + ' (Default: 10000)')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='The number of examples in each mini-batch (Default: 100)')
    parser.add_argument('--dropout', type=float, default=0.0, 
                        help='Dropout ratio for RNN vertical layers (Default: 0.0)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')

    # model parameters
    parser.add_argument('--embed_dimension', '-d', type=int, default=300,
                        help='The number of dimension of embedding lalyer (Default: 300)')
    parser.add_argument('--rnn_layers', '-l', type=int, default=1, 
                        help='The number of RNN layers (Default: 1)')
    parser.add_argument('--rnn_unit_type', default='lstm',
                        help='Choose unit type of RNN from among \'lstm\', \'gru\' and \'plain\''
                        + ' (Default: lstm)')
    parser.add_argument('--rnn_hidden_units', '-u', type=int, default=800,
                        help='The number of hidden units of RNN (Default: 800)')
    parser.add_argument('--rnn_bidirection', action='store_true', 
                        help='Use bidirectional RNN')
    parser.add_argument('--inference_layer', default='crf',
                        help='Choose type of inference layer from between \'softmax\' and \'crf\''
                        + ' (Default: crf)')

    # optimizer parameters
    parser.add_argument('--optimizer', default='sgd',
                        help='Choose optimizing algorithm from among \'sgd\', \'adam\' and \'adagrad\''
                        + ' (Default: sgd)')
    parser.add_argument('--learning_rate', type=float, default=1.0, 
                        help='Initial learning rate (Default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum ratio for SGD')
    parser.add_argument('--lrdecay', default='', 
                        help='Specify information on learning rate decay'
                        + ' by format \'start:width:rate\''
                        + ' where start indicates the number of epochs to start decay,'
                        + ' width indicates the number epochs maintaining the same decayed learning rate,'
                        + ' and rate indicates the decay late to multipy privious learning late')
    parser.add_argument('--weightdecay', type=float, default=0.0, 
                        help='Weight decay ratio (Default: 0.0)')

    # options to resume training
    parser.add_argument('--model_path', '-m',
                        help='npz File path of the trained model. \'xxx.hyp\' and \'xxx.s2i\' files'
                        + 'are also read simultaneously if you specify \'xxx_izzz.npz\' file.')

    # data paths and related options
    parser.add_argument('--data_format', '-f', 
                        help='Choose format of input data among from'
                        + ' \'seg\' (word segmentation on data where words are split with single spaces),'
                        + ' \'bccwj_seg\' (word segmentation w/ or w/o POS tagging on data with BCCWJ format),'
                        + ' \'bccwj_tag\' (POS tagging from gold words on data with BCCWJ format),'
                        + ' \'wsj\' (POS tagging on data with CoNLL-2005 format),'
                        + ' \'conll2003\' (NER on data with CoNLL-2003 format),')
    parser.add_argument('--subpos_depth', '-s', type=int, default=-1,
                        help='Set 0 (WS only), positive integer (use POS up to i-th hierarcy)'
                        + ' or -1 (use POS with all sub POS) when set \'bccwj_seg\' to data_format')
    parser.add_argument('--path_prefix', '-p', help='Path prefix of input data')
    parser.add_argument('--train_data', '-t', default='',
                        help='File path succeeding \'path_prefix\' of training data')
    parser.add_argument('--valid_data', '-v', default='',
                        help='File path succeeding \'path_prefix\' of validation data')
    parser.add_argument('--test_data', default='',
                        help='File path succeeding \'path_prefix\' of test data')
    parser.add_argument('--raw_data', '-r', default='',
                        help='File path of input raw text which succeeds \'path_prefix\'')
    parser.add_argument('--label_reference_data', default='',
                        help='File path succeeding \'path_prefix\''
                        + ' of data with the same format as training data to load pre-defined labels')
    parser.add_argument('--output_path', '-o', default='',
                        help='File path to output parsed text')
    parser.add_argument('--embed_model_path', 
                        help='File path of pretrained model of word (character or other unit) embedding')
    parser.add_argument('--dict_path', 
                        help='File path of word dictionary that lists words, or words and POS')
    parser.add_argument('--use_dict_feature', action='store_true',
                        help='Use dictionary features based on training data and specified word dictionary')
    parser.add_argument('--dump_train_data', action='store_true',
                        help='Dump data specified as \'train_data\''
                        + ' using new file name endding with \'.pickle\'')

    # options for data preprocessing
    parser.add_argument('--lowercase',  action='store_true',
                        help='Lowercase alphabets in the case of using English data')
    parser.add_argument('--normalize_digits',  action='store_true',
                        help='Normalize digits by the same symbol in the case of using English data')
    
    # other options
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Do not output log file and serialized model file')

    #parser.add_argument('--limit', type=int, default=-1, help='Limit the number of samples for quick tests')
    #parser.add_argument('--joint_type', '-j', default='')

    args = parser.parse_args()
    if args.lrdecay:
        parser.add_argument('lrdecay_start')
        parser.add_argument('lrdecay_width')
        parser.add_argument('lrdecay_rate')
        array = args.lrdecay.split(':')
        args.lrdecay_start = int(array[0])
        args.lrdecay_width = int(array[1])
        args.lrdecay_rate = float(array[2])
    if args.execute_mode == 'interactive':
        args.quiet = True

    return args


def init_hyperparameters(args):
    hparams = {
        #'joint_type' : args.joint_type,
        'embed_dimension' : args.embed_dimension,
        'rnn_layers' : args.rnn_layers,
        'rnn_hidden_units' : args.rnn_hidden_units,
        'rnn_unit_type' : args.rnn_unit_type,
        'rnn_bidirection' : args.rnn_bidirection,
        'inference_layer' : args.inference_layer,
        'dropout' : args.dropout,
        'use_dict_feature' : args.use_dict_feature,
        'data_format' : args.data_format,
        'subpos_depth' : args.subpos_depth,
        'lowercase' : args.lowercase,
        'normalize_digits' : args.normalize_digits,
    }

    return hparams


class Trainer(object):
    def __init__(self, args, logger=sys.stderr):
        err_msgs = []
        if args.execute_mode == 'train':
            if not args.train_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    'train', '--train_data/-t')
                err_msgs.append(msg)
            if not args.data_format:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    'train', '--data_format/-f')
                err_msgs.append(msg)

        elif args.execute_mode == 'eval':
            if not args.model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    'eval', '--model_path/-p')
                err_msgs.append(msg)

            if not args.test_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    'eval', '--test_data')
                err_msgs.append(msg)

        elif args.execute_mode == 'decode':
            if not args.model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    'decode', '--model_path/-p')
                err_msgs.append(msg)

            if not args.raw_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    'raw', '--model_path/-p')
                err_msgs.append(msg)

        elif args.execute_mode == 'interactive':
            if not args.model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    'interactive', '--model_path/-p')
                err_msgs.append(msg)

        else:
            msg = 'Error: invalid execute mode: {}'.format(args.execute_mode)
            err_msgs.append(msg)

        if err_msgs:
            for msg in err_msgs:
                print(msg, file=sys.stderr)
            sys.exit()

        self.args = args
        self.start_time = datetime.now().strftime('%Y%m%d_%H%M')
        self.logger = logger    # output execute log
        self.reporter = None    # output evaluation results
        self.train = None
        self.train_t = None
        self.train_p = None
        self.val = None
        self.val_t = None
        self.val_p = None
        self.test = None
        self.test_t = None
        self.test_p = None
        self.hparams = None
        self.tagger = None
        self.optimizer = None
        self.decode_type = None

        self.log('Start time: {}\n'.format(self.start_time))
        if not self.args.quiet:
            self.reporter = open('{}/{}.log'.format(LOG_DIR, self.start_time), 'a')

    def report(self, message):
        if not self.args.quiet:
            print(message, file=self.reporter)


    def log(self, message):
        print(message, file=self.logger)


    def close(self):
        if not self.args.quiet:
            self.reporter.close()
    

    def init_hparams(self, hparams):
        self.hparams = hparams

        if hparams['data_format'] == 'seg':
            self.decode_type = 'seg'

        elif hparams['data_format'] == 'bccwj_seg':
            if hparams['subpos_depth'] == 0:
                self.decode_type = 'seg'
            else:
                self.decode_type = 'seg_tag'

        else:
            self.decode_type == 'tag'


    def set_tagger(self, tagger):
        self.tagger = tagger


    def load_model(self, model_path, use_gpu=False):
        array = model_path.split('_i')
        indices_path = '{}.s2i'.format(array[0])
        hparam_path = '{}.hyp'.format(array[0])
        param_path = model_path
    
        with open(indices_path, 'rb') as f:
            indices = pickle.load(f)
        self.log('Load indices: {}'.format(indices_path))
        self.log('Vocab size: {}\n'.format(len(indices.token_indices)))

        hparams = {}
        with open(hparam_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue

                kv = line.split('=')
                key = kv[0]
                val = kv[1]

                if (key == 'embed_dimension' or
                    key == 'rnn_layers' or
                    key == 'rnn_hidden_units' or
                    key == 'subpos_depth'
                ):
                    val = int(val)

                elif key == 'dropout':
                    val = float(val)

                elif (key == 'rnn_bidirection' or
                      key == 'use_dict_feature' or
                      key == 'lowercase' or
                      key == 'normalize_digits'
                ):
                    val = (val.lower() == 'true')

                hparams[key] = val

        self.log('Load hyperparameters: {}\n'.format(hparam_path))
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            if k in hparams and v != hparams[k]:
                message = '{}={} (input option value {} was discarded)'.format(k, hparams[k], v)
            else:
                message = '{}={}'.format(k, v)

            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')

        self.log('Initialize model from hyperparameters\n')
        tagger = models.init_tagger(indices, hparams, use_gpu=use_gpu)
        chainer.serializers.load_npz(model_path, tagger)
        self.log('Load model parameters: {}\n'.format(model_path))

        self.init_hparams(hparams)
        self.tagger = tagger


    def load_data_for_training(self, indices=None, embed_model=None):
        args = self.args
        hparams = self.hparams
        prefix = args.path_prefix if args.path_prefix else ''
            
        refer_path   = prefix + args.label_reference_data
        train_path   = prefix + args.train_data
        val_path     = prefix + (args.valid_data if args.valid_data else '')
        refer_vocab = embed_model.wv if embed_model else set()
        read_pos = hparams['subpos_depth'] != 0

        if args.label_reference_data:
            _, _, _, indices = data.load_data(
                hparams['data_format'], refer_path, read_pos=read_pos, update_token=False,
                subpos_depth=hparams['subpos_depth'], lowercase=hparams['lowercase'],
                normalize_digits=hparams['normalize_digits'], indices=indices)
            self.log('Load label set from reference data: {}'.format(refer_path))

        if args.train_data.endswith('pickle'):
            name, ext = os.path.splitext(train_path)
            train, train_t, train_p = data.load_pickled_data(name)
            self.log('Load dumpped data:'.format(train_path))
            #TODO 別データで再学習の場合は要エラー処理

        else:
            train, train_t, train_p, indices = data.load_data(
            hparams['data_format'], train_path, read_pos=read_pos,
            subpos_depth=hparams['subpos_depth'], lowercase=hparams['lowercase'],
            normalize_digits=hparams['normalize_digits'], indices=indices, refer_vocab=refer_vocab)
        
        if self.args.dump_train_data:
            name, ext = os.path.splitext(train_path)
            data.dump_pickled_data(name, train, train_t, train_p)
            self.log('Dumpped training data: {}.pickle'.format(train_path))

        self.log('Load training data: {}'.format(train_path))
        self.log('Data length: {}'.format(len(train)))
        self.log('Vocab size: {}\n'.format(len(indices.token_indices)))

        if args.valid_data:
            # indices can be updated if embed_model are used
            val, val_t, val_p, indices = data.load_data(
                hparams['data_format'], val_path, read_pos=read_pos, update_token=False, update_label=False,
                subpos_depth=hparams['subpos_depth'], lowercase=hparams['lowercase'],
                normalize_digits=hparams['normalize_digits'], indices=indices, refer_vocab=refer_vocab)

            self.log('Load validation data: {}'.format(val_path))
            self.log('Data length: {}'.format(len(val)))
            self.log('Vocab size: {}\n'.format(len(indices.token_indices)))
        else:
            val, val_t, val_p = [], [], []
        
        self.train = train
        self.train_t = train_t
        self.train_p = train_p
        self.val = val
        self.val_t = val_t
        self.val_p = val_p

        n_train = len(train)
        t2i_tmp = list(indices.token_indices.token2id.items())
        id2label = {v:k for k,v in indices.label_indices.token2id.items()}
        if train_p:
            id2pos = {v:k for k,v in indices.pos_indices.token2id.items()}

        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train[:3], train[n_train-3:]))
        self.log('# train_gold: {} ... {}\n'.format(train_t[:3], train_t[n_train-3:]))
        if train_p:
            self.log('# POS labels: {} ... {}\n'.format(train_p[:3], train_p[n_train-3:]))
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))
        self.log('# labels: {}\n'.format(id2label))
        if train_p:
            self.log('poss: {}\n'.format(id2pos))
        
        self.report('[INFO] vocab: {}'.format(len(indices.token_indices)))
        self.report('[INFO] data length: train={} val={}'.format(len(train), len(val)))

        return indices


    def load_data_for_test(self, indices=None, embed_model=None):
        args = self.args
        hparams = self.hparams
        test_path = (args.path_prefix if args.path_prefix else '') + args.test_data
        refer_vocab = embed_model.wv if embed_model else set()
        read_pos = hparams['subpos_depth'] != 0
            
        test, test_t, test_p, indices = data.load_data(
            hparams['data_format'], test_path, read_pos=read_pos, update_token=False, update_label=False,
            subpos_depth=hparams['subpos_depth'], lowercase=hparams['lowercase'],
            normalize_digits=hparams['normalize_digits'], indices=indices, refer_vocab=refer_vocab)

        self.log('Load test data: {}'.format(test_path))
        self.log('Data length: {}'.format(len(test)))
        self.log('Vocab size: {}\n'.format(len(indices.token_indices)))

        self.test = test
        self.test_t = test_t
        self.test_p = test_p

        return indices


    def setup_optimizer(self):
        if self.args.optimizer == 'sgd':
            if self.args.momentum <= 0:
                optimizer = chainer.optimizers.SGD(lr=self.args.learning_rate)
            else:
                optimizer = chainer.optimizers.MomentumSGD(
                    lr=self.args.learning_rate, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = chainer.optimizers.Adam()
        elif self.args.optimizer == 'adagrad':
            optimizer = chainer.optimizers.AdaGrad(lr=self.args.learning_rate)

        optimizer.setup(self.tagger)
        optimizer.add_hook(chainer.optimizer.GradientClipping(self.args.gradclip))
        if self.args.weightdecay > 0:
            optimizer.add_hook(chainer.optimizer.WeightDecay(self.args.weightdecay))

        self.optimizer = optimizer

        return self.optimizer


    def run_evaluation(self):
        xp = cuda.cupy if args.gpu >= 0 else np

        self.log('<test result>')
        te_stat, te_res = evaluate(
            self.tagger, self.test, self.test_t, batchsize=self.args.batchsize, xp=xp, stream=self.logger)
        
        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.log('Finish: %s\n' % time)
        self.report('Finish: %s\n' % time)


    def run_decoding(self):
        xp = cuda.cupy if args.gpu >= 0 else np

        raw_path = (args.path_prefix if args.path_prefix else '') + self.args.raw_data
        if self.decode_type == 'tag':
            instances = data.load_raw_text_for_tagging(raw_path, self.tagger.indices)
        else:
            instances = data.load_raw_text_for_segmentation(raw_path, self.tagger.indices)

        stream = open(self.args.output_path, 'w') if self.args.output_path else sys.stdout
        decode(self.tagger, instances, batchsize=self.args.batchsize, 
               decode_type=self.decode_type, stream=stream, xp=xp)

        if self.args.output_path:
            stream.close()


    def run_interactive_decoding(self):
        xp = cuda.cupy if args.gpu >= 0 else np

        print('Please input text or type \'q\' to quit this mode:')
        while True:
            line = sys.stdin.readline().rstrip()
            if len(line) == 0:
                continue
            elif line == 'q':
                break

            if self.decode_type == 'tag':
                array = line.split(' ')
                ins = [self.tagger.indices.get_token_id(word) for word in array]
            else:
                ins = [self.tagger.indices.get_token_id(char) for char in line]

            xs = [xp.asarray(ins, dtype=np.int32)]
            decode_batch(xs, self.tagger, decode_type=self.decode_type)
            print()


    def run_training(self):
        if not self.args.quiet:
            hparam_path = '{}/{}.hyp'.format(MODEL_DIR, self.start_time)
            with open(hparam_path, 'w') as f:
                for key, val in self.hparams.items():
                    print('{}={}'.format(key, val), file=f)
                self.log('save hyperparameters: {}'.format(hparam_path))

            indices_path = '{}/{}.s2i'.format(MODEL_DIR, self.start_time)
            with open(indices_path, 'wb') as f:
                pickle.dump(self.tagger.indices, f)
            self.log('save string2index table: {}'.format(indices_path))

        xp = cuda.cupy if args.gpu >= 0 else np
        n_train = len(self.train)
        n_iter_report = self.args.iterations
        n_iter = 0

        for e in range(max(1, self.args.epoch_begin), self.args.epoch_end+1):
            time = datetime.now().strftime('%Y%m%d_%H%M')
            self.log('Start epoch {}: {}\n'.format(e, time))
            self.report('[INFO] Start epoch {} at {}'.format(e, time))

            # learning rate decay
            if self.args.lrdecay and self.args.optimizer == 'sgd':
                if (e - lrdecay_start) % lrdecay_width == 0:
                    lr_tmp = self.optimizer.lr
                    self.optimizer.lr = self.optimizer.lr * lrdecay_rate
                    self.log('Learning rate decay: {} -> {}\n'.format(lr_tmp, self.optimizer.lr))
                    self.report('Learning rate decay: {} -> {}'.format(lr_tmp, self.optimizer.lr))

            count = 0
            num_tokens = 0
            total_loss = 0
            total_ecounts = conlleval.EvalCounts()

            i = 0
            for xs, ts in batch_generator(self.train, self.train_t, label_seqs2=None, 
                                          batchsize=self.args.batchsize, shuffle=True, xp=xp):

                t0 = datetime.now()
                loss, ecounts = self.tagger(xs, ts, train=True)
                t1 = datetime.now()

                num_tokens += sum([len(x) for x in xs])
                count += len(xs)
                total_loss += loss.data
                total_ecounts = conlleval.merge_counts(total_ecounts, ecounts)
                i_max = min(i + self.args.batchsize, n_train)
                self.log('* batch %d-%d loss: %.4f' % ((i+1), i_max, loss.data))
                i = i_max
                n_iter += 1

                t2 = datetime.now()
                self.optimizer.target.cleargrads() # Clear the parameter gradients
                loss.backward()               # Backprop
                loss.unchain_backward()       # Truncate the graph
                self.optimizer.update()            # Update the parameters
                t3 = datetime.now()
                # print('train    {} ins: {}'.format((t1-t0).seconds+(t1-t0).microseconds/10**6, len(xs)))
                # print('backprop {} ins: {}'.format((t3-t2).seconds+(t3-t2).microseconds/10**6, len(xs)))
                # print('total    {} ins: {}'.format((t3-t0).seconds+(t3-t0).microseconds/10**6, len(xs)))

                # Evaluation
                if (n_iter * self.args.batchsize) % n_iter_report == 0: # or i == n_train:

                    now_e = '%.3f' % (n_iter * self.args.batchsize / n_train)
                    time = datetime.now().strftime('%Y%m%d_%H%M')
                    self.log('\n### iteration %s (epoch %s)' % ((n_iter * self.args.batchsize), now_e))
                    self.log('<training result for previous iterations>')

                    ave_loss = total_loss / num_tokens
                    c = total_ecounts
                    acc = conlleval.calculate_accuracy(c.correct_tags, c.token_counter)
                    overall = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)
                    print_results(total_loss, ave_loss, count, c, acc, overall, stream=self.logger)

                    t_stat = 'n_sen: %d, n_token: %d, n_chunk: %d, n_chunk_p: %d' % (
                        count, c.token_counter, c.found_correct, c.found_guessed)
                    t_res = '%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%d\t%.4f\t%.4f' % (
                        (100.*acc), (100.*overall.prec), (100.*overall.rec), (100.*overall.fscore), 
                        overall.tp, overall.fp, overall.fn, total_loss, ave_loss)
                    self.report('[INFO] train - %s' % t_stat)
                    if n_iter == 1:
                        self.report(
                            'data\titer\tep\tacc\tprec\trec\tfb1\tTP\tFP\tFN\ttloss\taloss')
                    self.report('train\t%d\t%s\t%s' % (n_iter, now_e, t_res))

                    if self.args.valid_data:
                        self.log('<validation result>')
                        v_stat, v_res = evaluate(
                            self.tagger, self.val, self.val_t, batchsize=self.args.batchsize, xp=xp,
                            stream=self.logger)

                        self.report('[INFO] valid - %s' % v_stat)
                        if n_iter == 1:
                            self.report(
                                'data\titer\tep\tacc\tprec\trec\tfb1\tTP\tFP\tFN\ttloss\taloss')
                        self.report('valid\t%d\t%s\t%s' % (n_iter, now_e, v_res))

                    # Save the model
                    mdl_path = '{}/{}_i{}.npz'.format(MODEL_DIR, self.start_time, now_e)
                    self.log('Save the model: %s\n' % mdl_path)
                    self.report('[INFO] save the model: %s\n' % mdl_path)
                    if not self.args.quiet:
                        serializers.save_npz(mdl_path, self.tagger)
        
                    # Reset counters
                    count = 0
                    num_tokens = 0
                    total_loss = 0
                    total_ecounts = conlleval.EvalCounts()

                    if not self.args.quiet:
                        self.reporter.close() 
                        self.reporter = open('{}/{}.log'.format(LOG_DIR, self.start_time), 'a')

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('Finish: %s\n' % time)


#TODO modify loggging 
class Trainer4JointMA(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super(Trainer4JointMA, self).__init__(args, logger=sys.stderr)


    def run_training(self):
        xp = cuda.cupy if args.gpu >= 0 else np

        n_train = len(self.train)
        n_iter_report = self.args.iter_to_report
        n_iter = 0

        for e in range(max(1, self.args.epoch_begin), self.args.epoch_end+1):
            time = datetime.now().strftime('%Y%m%d_%H%M')
            self.report('[INFO] Start epoch %d at %s\n' % (e, time))
            self.log('Start epoch %d: %s' % (e, time))

            # learning rate decay: not implemented

            count = 0
            num_tokens = 0
            total_loss = 0
            total_ecounts_seg = conlleval.EvalCounts()
            total_ecounts_pos = conlleval.EvalCounts()

            i = 0
            for xs, ts_seg, ts_pos in batch_generator(
                    self.train, self.train_t, self.train_p, self.args.batchsize, shuffle=False, xp=xp):

                #self.tagger.predictor.lattice_crf.debug = True
                #self.tagger.indices.chunk_trie.debug = True
                
                t0 = datetime.now()
                loss, ecounts_seg, ecounts_pos = self.tagger(xs, ts_seg, ts_pos, train=True)
                t1 = datetime.now()

                num_tokens += sum([len(x) for x in xs])
                count += len(xs)
                total_loss += loss.data
                total_ecounts_seg = conlleval.merge_counts(total_ecounts_seg, ecounts_seg)
                total_ecounts_pos = conlleval.merge_counts(total_ecounts_pos, ecounts_pos)
                i_max = min(i + self.args.batchsize, n_train)
                self.log('* batch %d-%d loss: %.4f' % ((i+1), i_max, loss.data))
                i = i_max
                n_iter += 1

                t2 = datetime.now()
                self.optimizer.target.cleargrads() # Clear the parameter gradients
                loss.backward()               # Backprop
                loss.unchain_backward()       # Truncate the graph
                self.optimizer.update()            # Update the parameters
                t3 = datetime.now()
                # print('train    {} ins: {}'.format((t1-t0).seconds+(t1-t0).microseconds/10**6, len(xs)))
                # print('backprop {} ins: {}'.format((t3-t2).seconds+(t3-t2).microseconds/10**6, len(xs)))
                # print('total    {} ins: {}'.format((t3-t0).seconds+(t3-t0).microseconds/10**6, len(xs)))

                # Evaluation
                if (n_iter * self.args.batchsize) % n_iter_report == 0:

                    now_e = '%.3f' % (n_iter * self.args.batchsize / n_train)
                    time = datetime.now().strftime('%Y%m%d_%H%M')
                    self.log('\n### iteration %s (epoch %s)' % ((n_iter * self.args.batchsize), now_e))
                    self.log('<training result for previous iterations>')

                    ave_loss = total_loss / num_tokens

                    seg_c = total_ecounts_seg
                    seg_acc = conlleval.calculate_accuracy(seg_c.correct_tags, seg_c.token_counter)
                    seg_overall = conlleval.calculate_metrics(
                        seg_c.correct_chunk, seg_c.found_guessed, seg_c.found_correct)

                    pos_c = total_ecounts_pos
                    pos_acc = conlleval.calculate_accuracy(pos_c.correct_tags, pos_c.token_counter)
                    pos_overall = conlleval.calculate_metrics(
                        pos_c.correct_chunk, pos_c.found_guessed, pos_c.found_correct)

                    print_results(
                        total_loss, ave_loss, count, seg_c, seg_acc, seg_overall, pos_acc, pos_overall)
                    print()

                    if not self.args.quiet:
                        t_stat = 'n_sen: %d, n_token: %d, n_chunk: %d, n_chunk_p: %d' % (
                            count, seg_c.token_counter, seg_c.found_correct, seg_c.found_guessed)
                        t_res = '%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f%.4f\t%.4f' % (
                            (100.*seg_acc), 
                            (100.*seg_overall.prec), (100.*seg_overall.rec), (100.*seg_overall.fscore),
                            (100.*pos_acc), 
                            (100.*pos_overall.prec), (100.*pos_overall.rec), (100.*pos_overall.fscore), 
                            total_loss, ave_loss)
                        self.logger.write('INFO: train - %s\n' % t_stat)
                        if n_iter == 1:
                            self.logger.write('data\titer\tep\t\s-acc\ts-prec\ts-rec\ts-fb1\t\p-acc\tp-prec\tp-rec\tp-fb1\tloss\taloss\n')
                        self.logger.write('train\t%d\t%s\t%s\n' % (n_iter, now_e, t_res))

                    if self.args.valid_data:
                        print('<validation result>')
                        v_stat, v_res = evaluate(
                            self.tagger, self.val, self.val_t, self.val_p, self.args.batchsize, xp=xp)
                        print()

                        if not self.args.quiet:
                            self.logger.write('INFO: valid - %s\n' % v_stat)
                            if n_iter == 1:
                                self.logger.write('data\titer\tep\t\s-acc\ts-prec\ts-rec\ts-fb1\t\p-acc\tp-prec\tp-rec\tp-fb1\tloss\taloss\n')
                            self.logger.write('valid\t%d\t%s\t%s\n' % (n_iter, now_e, v_res))

                    # Save the model
                    mdl_path = '{}/{}_i{}.npz'.format(MODEL_DIR, self.start_time, now_e)
                    if not self.args.quiet:
                        print('save the model: %s\n' % mdl_path)
                        self.logger.write('INFO: save the model: %s\n' % mdl_path)
                        serializers.save_npz(mdl_path, self.tagger)
        
                    # Reset counters
                    count = 0
                    total_loss = 0
                    num_tokens = 0
                    total_ecounts = conlleval.EvalCounts()

                    if not self.args.quiet:
                        self.logger.close() # 一度保存しておく
                        self.logger = open('{}/{}.log'.format(LOG_DIR, self.start_time), 'a')

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.log('finish: %s\n' % time)


if __name__ == '__main__':
    if chainer.__version__[0] != '2':
        print("chainer version>=2.0.0 is required.")
        sys.exit()


    ################################
    # Make necessary directories

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    ################################
    # Get arguments and prepare GPU

    args = parse_arguments()
    use_gpu = args.gpu >= 0
    if use_gpu:
        # Make the specified GPU current
        cuda.get_device_from_id(args.gpu).use()
        chainer.config.cudnn_deterministic = args.use_cudnn

    ################################
    # Intialize trainer and check arguments

    trainer = Trainer(args)

    ################################
    # Load tagger model

    if args.model_path:
        trainer.load_model(args.model_path, use_gpu=use_gpu)
        id2token_org = copy.deepcopy(trainer.tagger.indices.id2token)
        id2label_org = copy.deepcopy(trainer.tagger.indices.id2label)
    else:
        hparams = init_hyperparameters(args)
        trainer.init_hparams(hparams)
        id2token_org = {}
        id2label_org = {}


    ################################
    # Load word embedding model
    
    if args.embed_model_path and args.execute_mode != 'interactive':
        embed_model = emb.read_model(args.embed_model_path)
        embed_dim = embed_model.wv.syn0[0].shape[0]
        if embed_dim != hparams['embed_dimension']:
            print('Given embedding model is discarded due to dimension confriction')
            embed_model = None
    else:
        embed_model = None

    ################################
    # Initialize indices of strings from external dictionary

    if not trainer.tagger and args.dict_path:
        dic_path = args.dict_path
        indices = lattice.load_dictionary(dic_path, read_pos=False)
        trainer.set_indices(indices)
        trainer.log('Load dictionary: {}'.format(dic_path))
        trainer.log('Vocab size: {}'.format(len(indices.token_indices)))

        if not dic_path.endswith('pickle'):
            base = dic_path.split('.')[0]
            dic_pic_path = base + '.pickle'
            with open(dic_pic_path, 'wb') as f:
                pickle.dump(indices, f)
            trainer.log('Dumpped dictionary: {}'.format(dic_pic_path))

    ################################
    # Load dataset and set up indices

    indices = trainer.tagger.indices if trainer.tagger else None
    if args.execute_mode == 'train':
        indices = trainer.load_data_for_training(indices, embed_model)
    elif args.execute_mode == 'eval':
        indices = trainer.load_data_for_test(indices, embed_model)
    indices.create_id2token()
    indices.create_id2label()

    ################################
    # Set up tagger and optimizer

    if not trainer.tagger:
        tagger = models.init_tagger(indices, trainer.hparams, use_gpu=use_gpu)
        trainer.set_tagger(tagger)
    else:
        trainer.tagger.indices = indices

    grow_vocab = id2token_org and (len(trainer.tagger.indices.id2token) > len(id2token_org))
    if embed_model or grow_vocab:
        trainer.tagger.grow_embedding_layer(trainer.tagger.indices.id2token, id2token_org, embed_model)
    grow_labels = id2label_org and (len(trainer.tagger.indices.id2label) > len(id2label_org))
    if grow_labels:
        trainer.tagger.grow_inference_layers(trainer.tagger.indices.id2label, id2label_org)

    if args.gpu >= 0:
        trainer.tagger.to_gpu()

    trainer.setup_optimizer()

    ################################
    # Run

    if args.execute_mode == 'train':
        trainer.run_training()
    elif args.execute_mode == 'eval':
        trainer.run_evaluation()
    elif args.execute_mode == 'decode':
        trainer.run_decoding()
    elif args.execute_mode == 'interactive':
        trainer.run_interactive_decoding()

    ################################
    # Terminate

    trainer.close()
