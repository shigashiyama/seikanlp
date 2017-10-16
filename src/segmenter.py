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
import embedding.read_embedding as emb
from tools import conlleval


LOG_DIR = 'log'
MODEL_DIR = 'models/nn'
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


def decode(model, instances, batchsize=100, seg=False, stream=sys.stdout, xp=np):
    model.predictor.train = False
    model.compute_fscore = False

    len_data = len(instances)
    for i in range(0, len_data, batchsize):
        i_max = min(i + batchsize, len_data)
        xs = [xp.asarray(instances[i], dtype=np.int32) for i in range(i, i_max)]
        decode_batch(xs, model, stream)


def decode_batch(xs, model, seg=False, stream=sys.stdout):
    ys = model.predictor.decode(xs)

    for x, y in zip(xs, ys):
        x_str = [model.indices.get_token(int(xi)) for xi in x]
        y_str = [model.indices.get_label(int(yi)) for yi in y]

        if seg:
            res = ['{}{}'.format(' ' if (yi_str == 'B' or yi_str == 'S') else '', 
                                 xi_str) for xi_str, yi_str in zip(x_str, y_str)]
            res = ''.join(res).lstrip()

        else:
            res = ['{}/{}'.format(xi_str, yi_str) for xi_str, yi_str in zip(x_str, y_str)]
            res = ' '.join(res)

        print(res, file=stream)


def evaluate(model, instances, slab_seqs, plab_seqs=None, batchsize=100, xp=np):
    eval_pos = plab_seqs != None

    evaluator = model.copy()          # to use different state
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
        total_loss, ave_loss, count, seg_c, seg_acc, seg_overall, pos_acc, pos_overall)
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


def print_results(total_loss, ave_loss, count, c, acc, overall, acc2=None, overall2=None):
    print('total loss: %.4f' % total_loss)
    print('ave loss: %.5f'% ave_loss)
    print('#sen, #token, #chunk, #chunk_pred: {} {} {} {}'.format(
        count, c.token_counter, c.found_correct, c.found_guessed))
    if acc2:
        print('TP, FP, FN: %d %d %d | %d %d %d' % (
            overall.tp, overall.fp, overall.fn, overall2.tp, overall2.fp, overall2.fn))
    else:
        print('TP, FP, FN: %d %d %d' % (overall.tp, overall.fp, overall.fn))
    if acc2:
        print('A, P, R, F | A:%6.2f %6.2f %6.2f %6.2f | A:%6.2f %6.2f %6.2f %6.2f' % 
              (100.*acc, 100.*overall.prec, 100.*overall.rec, 100.*overall.fscore,
               100.*acc2, 100.*overall2.prec, 100.*overall2.rec, 100.*overall2.fscore))
    else:
        print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' % 
              (100.*acc, 100.*overall.prec, 100.*overall.rec, 100.*overall.fscore))


def parse_arguments():
    #bcedfgilmnoprstvx

    parser = argparse.ArgumentParser()

    # mandatory options
    parser.add_argument('--execute_mode', '-x', required=True,
                        help='Choose a mode from among \'train\', \'eval\', \'decode\' and \'interactive\'')
    
    # gpu options
    parser.add_argument('--gpu', '-g', type=int, default=-1, help=
                        'GPU device ID (Use CPU if unspecify any value or specify negative value)')
    parser.add_argument('--cudnn', dest='use_cudnn', action='store_true',
                        help='Use cuDNN')

    # training parameters
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='The number of sweeps over the dataset to train (Default: 5)')
    parser.add_argument('--iteration_to_report', '-i', type=int, default=10000, 
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
    #parser.add_argument('--linear_activation', '-a', default='identity')

    # optimizer parameters
    parser.add_argument('--optimizer', default='sgd',
                        help='Choose optimizing algorithm from among \'sgd\', \'adam\' and \'adagrad\'')
    parser.add_argument('--learning_rate', type=float, default=1.0, 
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum ratio for SGD')
    parser.add_argument('--lrdecay', default='', 
                        help='Specify information on learning rate decay'
                        + ' by format \'start:width:rate\''
                        + ' where start_epoch indicates the number of epochs to start decay,'
                        + ' width indicates the number epochs maintaining the same decayed learning rate,'
                        + ' rate indicates decay late to multipy privious learning late')
    parser.add_argument('--weightdecay', type=float, default=0, 
                        help='Weight decay ratio')

    # options to resume training
    parser.add_argument('--model_to_load', '-m',
                        help='File path of the trained model to resume the training')
    parser.add_argument('--epoch_to_resume', type=int, default=1, 
                        help='Specify how manieth epoch from which you resume the training')

    # data paths and related options
    parser.add_argument('--data_format', '-f', 
                        help='Choose format of input data among from'
                        + ' \'ws\' (word segmentation on data where words are split with single spaces),'
                        + ' \'bccwj_ws\' (word segmentation w/ or w/o POS tagging on data with BCCWJ format),'
                        + ' \'bccwj_pos\' (POS tagging from gold words on data with BCCWJ format),'
                        + ' \'wsj\' (POS tagging on data with CoNLL-2005 format),'
                        + ' \'conll2003\' (NER on data with CoNLL-2003 format),')
    parser.add_argument('--subpos_depth', '-s', type=int, default=-1,
                        help='Set 0 (WS only), positive integer (use POS up to i-th hierarcy)'
                        + ' or -1 (use POS with all sub POS) when set \'bccwj_ws\' to data_format')
    parser.add_argument('--path_prefix', '-p', help='Path prefix of input data')
    parser.add_argument('--pretrain_data', default='',
                        help='File path succeeding \'path_prefix\' of pretrain data')
    parser.add_argument('--train_data', '-t', default='',
                        help='File path succeeding \'path_prefix\' of training data')
    parser.add_argument('--validation_data', '-v', default='',
                        help='File path succeeding \'path_prefix\' of validation data')
    parser.add_argument('--test_data', default='',
                        help='File path  succeeding \'path_prefix\' of test data')
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
                        help='Lowercase alphabets when use English data')
    parser.add_argument('--normalize_digits',  action='store_true',
                        help='Normalize digits by the same symbol when use English data')
    
    # other options
    parser.add_argument('--nolog', '-n', action='store_true',
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
    parser.add_argument('linear_activation')
    args.linear_activation = 'identity'

    for k, v in args.__dict__.items():
        print('# {}={}'.format(k, v))
    print('')

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.start_time = datetime.now().strftime('%Y%m%d_%H%M')
        self.logger = None
        self.trained = None
        self.trained_t = None
        self.trained_p = None
        self.train = None
        self.train_t = None
        self.train_p = None
        self.val = None
        self.val_t = None
        self.val_p = None
        self.test = None
        self.test_t = None
        self.test_p = None
        self.dic = None
        self.hparams = None
        self.token_indices_org = None
        self.model = None
        self.optimizer = None

        print('INFO: start: {}\n'.format(self.start_time))
        if not self.args.nolog:
            self.logger = open('{}/{}.log'.format(LOG_DIR, self.start_time), 'a')
            self.logger.write('{}\n'.format(args))
        
    def log(self, message):
        if not self.args.nolog:
            self.logger.write(message)


    def prepare_gpu(self):
        if self.args.gpu >= 0:
            # Make the specified GPU current
            cuda.get_device_from_id(self.args.gpu).use()
            chainer.config.cudnn_deterministic = self.args.use_cudnn


    def load_data(self, dic=None, embed_model=None):
        args = self.args

        refer_path   = '{}/{}'.format(args.path_prefix, args.label_reference_data)
        trained_path = '{}/{}'.format(args.path_prefix, args.pretrain_data)
        train_path   = '{}/{}'.format(args.path_prefix, args.train_data)
        val_path     = '{}/{}'.format(args.path_prefix, (args.validation_data if args.validation_data else ''))
        test_path    = '{}/{}'.format(args.path_prefix, (args.test_data if args.test_data else ''))
        refer_vocab = embed_model.wv if embed_model else set()
        hparams = {}

        token_indices_org = None

        read_pos = args.subpos_depth != 0
        limit = -1 #args.limit if args.limit > 0 else -1

        if args.label_reference_data:
            _, _, _, dic = data.load_data(
                args.data_format, refer_path, read_pos=read_pos, update_token=False,
                subpos_depth=args.subpos_depth, 
                lowercase=args.lowercase, normalize_digits=args.normalize_digits,
                dic=dic, limit=limit)
            print('load label set from reference data')

        if args.pretrain_data:
            if args.pretrain_data.endswith('pickle'):
                name, ext = os.path.splitext('{}/{}'.format(args.path_prefix, args.pretrain_data))
                trained, trained_t, trained_p, dic = data.load_pickled_data(name)
            else:
                trained, trained_t, trained_p, dic = data.load_data(
                    args.data_format, trained_path, read_pos=read_pos, subpos_depth=args.subpos_depth, 
                    lowercase=args.lowercase, normalize_digits=args.normalize_digits,
                    dic=dic, refer_vocab=refer_vocab, limit=limit)

            token_indices_org = copy.deepcopy(dic.token_indices)
            print('vocab size:', len(dic.token_indices))
        else:
            trained = []
            trained_t = []
            trained_p = []

        if args.train_data.endswith('pickle'):
            name, ext = os.path.splitext('{}/{}'.format(args.path_prefix, args.train_data))
            train, train_t, train_p, dic = data.load_pickled_data(name)

        else:
            train, train_t, train_p, dic = data.load_data(
                args.data_format, train_path, read_pos=read_pos, subpos_depth=args.subpos_depth, 
                lowercase=args.lowercase, normalize_digits=args.normalize_digits,
                dic=dic, refer_vocab=refer_vocab, limit=limit)

            if self.args.dump_train_data:
                name, ext = os.path.splitext('{}/{}'.format(self.args.path_prefix, self.args.train_data))
                data.dump_pickled_data(name, train, train_t, train_p, dic)
                print('pickled training data')

        print('vocab size:', len(dic.token_indices))
        if not token_indices_org:
            token_indices_org = dic.token_indices

        if args.validation_data:
            if args.validation_data.endswith('pickle'):
                name, ext = os.path.splitext('{}/{}'.format(args.path_prefix, args.validation_data))
                pass
            else:
                val, val_t, val_p, dic = data.load_data(
                    args.data_format, val_path, read_pos=read_pos, update_token=False, update_label=False, 
                    subpos_depth=args.subpos_depth, 
                    lowercase=args.lowercase, normalize_digits=args.normalize_digits,
                    dic=dic, refer_vocab=refer_vocab, limit=limit)

            print('vocab size:', len(dic.token_indices))        
        else:
            val = []
            val_t = []
            val_p = []
            
        if args.test_data:
            if args.test_data.endswith('pickle'):
                name, ext = os.path.splitext('{}/{}'.format(args.path_prefix, args.test_data))
                pass
            else:
                test, test_t, test_p, dic = data.load_data(
                    args.data_format, test_path, read_pos=read_pos, update_token=False, update_label=False,
                    subpos_depth=args.subpos_depth, 
                    lowercase=args.lowercase, normalize_digits=args.normalize_digits,
                    dic=dic, refer_vocab=refer_vocab, limit=limit)
            print('vocab size:', len(dic.token_indices))
        else:
            test = []
            test_t = []
            test_p = []

        n_train = len(train)
        t2i_tmp = list(dic.token_indices.token2id.items())
        id2label = {v:k for k,v in dic.label_indices.token2id.items()}
        if train_p:
            id2pos = {v:k for k,v in dic.pos_indices.token2id.items()}

        print('vocab =', len(dic.token_indices))
        print('data length:', len(train))
        print()
        print('train:', train[:3], '...', train[n_train-3:])
        print('labels:', train_t[:3], '...', train_t[n_train-3:])
        if train_p:
            print('pos labels:', train_p[:3], '...', train_p[n_train-3:])
        print()
        print('token2id:', t2i_tmp[:10], '...', t2i_tmp[len(t2i_tmp)-10:])
        print('labels:', id2label)
        if train_p:
            print('poss:', id2pos)
        
        self.log('INFO: vocab = %d\n' % len(dic.token_indices))
        self.log('INFO: data length: train=%d val=%d\n' % (len(train), len(val)))

        self.trained = trained
        self.trained_t = trained_t
        self.trained_p = trained_p
        self.train = train
        self.train_t = train_t
        self.train_p = train_p
        self.val = val
        self.val_t = val_t
        self.val_p = val_p
        self.test = test
        self.test_t = test_t
        self.test_p = test_p
        self.dic = dic
        self.hparams = hparams
        self.token_indices_org = token_indices_org
        self.segment = (args.data_format == 'ws' or
                        (args.data_format == 'bccwj_ws' and args.subpos_depth == 0))


    def prepare_model_and_hparameters(self, embed_model=None):
        # parameter
        if not self.hparams:
            self.hparams = {#'joint_type' : self.args.joint_type,
                           'embed_dimension' : self.args.embed_dimension,
                           'rnn_layers' : self.args.rnn_layers,
                           'rnn_hidden_units' : self.args.rnn_hidden_units,
                           'rnn_bidirection' : self.args.rnn_bidirection,
                           'inference_layer' : self.args.inference_layer,
                           'linear_activation' : self.args.linear_activation,
                           'use_dict_feature' : self.args.use_dict_feature,
                           'dropout' : self.args.dropout,
            }
        if self.args.embed_model_path:
            self.hparams.update({'embed_model_path' : self.args.embed_model_path})

        # model
        grow_lookup = self.args.execute_mode == 'train' and len(self.dic.token_indices) > len(self.token_indices_org)
        if grow_lookup:
            ti_updated=self.dic.token_indices
            self.dic.token_indices = self.token_indices_org
        else:
            ti_updated=None

        model = data.load_model_from_hparams(
            self.hparams, model_path=self.args.model_to_load, dic=self.dic, 
            token_indices_updated=ti_updated, embed_model=embed_model, gpu=self.args.gpu)
        model.compute_fscore = True

        if grow_lookup:
            self.dic.token_indices = ti_updated

        self.model = model

        return self.model


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

        optimizer.setup(self.model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(self.args.gradclip))
        if self.args.weightdecay > 0:
            optimizer.add_hook(chainer.optimizer.WeightDecay(self.args.weightdecay))

        self.optimizer = optimizer

        return self.optimizer


    # unused
    def write_params_and_indices(self):
        if not self.args.nolog:
            self.hparams.update({
                'token2id_path' : 'nn_model/vocab_' + self.start_time + '.t2i.txt',
                'label2id_path' : 'nn_model/vocab_' + self.start_time + '.l2i.txt',
                'model_date': self.start_time
            })
            data.write_param_file(self.hparams, 'nn_model/param_' + self.start_time + '.txt')
            data.write_map(self.dic.token_indices.token2id, self.hparams['token2id_path'])
            data.write_map(self.dic.label_indices.token2id, self.hparams['label2id_path'])


    def run(self):
        if self.args.execute_mode == 'train':
            self.run_training()

        elif self.args.execute_mode == 'eval':
            self.run_evaluation()

        elif self.args.execute_mode == 'decode':
            self.run_decoding()

        elif self.args.execute_mode == 'interactive':
            self.run_interactive_decoding()

        else:
            print('Invalid execute mode: {}'.format(self.args.execute_mode))
            sys.exit()


    def run_evaluation(self):
        xp = cuda.cupy if args.gpu >= 0 else np

        print('<test result>')
        te_stat, te_res = evaluate(
            self.model, self.test, self.test_t, batchsize=self.args.batchsize, xp=xp)
        print()

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.log('finish: %s\n' % time)
        print('finish: %s\n' % time)


    def run_decoding(self):
        xp = cuda.cupy if args.gpu >= 0 else np

        raw_path = '{}/{}'.format(self.args.path_prefix, self.args.raw_data)
        if self.segment:
            instances = data.load_raw_text_for_segmentation(raw_path, self.dic)
        else:
            instances = data.load_raw_text(raw_path)

        stream = open(self.args.output_path, 'w') if self.args.output_path else sys.stdout
        decode(self.model, instances, batchsize=self.args.batchsize, seg=self.segment, stream=stream, xp=xp)

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

            if self.segment:
                ins = [dic.get_token_id(char) for char in line]
            else:
                array = line.split(' ')
                ins = [self.model.dic.get_token_id(word) for word in array]

            xs = [xp.asarray(ins, dtype=np.int32)]
            decode_batch(xs, self.model, seg=self.segment)
            print()


    def run_training(self):
        xp = cuda.cupy if args.gpu >= 0 else np

        n_train = len(self.train)
        n_iter_report = self.args.iteration_to_report
        n_iter = 0

        for e in range(max(1, self.args.epoch_to_resume), self.args.epoch+1):
            time = datetime.now().strftime('%Y%m%d_%H%M')
            self.log('INFO: start epoch %d at %s\n' % (e, time))
            print('start epoch %d: %s' % (e, time))

            # learning rate decay

            if self.args.lrdecay and self.args.optimizer == 'sgd':
                if (e - lrdecay_start) % lrdecay_width == 0:
                    lr_tmp = optimizer.lr
                    optimizer.lr = optimizer.lr * lrdecay_rate
                    if not self.args.nolog:
                        self.logger.write('learning decay: %f -> %f' % (lr_tmp, optimizer.lr))
                    print('learning decay: %f -> %f' % (lr_tmp, optimizer.lr))

            count = 0
            num_tokens = 0
            total_loss = 0
            total_ecounts = conlleval.EvalCounts()

            i = 0
            for xs, ts in batch_generator(self.train, self.train_t, label_seqs2=None, 
                                          batchsize=self.args.batchsize, shuffle=True, xp=xp):

                t0 = datetime.now()
                loss, ecounts = self.model(xs, ts, train=True)
                t1 = datetime.now()

                num_tokens += sum([len(x) for x in xs])
                count += len(xs)
                total_loss += loss.data
                total_ecounts = conlleval.merge_counts(total_ecounts, ecounts)
                i_max = min(i + self.args.batchsize, n_train)
                print('* batch %d-%d loss: %.4f' % ((i+1), i_max, loss.data))
                i = i_max
                n_iter += 1

                t2 = datetime.now()
                optimizer.target.cleargrads() # Clear the parameter gradients
                loss.backward()               # Backprop
                loss.unchain_backward()       # Truncate the graph
                optimizer.update()            # Update the parameters
                t3 = datetime.now()
                # print('train    {} ins: {}'.format((t1-t0).seconds+(t1-t0).microseconds/10**6, len(xs)))
                # print('backprop {} ins: {}'.format((t3-t2).seconds+(t3-t2).microseconds/10**6, len(xs)))
                # print('total    {} ins: {}'.format((t3-t0).seconds+(t3-t0).microseconds/10**6, len(xs)))

                # Evaluation
                if (n_iter * self.args.batchsize) % n_iter_report == 0: # or i == n_train:

                    now_e = '%.3f' % (n_iter * self.args.batchsize / n_train)
                    time = datetime.now().strftime('%Y%m%d_%H%M')
                    print('\n### iteration %s (epoch %s)' % ((n_iter * self.args.batchsize), now_e))
                    print('<training result for previous iterations>')

                    ave_loss = total_loss / num_tokens
                    c = total_ecounts
                    acc = conlleval.calculate_accuracy(c.correct_tags, c.token_counter)
                    overall = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)
                    print_results(total_loss, ave_loss, count, c, acc, overall)
                    print()

                    if not self.args.nolog:
                        t_stat = 'n_sen: %d, n_token: %d, n_chunk: %d, n_chunk_p: %d' % (
                            count, c.token_counter, c.found_correct, c.found_guessed)
                        t_res = '%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%d\t%.4f\t%.4f' % (
                            (100.*acc), (100.*overall.prec), (100.*overall.rec), (100.*overall.fscore), 
                            overall.tp, overall.fp, overall.fn, total_loss, ave_loss)
                        self.logger.write('INFO: train - %s\n' % t_stat)
                        if n_iter == 1:
                            self.logger.write(
                                'data\titer\tep\tacc\tprec\trec\tfb1\tTP\tFP\tFN\ttloss\taloss\n')
                        self.logger.write('train\t%d\t%s\t%s\n' % (n_iter, now_e, t_res))

                    if self.args.validation_data:
                        print('<validation result>')
                        v_stat, v_res = evaluate(
                            self.model, self.val, self.val_t, batchsize=self.args.batchsize, xp=xp)
                        print()

                        if not self.args.nolog:
                            self.logger.write('INFO: valid - %s\n' % v_stat)
                            if n_iter == 1:
                                self.logger.write(
                                    'data\titer\tep\tacc\tprec\trec\tfb1\tTP\tFP\tFN\ttloss\taloss\n')
                            self.logger.write('valid\t%d\t%s\t%s\n' % (n_iter, now_e, v_res))

                    # Save the model
                    #mdl_path = '{}/nn_model/nn_{}_i{}.mdl'.format(HOME_DIR, self.start_time, now_e)
                    mdl_path = '{}/nn_{}_i{}.mdl'.format(MODEL_DIR, self.start_time, now_e)
                    if not self.args.nolog:
                        print('save the model: %s\n' % mdl_path)
                        self.logger.write('INFO: save the model: %s\n' % mdl_path)
                        serializers.save_npz(mdl_path, self.model)
        
                    # Reset counters
                    count = 0
                    num_tokens = 0
                    total_loss = 0
                    total_ecounts = conlleval.EvalCounts()

                    if not self.args.nolog:
                        self.logger.close() # 一度保存しておく
                        self.logger = open('{}/{}.log'.format(LOG_DIR, self.start_time), 'a')

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.log('finish: %s\n' % time)


class Trainer4JointMA(Trainer):
    def __init__(self, args):
        super(Trainer4JointMA, self).__init__(args)


    def run_training(self):
        xp = cuda.cupy if args.gpu >= 0 else np

        n_train = len(self.train)
        n_iter_report = self.args.iter_to_report
        n_iter = 0

        for e in range(max(1, self.args.epoch_to_resume), self.args.epoch+1):
            time = datetime.now().strftime('%Y%m%d_%H%M')
            self.log('INFO: start epoch %d at %s\n' % (e, time))
            print('start epoch %d: %s' % (e, time))

            # learning rate decay: not implemented

            count = 0
            num_tokens = 0
            total_loss = 0
            total_ecounts_seg = conlleval.EvalCounts()
            total_ecounts_pos = conlleval.EvalCounts()

            i = 0
            for xs, ts_seg, ts_pos in batch_generator(
                    self.train, self.train_t, self.train_p, self.args.batchsize, shuffle=False, xp=xp):

                #self.model.predictor.lattice_crf.debug = True
                #self.dic.chunk_trie.debug = True
                
                t0 = datetime.now()
                loss, ecounts_seg, ecounts_pos = self.model(xs, ts_seg, ts_pos, train=True)
                t1 = datetime.now()

                num_tokens += sum([len(x) for x in xs])
                count += len(xs)
                total_loss += loss.data
                total_ecounts_seg = conlleval.merge_counts(total_ecounts_seg, ecounts_seg)
                total_ecounts_pos = conlleval.merge_counts(total_ecounts_pos, ecounts_pos)
                i_max = min(i + self.args.batchsize, n_train)
                print('* batch %d-%d loss: %.4f' % ((i+1), i_max, loss.data))
                i = i_max
                n_iter += 1

                t2 = datetime.now()
                optimizer.target.cleargrads() # Clear the parameter gradients
                loss.backward()               # Backprop
                loss.unchain_backward()       # Truncate the graph
                optimizer.update()            # Update the parameters
                t3 = datetime.now()
                # print('train    {} ins: {}'.format((t1-t0).seconds+(t1-t0).microseconds/10**6, len(xs)))
                # print('backprop {} ins: {}'.format((t3-t2).seconds+(t3-t2).microseconds/10**6, len(xs)))
                # print('total    {} ins: {}'.format((t3-t0).seconds+(t3-t0).microseconds/10**6, len(xs)))

                # Evaluation
                if (n_iter * self.args.batchsize) % n_iter_report == 0:

                    now_e = '%.3f' % (n_iter * self.args.batchsize / n_train)
                    time = datetime.now().strftime('%Y%m%d_%H%M')
                    print('\n### iteration %s (epoch %s)' % ((n_iter * self.args.batchsize), now_e))
                    print('<training result for previous iterations>')

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

                    if not self.args.nolog:
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

                    if self.args.validation_data:
                        print('<validation result>')
                        v_stat, v_res = evaluate(
                            self.model, self.val, self.val_t, self.val_p, self.args.batchsize, xp=xp)
                        print()

                        if not self.args.nolog:
                            self.logger.write('INFO: valid - %s\n' % v_stat)
                            if n_iter == 1:
                                self.logger.write('data\titer\tep\t\s-acc\ts-prec\ts-rec\ts-fb1\t\p-acc\tp-prec\tp-rec\tp-fb1\tloss\taloss\n')
                            self.logger.write('valid\t%d\t%s\t%s\n' % (n_iter, now_e, v_res))

                    # Save the model
                    mdl_path = 'nn_model/rnn_%s_i%s.mdl' % (self.start_time, now_e)
                    if not self.args.nolog:
                        print('save the model: %s\n' % mdl_path)
                        self.logger.write('INFO: save the model: %s\n' % mdl_path)
                        serializers.save_npz(mdl_path, self.model)
        
                    # Reset counters
                    count = 0
                    total_loss = 0
                    num_tokens = 0
                    total_ecounts = conlleval.EvalCounts()

                    if not self.args.nolog:
                        self.logger.close() # 一度保存しておく
                        self.logger = open('{}/{}.log'.format(LOG_DIR, self.start_time), 'a')

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.log('finish: %s\n' % time)


if __name__ == '__main__':
    if chainer.__version__[0] != '2':
        print("chainer version>=2.0.0 is required.")
        sys.exit()

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # get arguments and set up trainer

    args = parse_arguments()
    # if args.joint_type == 'lattice':
    #     trainer = Trainer4JointMA(args)
    #     read_pos = True
    # else:

    trainer = Trainer(args)
    read_pos = False
    trainer.prepare_gpu()

    # Load word embedding model, dictionary and dataset

    embed_model = emb.read_model(args.embed_model_path) if args.embed_model_path else None

    if args.dict_path:
        dic_path = args.dict_path
        dic = lattice.load_dictionary(dic_path, read_pos=read_pos)
        print('load dic:', dic_path)
        print('vocab size:', len(dic.token_indices))

        if not dic_path.endswith('pickle'):
            base = dic_path.split('.')[0]
            dic_pic_path = base + '.pickle'
            with open(dic_pic_path, 'wb') as f:
                pickle.dump(dic, f)
    else:
        dic = None

    trainer.load_data(dic, embed_model)

    # Set up model and optimizer

    model = trainer.prepare_model_and_hparameters(embed_model=embed_model)
    optimizer = trainer.setup_optimizer()

    # Run

    trainer.run()
    if not args.nolog:
        trainer.logger.close()
