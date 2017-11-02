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

import data_io
import lattice
import models
import features
import embedding.read_embedding as emb
from tools import conlleval


LOG_DIR = 'log'
MODEL_DIR = 'models/main'
OUT_DIR = 'out'


def batch_generator(len_data, batchsize=100, shuffle=True):
    perm = np.random.permutation(len_data) if shuffle else list(range(0, len_data))
    for i in range(0, len_data, batchsize):
        i_max = min(i + batchsize, len_data)
        yield perm[i:i_max+1]

    raise StopIteration


def show_results(total_loss, ave_loss, count, c, acc, overall, acc2=None, overall2=None, 
                  stream=sys.stderr):
    print('total loss: %.4f' % total_loss, file=stream)
    print('ave loss: %.5f'% ave_loss, file=stream)
    print('sen, token, chunk, chunk_pred: {} {} {} {}'.format(
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
    # used letters: bcedfglmnopqrstvx

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
    parser.add_argument('--evaluation_size', type=int, default=10000, 
                        help='The number of examples which'
                        + ' trained model is evaluated and saved at each multiple of'
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
                        + ' \'seg_tag\' (word segmentation w/ POS tagging on data '
                        + ' where \'words_pos\' pairs are split with single spaces),'
                        + ' \'bccwj_seg\' (word segmentation on data with BCCWJ format),'
                        + ' \'bccwj_seg_tag\' (word segmentation w/ POS tagging on data with BCCWJ format),'
                        + ' \'bccwj_tag\' (POS tagging from gold words on data with BCCWJ format),'
                        # + ' \'wsj\' (POS tagging on data with CoNLL-2005 format),'
                        # + ' \'conll2003\' (NER on data with CoNLL-2003 format),'
    )
    parser.add_argument('--subpos_depth', '-s', type=int, default=-1,
                        help='Set positive integer (use POS up to i-th hierarcy)'
                        + ' or other value (use POS with all sub POS) when set \'bccwj_seg_tag\''
                        + ' to data_format')
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
    parser.add_argument('--feature_template', default='',
                        help='Use dictionary features based on given feature template file.'
                        + ' Specify \'defualt\' to use default features defined for each task,'
                        + ' or specify the path of castomized template file.')
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


class Trainer(object):
    def __init__(self, args, logger=sys.stderr):
        err_msgs = []
        if args.execute_mode == 'train':
            if not args.train_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    'train', '--train_data/-t')
                err_msgs.append(msg)
            if not args.data_format and not args.model_path:
                msg = 'Error: the following argument is required for {} mode unless resuming a trained model: {}'.format('train', '--data_format/-f')
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
        self.val = None
        self.test = None
        self.hparams = None
        self.tagger = None
        self.feat_extractor = None
        self.optimizer = None
        self.decode_type = None
        self.n_iter = 0

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
    

    def init_feat_extractor(self, use_gpu=False):
        template = self.hparams['feature_template']
        feat_dim = 0

        if template:
            self.feat_extractor = features.DictionaryFeatureExtractor(template, use_gpu=use_gpu)

            if 'add_feat_dimension' in self.hparams:
                feat_dim = self.hparams['add_feat_dimension']
            else:
                feat_dim = self.feat_extractor.dim

        self.hparams.update({'add_feat_dimension' : feat_dim})
                
        

    def init_hyperparameters(self, args):
        hparams = {
            'embed_dimension' : args.embed_dimension,
            'rnn_layers' : args.rnn_layers,
            'rnn_hidden_units' : args.rnn_hidden_units,
            'rnn_unit_type' : args.rnn_unit_type,
            'rnn_bidirection' : args.rnn_bidirection,
            'inference_layer' : args.inference_layer,
            'dropout' : args.dropout,
            'feature_template' : args.feature_template,
            'data_format' : args.data_format,
            'subpos_depth' : args.subpos_depth,
            'lowercase' : args.lowercase,
            'normalize_digits' : args.normalize_digits,
        }
        self.finalize_hparams(hparams)

        self.log('Init hyperparameters')
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def finalize_hparams(self, hparams):
        self.hparams = hparams

        if (hparams['data_format'] == 'wl_seg' or
            hparams['data_format'] == 'sl_seg'):
            self.decode_type = 'seg'

        elif (hparams['data_format'] == 'wl_seg_tag' or
              hparams['data_format'] == 'sl_seg_tag'):
            self.decode_type = 'seg_tag'

        else:
            self.decode_type = 'tag'


    def set_tagger(self, tagger):
        self.tagger = tagger


    def load_model(self, model_path, use_gpu=False):
        array = model_path.split('_e')
        indices_path = '{}.s2i'.format(array[0])
        hparam_path = '{}.hyp'.format(array[0])
        param_path = model_path
    
        with open(indices_path, 'rb') as f:
            indices = pickle.load(f)
        self.log('Load indices: {}'.format(indices_path))
        self.log('Vocab size: {}\n'.format(len(indices.token_indices)))

        # hyper parameters
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
                    key == 'add_feat_dimension' or
                    key == 'rnn_layers' or
                    key == 'rnn_hidden_units' or
                    key == 'subpos_depth'
                ):
                    val = int(val)

                elif key == 'dropout':
                    val = float(val)

                elif (key == 'rnn_bidirection' or
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

        # model
        self.log('Initialize model from hyperparameters\n')
        tagger = models.init_tagger(indices, hparams, use_gpu=use_gpu)
        chainer.serializers.load_npz(model_path, tagger)
        self.log('Load model parameters: {}\n'.format(model_path))

        self.finalize_hparams(hparams)
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
            _, indices = data_io.load_data(
                hparams['data_format'], refer_path, read_pos=read_pos, update_token=False,
                ws_dict_feat=hparams['feature_template'],
                subpos_depth=hparams['subpos_depth'], lowercase=hparams['lowercase'],
                normalize_digits=hparams['normalize_digits'], indices=indices)
            self.log('Load label set from reference data: {}'.format(refer_path))

        if args.train_data.endswith('pickle'):
            name, ext = os.path.splitext(train_path)
            train = data_io.load_pickled_data(name)
            self.log('Load dumpped data:'.format(train_path))
            #TODO 別データで再学習の場合は要エラー処理

        else:
            train, indices = data_io.load_data(
            hparams['data_format'], train_path, read_pos=read_pos,
            ws_dict_feat=hparams['feature_template'],
            subpos_depth=hparams['subpos_depth'], lowercase=hparams['lowercase'],
            normalize_digits=hparams['normalize_digits'], indices=indices, refer_vocab=refer_vocab)
        
        if self.args.dump_train_data:
            name, ext = os.path.splitext(train_path)
            data_io.dump_pickled_data(name, train)
            self.log('Dumpped training data: {}.pickle'.format(train_path))

        self.log('Load training data: {}'.format(train_path))
        self.log('Data length: {}'.format(len(train.instances)))
        self.log('Vocab size: {}\n'.format(len(indices.token_indices)))
        if self.hparams['feature_template']:
            self.log('Start feature extraction for training data\n')
            self.extract_features(train, indices)

        if args.valid_data:
            # indices can be updated if embed_model are used
            val, indices = data_io.load_data(
                hparams['data_format'], val_path, read_pos=read_pos, update_token=False, update_label=False,
                ws_dict_feat=hparams['feature_template'],
                subpos_depth=hparams['subpos_depth'], lowercase=hparams['lowercase'],
                normalize_digits=hparams['normalize_digits'], indices=indices, refer_vocab=refer_vocab)

            self.log('Load validation data: {}'.format(val_path))
            self.log('Data length: {}'.format(len(val.instances)))
            self.log('Vocab size: {}\n'.format(len(indices.token_indices)))
            if self.hparams['feature_template']:
                self.log('Start feature extraction for validation data\n')
                self.extract_features(val, indices)
        else:
            val = None
        
        self.train = train
        self.val = val

        n_train = len(train.instances)
        t2i_tmp = list(indices.token_indices.token2id.items())
        id2label = {v:k for k,v in indices.label_indices.token2id.items()}
        # if train_p:
        #     id2pos = {v:k for k,v in indices.pos_indices.token2id.items()}

        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.instances[:3], train.instances[n_train-3:]))
        self.log('# train_gold: {} ... {}\n'.format(train.labels[0][:3], train.labels[0][n_train-3:]))
        # if train_p:
        #     self.log('# POS labels: {} ... {}\n'.format(train_p[:3], train_p[n_train-3:]))
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))
        self.log('# labels: {}\n'.format(id2label))
        # if train_p:
        #     self.log('poss: {}\n'.format(id2pos))
        
        self.report('[INFO] vocab: {}'.format(len(indices.token_indices)))
        self.report('[INFO] data length: train={} val={}'.format(
            len(train.instances), len(val.instances) if val else 0))

        return indices


    def load_data_for_test(self, indices=None, embed_model=None):
        args = self.args
        hparams = self.hparams
        test_path = (args.path_prefix if args.path_prefix else '') + args.test_data
        refer_vocab = embed_model.wv if embed_model else set()
        read_pos = hparams['subpos_depth'] != 0
            
        test, indices = data_io.load_data(
            hparams['data_format'], test_path, read_pos=read_pos, update_token=False, update_label=False,
            ws_dict_feat=hparams['feature_template'],
            subpos_depth=hparams['subpos_depth'], lowercase=hparams['lowercase'],
            normalize_digits=hparams['normalize_digits'], indices=indices, refer_vocab=refer_vocab)

        self.log('Load test data: {}'.format(test_path))
        self.log('Data length: {}'.format(len(test.instances)))
        self.log('Vocab size: {}\n'.format(len(indices.token_indices)))
        if self.hparams['feature_template']:
            self.log('Start feature extraction for test data\n')
            self.extract_features(test, indices)

        self.test = test

        return indices


    def extract_features(self, data, indices):
        features = self.feat_extractor.extract_features(data.instances, indices)
        data.set_features(features)


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


    def decode(self, data, stream=sys.stdout):
        xp = cuda.cupy if args.gpu >= 0 else np

        n_ins = len(data.instances)
        for ids in batch_generator(n_ins, batchsize=self.args.batchsize, shuffle=False):
            xs = [xp.asarray(data.instances[j], dtype=np.int32) for j in ids]
            if self.hparams['feature_template']:
                fs = [data.features[j] for j in ids]
            else:
                fs = None

            self.decode_batch(xs, fs)


    def decode_batch(self, xs, fs, stream=sys.stdout):
        ys = self.tagger.decode(xs, fs)

        for x, y in zip(xs, ys):
            x_str = [self.tagger.indices.get_token(int(xi)) for xi in x]
            y_str = [self.tagger.indices.get_label(int(yi)) for yi in y]

            if self.decode_type == 'tag':
                res = ['{}/{}'.format(xi_str, yi_str) for xi_str, yi_str in zip(x_str, y_str)]
                res = ' '.join(res)

            elif self.decode_type == 'seg':
                res = ['{}{}'.format(xi_str, ' ' if (yi_str == 'E' or yi_str == 'S') else '') for xi_str, yi_str in zip(x_str, y_str)]
                res = ''.join(res).rstrip()

            elif self.decode_type == 'seg_tag':
                res = ['{}{}'.format(
                    xi_str, 
                    ('/'+yi_str[2:]+' ') if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
                ) for xi_str, yi_str in zip(x_str, y_str)]
                res = ''.join(res).rstrip()

            else:
                print('Error: Invalid decode type', file=self.logger)
                sys.exit()

            print(res, file=stream)


    def run_epoch(self, data, train=False):
        xp = cuda.cupy if args.gpu >= 0 else np

        if train:
            tagger = self.tagger
        else:
            tagger = self.tagger.copy()

        count = 0
        num_tokens = 0
        total_loss = 0
        total_ecounts = conlleval.EvalCounts()
        #total_ecounts_pos = conlleval.EvalCounts()

        i = 0
        n_ins = len(data.instances)
        shuffle = True if train else False
        for ids in batch_generator(n_ins, batchsize=self.args.batchsize, shuffle=shuffle):
            xs = [xp.asarray(data.instances[j], dtype=np.int32) for j in ids]
            ts = [xp.asarray(data.labels[0][j], dtype=np.int32) for j in ids]
            if self.hparams['feature_template']:
                fs = [data.features[j] for j in ids]
            else:
                fs = None

            if train:
                loss, ecounts = tagger(xs, fs, ts)

                self.optimizer.target.cleargrads() # Clear the parameter gradients
                loss.backward()                    # Backprop
                loss.unchain_backward()            # Truncate the graph
                self.optimizer.update()            # Update the parameters

                i_max = min(i + self.args.batchsize, n_ins)
                self.log('* batch %d-%d loss: %.4f' % ((i+1), i_max, loss.data))
                i = i_max
                self.n_iter += 1

            else:
                with chainer.no_backprop_mode():
                    loss, ecounts = tagger(xs, fs, ts)

            num_tokens += sum([len(x) for x in xs])
            count += len(xs)
            total_loss += loss.data
            total_ecounts = conlleval.merge_counts(total_ecounts, ecounts)
            # if eval_pos:
            #     total_ecounts_pos = conlleval.merge_counts(total_ecounts_pos, ecounts_pos)

            if train and ((self.n_iter * self.args.batchsize) % self.args.evaluation_size == 0):
                now_e = '%.3f' % (self.n_iter * self.args.batchsize / n_ins)
                time = datetime.now().strftime('%Y%m%d_%H%M')

                self.log('\n### Finish %s iterations (%s examples: %s epoch)' % (
                    self.n_iter, (self.n_iter * self.args.batchsize), now_e))
                self.log('<training result for previous iterations>')            
                stat, res = self.evaluate(count, num_tokens, total_ecounts, total_loss)
                self.report('[INFO] train - %s' % stat)
                self.report('train\t%d\t%s\t%s' % (self.n_iter, now_e, res))

                if self.args.valid_data:
                    self.log('<validation result>')
                    v_stat, v_res = self.run_epoch(self.val, train=False)
                    self.report('[INFO] valid - %s' % v_stat)
                    self.report('valid\t%d\t%s\t%s' % (self.n_iter, now_e, v_res))

                # Save the model
                if not self.args.quiet:
                    mdl_path = '{}/{}_e{}.npz'.format(MODEL_DIR, self.start_time, now_e)
                    self.log('Save the model: %s\n' % mdl_path)
                    self.report('[INFO] save the model: %s\n' % mdl_path)
                    serializers.save_npz(mdl_path, self.tagger)
        
                if not self.args.quiet:
                    self.reporter.close() 
                    self.reporter = open('{}/{}.log'.format(LOG_DIR, self.start_time), 'a')

                # Reset counters
                count = 0
                num_tokens = 0
                total_loss = 0
                total_ecounts = conlleval.EvalCounts()

        if not train:
            stat, res = self.evaluate(count, num_tokens, total_ecounts, total_loss)
        else:
            stat = res = None
    
        return stat, res


    def evaluate(self, count, num_tokens, total_ecounts, total_loss):
        ave_loss = total_loss / num_tokens
        c = total_ecounts
        acc = conlleval.calculate_accuracy(c.correct_tags, c.token_counter)
        overall = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)
        show_results(total_loss, ave_loss, count, c, acc, overall, stream=self.logger)

        # if eval_pos:
        #     res = '%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f' % (
        #         (100.*seg_acc), (100.*seg_overall.prec), (100.*seg_overall.rec), 
        #         (100.*seg_overall.fscore), total_loss, ave_loss)
        # else:

        stat = 'n_sen: %d, n_token: %d, n_chunk: %d, n_chunk_p: %d' % (
            count, c.token_counter, c.found_correct, c.found_guessed)
        res = '%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f' % (
            (100.*acc), (100.*overall.prec), (100.*overall.rec), 
            (100.*overall.fscore), total_loss, ave_loss)

        return stat, res


    def run_train_mode(self):
        # save model
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

        #self.report('data\titer\tep\tacc\tprec\trec\tfb1\tTP\tFP\tFN\ttloss\taloss')

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

            # learning and evaluation for each epoch
            self.run_epoch(self.train, train=True)


        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('Finish: %s\n' % time)


    def run_eval_mode(self):
        self.log('<test result>')
        self.run_epoch(self.test, train=False)
        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.log('Finish: %s\n' % time)
        self.report('Finish: %s\n' % time)


    def run_decode_mode(self):
        xp = cuda.cupy if args.gpu >= 0 else np

        raw_path = (args.path_prefix if args.path_prefix else '') + self.args.raw_data
        if self.decode_type == 'tag':
            instances = data_io.load_raw_text_for_tagging(raw_path, self.tagger.indices)
        else:
            instances = data_io.load_raw_text_for_segmentation(raw_path, self.tagger.indices)
        data = data_io.Data(instances, None)

        if self.hparams['feature_template']:
            self.extract_features(data, self.tagger.indices)

        stream = open(self.args.output_path, 'w') if self.args.output_path else sys.stdout
        self.decode(data, stream=stream)

        if self.args.output_path:
            stream.close()


    def run_interactive_mode(self):
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

            if self.hparams['feature_template']:
                fs = self.feat_extractor.extract_features([ins], self.tagger.indices)
            else:
                fs = None

            self.decode_batch(xs, fs)
            print()


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
    # Load feature extractor and tagger model

    if args.model_path:
        trainer.load_model(args.model_path, use_gpu=use_gpu)
        indices = trainer.tagger.indices
        id2token_org = copy.deepcopy(trainer.tagger.indices.id2token)
        id2label_org = copy.deepcopy(trainer.tagger.indices.id2label)
    else:
        trainer.init_hyperparameters(args)
        indices = None
        id2token_org = {}
        id2label_org = {}
    trainer.init_feat_extractor(use_gpu=use_gpu)


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
        trainer.run_train_mode()
    elif args.execute_mode == 'eval':
        trainer.run_eval_mode()
    elif args.execute_mode == 'decode':
        trainer.run_decode_mode()
    elif args.execute_mode == 'interactive':
        trainer.run_interactive_mode()

    ################################
    # Terminate

    trainer.close()
