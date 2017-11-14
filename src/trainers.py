import sys
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

import util
import constants
import data_io
import classifiers
import evaluators
import features
import embedding.read_embedding as emb
from tools import conlleval


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
        self.indices = None
        self.classifier = None
        self.evaluator = None
        self.feat_extractor = None
        self.optimizer = None
        self.decode_type = None
        self.n_iter = 0

        self.log('Start time: {}\n'.format(self.start_time))
        if not self.args.quiet:
            self.reporter = open('{}/{}.log'.format(constants.LOG_DIR, self.start_time), 'a')


    def report(self, message):
        if not self.args.quiet:
            print(message, file=self.reporter)


    def log(self, message):
        print(message, file=self.logger)


    def close(self):
        if not self.args.quiet:
            self.reporter.close()


    def init_feat_extractor(self, use_gpu=False):
        if 'feature_template' in self.hparams:
            template = self.hparams['feature_template']
            feat_dim = 0

            if template:
                self.feat_extractor = features.DictionaryFeatureExtractor(template, use_gpu=use_gpu)

                if 'additional_feat_dim' in self.hparams:
                    feat_dim = self.hparams['additional_feat_dim']
                else:
                    feat_dim = self.feat_extractor.dim

            self.hparams.update({'additional_feat_dim' : feat_dim})


    def init_model(self):
        self.log('Initialize model from hyperparameters\n')
        self.classifier = classifiers.init_classifier(
            self.decode_type, self.hparams, self.indices)
        self.evaluator = classifiers.init_evaluator(self.decode_type, self.indices)


    def load_model(self, model_path):
        array = model_path.split('_e')
        indices_path = '{}.s2i'.format(array[0])
        hparam_path = '{}.hyp'.format(array[0])
        param_path = model_path
    
        with open(indices_path, 'rb') as f:
            self.indices = pickle.load(f)
        self.log('Load indices: {}'.format(indices_path))
        self.log('Vocab size: {}\n'.format(len(self.indices.token_indices)))

        # hyper parameters
        self.hparams = self.load_hyperparameters(hparam_path)
        self.log('Load hyperparameters: {}\n'.format(hparam_path))
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            if k in self.hparams and v != self.hparams[k]:
                message = '{}={} (input option value {} was discarded)'.format(k, self.hparams[k], v)
            else:
                message = '{}={}'.format(k, v)

            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')

        # model
        self.init_model()
        chainer.serializers.load_npz(model_path, self.classifier)
        self.log('Load model parameters: {}\n'.format(model_path))


    def init_hyperparameters(self, args):
        # to be implemented in sub-class
        self.hparams = {}
        

    def load_hyperparameters(self, hparams_path):
        # to be implemented in sub-class
        return {}


    def check_data_format(self, data_format):
        # to be implemented in sub-class
        pass


    def show_training_data(self):
        # to be implemented in sub-class
        pass

    def load_data_for_training(self, embed_model=None):
        args = self.args
        hparams = self.hparams
        prefix = args.path_prefix if args.path_prefix else ''
            
        refer_path   = prefix + args.label_reference_data
        train_path   = prefix + args.train_data
        val_path     = prefix + (args.valid_data if args.valid_data else '')
        refer_vocab = embed_model.wv if embed_model else set()

        data_format = hparams['data_format']
        read_pos = hparams['subpos_depth'] != 0
        create_word_trie = ('feature_template' in hparams and hparams['feature_template'])
        subpos_depth = hparams['subpos_depth'] if 'subpos_depth' in hparams else -1
        lowercase = hparams['lowercase'] if 'lowercase' in hparams else False
        normalize_digits = hparams['normalize_digits'] if 'normalize_digits' else False
        
        if args.label_reference_data:
            _, self.indices = data_io.load_data(
                data_format, refer_path, read_pos=read_pos, update_token=False,
                create_word_trie=create_word_trie, subpos_depth=subpos_depth, 
                lowercase=lowercase, normalize_digits=normalize_digits, 
                indices=self.indices)
            self.log('Load label set from reference data: {}'.format(refer_path))

        if args.train_data.endswith('pickle'):
            name, ext = os.path.splitext(train_path)
            train = data_io.load_pickled_data(name)
            self.log('Load dumpped data:'.format(train_path))
            #TODO error checking for re-training from another training data

        else:
            train, self.indices = data_io.load_data(
                data_format, train_path, read_pos=read_pos,
                create_word_trie=create_word_trie, subpos_depth=subpos_depth, 
                lowercase=lowercase, normalize_digits=normalize_digits, 
                indices=self.indices, refer_vocab=refer_vocab)
        
        if self.args.dump_train_data:
            name, ext = os.path.splitext(train_path)
            data_io.dump_pickled_data(name, train)
            self.log('Dumpped training data: {}.pickle'.format(train_path))

        self.log('Load training data: {}'.format(train_path))
        self.log('Data length: {}'.format(len(train.instances)))
        self.log('Vocab size: {}\n'.format(len(self.indices.token_indices)))
        if 'featre_template' in self.hparams and self.hparams['feature_template']:
            self.log('Start feature extraction for training data\n')
            self.extract_features(train)

        if args.valid_data:
            # indices can be updated if embed_model are used
            val, self.indices = data_io.load_data(
                data_format, val_path, read_pos=read_pos, update_token=False, update_label=False,
                create_word_trie=create_word_trie, subpos_depth=subpos_depth, 
                lowercase=lowercase, normalize_digits=normalize_digits, 
                indices=self.indices, refer_vocab=refer_vocab)

            self.log('Load validation data: {}'.format(val_path))
            self.log('Data length: {}'.format(len(val.instances)))
            self.log('Vocab size: {}\n'.format(len(self.indices.token_indices)))
            if 'featre_template' in self.hparams and self.hparams['feature_template']:
                self.log('Start feature extraction for validation data\n')
                self.extract_features(val)
        else:
            val = None
        
        self.train = train
        self.val = val
        self.show_training_data()


    def load_data_for_test(self, embed_model=None):
        args = self.args
        hparams = self.hparams
        test_path = (args.path_prefix if args.path_prefix else '') + args.test_data
        refer_vocab = embed_model.wv if embed_model else set()

        data_format = hparams['data_format']
        read_pos = hparams['subpos_depth'] != 0
        create_word_trie = ('feature_template' in hparams and hparams['feature_template'])
        subpos_depth = hparams['subpos_depth'] if 'subpos_depth' in hparams else -1
        lowercase = hparams['lowercase'] if 'lowercase' in hparams else False
        normalize_digits = hparams['normalize_digits'] if 'normalize_digits' else False
            
        test, self.indices = data_io.load_data(
            data_format, test_path, read_pos=read_pos, update_token=False, update_label=False,
            ws_dict_feat=hparams['feature_template'],
            subpos_depth=hparams['subpos_depth'], lowercase=hparams['lowercase'],
            normalize_digits=hparams['normalize_digits'], indices=self.indices, refer_vocab=refer_vocab)

        self.log('Load test data: {}'.format(test_path))
        self.log('Data length: {}'.format(len(test.instances)))
        self.log('Vocab size: {}\n'.format(len(self.indices.token_indices)))
        if self.hparams['feature_template']:
            self.log('Start feature extraction for test data\n')
            self.extract_features(test)

        self.test = test


    def extract_features(self, data):
        features = self.feat_extractor.extract_features(data.instances, self.indices)
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

        optimizer.setup(self.classifier)
        optimizer.add_hook(chainer.optimizer.GradientClipping(self.args.gradclip))
        if self.args.weightdecay > 0:
            optimizer.add_hook(chainer.optimizer.WeightDecay(self.args.weightdecay))

        self.optimizer = optimizer

        return self.optimizer


    def run_train_mode(self):
        # save model
        if not self.args.quiet:
            hparam_path = '{}/{}.hyp'.format(constants.MODEL_DIR, self.start_time)
            with open(hparam_path, 'w') as f:
                for key, val in self.hparams.items():
                    print('{}={}'.format(key, val), file=f)
                self.log('save hyperparameters: {}'.format(hparam_path))

            indices_path = '{}/{}.s2i'.format(constants.MODEL_DIR, self.start_time)
            with open(indices_path, 'wb') as f:
                pickle.dump(self.indices, f)
            self.log('save string2index table: {}'.format(indices_path))

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
        res = self.run_epoch(self.test, train=False)
        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('test\t%s\n' % res)
        self.report('Finish: %s\n' % time)
        self.log('Finish: %s\n' % time)


    def run_decode_mode(self):
        xp = cuda.cupy if args.gpu >= 0 else np

        raw_path = (args.path_prefix if args.path_prefix else '') + self.args.raw_data
        if self.decode_type == 'tag':
            instances = data_io.load_raw_text_for_tagging(raw_path, self.classifier.indices)
        else:
            instances = data_io.load_raw_text_for_segmentation(raw_path, self.classifier.indices)
        data = data_io.Data(instances, None)

        if 'feature_template' in self.hparams and self.hparams['feature_template']:
            self.extract_features(data, self.classifier.indices)

        stream = open(self.args.output_path, 'w') if self.args.output_path else sys.stdout
        self.decode(data, stream=stream)

        if self.args.output_path:
            stream.close()


    # TODO modify
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
                ins = [self.classifier.indices.get_token_id(word) for word in array]
            else:
                ins = [self.classifier.indices.get_token_id(char) for char in line]
            xs = [xp.asarray(ins, dtype=np.int32)]

            if self.hparams['feature_template']:
                fs = self.feat_extractor.extract_features([ins], self.classifier.indices)
            else:
                fs = None

            self.decode_batch(xs, fs)
            print()


    def gen_inputs(self, data, ids):
        # to be implemented in sub-class
        return None


    def run_epoch(self, data, train=False):
        xp = cuda.cupy if self.args.gpu >= 0 else np

        if train:
            classifier = self.classifier
        else:
            classifier = self.classifier.copy()

        n_sen = 0
        total_loss = 0
        total_counts = None

        i = 0
        n_ins = len(data.instances)
        shuffle = True if train else False
        for ids in batch_generator(n_ins, batchsize=self.args.batchsize, shuffle=shuffle):
            inputs = self.gen_inputs(data, ids)
            xs = inputs[0]
            golds = inputs[2:]
            # timer = util.Timer()
            # timer.start()
            if train:
                ret = self.classifier(*inputs, train=True)
            else:
                with chainer.no_backprop_mode():
                    ret = self.classifier(*inputs)
            loss = ret[0]
            outputs = ret[1:]
            # timer.stop()
            # print('time: predict:', timer.elapsed)
            # timer.reset()

            if train:
                # timer.start()
                self.optimizer.target.cleargrads() # Clear the parameter gradients
                loss.backward()                    # Backprop
                loss.unchain_backward()            # Truncate the graph
                self.optimizer.update()            # Update the parameters
                # timer.stop()
                # print('time: backprop', timer.elapsed)
                # timer.reset()

                i_max = min(i + self.args.batchsize, n_ins)
                self.log('* batch %d-%d loss: %.4f' % ((i+1), i_max, loss.data))
                i = i_max
                self.n_iter += 1

            n_sen += len(xs)
            total_loss += loss.data
            counts = self.evaluator.calculate(*[xs], *golds, *outputs)
            total_counts = evaluators.merge_counts(total_counts, counts)

            if train and ((self.n_iter * self.args.batchsize) % self.args.evaluation_size == 0):
                now_e = '%.3f' % (self.n_iter * self.args.batchsize / n_ins)
                time = datetime.now().strftime('%Y%m%d_%H%M')

                self.log('\n### Finish %s iterations (%s examples: %s epoch)' % (
                    self.n_iter, (self.n_iter * self.args.batchsize), now_e))
                self.log('<training result for previous iterations>')            
                res = self.evaluator.report_results(n_sen, total_counts, total_loss, stream=self.logger)
                self.report('train\t%d\t%s\t%s' % (self.n_iter, now_e, res))

                if self.args.valid_data:
                    self.log('\n<validation result>')
                    v_res = self.run_epoch(self.val, train=False)
                    self.report('valid\t%d\t%s\t%s' % (self.n_iter, now_e, v_res))

                # Save the model
                if not self.args.quiet:
                    mdl_path = '{}/{}_e{}.npz'.format(constants.MODEL_DIR, self.start_time, now_e)
                    self.log('Save the model: %s\n' % mdl_path)
                    self.report('[INFO] save the model: %s\n' % mdl_path)
                    serializers.save_npz(mdl_path, self.classifier)
        
                if not self.args.quiet:
                    self.reporter.close() 
                    self.reporter = open('{}/{}.log'.format(constants.LOG_DIR, self.start_time), 'a')

                # Reset counters
                count = 0
                total_loss = 0
                total_counts = None

        if not train:
            res = self.evaluator.report_results(n_sen, total_counts, total_loss, stream=self.logger)
        else:
            res = None

        return res


class TaggerTrainer(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super(TaggerTrainer, self).__init__(args, logger)


    def init_hyperparameters(self, args):
        self.hparams = {
            'unit_embed_dim' : args.unit_embed_dim,
            'rnn_unit_type' : args.rnn_unit_type,
            'rnn_bidirection' : args.rnn_bidirection,
            'rnn_layers' : args.rnn_layers,
            'rnn_hidden_units' : args.rnn_hidden_units,
            'inference_layer' : args.inference_layer,
            'dropout' : args.dropout,
            'feature_template' : args.feature_template,
            'data_format' : args.data_format,
            'subpos_depth' : args.subpos_depth,
            'lowercase' : args.lowercase,
            'normalize_digits' : args.normalize_digits,
        }
        self.check_data_format(self.hparams['data_format'])

        self.log('Init hyperparameters')
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def load_hyperparameters(self, hparams_path):
        hparams = {}
        with open(hparams_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue

                kv = line.split('=')
                key = kv[0]
                val = kv[1]

                if (key == 'unit_embed_dim' or
                    key == 'additional_feat_dim' or
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

        self.check_data_format(hparams['data_format'])

        return hparams


    def check_data_format(self, data_format):
        if (data_format == 'wl_seg' or
            data_format == 'sl_seg'):
            self.decode_type = 'seg'

        elif (data_format == 'wl_seg_tag' or
              data_format == 'sl_seg_tag'):
            self.decode_type = 'seg_tag'

        elif (data_format == 'wl_tag' or
              data_format == 'sl_tag'):
            self.decode_type = 'tag'

        else:
            print('Error: invalid data format: {}'.format(data_format), file=sys.stderr)
            sys.exit()


    def show_training_data(self):
        train = self.train
        val = self.val
        n_train = len(train.instances)
        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.instances[:3], train.instances[n_train-3:]))
        self.log('# train_gold: {} ... {}\n'.format(train.labels[0][:3], train.labels[0][n_train-3:]))
        t2i_tmp = list(self.indices.token_indices.str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))
        if self.indices.seg_label_indices:
            id2seg = {v:k for k,v in self.indices.seg_label_indices.str2id.items()}
            self.log('# seg labels: {}\n'.format(id2seg))
        if self.indices.pos_label_indices:
            id2pos = {v:k for k,v in self.indices.pos_label_indices.str2id.items()}
            self.log('# pos labels: {}\n'.format(id2pos))
        
        self.report('[INFO] vocab: {}'.format(len(self.indices.token_indices)))
        self.report('[INFO] data length: train={} val={}'.format(
            len(train.instances), len(val.instances) if val else 0))


    def gen_inputs(self, data, ids):
        xp = cuda.cupy if self.args.gpu >= 0 else np

        xs = [xp.asarray(data.instances[j], dtype='i') for j in ids]
        ts = [xp.asarray(data.labels[0][j], dtype='i') for j in ids]
        if self.hparams['feature_template']:
            fs = [data.features[j] for j in ids]
        else:
            fs = None

        return xs, fs, ts
        

    def decode(self, data, stream=sys.stdout):
        xp = cuda.cupy if args.gpu >= 0 else np

        n_ins = len(data.instances)
        for ids in batch_generator(n_ins, batchsize=self.args.batchsize, shuffle=False):
            xs = [xp.asarray(data.instances[j], dtype='i') for j in ids]
            if self.hparams['feature_template']:
                fs = [data.features[j] for j in ids]
            else:
                fs = None

            self.decode_batch(xs, fs)


    def decode_batch(self, xs, fs, stream=sys.stdout):
        ys = self.classifier.decode(xs, fs)

        for x, y in zip(xs, ys):
            x_str = [self.classifier.indices.get_token(int(xi)) for xi in x]
            y_str = [self.classifier.indices.get_label(int(yi)) for yi in y]

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


class ParserTrainer(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super(ParserTrainer, self).__init__(args, logger)


    def init_hyperparameters(self, args):
        self.hparams = {
            'unit_embed_dim' : args.unit_embed_dim,
            'pos_embed_dim' : args.pos_embed_dim,
            'rnn_unit_type' : args.rnn_unit_type,
            'rnn_bidirection' : args.rnn_bidirection,
            'rnn_layers' : args.rnn_layers,
            'rnn_hidden_units' : args.rnn_hidden_units,
            'affine_dim' : args.affine_dim,
            'dropout' : args.dropout,
            'data_format' : args.data_format,
            'subpos_depth' : args.subpos_depth,
            'lowercase' : args.lowercase,
            'normalize_digits' : args.normalize_digits,
        }
        self.check_data_format(self.hparams['data_format'])

        self.log('Init hyperparameters')
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def load_hyperparameters(self, hparams_path):
        hparams = {}
        with open(hparams_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue

                kv = line.split('=')
                key = kv[0]
                val = kv[1]

                if (key == 'unit_embed_dim' or
                    key == 'pos_embed_dim' or
                    key == 'rnn_layers' or
                    key == 'rnn_hidden_units' or
                    key == 'affine_dim' or
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

        self.check_data_format(hparams['data_format'])

        return hparams


    def check_data_format(self, data_format):
        if (data_format == 'wl_dep'):
            self.decode_type = 'dep'
            
        elif (data_format == 'wl_tdep'):
            self.decode_type = 'tdep'

        else:
            print('Error: invalid data format: {}'.format(data_format), file=sys.stderr)
            sys.exit()


    def show_training_data(self):
        train = self.train
        val = self.val
        n_train = len(train.instances)
        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.instances[:3], train.instances[n_train-3:]))
        self.log('# train_gold_head: {} ... {}\n'.format(train.labels[1][:3], train.labels[1][n_train-3:]))
        self.log('# train_gold_label: {} ... {}\n'.format(train.labels[2][:3], train.labels[2][n_train-3:]))
        t2i_tmp = list(self.indices.token_indices.str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))
        if self.indices.pos_label_indices:
            id2pos = {v:k for k,v in self.indices.pos_label_indices.str2id.items()}
            self.log('# pos labels: {}\n'.format(id2pos))
        if self.indices.arc_label_indices:
            id2arc = {v:k for k,v in self.indices.arc_label_indices.str2id.items()}
            self.log('# arc labels: {}\n'.format(id2arc))
        
        self.report('[INFO] vocab: {}'.format(len(self.indices.token_indices)))
        self.report('[INFO] data length: train={} val={}'.format(
            len(train.instances), len(val.instances) if val else 0))


    def gen_inputs(self, data, ids):
        xp = cuda.cupy if self.args.gpu >= 0 else np

        ws = [xp.asarray(data.instances[j], dtype='i') for j in ids]
        ps = [xp.asarray(data.labels[0][j], dtype='i') for j in ids]
        ths = [xp.asarray(data.labels[1][j], dtype='i') for j in ids]
        tls = [xp.asarray(data.labels[2][j], dtype='i') for j in ids]

        return ws, ps, ths, tls
        

    def decode(self, data, stream=sys.stdout):
        pass


def batch_generator(len_data, batchsize=100, shuffle=True):
    perm = np.random.permutation(len_data) if shuffle else list(range(0, len_data))
    for i in range(0, len_data, batchsize):
        i_max = min(i + batchsize, len_data)
        yield perm[i:i_max]

    raise StopIteration
