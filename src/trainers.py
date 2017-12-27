import sys
import copy
import os
import pickle
from datetime import datetime
from collections import Counter


import numpy as np
import chainer
from chainer import serializers
from chainer import cuda
import gensim
from gensim.models import keyedvectors
from gensim.models import word2vec

import util
import constants
import data_io
import classifiers
import evaluators
import features
import optimizers
from tools import conlleval


class Trainer(object):
    def __init__(self, args, logger=sys.stderr):
        err_msgs = []
        if args.execute_mode == 'train':
            if not args.train_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--train_data/-t')
                err_msgs.append(msg)
            if not args.data_format and not args.model_path:
                msg = 'Error: the following argument is required for {} mode unless resuming a trained model: {}'.format('train', '--data_format/-f')
                err_msgs.append(msg)

        elif args.execute_mode == 'eval':
            if not args.model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--model_path/-m')
                err_msgs.append(msg)

            if not args.test_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--test_data')
                err_msgs.append(msg)

        elif args.execute_mode == 'decode':
            if not args.model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--model_path/-m')
                err_msgs.append(msg)

            if not args.input_text:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--input_text')
                err_msgs.append(msg)

        elif args.execute_mode == 'interactive':
            if not args.model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--model_path/-m')
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
        self.dic = None
        self.classifier = None
        self.evaluator = None
        self.feat_extractor = None
        self.optimizer = None
        self.decode_type = None
        self.n_iter = args.epoch_begin
        self.label_begin_index = 2

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


    def init_model(self, pretrained_token_embed_dim=0):
        self.log('Initialize model from hyperparameters\n')
        self.classifier = classifiers.init_classifier(
            self.decode_type, self.hparams, self.dic, pretrained_token_embed_dim,
            self.args.predict_parent_existence)


    def load_dic(self, dic_path):
        with open(dic_path, 'rb') as f:
            self.dic = pickle.load(f)
        self.log('Load dic: {}'.format(dic_path))
        self.log('Vocab size: {}\n'.format(len(self.dic.token_indices)))


    def load_model(self, model_path, pretrained_token_embed_dim=0):
        array = model_path.split('_e')
        dic_path = '{}.s2i'.format(array[0])
        hparam_path = '{}.hyp'.format(array[0])
        param_path = model_path

        # dictionary
        self.load_dic(dic_path)

        # hyper parameters
        self.hparams = self.load_hyperparameters(hparam_path)
        self.log('Load hyperparameters: {}\n'.format(hparam_path))
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            if 'dropout' in k:
                self.hparams[k] = v
                message = '{}={}'.format(k, v)            

            elif k in self.hparams and v != self.hparams[k]:
                message = '{}={} (input option value {} was discarded)'.format(k, self.hparams[k], v)

            else:
                message = '{}={}'.format(k, v)

            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')

        # model
        self.init_model(pretrained_token_embed_dim)
        chainer.serializers.load_npz(model_path, self.classifier)
        self.log('Load model parameters: {}\n'.format(model_path))


    def finalize_model(self):
        # to be implemented in sub-class if nessesary
        pass


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
        use_subtoken = hparams["subtoken_embed_dim"] > 0
        create_chunk_trie = ('feature_template' in hparams and hparams['feature_template'])
        subpos_depth = hparams['subpos_depth'] if 'subpos_depth' in hparams else -1
        pred_parex = args.predict_parent_existence
        lowercase = hparams['lowercase'] if 'lowercase' in hparams else False
        normalize_digits = hparams['normalize_digits'] if 'normalize_digits' else False
        
        # if args.label_reference_data:
        #     _, self.dic = data_io.load_data(
        #         data_format, refer_path, read_pos=read_pos, update_tokens=False,
        #         create_chunk_trie=create_chunk_trie, subpos_depth=subpos_depth, 
        #         lowercase=lowercase, normalize_digits=normalize_digits, 
        #         dic=self.dic)
        #     self.log('Load label set from reference data: {}'.format(refer_path))

        if args.train_data.endswith('pickle'):
            name, ext = os.path.splitext(train_path)
            train = data_io.load_pickled_data(name)
            self.log('Load dumpped data:'.format(train_path))
            #TODO update dic when re-training using another training data

        else:
            train, self.dic = data_io.load_data(
                data_format, train_path, read_pos=read_pos,
                create_chunk_trie=create_chunk_trie, use_subtoken=use_subtoken, subpos_depth=subpos_depth, 
                predict_parent_existence=pred_parex,
                lowercase=lowercase, normalize_digits=normalize_digits, 
                max_vocab_size=args.max_vocab_size, freq_threshold=args.freq_threshold,
                dic=self.dic, refer_vocab=refer_vocab)
        
            if self.args.dump_train_data:
                name, ext = os.path.splitext(train_path)
                if args.max_vocab_size > 0 or args.freq_threshold > 1:
                    name = '{}_{}k'.format(name, int(len(self.dic.token_indices) / 1000))
                data_io.dump_pickled_data(name, train)
                self.log('Dumpped training data: {}.pickle'.format(train_path))

        self.log('Load training data: {}'.format(train_path))
        self.log('Data length: {}'.format(len(train.tokenseqs)))
        self.log('Vocab size: {}\n'.format(len(self.dic.token_indices)))
        if 'featre_template' in self.hparams and self.hparams['feature_template']:
            self.log('Start feature extraction for training data\n')
            self.extract_features(train)

        if args.valid_data:
            # dic can be updated if embed_model are used
            val, self.dic = data_io.load_data(
                data_format, val_path, read_pos=read_pos, update_tokens=False, update_labels=False,
                create_chunk_trie=create_chunk_trie, use_subtoken=use_subtoken, subpos_depth=subpos_depth, 
                predict_parent_existence=pred_parex,
                lowercase=lowercase, normalize_digits=normalize_digits, 
                max_vocab_size=args.max_vocab_size, freq_threshold=args.freq_threshold,
                dic=self.dic, refer_vocab=refer_vocab)

            self.log('Load validation data: {}'.format(val_path))
            self.log('Data length: {}'.format(len(val.tokenseqs)))
            self.log('Vocab size: {}\n'.format(len(self.dic.token_indices)))
            if 'featre_template' in self.hparams and self.hparams['feature_template']:
                self.log('Start feature extraction for validation data\n')
                self.extract_features(val)
        else:
            val = None
        
        self.train = train
        self.val = val
        self.dic.create_id2strs()
        self.show_training_data()


    def load_data_for_test(self, embed_model=None):
        args = self.args
        hparams = self.hparams
        test_path = (args.path_prefix if args.path_prefix else '') + args.test_data
        refer_vocab = embed_model.wv if embed_model else set()

        data_format = hparams['data_format']
        read_pos = hparams['subpos_depth'] != 0
        use_subtoken = hparams['subtoken_embed_dim'] > 0
        create_chunk_trie = ('feature_template' in hparams and hparams['feature_template'])
        subpos_depth = hparams['subpos_depth'] if 'subpos_depth' in hparams else -1
        pred_parex = args.predict_parent_existence
        lowercase = hparams['lowercase'] if 'lowercase' in hparams else False
        normalize_digits = hparams['normalize_digits'] if 'normalize_digits' else False

        test, self.dic = data_io.load_data(
            data_format, test_path, read_pos=read_pos, update_tokens=False, update_labels=False,
            create_chunk_trie=create_chunk_trie, use_subtoken=use_subtoken, subpos_depth=subpos_depth, 
            predict_parent_existence=pred_parex,
            lowercase=lowercase, normalize_digits=normalize_digits, 
            max_vocab_size=args.max_vocab_size, freq_threshold=args.freq_threshold,
            dic=self.dic, refer_vocab=refer_vocab)

        self.log('Load test data: {}'.format(test_path))
        self.log('Data length: {}'.format(len(test.tokenseqs)))
        self.log('Vocab size: {}\n'.format(len(self.dic.token_indices)))
        if 'featre_template' in self.hparams and self.hparams['feature_template']:
            self.log('Start feature extraction for test data\n')
            self.extract_features(test)

        self.test = test
        self.dic.create_id2strs()


    def extract_features(self, data):
        features = self.feat_extractor.extract_features(data.tokenseqs, self.dic)
        data.set_features(features)


    def setup_optimizer(self):
        if self.args.optimizer == 'sgd':
            if self.args.momentum <= 0:
                optimizer = chainer.optimizers.SGD(lr=self.args.learning_rate)
            else:
                optimizer = chainer.optimizers.MomentumSGD(
                    lr=self.args.learning_rate, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = chainer.optimizers.Adam(
                alpha=self.args.adam_alpha,
                beta1=self.args.adam_beta1,
                beta2=self.args.adam_beta2
            )
        elif self.args.optimizer == 'adagrad':
            optimizer = chainer.optimizers.AdaGrad(lr=self.args.learning_rate)

        optimizer.setup(self.classifier)
        delparams = []
        if self.args.fix_pretrained_embed:
            delparams.append('pretrained_token_embed/W')
            self.log('Fix parameters: pretrained_token_embed/W')
        if 'dual' in self.decode_type:
            delparams.append('su_tagger')
            self.log('Fix parameters: su_tagger/*')
        if delparams:
            optimizer.add_hook(optimizers.DeleteGradient(delparams))

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

            dic_path = '{}/{}.s2i'.format(constants.MODEL_DIR, self.start_time)
            with open(dic_path, 'wb') as f:
                pickle.dump(self.dic, f)
            self.log('save string2index table: {}'.format(dic_path))

        for e in range(max(1, self.args.epoch_begin), self.args.epoch_end+1):
            time = datetime.now().strftime('%Y%m%d_%H%M')
            self.log('Start epoch {}: {}\n'.format(e, time))
            self.report('[INFO] Start epoch {} at {}'.format(e, time))

            # learning rate decay
            if self.args.lrdecay and self.args.optimizer == 'sgd':
                if (e >= self.args.lrdecay_start and
                    (e - self.args.lrdecay_start) % self.args.lrdecay_width == 0):
                    lr_tmp = self.optimizer.lr
                    self.optimizer.lr = self.optimizer.lr * self.args.lrdecay_rate
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
        text_path = (self.args.path_prefix if self.args.path_prefix else '') + self.args.input_text
        if self.decode_type == 'tag' or self.decode_type == 'dual_tag':
            tokenseqs = data_io.load_raw_text_for_tagging(text_path, self.dic)

        elif (self.decode_type == 'seg' or self.decode_type == 'dual_seg' or
              self.decode_type == 'seg_tag' or self.decode_type == 'dual_seg_tag'):
            tokenseqs = data_io.load_raw_text_for_segmentation(text_path, self.dic)
        else:
            # to be implemented
            pass

        data = data_io.Data(tokenseqs, None)

        if 'feature_template' in self.hparams and self.hparams['feature_template']:
            self.extract_features(data, self.dic)

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
                ins = [self.dic.get_token_id(word) for word in array]
            else:
                ins = [self.dic.get_token_id(char) for char in line]
            xs = [xp.asarray(ins, dtype=np.int32)]

            if self.hparams['feature_template']:
                fs = self.feat_extractor.extract_features([ins], self.dic)
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
            classifier.change_dropout_ratio(0)

        n_sen = 0
        total_loss = 0
        total_counts = None

        i = 0
        n_ins = len(data.tokenseqs)
        # shuffle = False
        shuffle = True if train else False
        for ids in batch_generator(n_ins, batchsize=self.args.batchsize, shuffle=shuffle):
            inputs = self.gen_inputs(data, ids)

            xs = inputs[0]
            golds = inputs[self.label_begin_index:]
            # timer = util.Timer()
            # timer.start()
            if train:
                ret = classifier(*inputs, train=True)
            else:
                with chainer.no_backprop_mode():
                    ret = classifier(*inputs, train=False)
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

            n_sen += len(xs)
            total_loss += loss.data
            counts = self.evaluator.calculate(*[xs], *golds, *outputs)
            total_counts = evaluators.merge_counts(total_counts, counts)

            if (train and 
                self.n_iter > 0 and 
                self.n_iter * self.args.batchsize % self.args.evaluation_size == 0):
                now_e = '%.3f' % (self.args.epoch_begin-1 + (self.n_iter * self.args.batchsize / n_ins))
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
                    serializers.save_npz(mdl_path, classifier)

                if not self.args.quiet:
                    self.reporter.close() 
                    self.reporter = open('{}/{}.log'.format(constants.LOG_DIR, self.start_time), 'a')

                # Reset counters
                n_sen = 0
                total_loss = 0
                counts = None
                total_counts = None

            if train:
                self.n_iter += 1


        res = None if train else self.evaluator.report_results(n_sen, total_counts, total_loss, stream=self.logger)
        return res


class TaggerTrainerBase(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)


    def load_hyperparameters(self, hparams_path, check_data_format=True):
        hparams = {}
        with open(hparams_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue

                kv = line.split('=')
                key = kv[0]
                val = kv[1]

                if (key == 'token_embed_dim' or
                    key == 'subtoken_embed_dim' or
                    key == 'additional_feat_dim' or
                    key == 'rnn_n_layers' or
                    key == 'rnn_n_units' or
                    key == 'mlp_n_layers'or
                    key == 'mlp_n_units' or
                    key == 'subpos_depth'
                ):
                    val = int(val)

                elif (key == 'rnn_dropout' or
                      key == 'mlp_dropout'
                ):
                    val = float(val)

                elif (key == 'rnn_bidirection' or
                      key == 'lowercase' or
                      key == 'normalize_digits'
                ):
                    val = (val.lower() == 'true')

                hparams[key] = val

        if check_data_format:
            self.check_data_format(hparams['data_format'])

        return hparams


    def show_training_data(self):
        train = self.train
        val = self.val
        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.tokenseqs[0], train.tokenseqs[-1]))
        self.log('# train_gold: {} ... {}\n'.format(train.labels[0][0], train.labels[0][-1]))
        t2i_tmp = list(self.dic.token_indices.str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))
        if self.dic.seg_label_indices:
            id2seg = {v:k for k,v in self.dic.seg_label_indices.str2id.items()}
            self.log('# seg labels: {}\n'.format(id2seg))
        if self.dic.pos_label_indices:
            id2pos = {v:k for k,v in self.dic.pos_label_indices.str2id.items()}
            self.log('# pos labels: {}\n'.format(id2pos))
        
        self.report('[INFO] vocab: {}'.format(len(self.dic.token_indices)))
        self.report('[INFO] data length: train={} val={}'.format(
            len(train.tokenseqs), len(val.tokenseqs) if val else 0))


    def gen_inputs(self, data, ids, evaluate=True):
        xp = cuda.cupy if self.args.gpu >= 0 else np

        xs = [xp.asarray(data.tokenseqs[j], dtype='i') for j in ids]
        ts = [xp.asarray(data.labels[0][j], dtype='i') for j in ids] if evaluate else None
        fs = [data.features[j] for j in ids] if self.hparams['feature_template'] else None

        if evaluate:
            return xs, fs, ts
        else:
            return xs, fs
        

    def decode(self, data, stream=sys.stdout):
        n_ins = len(data.tokenseqs)
        for ids in batch_generator(n_ins, batchsize=self.args.batchsize, shuffle=False):
            xs, fs = self.gen_inputs(data, ids, evaluate=False)
            self.decode_batch(xs, fs)


class TaggerTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)


    def load_model(self, model_path, pretrained_token_embed_dim=0):
        super().load_model(model_path, pretrained_token_embed_dim)
        if 'rnn_dropout' in self.hparams:
            self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
        if 'hidden_mlp_dropout' in self.hparams:
            self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])


    def init_hyperparameters(self, args, pretrained_token_embed_dim=0):
        self.hparams = {
            'token_embed_dim' : args.token_embed_dim,
            'subtoken_embed_dim' : args.subtoken_embed_dim,
            'rnn_unit_type' : args.rnn_unit_type,
            'rnn_bidirection' : args.rnn_bidirection,
            'rnn_n_layers' : args.rnn_n_layers,
            'rnn_n_units' : args.rnn_n_units,
            'mlp_n_layers' : args.mlp_n_layers,
            'mlp_n_units' : args.mlp_n_units,
            'inference_layer' : args.inference_layer,
            'rnn_dropout' : args.rnn_dropout,
            'mlp_dropout' : args.mlp_dropout,
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


    def check_data_format(self, data_format):
        if (data_format == 'wl_seg' or
            data_format == 'sl_seg'):
            self.decode_type = 'seg'
            self.label_begin_index = 2

        elif (data_format == 'wl_seg_tag' or
              data_format == 'sl_seg_tag'):
            self.decode_type = 'seg_tag'
            self.label_begin_index = 2

        elif (data_format == 'wl_tag' or
              data_format == 'sl_tag'):
            self.decode_type = 'tag'
            self.label_begin_index = 2

        else:
            print('Error: invalid data format: {}'.format(data_format), file=sys.stderr)
            sys.exit()


    def setup_evaluator(self):
        if self.decode_type == 'tag' and self.args.ignore_labels:
            tmp = self.args.ignore_labels
            self.args.ignore_labels = set()

            for label in tmp.split(','):
                label_id = self.dic.get_pos_label_id(label)
                if label_id >= 0:
                    self.args.ignore_labels.add(label_id)

            self.log('Setup evaluator: labels to be ignored={}\n'.format(self.args.ignore_labels))

        else:
            self.args.ignore_labels = set()

        self.evaluator = classifiers.init_evaluator(self.decode_type, self.dic, self.args.ignore_labels)


    def decode_batch(self, xs, fs, stream=sys.stdout):
        ys = self.classifier.decode(xs, fs)

        for x, y in zip(xs, ys):
            x_str = [self.dic.get_token(int(xi)) for xi in x]
            y_str = [self.dic.get_label(int(yi)) for yi in y]

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


class DualTaggerTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.su_tagger = None


    def load_sub_model(self, model_path, pretrained_token_embed_dim=0):
        array = model_path.split('_e')
        dic_path = '{}.s2i'.format(array[0])
        hparam_path = '{}.hyp'.format(array[0])
        param_path = model_path

        # dictionary
        self.load_dic(dic_path)

        # decode type
        self.check_data_format(self.args.data_format)
        decode_type = self.decode_type.split('dual_')[1]

        # hyper parameters
        self.hparams = self.load_hyperparameters(hparam_path, False)
        self.log('Load hyperparameters from short unit model: {}\n'.format(hparam_path))
        for k, v in self.args.__dict__.items():
            if 'dropout' in k:
                self.hparams[k] = 0.0
            elif k == 'data_fromat':
                self.hparams[k] = v

        # sub model
        hparams_sub = self.hparams.copy()
        self.log('Initialize short unit model from hyperparameters\n')
        classifier_tmp = classifiers.init_classifier(
            decode_type, hparams_sub, self.dic, pretrained_token_embed_dim)
        chainer.serializers.load_npz(model_path, classifier_tmp)
        self.su_tagger = classifier_tmp.predictor
        self.log('Load short unit model parameters: {}\n'.format(model_path))


    def init_hyperparameters(self, args, pretrained_token_embed_dim=0):
        self.log('Initialize hyperparameters for full model\n')

        # keep most loaded params from sub model and update some params from args
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            if 'dropout' in k:  # update
                self.hparams[k] = v
                message = '{}={}'.format(k, v)            

            elif k == 'data_format': # udpate
                self.hparams[k] = v
                message = '{}={}'.format(k, v)            

            elif k in self.hparams and v != self.hparams[k]:
                message = '{}={} (input option value {} was discarded)'.format(k, self.hparams[k], v)

            else:
                message = '{}={}'.format(k, v)

            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def load_model(self, model_path, pretrained_token_embed_dim=0):
        super().load_model(model_path, pretrained_token_embed_dim)
        if 'rnn_dropout' in self.hparams:
            self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
        if 'hidden_mlp_dropout' in self.hparams:
            self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])


    def finalize_model(self):
        if self.su_tagger is not None:
            self.classifier.integrate_submodel(self.su_tagger)


    def check_data_format(self, data_format):
        if (data_format == 'wl_dual_seg' or
            data_format == 'sl_dual_seg'):
            self.decode_type = 'dual_seg'
            self.label_begin_index = 2

        elif (data_format == 'wl_dual_seg_tag' or
              data_format == 'sl_dual_seg_tag'):
            self.decode_type = 'dual_seg_tag'
            self.label_begin_index = 2

        elif (data_format == 'wl_dual_tag' or
              data_format == 'sl_dual_tag'):
            self.decode_type = 'dual_tag'
            self.label_begin_index = 2

        else:
            print('Error: invalid data format: {}'.format(data_format), file=sys.stderr)
            sys.exit()


    def setup_evaluator(self):
        if self.decode_type == 'dual_tag' and self.args.ignore_labels:
            tmp = self.args.ignore_labels
            self.args.ignore_labels = set()

            for label in tmp.split(','):
                label_id = self.dic.get_pos_label_id(label)
                if label_id >= 0:
                    self.args.ignore_labels.add(label_id)

            self.log('Setup evaluator: labels to be ignored={}\n'.format(self.args.ignore_labels))

        else:
            self.args.ignore_labels = set()

        self.evaluator = classifiers.init_evaluator(self.decode_type, self.dic, self.args.ignore_labels)


    def decode_batch(self, xs, fs, stream=sys.stdout):
        sys, lys = self.classifier.decode(xs, fs)

        get_token = self.dic.get_token
        get_label = self.dic.get_seg_label if 'seg' in self.decode_type else self.dic.get_pos_label

        for x, sy, ly in zip(xs, sys, lys):
            x_str = [get_token(int(xi)) for xi in x]
            sy_str = [get_label(int(yi)) for yi in sy]
            ly_str = [get_label(int(yi)) for yi in ly]

            if self.decode_type == 'dual_tag':
                res1 = ['{}/{}'.format(xi_str, yi_str) for xi_str, yi_str in zip(x_str, sy_str)]
                res1 = ' '.join(res1)
                res2 = ['{}/{}'.format(xi_str, yi_str) for xi_str, yi_str in zip(x_str, ly_str)]
                res2 = ' '.join(res2)

            elif self.decode_type == 'dual_seg':
                res1 = ['{}{}'.format(
                    xi_str, ' ' if (yi_str == 'E' or yi_str == 'S') else ''
                ) for xi_str, yi_str in zip(x_str, sy_str)]
                res1 = ''.join(res1).rstrip()
                res2 = ['{}{}'.format(
                    xi_str, ' ' if (yi_str == 'E' or yi_str == 'S') else ''
                ) for xi_str, yi_str in zip(x_str, ly_str)]
                res2 = ''.join(res2).rstrip()

            elif self.decode_type == 'dual_seg_tag':
                res1 = ['{}{}'.format(
                    xi_str, ('/'+yi_str[2:]+' ') if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
                ) for xi_str, yi_str in zip(x_str, sy_str)]
                res1 = ''.join(res1).rstrip()
                res2 = ['{}{}'.format(
                    xi_str, ('/'+yi_str[2:]+' ') if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
                ) for xi_str, yi_str in zip(x_str, ly_str)]
                res2 = ''.join(res2).rstrip()

            else:
                print('Error: Invalid decode type', file=self.logger)
                sys.exit()

            print('<SU>', res1, file=stream)
            print('<LU>', res2, file=stream)


class ParserTrainer(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)


    def load_model(self, model_path, pretrained_token_embed_dim=0):
        # TODO give args.predict_parent_existence to init classifier
        super().load_model(model_path, pretrained_token_embed_dim)
        if 'rnn_dropout' in self.hparams:
            self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
        if 'hidden_mlp_dropout' in self.hparams:
            self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])
        if 'pred_layers_dropout' in self.hparams:
            self.classifier.change_pred_layers_dropout_ratio(self.hparams['pred_layers_dropout'])
            

    def init_hyperparameters(self, args):
        self.hparams = {
            'token_embed_dim' : args.token_embed_dim,
            'subtoken_embed_dim' : args.subtoken_embed_dim,
            'pos_embed_dim' : args.pos_embed_dim,
            'rnn_unit_type' : args.rnn_unit_type,
            'rnn_bidirection' : args.rnn_bidirection,
            'rnn_n_layers' : args.rnn_n_layers,
            'rnn_n_units' : args.rnn_n_units,
            'mlp4pospred_n_layers' : args.mlp4pospred_n_layers,
            'mlp4pospred_n_units' : args.mlp4pospred_n_units,
            'mlp4arcrep_n_layers' : args.mlp4arcrep_n_layers,
            'mlp4arcrep_n_units' : args.mlp4arcrep_n_units,
            'mlp4labelrep_n_layers' : args.mlp4labelrep_n_layers,
            'mlp4labelrep_n_units' : args.mlp4labelrep_n_units,
            'mlp4labelpred_n_layers' : args.mlp4labelpred_n_layers,
            'mlp4labelpred_n_units' : args.mlp4labelpred_n_units,
            'rnn_dropout' : args.rnn_dropout,
            'hidden_mlp_dropout' : args.hidden_mlp_dropout,
            'pred_layers_dropout' : args.pred_layers_dropout,
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

                if (key == 'token_embed_dim' or
                    key == 'subtoken_embed_dim' or
                    key == 'pretrained_token_embed_dim' or
                    key == 'pos_embed_dim' or
                    key == 'rnn_n_layers' or
                    key == 'rnn_n_units' or
                    key == 'mlp4pospred_n_layers' or
                    key == 'mlp4pospred_n_units' or
                    key == 'mlp4arcrep_n_layers' or
                    key == 'mlp4arcrep_n_units' or
                    key == 'mlp4labelrep_n_layers' or
                    key == 'mlp4labelrep_n_units' or
                    key == 'mlp4labelpred_n_layers' or
                    key == 'mlp4labelpred_n_units' or
                    key == 'subpos_depth'
                ):
                    val = int(val)

                elif (key == 'rnn_dropout' or
                      key == 'hidden_mlp_dropout' or
                      key == 'pred_layers_dropout'
                ):
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
            self.label_begin_index = 3
            
        elif (data_format == 'wl_tdep'):
            self.decode_type = 'tdep'
            self.label_begin_index = 3

        elif (data_format == 'wl_tag_dep'):
            self.decode_type = 'tag_dep'
            self.label_begin_index = 2

        elif (data_format == 'wl_tag_tdep'):
            self.decode_type = 'tag_tdep'
            self.label_begin_index = 2

        else:
            print('Error: invalid data format: {}'.format(data_format), file=sys.stderr)
            sys.exit()


    def show_training_data(self):
        train = self.train
        val = self.val
        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.tokenseqs[0], train.tokenseqs[-1]))
        self.log('# train_gold_head: {} ... {}\n'.format(train.labels[1][0], train.labels[1][-1]))
        self.log('# train_gold_label: {} ... {}\n'.format(train.labels[2][0], train.labels[2][-1]))
        t2i_tmp = list(self.dic.token_indices.str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))

        if self.dic.subtoken_indices:
            self.log('# train_subtokens: {} ... {}\n'.format(
                train.subtokenseqs[0], train.subtokenseqs[-1]))
            st2i_tmp = list(self.dic.subtoken_indices.str2id.items())
            self.log('# subtoken2id: {} ... {}\n'.format(st2i_tmp[:10], st2i_tmp[len(st2i_tmp)-10:]))

        if self.dic.pos_label_indices:
            id2pos = {v:k for k,v in self.dic.pos_label_indices.str2id.items()}
            self.log('# pos labels: {}\n'.format(id2pos))

        if self.dic.arc_label_indices:
            id2arc = {v:k for k,v in self.dic.arc_label_indices.str2id.items()}
            self.log('# arc labels: {}\n'.format(id2arc))
        
        self.report('[INFO] vocab: {}'.format(len(self.dic.token_indices)))
        self.report('[INFO] data length: train={} val={}'.format(
            len(train.tokenseqs), len(val.tokenseqs) if val else 0))


    def setup_evaluator(self):
        if (self.decode_type == 'tdep' or self.decode_type == 'tag_tdep') and self.args.ignore_labels:
            tmp = self.args.ignore_labels
            self.args.ignore_labels = set()

            for label in tmp.split(','):
                label_id = self.dic.get_arc_label_id(label)
                if label_id >= 0:
                    self.args.ignore_labels.add(label_id)

            self.log('Setup evaluator: labels to be ignored={}\n'.format(self.args.ignore_labels))

        else:
            self.args.ignore_labels = set()

        self.evaluator = classifiers.init_evaluator(self.decode_type, self.dic, self.args.ignore_labels)


    def gen_inputs(self, data, ids):
        xp = cuda.cupy if self.args.gpu >= 0 else np
        ws = [xp.asarray(data.tokenseqs[j], dtype='i') for j in ids]
        cs = ([[xp.asarray(subs, dtype='i') for subs in data.subtokenseqs[j]] for j in ids] 
              if data.subtokenseqs else None)
        ps = [xp.asarray(data.labels[0][j], dtype='i') for j in ids] if data.labels[0] else None
        ths = [xp.asarray(data.labels[1][j], dtype='i') for j in ids]

        if self.decode_type == 'tdep' or self.decode_type == 'tag_tdep':
            tls = [xp.asarray(data.labels[2][j], dtype='i') for j in ids]
            return ws, cs, ps, ths, tls
        else:
            return ws, cs, ps, ths
        

    def decode(self, data, stream=sys.stdout):
        pass


def batch_generator(len_data, batchsize=100, shuffle=True):
    perm = np.random.permutation(len_data) if shuffle else list(range(0, len_data))
    for i in range(0, len_data, batchsize):
        i_max = min(i + batchsize, len_data)
        yield perm[i:i_max]

    raise StopIteration


def load_embedding_model(model_path):
    if model_path.endswith('model'):
        model = word2vec.Word2Vec.load(model_path)        
    elif model_path.endswith('bin'):
        model = keyedvectors.KeyedVectors.load_word2vec_format(model_path, binary=True)
    elif model_path.endswith('txt'):
        model = keyedvectors.KeyedVectors.load_word2vec_format(model_path, binary=False)
    else:
        print("unsuported format of word embedding model.", file=sys.stderr)
        return

    print("load embedding model: vocab=%d\n" % len(model.wv.vocab))
    return model
