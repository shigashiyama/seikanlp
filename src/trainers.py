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
import models.attribute_annotator
from tools import conlleval


class Trainer(object):
    def __init__(self, args, logger=sys.stderr):
        err_msgs = []
        if args.execute_mode == 'train':
            if not args.train_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--train_data')
                err_msgs.append(msg)
            if not args.task and not args.model_path:
                msg = 'Error: the following argument is required for {} mode unless resuming a trained model: {}'.format('train', '--task')
                err_msgs.append(msg)

            # if not args.input_data_format:
            #     msg = ''
            #     err_msgs.append(msg)

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

            if not args.decode_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--decode_data')
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
        self.task = args.task
        self.train = None
        self.dev = None
        self.test = None
        self.hparams = None
        self.dic = None
        self.classifier = None
        self.evaluator = None
        self.feat_extractor = None
        self.optimizer = None
        self.external_embed_model = None
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


    def load_unigram_embedding_model(self, model_path):
        self.external_embed_model = load_embedding_model(model_path)


    def init_model(self):
        self.log('Initialize model from hyperparameters\n')
        self.classifier = classifiers.init_classifier(self.task, self.hparams, self.dic)
        if self.external_embed_model:
            self.classifier.load_pretrained_embedding_layer(
                self.dic, self.external_embed_model, finetuning=(not self.args.fix_pretrained_embed))


    def load_dic(self, dic_path):
        with open(dic_path, 'rb') as f:
            self.dic = pickle.load(f)
        self.log('Load dic: {}'.format(dic_path))
        self.log('Vocab size: {}\n'.format(len(self.dic.tables['unigram'])))


    def load_model(self, model_path):
        array = model_path.split('_e')
        dic_path = '{}.s2i'.format(array[0])
        hparam_path = '{}.hyp'.format(array[0])
        param_path = model_path

        # dictionary
        self.load_dic(dic_path)

        # hyper parameters
        self.load_hyperparameters(hparam_path)
        self.log('Load hyperparameters: {}\n'.format(hparam_path))
        self.show_hyperparameters()

        # model
        self.init_model()
        chainer.serializers.load_npz(model_path, self.classifier)
        self.log('Load model parameters: {}\n'.format(model_path))


    def show_hyperparameters(self):
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            if (k in self.hparams and
                ('dropout' in k or k == 'freq_threshold' or k == 'max_vocab_size')):
                update = self.hparams[k] != v
                message = '{}={}{}'.format(
                    k, v, ' (original value ' + str(self.hparams[k]) + ' was updated)' if update else '')
                self.hparams[k] = v

            elif (k == 'task' and v == 'seg' and self.hparams[k] == 'seg_tag' and 
                  (self.args.execute_mode == 'decode' or self.args.execute_mode == 'interactive')):
                self.task = self.hparams[k] = v
                message = '{}={}'.format(k, v)            

            elif k in self.hparams and v != self.hparams[k]:
                message = '{}={} (input option value {} was discarded)'.format(k, self.hparams[k], v)

            else:
                message = '{}={}'.format(k, v)

            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def update_model(self):
        if (self.args.execute_mode == 'train' or
            self.args.execute_mode == 'eval'):

            self.classifier.grow_embedding_layers(
                self.dic, self.external_embed_model, train=(self.args.execute_mode=='train'))
            self.classifier.grow_inference_layers(self.dic)


    def init_hyperparameters(self, args):
        # to be implemented in sub-class
        self.hparams = {}
        

    def load_hyperparameters(self, hparams_path):
        # to be implemented in sub-class
        pass


    # def check_data_format(self, data_format):
    #     # to be implemented in sub-class
    #     pass


    def show_training_data(self):
        # to be implemented in sub-class
        pass


    def load_data_for_training(self):
        args = self.args
        hparams = self.hparams
        prefix = args.path_prefix if args.path_prefix else ''
            
        # refer_path   = prefix + args.label_reference_data
        train_path   = prefix + args.train_data
        dev_path     = prefix + (args.devel_data if args.devel_data else '')
        refer_vocab = self.external_embed_model.wv if self.external_embed_model else set()
        input_data_format = args.input_data_format

        task = hparams['task']
        if 'tag' in self.task:
            read_pos = True
        elif 'dep' in self.task and 'pos_embed_dim' in hparams and hparams['pos_embed_dim'] > 0:
            read_pos = True
        elif self.task == 'attr':
            read_pos = True
        else:
            read_pos = False
        use_subtoken = ('subtoken_embed_dim' in hparams and hparams['subtoken_embed_dim'] > 0)
        create_chunk_trie = ('feature_template' in hparams and hparams['feature_template'] is not None)
        subpos_depth = hparams['subpos_depth'] if 'subpos_depth' in hparams else -1
        lowercase = hparams['lowercase'] if 'lowercase' in hparams else False
        normalize_digits = hparams['normalize_digits'] if 'normalize_digits' else False
        freq_threshold = hparams['freq_threshold'] if 'freq_threshold' in hparams else 1
        max_vocab_size = hparams['max_vocab_size'] if 'max_vocab_size' in hparams else -1
        
        if args.train_data.endswith('pickle'):
            name, ext = os.path.splitext(train_path)
            train = data_io.load_pickled_data(name)
            self.log('Load dumpped data:'.format(train_path))
            #TODO update dic when re-training using another training data

        else:
            train, self.dic = data_io.load_annotated_data(
                train_path, task, input_data_format, read_pos=read_pos,
                create_chunk_trie=create_chunk_trie, use_subtoken=use_subtoken, subpos_depth=subpos_depth, 
                lowercase=lowercase, normalize_digits=normalize_digits, 
                max_vocab_size=max_vocab_size, freq_threshold=freq_threshold,
                dic=self.dic, refer_vocab=refer_vocab)
        
            if self.args.dump_train_data:
                name, ext = os.path.splitext(train_path)
                if args.max_vocab_size > 0 or args.freq_threshold > 1:
                    name = '{}_{}k'.format(name, int(len(self.dic.tables['unigram']) / 1000))
                data_io.dump_pickled_data(name, train)
                self.log('Dumpped training data: {}.pickle'.format(name))

        self.log('Load training data: {}'.format(train_path))
        self.log('Data length: {}'.format(len(train.inputs[0])))
        self.log('Vocab size: {}\n'.format(len(self.dic.tables['unigram'])))
        if 'feature_template' in self.hparams and self.hparams['feature_template']:
            self.log('Start feature extraction for training data\n')
            self.extract_features(train)

        if args.devel_data:
            read_pos = 'pos_label' in self.dic.tables

            # dic can be updated if embed_model are used
            dev, self.dic = data_io.load_annotated_data(
                dev_path, task, input_data_format, read_pos=read_pos, update_tokens=False, update_labels=False,
                create_chunk_trie=create_chunk_trie, use_subtoken=use_subtoken, subpos_depth=subpos_depth, 
                lowercase=lowercase, normalize_digits=normalize_digits, 
                dic=self.dic, refer_vocab=refer_vocab)

            self.log('Load development data: {}'.format(dev_path))
            self.log('Data length: {}'.format(len(dev.inputs[0])))
            self.log('Vocab size: {}\n'.format(len(self.dic.tables['unigram'])))
            if 'feature_template' in self.hparams and self.hparams['feature_template']:
                self.log('Start feature extraction for development data\n')
                self.extract_features(dev)
        else:
            dev = None
        
        self.train = train
        self.dev = dev
        self.dic.create_id2strs()
        self.show_training_data()


    def load_data_for_test(self):
        args = self.args
        hparams = self.hparams
        test_path = (args.path_prefix if args.path_prefix else '') + args.test_data
        refer_vocab = self.external_embed_model.wv if self.external_embed_model else set()
        input_data_format = args.input_data_format

        task = hparams['task']
        read_pos = 'pos_label' in self.dic.tables
        # if 'tag' in self.task:
        #     read_pos = True
        # elif 'dep' in self.task and 'pos_embed_dim' in hparams and hparams['pos_embed_dim'] > 0:
        #     read_pos = True
        # elif self.task == 'attr':
        #     read_pos = True
        # else:
        #     read_pos = False
        use_subtoken = ('subtoken_embed_dim' in hparams and hparams['subtoken_embed_dim'] > 0)
        create_chunk_trie = ('feature_template' in hparams and hparams['feature_template'] is not None)
        subpos_depth = hparams['subpos_depth'] if 'subpos_depth' in hparams else -1
        lowercase = hparams['lowercase'] if 'lowercase' in hparams else False
        normalize_digits = hparams['normalize_digits'] if 'normalize_digits' else False

        test, self.dic = data_io.load_annotated_data(
            test_path, task, input_data_format, read_pos=read_pos, update_tokens=False, update_labels=False,
            create_chunk_trie=create_chunk_trie, use_subtoken=use_subtoken, subpos_depth=subpos_depth, 
            lowercase=lowercase, normalize_digits=normalize_digits, 
            dic=self.dic, refer_vocab=refer_vocab)

        self.log('Load test data: {}'.format(test_path))
        self.log('Data length: {}'.format(len(test.inputs[0])))
        self.log('Vocab size: {}\n'.format(len(self.dic.tables['unigram'])))
        if 'feature_template' in self.hparams and self.hparams['feature_template']:
            self.log('Start feature extraction for test data\n')
            self.extract_features(test)

        self.test = test
        self.dic.create_id2strs()


    def extract_features(self, data):
        featvecs = self.feat_extractor.extract_features(data.inputs[0], self.dic.tries['chunk'])
        data.featvecs = featvecs


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
            delparams.append('pretrained_unigram_embed/W')
            self.log('Fix parameters: pretrained_unigram_embed/W')
        if 'dual' in self.task:
            delparams.append('su_tagger')
            self.log('Fix parameters: su_tagger/*')
        if delparams:
            optimizer.add_hook(optimizers.DeleteGradient(delparams))

        optimizer.add_hook(chainer.optimizer.GradientClipping(self.args.grad_clip))

        if self.args.weight_decay > 0:
            optimizer.add_hook(chainer.optimizer.WeightDecay(self.args.weight_decay))

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
            if self.args.lr_decay and self.args.optimizer == 'sgd':
                if (e >= self.args.lr_decay_start and
                    (e - self.args.lr_decay_start) % self.args.lr_decay_width == 0):
                    lr_tmp = self.optimizer.lr
                    self.optimizer.lr = self.optimizer.lr * self.args.lr_decay_rate
                    self.log('Learning rate decay: {} -> {}\n'.format(lr_tmp, self.optimizer.lr))
                    self.report('Learning rate decay: {} -> {}'.format(lr_tmp, self.optimizer.lr))

            # learning and evaluation for each epoch
            self.run_epoch(self.train, train=True)

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('Finish: %s\n' % time)
        self.log('Finish: %s\n' % time)


    def run_eval_mode(self):
        self.log('<test result>')
        res = self.run_epoch(self.test, train=False)
        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('test\t%s\n' % res)
        self.report('Finish: %s\n' % time)
        self.log('Finish: %s\n' % time)


    def run_decode_mode(self):
        text_path = (self.args.path_prefix if self.args.path_prefix else '') + self.args.decode_data
        read_pos = 'pos_label' in self.dic.tables
        # read_pos = 'tag' in self.task or ('pos_embed_dim' in self.hparams and self.hparams['pos_embed_dim'] > 0)
        use_subtoken = ('subtoken_embed_dim' in self.hparams and self.hparams['subtoken_embed_dim'] > 0)
        subpos_depth = self.hparams['subpos_depth'] if 'subpos_depth' in self.hparams else -1
        lowercase = self.hparams['lowercase'] if 'lowercase' in self.hparams else False
        normalize_digits = self.hparams['normalize_digits'] if 'normalize_digits' else False

        rdata = data_io.load_decode_data(
            text_path, self.dic, self.task, data_format=self.args.input_data_format,
            read_pos=read_pos, use_subtoken=use_subtoken, subpos_depth=subpos_depth,
            lowercase=lowercase, normalize_digits=normalize_digits)

        if 'feature_template' in self.hparams and self.hparams['feature_template']:
            self.extract_features(rdata)

        file = open(self.args.output_data_path, 'w') if self.args.output_data_path else sys.stdout
        self.decode(rdata, read_pos=read_pos, file=file)

        if self.args.output_data_path:
            file.close()


    def run_interactive_mode(self):
        # to be implemented in sub-class
        pass

    
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
        n_ins = len(data.inputs[0])
        shuffle = True if train else False
        for ids in batch_generator(n_ins, batch_size=self.args.batch_size, shuffle=shuffle):
            inputs = self.gen_inputs(data, ids)

            xs = inputs[0]
            golds = inputs[self.label_begin_index:]
            if train:
                ret = classifier(*inputs, train=True)
            else:
                with chainer.no_backprop_mode():
                    ret = classifier(*inputs, train=False)
            loss = ret[0]
            outputs = ret[1:]

            if train:
                # timer.start()
                self.optimizer.target.cleargrads() # Clear the parameter gradients
                loss.backward()                    # Backprop
                loss.unchain_backward()            # Truncate the graph
                self.optimizer.update()            # Update the parameters
                # timer.stop()
                # print('time: backprop', timer.elapsed)
                # timer.reset()

                i_max = min(i + self.args.batch_size, n_ins)
                self.log('* batch %d-%d loss: %.4f' % ((i+1), i_max, loss.data))

                i = i_max

            n_sen += len(xs)
            total_loss += loss.data
            counts = self.evaluator.calculate(*[xs], *golds, *outputs)
            total_counts = evaluators.merge_counts(total_counts, counts)

            if (train and 
                self.n_iter > 0 and 
                self.n_iter * self.args.batch_size % self.args.step_size == 0):
                now_e = '%.3f' % (self.args.epoch_begin-1 + (self.n_iter * self.args.batch_size / n_ins))
                time = datetime.now().strftime('%Y%m%d_%H%M')

                self.log('\n### Finish %s iterations (%s examples: %s epoch)' % (
                    self.n_iter, (self.n_iter * self.args.batch_size), now_e))
                self.log('<training result for previous iterations>')
                res = self.evaluator.report_results(n_sen, total_counts, total_loss, file=self.logger)
                self.report('train\t%d\t%s\t%s' % (self.n_iter, now_e, res))

                if self.args.devel_data:
                    self.log('\n<development result>')
                    v_res = self.run_epoch(self.dev, train=False)
                    self.report('devel\t%d\t%s\t%s' % (self.n_iter, now_e, v_res))

                # save model
                if not self.args.quiet:
                    mdl_path = '{}/{}_e{}.npz'.format(constants.MODEL_DIR, self.start_time, now_e)
                    self.log('Save the model: %s\n' % mdl_path)
                    self.report('[INFO] save the model: %s\n' % mdl_path)
                    serializers.save_npz(mdl_path, classifier)
                    # print(self.classifier.predictor.su_tagger.unigram_embed.W.shape)
                    # print(self.classifier.predictor.su_tagger.mlp.layers[-1].W.shape)

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

        res = None if train else self.evaluator.report_results(n_sen, total_counts, total_loss, file=self.logger)
        return res


class TaggerTrainerBase(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)


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

                if (key == 'pretrained_unigram_embed_dim' or
                    key == 'unigram_embed_dim' or
                    key == 'subtoken_embed_dim' or
                    key == 'additional_feat_dim' or
                    key == 'rnn_n_layers' or
                    key == 'rnn_n_units' or
                    key == 'mlp_n_layers'or
                    key == 'mlp_n_units' or
                    key == 'subpos_depth' or
                    key == 'freq_threshold' or
                    key == 'max_vocab_size'
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

        self.hparams = hparams

        self.task = self.hparams['task']
        if self.external_embed_model:
            pretrained_unigram_embed_dim = self.embed_model.wv.syn0[0].shape[0]
            if hparams['pretrained_unigram_embed_dim'] != pretrained_unigram_embed_dim:
                self.external_embed_model = None
                self.log('Warning: loaded embedding model is discarded due to dimension confliction.')
        

    def show_training_data(self):
        train = self.train
        dev = self.dev
        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.inputs[0][0], train.inputs[0][-1]))
        self.log('# train_gold: {} ... {}\n'.format(train.outputs[0][0], train.outputs[0][-1]))
        t2i_tmp = list(self.dic.tables['unigram'].str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))
        if self.dic.has_table('subtoken'):
            st2i_tmp = list(self.dic.tables['subtoken'].str2id.items())
            self.log('# subtoken2id: {} ... {}\n'.format(st2i_tmp[:10], st2i_tmp[len(st2i_tmp)-10:]))
        if self.dic.has_table('seg_label'):
            id2seg = {v:k for k,v in self.dic.tables['seg_label'].str2id.items()}
            self.log('# seg labels: {}\n'.format(id2seg))
        if self.dic.has_table('pos_label'):
            id2pos = {v:k for k,v in self.dic.tables['pos_label'].str2id.items()}
            self.log('# pos labels: {}\n'.format(id2pos))
        
        self.report('[INFO] vocab: {}'.format(len(self.dic.tables['unigram'])))
        self.report('[INFO] data length: train={} devel={}'.format(
            len(train.inputs[0]), len(dev.inputs[0]) if dev else 0))


    def gen_inputs(self, data, ids, evaluate=True):
        xp = cuda.cupy if self.args.gpu >= 0 else np

        xs = [xp.asarray(data.inputs[0][j], dtype='i') for j in ids]
        ts = [xp.asarray(data.outputs[0][j], dtype='i') for j in ids] if evaluate else None
        fs = [data.featvecs[j] for j in ids] if self.hparams['feature_template'] else None

        if evaluate:
            return xs, fs, ts
        else:
            return xs, fs
        

    def decode(self, rdata, read_pos=False, file=sys.stdout):
        n_ins = len(rdata.inputs[0])
        orgseqs = rdata.orgdata[0]

        timer = util.Timer()
        timer.start()
        for ids in batch_generator(n_ins, batch_size=self.args.batch_size, shuffle=False):
            xs, fs = self.gen_inputs(rdata, ids, evaluate=False)
            os = [orgseqs[j] for j in ids]
            self.decode_batch(xs, fs, os, file=file)
        timer.stop()

        print('Parsed %d sentences. Elapsed time: %.4f sec (total) / %.4f sec (ave)' % (
            n_ins, timer.elapsed, timer.elapsed/n_ins), file=sys.stderr)


    def run_interactive_mode(self):
        read_pos = False
        lowercase = self.hparams['lowercase'] if 'lowercase' in self.hparams else False
        normalize_digits = self.hparams['normalize_digits'] if 'normalize_digits' else False
        use_subtoken = ('subtoken_embed_dim' in self.hparams and self.hparams['subtoken_embed_dim'] > 0)
        subpos_depth = self.hparams['subpos_depth'] if 'subpos_depth' in self.hparams else -1

        print('Please input text or type \'q\' to quit this mode:')
        while True:
            line = sys.stdin.readline().rstrip()
            if len(line) == 0:
                continue
            elif line == 'q':
                break

            rdata = data_io.parse_commandline_input(
                line, self.dic, self.task, read_pos=read_pos,
                lowercase=lowercase, normalize_digits=normalize_digits,
                use_subtoken=use_subtoken, subpos_depth=subpos_depth)

            if self.hparams['feature_template']:
                rdata.featvecs = self.feat_extractor.extract_features(ws[0], self.dic.tries['chunk'])

            xs, fs = self.gen_inputs(rdata, [0], evaluate=False)
            orgseqs = rdata.orgdata[0]

            self.decode_batch(xs, fs, orgseqs)
            if not self.args.output_empty_line:
                print(file=sys.stdout)


    def load_external_dictionary(self):
        if not self.dic and self.args.external_dic_path:
            edic_path = self.args.external_dic_path
            self.dic = data_io.load_external_dictionary(edic_path, read_pos=True)
            self.log('Load external dictionary: {}'.format(edic_path))
            self.log('Vocab size: {}'.format(len(self.dic.tables['unigram'])))
            
            if not edic_path.endswith('pickle'):
                base = edic_path.split('.')[0]
                edic_pic_path = base + '.pickle'
                with open(edic_pic_path, 'wb') as f:
                    pickle.dump(self.dic, f)
                self.log('Dumpped dictionary: {}'.format(edic_pic_path))
            self.log('')


class TaggerTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.label_begin_index = 2


    def load_model(self, model_path):
        super().load_model(model_path)
        if 'rnn_dropout' in self.hparams:
            self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
        if 'hidden_mlp_dropout' in self.hparams:
            self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])


    def init_hyperparameters(self, args):
        if self.external_embed_model:
            pretrained_unigram_embed_dim = self.embed_model.wv.syn0[0].shape[0]
        else:
            pretrained_unigram_embed_dim = 0

        self.hparams = {
            'pretrained_unigram_embed_dim' : pretrained_unigram_embed_dim,
            'unigram_embed_dim' : args.unigram_embed_dim,
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
            'task' : args.task,
            'subpos_depth' : args.subpos_depth,
            'lowercase' : args.lowercase,
            'normalize_digits' : args.normalize_digits,
            'freq_threshold' : args.freq_threshold,
            'max_vocab_size' : args.max_vocab_size,
        }

        self.log('Init hyperparameters')
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def setup_evaluator(self):
        if self.task == 'tag' and self.args.ignore_labels:
            tmp = self.args.ignore_labels
            self.args.ignore_labels = set()

            for label in tmp.split(','):
                label_id = self.dic.get_pos_label_id(label)
                if label_id >= 0:
                    self.args.ignore_labels.add(label_id)

            self.log('Setup evaluator: labels to be ignored={}\n'.format(self.args.ignore_labels))

        else:
            self.args.ignore_labels = set()

        self.evaluator = classifiers.init_evaluator(self.task, self.dic, self.args.ignore_labels)


    def decode_batch(self, xs, fs, orgseqs, file=sys.stdout):
        ys = self.classifier.decode(xs, fs)
        id2label = self.dic.tables['seg_label' if 'seg' in self.task else 'pos_label'].id2str

        for x_str, y in zip(orgseqs, ys):
            y_str = [id2label[int(yi)] for yi in y]

            if self.task == 'tag':
                res = ['{}{}{}'.format(xi_str, self.args.output_attr_delim, yi_str) 
                       for xi_str, yi_str in zip(x_str, y_str)]
                res = self.args.output_token_delim.join(res)

            elif self.task == 'seg':
                res = ['{}{}'.format(xi_str, self.args.output_token_delim 
                                     if (yi_str.startswith('E') or yi_str.startswith('S')) 
                                     else '') for xi_str, yi_str in zip(x_str, y_str)]
                res = ''.join(res).rstrip()

            elif self.task == 'seg_tag':
                res = ['{}{}'.format(
                    xi_str, 
                    (self.args.output_attr_delim+yi_str[2:]+self.args.output_token_delim) 
                    if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
                ) for xi_str, yi_str in zip(x_str, y_str)]
                res = ''.join(res).rstrip()

            else:
                print('Error: Invalid decode type', file=self.logger)
                sys.exit()

            print(res, file=file)
            if self.args.output_empty_line:
                print(file=file)


class DualTaggerTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.label_begin_index = 2
        self.su_tagger = None


    def load_submodel(self, model_path, pretrained_unigram_embed_dim=0):
        array = model_path.split('_e')
        dic_path = '{}.s2i'.format(array[0])
        hparam_path = '{}.hyp'.format(array[0])
        param_path = model_path

        # dictionary
        self.load_dic(dic_path)

        # hyper parameters
        self.load_hyperparameters(hparam_path)
        self.log('Load hyperparameters from short unit model: {}\n'.format(hparam_path))
        for k, v in self.args.__dict__.items():
            if 'dropout' in k:
                self.hparams[k] = 0.0
            elif k == 'task':
                task = 'dual_{}'.format(self.hparams[k])
                self.task = self.hparams[k] = task

        # sub model
        hparams_sub = self.hparams.copy()
        self.log('Initialize short unit model from hyperparameters\n')
        task_tmp = self.task.split('dual_')[1]
        classifier_tmp = classifiers.init_classifier(
            task_tmp, hparams_sub, self.dic)
        chainer.serializers.load_npz(model_path, classifier_tmp)
        self.su_tagger = classifier_tmp.predictor
        self.log('Load short unit model parameters: {}\n'.format(model_path))


    def load_hyperparameters(self, hparams_path):
        super().load_hyperparameters(hparams_path)


    def init_hyperparameters(self, args, pretrained_unigram_embed_dim=0):
        self.log('Initialize hyperparameters for full model\n')

        # keep most of loaded params from sub model and update some params from args
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            if 'dropout' in k:  # update
                self.hparams[k] = v
                message = '{}={}'.format(k, v)            

            elif k == 'submodel_usage':
                self.hparams[k] = v
                message = '{}={}'.format(k, v)            

            elif k in self.hparams and v != self.hparams[k]:
                message = '{}={} (input option value {} was discarded)'.format(k, self.hparams[k], v)

            else:
                message = '{}={}'.format(k, v)

            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def init_model(self, pretrained_unigram_embed_dim=0):
        self.log('Initialize model from hyperparameters\n')
        self.classifier = classifiers.init_classifier(self.task, self.hparams, self.dic)


    def load_model(self, model_path):
        super().load_model(model_path)
        if 'rnn_dropout' in self.hparams:
            self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
        if 'hidden_mlp_dropout' in self.hparams:
            self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])


    def update_model(self):
        if (self.args.execute_mode == 'train' or
            self.args.execute_mode == 'eval'):

            super().update_model()
            if self.su_tagger is not None:
                self.classifier.integrate_submodel(self.su_tagger)


    def setup_evaluator(self):
        if self.task == 'dual_tag' and self.args.ignore_labels:
            tmp = self.args.ignore_labels
            self.args.ignore_labels = set()

            for label in tmp.split(','):
                label_id = self.dic.get_pos_label_id(label)
                if label_id >= 0:
                    self.args.ignore_labels.add(label_id)

            self.log('Setup evaluator: labels to be ignored={}\n'.format(self.args.ignore_labels))

        else:
            self.args.ignore_labels = set()

        self.evaluator = classifiers.init_evaluator(self.task, self.dic, self.args.ignore_labels)


    def decode_batch(self, xs, fs, orgseqs, file=sys.stdout):
        su_sym = 'S\t' if self.args.output_data_format == 'wl' else ''
        lu_sym = 'L\t' if self.args.output_data_format == 'wl' else ''

        ys_short, ys_long = self.classifier.decode(xs, fs)

        id2token = self.dic.tables['unigram'].id2str
        id2label = self.dic.tables['seg_label' if 'seg' in self.task else 'pos_label'].id2str

        for x_str, sy, ly in zip(orgseqs, ys_short, ys_long):
            sy_str = [id2label[int(yi)] for yi in sy]
            ly_str = [id2label[int(yi)] for yi in ly]

            if self.task == 'dual_tag':
                res1 = ['{}{}{}'.format(xi_str, self.args.output_attr_delim, yi_str) 
                        for xi_str, yi_str in zip(x_str, sy_str)]
                res1 = ' '.join(res1)
                res2 = ['{}{}{}'.format(xi_str, self.args.output_attr_delim, yi_str) 
                        for xi_str, yi_str in zip(x_str, ly_str)]
                res2 = ' '.join(res2)

            elif self.task == 'dual_seg':
                res1 = [xi_str + self.args.output_token_delim + su_sym
                         if yi_str.startswith('E') or yi_str.startswith('S') else xi_str
                        for xi_str, yi_str in zip(x_str, ly_str)]
                res1 = ''.join(res1).rstrip('S\t\n')
                res2 = [xi_str + self.args.output_token_delim + lu_sym
                        if yi_str.startswith('E') or yi_str.startswith('S') else xi_str
                        for xi_str, yi_str in zip(x_str, sy_str)]
                res2 = ''.join(res2).rstrip('L\t\n')

            elif self.task == 'dual_seg_tag':
                res1 = ['{}{}'.format(
                    xi_str, 
                    (self.args.output_attr_delim+yi_str[2:]+self.args.output_token_delim) 
                    if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
                ) for xi_str, yi_str in zip(x_str, sy_str)]
                res1 = ''.join(res1).rstrip()
                res2 = ['{}{}'.format(
                    xi_str, 
                    (self.args.output_attr_delim+yi_str[2:]+self.args.output_token_delim) 
                    if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
                ) for xi_str, yi_str in zip(x_str, ly_str)]
                res2 = ''.join(res2).rstrip()

            else:
                print('Error: Invalid decode type', file=self.logger)
                sys.exit()

            if self.args.output_data_format == 'sl':
                print('S:{}'.format(res1), file=file)
                print('L:{}'.format(res2), file=file)
            else:
                print('S\t{}'.format(res1), file=file)
                print('L\t{}'.format(res2), file=file)
                print(file=file)


class ParserTrainer(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.label_begin_index = 3
        # elif args.task == 'tag_dep' or args.task == 'tag_tdep':
        #     self.label_begin_index = 2


    def load_model(self, model_path):
        super().load_model(model_path)
        if 'rnn_dropout' in self.hparams:
            self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
        if 'hidden_mlp_dropout' in self.hparams:
            self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])
        if 'pred_layers_dropout' in self.hparams:
            self.classifier.change_pred_layers_dropout_ratio(self.hparams['pred_layers_dropout'])
            

    def init_hyperparameters(self, args):
        if self.external_embed_model:
            pretrained_unigram_embed_dim = self.embed_model.wv.syn0[0].shape[0]
        else:
            pretrained_unigram_embed_dim = 0

        self.hparams = {
            'pretrained_unigram_embed_dim' : pretrained_unigram_embed_dim,
            'unigram_embed_dim' : args.unigram_embed_dim,
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
            'task' : args.task,
            'subpos_depth' : args.subpos_depth,
            'lowercase' : args.lowercase,
            'normalize_digits' : args.normalize_digits,
            'freq_threshold' : args.freq_threshold,
            'max_vocab_size' : args.max_vocab_size,
        }

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

                if (key == 'pretrained_unigram_embed_dim' or
                    key == 'unigram_embed_dim' or
                    key == 'subtoken_embed_dim' or
                    key == 'pretrained_unigram_embed_dim' or
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
                    key == 'subpos_depth' or
                    key == 'freq_threshold' or
                    key == 'max_vocab_size'
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

        self.hparams = hparams
        self.task = self.hparams['task']


    def show_training_data(self):
        train = self.train
        dev = self.dev
        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.inputs[0][0], train.inputs[0][-1]))
        self.log('# train_gold_head: {} ... {}\n'.format(train.outputs[1][0], train.outputs[1][-1]))
        if self.dic.has_table('arc_label'):
            self.log('# train_gold_label: {} ... {}\n'.format(train.outputs[2][0], train.outputs[2][-1]))
        t2i_tmp = list(self.dic.tables['unigram'].str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))

        if self.dic.has_table('subtoken'):
            st2i_tmp = list(self.dic.tables['subtoken'].str2id.items())
            self.log('# subtoken2id: {} ... {}\n'.format(st2i_tmp[:10], st2i_tmp[len(st2i_tmp)-10:]))
        if self.dic.has_table('seg_label'):
            id2seg = {v:k for k,v in self.dic.tables['seg_label'].str2id.items()}
            self.log('# seg labels: {}\n'.format(id2seg))
        if self.dic.has_table('pos_label'):
            id2pos = {v:k for k,v in self.dic.tables['pos_label'].str2id.items()}
            self.log('# pos labels: {}\n'.format(id2pos))
        if self.dic.has_table('arc_label'):
            id2arc = {v:k for k,v in self.dic.tables['arc_label'].str2id.items()}
            self.log('# arc labels: {}\n'.format(id2arc))
        
        self.report('[INFO] vocab: {}'.format(len(self.dic.tables['unigram'])))
        self.report('[INFO] data length: train={} devel={}'.format(
            len(train.inputs[0]), len(dev.inputs[0]) if dev else 0))


    def setup_evaluator(self):
        if (self.task == 'tdep' or self.task == 'tag_tdep') and self.args.ignore_labels:
            tmp = self.args.ignore_labels
            self.args.ignore_labels = set()

            for label in tmp.split(','):
                label_id = self.dic.tables['arc_label'].get_id(label)
                if label_id >= 0:
                    self.args.ignore_labels.add(label_id)

            self.log('Setup evaluator: labels to be ignored={}\n'.format(self.args.ignore_labels))

        else:
            self.args.ignore_labels = set()

        self.evaluator = classifiers.init_evaluator(self.task, self.dic, self.args.ignore_labels)


    def gen_inputs(self, data, ids, evaluate=True):
        xp = cuda.cupy if self.args.gpu >= 0 else np
        ws = [xp.asarray(data.inputs[0][j], dtype='i') for j in ids]
        cs = ([[xp.asarray(subs, dtype='i') for subs in data.inputs[1][j]] for j in ids] 
              if len(data.inputs) >= 2 else None)
        ps = ([xp.asarray(data.outputs[0][j], dtype='i') for j in ids] 
              if len(data.outputs) > 0 and data.outputs[0] else None)

        if evaluate:
            ths = [xp.asarray(data.outputs[1][j], dtype='i') for j in ids]
            if self.task == 'tdep' or self.task == 'tag_tdep':
                tls = [xp.asarray(data.outputs[2][j], dtype='i') for j in ids]
                return ws, cs, ps, ths, tls
            else:
                return ws, cs, ps, ths
        else:
            return ws, cs, ps


    def decode(self, rdata, read_pos=False, file=sys.stdout):
        n_ins = len(rdata.inputs[0])
        orgseqs = rdata.orgdata[0]
        orgposs = rdata.orgdata[1] if read_pos and len(rdata.orgdata) > 1 else None

        timer = util.Timer()
        timer.start()
        for ids in batch_generator(n_ins, batch_size=self.args.batch_size, shuffle=False):
            ws, cs, ps  = self.gen_inputs(rdata, ids, evaluate=False)
            ows = [orgseqs[j] for j in ids]
            ops = [orgposs[j] for j in ids] if orgposs else None
            self.decode_batch(ws, cs, ps, ows, ops, file=file)
        timer.stop()

        print('Parsed %d sentences. Elapsed time: %.4f sec (total) / %.4f sec (ave)' % (
            n_ins, timer.elapsed, timer.elapsed/n_ins), file=sys.stderr)


    def decode_batch(self, ws, cs, ps, orgseqs, orgposs, file=sys.stdout):
        use_pos = ps is not None
        label_prediction = self.task == 'tdep'

        n_ins = len(ws)
        ret = self.classifier.decode(ws, cs, ps, label_prediction)

        hs = ret[0]
        if label_prediction:
            ls = ret[1]
            id2label = self.dic.tables['arc_label'].id2str
        else:
            ls = [None] * len(ws)
        if not use_pos:
            ps = [None] * len(ws)
            orgposs = [None] * len(ws)

        for x_str, p_str, h, l in zip(orgseqs, orgposs, hs, ls):
            x_str = x_str[1:]
            p_str = p_str[1:] if p_str else None
            h = h[1:]
            if label_prediction:
                l = l[1:]
                l_str = [id2label[int(li)] for li in l] 

            if label_prediction:
                if use_pos:
                    res = ['{}\t{}\t{}\t{}'.format(
                        xi_str, pi_str, hi, li_str) for xi_str, pi_str, hi, li_str in zip(
                            x_str, p_str, h, l_str)]
                        
                else:
                    res = ['{}\t{}\t{}'.format(
                        xi_str, hi, li_str) for xi_str, hi, li_str in zip(x_str, h, l_str)]
            else:
                if use_pos:
                    res = ['{}\t{}\t{}'.format(
                        xi_str, pi_str, hi) for xi_str, pi_str, hi in zip(x_str, p_str, h)]
                else:
                    res = ['{}\t{}'.format(
                        xi_str, hi) for xi_str, hi in zip(x_str, h)]
            res = '\n'.join(res).rstrip()

            print(res+'\n', file=file)


    def run_interactive_mode(self):
        read_pos = 'pos_label' in self.dic.tables
        lowercase = self.hparams['lowercase'] if 'lowercase' in self.hparams else False
        normalize_digits = self.hparams['normalize_digits'] if 'normalize_digits' else False
        use_subtoken = ('subtoken_embed_dim' in self.hparams and self.hparams['subtoken_embed_dim'] > 0)
        subpos_depth = self.hparams['subpos_depth'] if 'subpos_depth' in self.hparams else -1

        print('Please input text or type \'q\' to quit this mode:')
        while True:
            line = sys.stdin.readline().rstrip()
            if len(line) == 0:
                continue
            elif line == 'q':
                break

            rdata = data_io.parse_commandline_input(
                line, self.dic, self.task, read_pos=read_pos and '/' in line,
                lowercase=lowercase, normalize_digits=normalize_digits,
                use_subtoken=use_subtoken, subpos_depth=subpos_depth)

            ws, cs, ps = self.gen_inputs(rdata, [0], evaluate=False)
            orgseqs = rdata.orgdata[0]
            orgposs = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None

            self.decode_batch(ws, cs, ps, orgseqs, orgposs)
            if not self.args.output_empty_line:
                print(file=sys.stdout)


class AttributeAnnotatorTrainer(Trainer):
    def __init__(self, args, logger=sys.stderr):
        args.task = 'attr'
        super().__init__(args, logger)
        self.label_begin_index = 2


    def init_hyperparameters(self, args):
        self.hparams = {
            'task' : args.task,
            'subpos_depth' : args.subpos_depth,
            'lowercase' : args.lowercase,
            'normalize_digits' : args.normalize_digits,
        }

        self.log('Init hyperparameters')
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def init_model(self):
        self.log('Initialize model from hyperparameters\n')
        self.classifier = classifiers.init_classifier(self.task, self.hparams, self.dic)


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

                if (key == 'subpos_depth'):
                    val = int(val)

                if (key == 'lowercase' or
                    key == 'normalize_digits'
                ):
                    val = (val.lower() == 'true')

                hparams[key] = val

        self.hparams = hparams
        self.task = self.hparams['task']


    def load_model(self, model_path):
        array = model_path.split('.')
        dic_path = '{}.s2i'.format(array[0])
        hparam_path = '{}.hyp'.format(array[0])
        param_path = model_path

        # dictionary
        self.load_dic(dic_path)

        # hyper parameters
        self.load_hyperparameters(hparam_path)
        self.log('Load hyperparameters: {}\n'.format(hparam_path))
        self.show_hyperparameters()

        # model
        self.init_model()
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        self.classifier = model
        self.log('Load model parameters: {}\n'.format(model_path))


    def update_model(self):
        # do nothing
        pass


    def show_training_data(self):
        train = self.train
        dev = self.dev
        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.inputs[0][0], train.inputs[0][-1]))
        self.log('# train_gold_attr: {} ... {}\n'.format(train.outputs[1][0], train.outputs[1][-1]))
        t2i_tmp = list(self.dic.tables['unigram'].str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))

        if self.dic.has_table('pos_label'):
            id2pos = {v:k for k,v in self.dic.tables['pos_label'].str2id.items()}
            self.log('# pos labels: {}\n'.format(id2pos))
        
        self.report('[INFO] vocab: {}'.format(len(self.dic.tables['unigram'])))
        self.report('[INFO] data length: train={} devel={}'.format(
            len(train.inputs[0]), len(dev.inputs[0]) if dev else 0))


    def gen_inputs(self, data, evaluate=True):
        ws = data.inputs[0]
        ps = data.outputs[0] if data.outputs[0] else None
        ls = data.outputs[1] if evaluate else None
        return ws, ps, ls


    def setup_evaluator(self):
        self.args.ignore_labels = set()
        self.evaluator = classifiers.init_evaluator(self.task, self.dic, self.args.ignore_labels)


    def setup_optimizer(self):
        # do nothing
        pass


    def run_step(self, data, train=True):
        ws, ps, ls = self.gen_inputs(data)
        if train:
            self.classifier.train(ws, ps, ls)
        ys = self.classifier.decode(ws, ps)
        counts = self.evaluator.calculate(*(ws, ls, ys))
        return counts


    def decode(self, rdata, file=sys.stdout):
        read_pos = True if len(rdata.outputs) > 0 else None
        ws = rdata.inputs[0]
        ps = rdata.outputs[0] if read_pos else None
        orgseqs = rdata.orgdata[0]
        orgposs = rdata.orgdata[1] if read_pos else None
        n_ins = len(ws)

        timer = util.Timer()
        timer.start()
        self.decode_batch(ws, ps, orgseqs, orgposs, file=file)
        timer.stop()

        print('Parsed %d sentences. Elapsed time: %.4f sec (total) / %.4f sec (ave)' % (
            n_ins, timer.elapsed, timer.elapsed/n_ins), file=sys.stderr)


    def decode_batch(self, ws, ps, orgseqs, orgposs, file=sys.stdout):
        ls = self.classifier.decode(ws, ps)

        use_pos = ps is not None
        if not use_pos:
            orgposs = [None] * len(ws)

        id2label = self.dic.tables['attr_label'].id2str

        for x_str, p_str, l in zip(orgseqs, orgposs, ls):
            l_str = [id2label[int(li)] for li in l] 

            if use_pos:
                res = ['{}\t{}\t{}'.format(
                    xi_str, pi_str, li_str) for xi_str, pi_str, li_str in zip(x_str, p_str, l_str)]
            else:
                res = ['{}\t{}'.format(xi_str, li_str) for xi_str, li_str in zip(x_str, l_str)]
            res = '\n'.join(res).rstrip()
                    
            print(res, file=file)


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

        # training
        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.log('Start epoch 1: {}\n'.format(time))
        self.report('[INFO] Start epoch 1 at {}'.format(time))

        t_counts = self.run_step(self.train)
        self.log('\n### Finish training (%s examples)' % (len(self.train.inputs[0])))
        self.log('<training result>')
        t_res = self.evaluator.report_results(
            len(self.train.inputs[0]), t_counts, loss=None, file=self.logger)
        self.report('train\t%s' % (t_res))

        if self.args.devel_data:
            self.log('\n<development result>')
            d_counts = self.run_step(self.dev, train=False)
            d_res = self.evaluator.report_results(
                len(self.dev.inputs[0]), d_counts, loss=None, file=self.logger)
            self.report('devel\t%s' % (d_res))

        # save model
        if not self.args.quiet:
            mdl_path = '{}/{}.pkl'.format(constants.MODEL_DIR, self.start_time)
            self.log('Save the model: %s\n' % mdl_path)
            self.report('[INFO] save the model: %s\n' % mdl_path)
            with open(mdl_path, 'wb') as f:
                pickle.dump(self.classifier, f)

        if not self.args.quiet:
            self.reporter.close() 
            self.reporter = open('{}/{}.log'.format(constants.LOG_DIR, self.start_time), 'a')

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('Finish: %s\n' % time)
        self.log('Finish: %s\n' % time)


    def run_eval_mode(self):
        self.log('<test result>')
        counts = self.run_step(self.test, train=False)
        res = self.evaluator.report_results(
            len(self.test.inputs[0]), counts, loss=None, file=self.logger)
        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('test\t%s\n' % res)
        self.report('Finish: %s\n' % time)
        self.log('Finish: %s\n' % time)


    def run_decode_mode(self):
        text_path = (self.args.path_prefix if self.args.path_prefix else '') + self.args.decode_data
        read_pos = 'pos_label' in self.dic.tables
        subpos_depth = self.hparams['subpos_depth'] if 'subpos_depth' in self.hparams else -1
        lowercase = self.hparams['lowercase'] if 'lowercase' in self.hparams else False
        normalize_digits = self.hparams['normalize_digits'] if 'normalize_digits' else False

        rdata = data_io.load_decode_data(
            text_path, self.dic, self.task, data_format=self.args.input_data_format,
            read_pos=read_pos, use_subtoken=False, subpos_depth=subpos_depth,
            lowercase=lowercase, normalize_digits=normalize_digits)

        file = open(self.args.output_data_path, 'w') if self.args.output_data_path else sys.stdout
        self.decode(rdata, file=file)

        if self.args.output_data_path:
            file.close()


    def run_interactive_mode(self):
        text_path = (self.args.path_prefix if self.args.path_prefix else '') + self.args.decode_data
        read_pos = 'pos_label' in self.dic.tables
        subpos_depth = self.hparams['subpos_depth'] if 'subpos_depth' in self.hparams else -1
        lowercase = self.hparams['lowercase'] if 'lowercase' in self.hparams else False
        normalize_digits = self.hparams['normalize_digits'] if 'normalize_digits' else False

        print('Please input text or type \'q\' to quit this mode:')
        while True:
            line = sys.stdin.readline().rstrip()
            if len(line) == 0:
                continue
            elif line == 'q':
                break

            read_pos_now = read_pos and '/' in line
            rdata = data_io.parse_commandline_input(
                line, self.dic, self.task, read_pos=read_pos_now,
                lowercase=lowercase, normalize_digits=normalize_digits,
                use_subtoken=False, subpos_depth=subpos_depth)

            ws = rdata.inputs[0]
            ps = rdata.outputs[0] if read_pos_now else None
            orgseqs = rdata.orgdata[0]
            orgposs = rdata.orgdata[1] if read_pos_now else None

            self.decode_batch(ws, ps, orgseqs, orgposs)
            if not self.args.output_empty_line:
                print(file=sys.stdout)


def batch_generator(len_data, batch_size=100, shuffle=True):
    perm = np.random.permutation(len_data) if shuffle else list(range(0, len_data))
    for i in range(0, len_data, batch_size):
        i_max = min(i + batch_size, len_data)
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
