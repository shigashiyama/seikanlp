import sys
import os
import pickle
from datetime import datetime

import chainer
from chainer import cuda
from gensim.models import word2vec, keyedvectors

import numpy as np

import common
import constants
import data_io
import classifiers
import evaluators
import features


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
        self.decode_txt = None
        self.hparams = None
        self.dic = None
        self.classifier = None
        self.evaluator = None
        self.feat_extractor = None
        self.optimizer = None
        self.n_iter = 1
        self.label_begin_index = 2

        self.log('Start time: {}\n'.format(self.start_time))
        if not self.args.quiet:
            self.reporter = open('{}/{}.log'.format(constants.LOG_DIR, self.start_time), 'a')


    def report(self, message):
        if not self.args.quiet:
            print(message, file=self.reporter)


    def log(self, message=''):
        print(message, file=self.logger)


    def close(self):
        if not self.args.quiet:
            self.reporter.close()


    def init_feature_extractor(self, use_gpu=False):
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


    def load_external_dictionary(self):
        # to be implemented in sub-class
        pass


    def load_external_embedding_models(self):
        # to be implemented in sub-class
        pass


    def init_model(self):
        self.log('Initialize model from hyperparameters\n')
        self.classifier = classifiers.init_classifier(self.task, self.hparams, self.dic)


    def load_dic(self, dic_path):
        with open(dic_path, 'rb') as f:
            self.dic = pickle.load(f)
        self.log('Load dic: {}'.format(dic_path))
        self.log('Num of tokens: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
        if self.dic.has_table(constants.BIGRAM):
            self.log('Num of bigrams: {}'.format(len(self.dic.tables[constants.BIGRAM])))
        if self.dic.has_table(constants.TOKEN_TYPE):
            self.log('Num of tokentypes: {}'.format(len(self.dic.tables[constants.TOKEN_TYPE])))
        if self.dic.has_table(constants.SUBTOKEN):
            self.log('Num of subtokens: {}'.format(len(self.dic.tables[constants.SUBTOKEN])))
        if self.dic.has_trie(constants.CHUNK):
            self.log('Num of chunks: {}'.format(len(self.dic.tries[constants.CHUNK])))
        if self.dic.has_table(constants.SEG_LABEL):
            self.log('Num of segmentation labels: {}'.format(len(self.dic.tables[constants.SEG_LABEL])))
        if self.dic.has_table(constants.POS_LABEL):
            self.log('Num of pos labels: {}'.format(len(self.dic.tables[constants.POS_LABEL])))
        if self.dic.has_table(constants.ARC_LABEL):
            self.log('Num of arc labels: {}'.format(len(self.dic.tables[constants.ARC_LABEL])))
        if self.dic.has_table(constants.ATTR_LABEL):
            self.log('Num of semantic attribute labels: {}'.format(len(self.dic.tables[constants.ATTR_LABEL])))
        self.log('')


    def load_model(self):
        model_path = self.args.model_path
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

            elif (k == 'task' and v == 'seg' and self.hparams[k] == constants.TASK_SEGTAG and 
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
        # to be implemented in sub-class
        pass


    def init_hyperparameters(self):
        # to be implemented in sub-class
        self.hparams = {}
        

    def load_hyperparameters(self, hparams_path):
        # to be implemented in sub-class
        pass


    def show_training_data(self):
        # to be implemented in sub-class
        pass


    def load_data_for_training(self, refer_tokens=set(), refer_chunks=set()):
        args = self.args
        hparams = self.hparams
        prefix = args.path_prefix if args.path_prefix else ''
            
        # refer_path   = prefix + args.label_reference_data
        train_path   = prefix + args.train_data
        dev_path     = prefix + (args.devel_data if args.devel_data else '')
        input_data_format = args.input_data_format

        task = hparams['task']
        use_pos = (
            common.is_tagging_task(task) or 
            (common.is_parsing_task(task) and 'pos_embed_dim' in hparams and hparams['pos_embed_dim'] > 0) or
            common.is_attribute_annotation_task(task))
        use_bigram = ('bigram_embed_dim' in hparams and hparams['bigram_embed_dim'] > 0)
        use_tokentype = ('tokentype_embed_dim' in hparams and hparams['tokentype_embed_dim'] > 0)        
        use_subtoken = ('subtoken_embed_dim' in hparams and hparams['subtoken_embed_dim'] > 0)
        use_chunk_trie = get_chunk_trie_flag(hparams)
        subpos_depth = hparams['subpos_depth'] if 'subpos_depth' in hparams else -1
        lowercase = hparams['lowercase'] if 'lowercase' in hparams else False
        normalize_digits = hparams['normalize_digits'] if 'normalize_digits' else False
        freq_threshold = hparams['freq_threshold'] if 'freq_threshold' in hparams else 1
        max_vocab_size = hparams['max_vocab_size'] if 'max_vocab_size' in hparams else -1
        max_chunk_len = self.hparams['max_chunk_len'] if 'max_chunk_len' in hparams else 0
        
        if args.train_data.endswith('pickle'):
            name, ext = os.path.splitext(train_path)
            train = data_io.load_pickled_data(name)
            self.log('Load dumpped data:'.format(train_path))
            #TODO update dic when re-training using another training data

        else:
            train, self.dic = data_io.load_annotated_data(
                train_path, task, input_data_format, train=True, use_pos=use_pos,
                use_bigram=use_bigram, use_tokentype=use_tokentype, use_subtoken=use_subtoken, 
                use_chunk_trie=use_chunk_trie, subpos_depth=subpos_depth, 
                lowercase=lowercase, normalize_digits=normalize_digits, 
                max_vocab_size=max_vocab_size, freq_threshold=freq_threshold, max_chunk_len=max_chunk_len, 
                dic=self.dic, refer_tokens=refer_tokens, refer_chunks=refer_chunks)
        
            # if self.args.dump_train_data:
            #     name, ext = os.path.splitext(train_path)
            #     if args.max_vocab_size > 0 or args.freq_threshold > 1:
            #         name = '{}_{}k'.format(name, int(len(self.dic.tables[constants.UNIGRAM]) / 1000))
            #     data_io.dump_pickled_data(name, train)
            #     self.log('Dumpped training data: {}.pickle'.format(name))

        self.log('Load training data: {}'.format(train_path))
        self.log('Data length: {}'.format(len(train.inputs[0])))
        self.log('Num of tokens: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
        if self.dic.has_table(constants.BIGRAM):
            self.log('Num of bigrams: {}'.format(len(self.dic.tables[constants.BIGRAM])))
        if self.dic.has_table(constants.TOKEN_TYPE):
            self.log('Num of tokentypes: {}'.format(len(self.dic.tables[constants.TOKEN_TYPE])))
        if self.dic.has_table(constants.SUBTOKEN):
            self.log('Num of subtokens: {}'.format(len(self.dic.tables[constants.SUBTOKEN])))
        if self.dic.has_trie(constants.CHUNK):
            self.log('Num of chunks: {}'.format(len(self.dic.tries[constants.CHUNK])))
        self.log()
        if 'feature_template' in self.hparams and self.hparams['feature_template']:
            self.log('Start feature extraction for training data\n')
            self.extract_features(train)

        if args.devel_data:
            use_pos = constants.POS_LABEL in self.dic.tables

            # dic can be updated if embed_model are used
            dev, self.dic = data_io.load_annotated_data(
                dev_path, task, input_data_format, train=False, use_pos=use_pos, 
                use_bigram=use_bigram, use_tokentype=use_tokentype, use_subtoken=use_subtoken, 
                use_chunk_trie=use_chunk_trie, subpos_depth=subpos_depth, 
                lowercase=lowercase, normalize_digits=normalize_digits, max_chunk_len=max_chunk_len, 
                dic=self.dic, refer_tokens=refer_tokens, refer_chunks=refer_chunks)

            self.log('Load development data: {}'.format(dev_path))
            self.log('Data length: {}'.format(len(dev.inputs[0])))
            self.log('Num of tokens: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
            if self.dic.has_table(constants.BIGRAM):
                self.log('Num of bigrams: {}'.format(len(self.dic.tables[constants.BIGRAM])))
            if self.dic.has_table(constants.TOKEN_TYPE):
                self.log('Num of tokentypes: {}'.format(len(self.dic.tables[constants.TOKEN_TYPE])))
            if self.dic.has_table(constants.SUBTOKEN):
                self.log('Num of subtokens: {}'.format(len(self.dic.tables[constants.SUBTOKEN])))
            if self.dic.has_trie(constants.CHUNK):
                self.log('Num of chunks: {}'.format(len(self.dic.tries[constants.CHUNK])))
            self.log()
            if 'feature_template' in self.hparams and self.hparams['feature_template']:
                self.log('Start feature extraction for development data\n')
                self.extract_features(dev)
        else:
            dev = None
        
        self.train = train
        self.dev = dev
        self.dic.create_id2strs()
        self.show_training_data()


    def load_test_data(self, refer_tokens=set(), refer_chunks=set()):
        args = self.args
        hparams = self.hparams
        test_path = (args.path_prefix if args.path_prefix else '') + args.test_data
        input_data_format = args.input_data_format

        task = hparams['task']
        use_pos = self.dic.has_table(constants.POS_LABEL)
        use_bigram = self.dic.has_table(constants.BIGRAM)
        use_tokentype = self.dic.has_table(constants.TOKEN_TYPE)
        use_subtoken = self.dic.has_table(constants.SUBTOKEN)
        use_chunk_trie = self.dic.has_trie(constants.CHUNK) #get_chunk_trie_flag(hparams)
        subpos_depth = hparams['subpos_depth'] if 'subpos_depth' in hparams else -1
        lowercase = hparams['lowercase'] if 'lowercase' in hparams else False
        normalize_digits = hparams['normalize_digits'] if 'normalize_digits' else False
        max_chunk_len = self.hparams['max_chunk_len'] if 'max_chunk_len' in hparams else 0

        test, self.dic = data_io.load_annotated_data(
            test_path, task, input_data_format, train=False, use_pos=use_pos,
            use_bigram=use_bigram, use_tokentype=use_tokentype,
            use_chunk_trie=use_chunk_trie, use_subtoken=use_subtoken, subpos_depth=subpos_depth, 
            lowercase=lowercase, normalize_digits=normalize_digits, max_chunk_len=max_chunk_len, 
            dic=self.dic, refer_tokens=refer_tokens, refer_chunks=refer_chunks)

        self.log('Load test data: {}'.format(test_path))
        self.log('Data length: {}'.format(len(test.inputs[0])))
        self.log('Num of tokens: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
        if self.dic.has_table(constants.BIGRAM):
            self.log('Num of bigrams: {}'.format(len(self.dic.tables[constants.BIGRAM])))
        if self.dic.has_table(constants.TOKEN_TYPE):
            self.log('Num of tokentypes: {}'.format(len(self.dic.tables[constants.TOKEN_TYPE])))
        if self.dic.has_table(constants.SUBTOKEN):
            self.log('Num of subtokens: {}'.format(len(self.dic.tables[constants.SUBTOKEN])))
        if self.dic.has_trie(constants.CHUNK):
            self.log('Num of chunks: {}'.format(len(self.dic.tries[constants.CHUNK])))
        self.log()

        if 'feature_template' in self.hparams and self.hparams['feature_template']:
            self.log('Start feature extraction for test data\n')
            self.extract_features(test)

        self.test = test
        self.dic.create_id2strs()


    def load_decode_data(self, refer_tokens=set(), refer_chunks=set()):
        text_path = (self.args.path_prefix if self.args.path_prefix else '') + self.args.decode_data
        if 'input_data_format' in self.args:
            data_format = self.args.input_data_format
        else:
            data_format = constants.SL_FORMAT

        use_pos = constants.POS_LABEL in self.dic.tables
        use_bigram = self.dic.has_table(constants.BIGRAM)
        use_tokentype = self.dic.has_table(constants.TOKEN_TYPE)
        use_subtoken = self.dic.has_table(constants.SUBTOKEN)
        use_chunk_trie = self.dic.has_trie(constants.CHUNK) #get_chunk_trie_flag(hparams)
        subpos_depth = self.hparams['subpos_depth'] if 'subpos_depth' in self.hparams else -1
        lowercase = self.hparams['lowercase'] if 'lowercase' in self.hparams else False
        normalize_digits = self.hparams['normalize_digits'] if 'normalize_digits' else False
        max_chunk_len = self.hparams['max_chunk_len'] if 'max_chunk_len' in self.hparams else 0

        rdata = data_io.load_decode_data(
            text_path, self.dic, self.task, data_format=data_format,
            use_pos=use_pos, use_bigram=use_bigram, use_tokentype=use_tokentype,
            use_chunk_trie=use_chunk_trie, use_subtoken=use_subtoken, 
            subpos_depth=subpos_depth, lowercase=lowercase, normalize_digits=normalize_digits, 
            max_chunk_len=max_chunk_len, refer_tokens=refer_tokens, refer_chunks=refer_chunks)

        self.log('Load decode data: {}'.format(text_path))
        self.log('Data length: {}'.format(len(rdata.inputs[0])))

        if 'feature_template' in self.hparams and self.hparams['feature_template']:
            self.log('Start feature extraction for decode data\n')
            self.extract_features(rdata)

        self.decode_data = rdata
        self.dic.create_id2strs()
        

    def extract_features(self, data):
        featvecs = self.feat_extractor.extract_features(data.inputs[0], self.dic.tries[constants.CHUNK])
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

        elif self.args.optimizer == 'adadelta':
            optimizer = chainer.optimizers.AdaDelta(rho=self.args.adadelta_rho)
            
        elif self.args.optimizer == 'rmsprop':
            optimizer = chainer.optimizers.RMSprop(lr=self.args.learning_rate, alpha=self.args.rmsprop_alpha)

        optimizer.setup(self.classifier)

        optimizer.add_hook(chainer.optimizer.GradientClipping(self.args.grad_clip))

        if self.args.weight_decay_ratio > 0:
            optimizer.add_hook(chainer.optimizer.WeightDecay(self.args.weight_decay_ratio))

        self.optimizer = optimizer

        return self.optimizer


    def run_train_mode(self):
        # save model
        if not self.args.quiet:
            hparam_path = '{}/{}.hyp'.format(constants.MODEL_DIR, self.start_time)
            with open(hparam_path, 'w') as f:
                for key, val in self.hparams.items():
                    print('{}={}'.format(key, val), file=f)
                self.log('Save hyperparameters: {}'.format(hparam_path))

            dic_path = '{}/{}.s2i'.format(constants.MODEL_DIR, self.start_time)
            with open(dic_path, 'wb') as f:
                pickle.dump(self.dic, f)
            self.log('Save string2index table: {}'.format(dic_path))

        for e in range(max(1, self.args.epoch_begin), self.args.epoch_end+1):
            time = datetime.now().strftime('%Y%m%d_%H%M')
            self.log('Start epoch {}: {}\n'.format(e, time))
            self.report('[INFO] Start epoch {} at {}'.format(e, time))

            # learning rate decay
            if self.args.optimizer == 'sgd' and self.args.sgd_lr_decay:
                if (e >= self.args.sgd_lr_decay_start and
                    (e - self.args.sgd_lr_decay_start) % self.args.sgd_lr_decay_width == 0):
                    lr_tmp = self.optimizer.lr
                    self.optimizer.lr = self.optimizer.lr * self.args.sgd_lr_decay_rate
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
        use_pos = constants.POS_LABEL in self.dic.tables
        file = open(self.args.output_data, 'w') if self.args.output_data else sys.stdout
        self.decode(self.decode_data, use_pos=use_pos, file=file)

        if self.args.output_data:
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
                self.n_iter * self.args.batch_size % self.args.break_point == 0):
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
                    self.report('[INFO] Save the model: %s\n' % mdl_path)
                    chainer.serializers.save_npz(mdl_path, classifier)

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


def get_chunk_trie_flag(hparams):
    if 'feature_template' in hparams and hparams['feature_template']:
        return True
    elif 'chunk_embed_dim' in hparams and hparams['chunk_embed_dim'] > 0:
        return True
    else:
        return False


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

    print("load embedding model: vocab=%d\n" % len(model.wv.vocab), file=sys.stderr)
    return model
