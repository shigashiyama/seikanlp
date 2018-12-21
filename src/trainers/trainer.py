import sys
import os
import pickle
import copy
from datetime import datetime

import chainer
from chainer import cuda
from gensim.models import word2vec, keyedvectors

import numpy as np

import common
import constants
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
        self.decode_data = None
        self.hparams = None
        self.dic = None
        self.dic_dev = None
        self.data_loader = None
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
        self.setup_classifier()


    def reinit_model(self):
        self.log('Re-initialize model from hyperparameters\n')
        self.setup_classifier()


    def setup_classifier(self):
        # to be implemented in sub-class
        pass


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
        for i in range(3):      # tmp
            if self.dic.has_table(constants.ATTR_LABEL(i)):
                self.log('Num of {}-th attribute labels: {}'.format(
                    i, len(self.dic.tables[constants.ATTR_LABEL(i)])))
        if self.dic.has_table(constants.ARC_LABEL):
            self.log('Num of arc labels: {}'.format(len(self.dic.tables[constants.ARC_LABEL])))
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
        self.reinit_model()
        chainer.serializers.load_npz(model_path, self.classifier)
        self.log('Load model parameters: {}\n'.format(model_path))


    def show_hyperparameters(self):
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            if (k in self.hparams and
                ('dropout' in k or 'freq_threshold' in k or 'max_vocab_size' in k or
                 k == 'attr_indexes' )):
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


    def update_model(self, classifier=None, dic=None):
        # to be implemented in sub-class
        pass


    def init_hyperparameters(self):
        # to be implemented in sub-class
        self.hparams = {}
        

    def load_hyperparameters(self, hparams_path):
        # to be implemented in sub-class
        pass


    def load_training_and_development_data(self):
        self.load_data('train')
        if self.args.devel_data:
            self.load_data('devel')
        self.show_training_data()


    def load_test_data(self):
        self.load_data('test')


    def load_decode_data(self):
        self.load_data('decode')


    def load_data(self, data_type):
        self.setup_data_loader()
        if data_type == 'train':
            data_path = self.args.path_prefix + self.args.train_data
            data, self.dic = self.data_loader.load_gold_data(
                data_path, self.args.input_data_format, dic=self.dic, train=True)
            self.dic.create_id2strs()
            self.train = data

        elif data_type == 'devel':
            data_path = self.args.path_prefix + self.args.devel_data
            self.dic_dev = copy.deepcopy(self.dic)
            data, self.dic_dev = self.data_loader.load_gold_data(
                data_path, self.args.input_data_format, dic=self.dic_dev, train=False)
            self.dic_dev.create_id2strs()
            self.dev = data

        elif data_type == 'test':
            data_path = self.args.path_prefix + self.args.test_data
            data, self.dic = self.data_loader.load_gold_data(
                data_path, self.args.input_data_format, dic=self.dic, train=False)
            self.dic.create_id2strs()
            self.test = data

        elif data_type == 'decode':
            data_path = self.args.path_prefix + self.args.decode_data
            data = self.data_loader.load_decode_data(
                data_path, self.args.input_data_format, dic=self.dic)
            self.dic.create_id2strs()
            self.decode_data = data
        
        else:
            print('Error: incorect data type: {}'.format(), file=sys.stderr)
            sys.exit()

        if self.hparams['feature_template'] and self.args.batch_feature_extraction: # for segmentation
            data.featvecs = self.feat_extractor.extract_features(data.inputs[0], self.dic.tries[constants.CHUNK])
            self.log('Extract dictionary features for {}'.format(data_type))

        self.log('Load {} data: {}'.format(data_type, data_path))
        self.show_data_info(data_type)
        

    def show_data_info(self, data_type):
        # to be implemented in sub-class
        pass


    def show_training_data(self):
        # to be implemented in sub-class
        pass


    def setup_data_loader(self):
        # to be implemented in sub-class
        pass


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
        self.update_model(classifier=self.classifier, dic=self.dic)
        self.decode(self.decode_data, 
                    file=open(self.args.output_data, 'w') if self.args.output_data else sys.stdout)
        if self.args.output_data:
            file.close()


    def run_interactive_mode(self):
        # to be implemented in sub-class
        pass

    
    def decode(self, rdata, file=sys.stdout):
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
            self.update_model(classifier=classifier, dic=self.dic_dev)
            classifier.change_dropout_ratio(0)

        n_sen = 0
        total_loss = 0
        total_counts = None

        i = 0
        n_ins = len(data.inputs[0])
        # shuffle = False         # tmp
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
                self.optimizer.target.cleargrads() # Clear the parameter gradients
                loss.backward()                    # Backprop
                loss.unchain_backward()            # Truncate the graph
                self.optimizer.update()            # Update the parameters
                i_max = min(i + self.args.batch_size, n_ins)

                if self.args.optimizer == 'sgd' and self.args.sgd_cyclical_lr:
                    lr_tmp = self.optimizer.lr
                    lr = get_triangular_lr(
                        self.n_iter * self.args.batch_size, self.args.clr_stepsize, 
                        self.args.clr_min_lr, self.args.clr_max_lr)
                    self.optimizer.lr = lr
                    self.log('* batch %d-%d loss: %.4f lr: %.4f -> %.4f' % (
                        (i+1), i_max, loss.data, lr_tmp, self.optimizer.lr))
                else:
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


def get_triangular_lr(n_ins, stepsize, min_lr, max_lr):
    slope = (max_lr - min_lr) / stepsize
    position = n_ins % (stepsize * 2)
    if position <= stepsize:
        lr = min_lr + slope * position
    else:
        lr = max_lr - slope * (position - stepsize)
    return lr


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
