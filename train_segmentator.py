from datetime import datetime
import sys
import logging
import argparse
import copy
import numpy as np
import os
import pickle
import gc

import chainer
from chainer import serializers
from chainer import cuda

import util
import read_embedding as emb
from conlleval import conlleval


def batch_generator(instances, labels, batchsize, shuffle=True, xp=np):

    len_data = len(instances)
    perm = np.random.permutation(len_data) if shuffle else range(0, len_data)

    for i in range(0, len_data, batchsize):
        i_max = min(i + batchsize, len_data)
        xs = [xp.asarray(instances[perm[i]], dtype=np.int32) for i in range(i, i_max)]
        ts = [xp.asarray(labels[perm[i]], dtype=np.int32) for i in range(i, i_max)]

        # if not shuffle:        
        #     print(i,i_max)
        #     if i == 2000:
        #         for i in range(len(xs)):
        #             x = xs[i]
        #             t = ts[i]
        #             if len(x) != len(t):
        #                 print('*WARN*')
        #             print(i)
        #             print('x', len(x), x)
        #             print('t', len(t), t)

        yield xs, ts
    raise StopIteration


def evaluate(model, instances, labels, batchsize, xp=np):
    evaluator = model.copy()          # to use different state
    evaluator.predictor.train = False # dropout does nothing
    evaluator.compute_fscore = True

    count = 0
    num_tokens = 0
    total_loss = 0
    total_ecounts = conlleval.EvalCounts()

    for xs, ts in batch_generator(instances, labels, batchsize, shuffle=False, xp=xp):
        with chainer.no_backprop_mode():
            loss, ecounts = evaluator(xs, ts, train=False)

        num_tokens += sum([len(x) for x in xs])
        count += len(xs)
        total_loss += loss.data
        ave_loss = total_loss / num_tokens
        total_ecounts = conlleval.merge_counts(total_ecounts, ecounts)
        c = total_ecounts
        acc = conlleval.calculate_accuracy(c.correct_tags, c.token_counter)
        overall = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)
           
    print_results(total_loss, ave_loss, count, c, acc, overall)
    stat = 'n_sen: %d, n_token: %d, n_chunk: %d, n_chunk_p: %d' % (count, c.token_counter, c.found_correct, c.found_guessed)
    res = '%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%d\t%.4f\t%.5f' % ((100.*acc), (100.*overall.prec), (100.*overall.rec), (100.*overall.fscore), overall.tp, overall.fp, overall.fn, total_loss, ave_loss)
            
    return stat, res


def print_results(total_loss, ave_loss, count, c, acc, overall):
    print('total loss: %.4f' % total_loss)
    print('ave loss: %.5f'% ave_loss)
    print('#sen, #token, #chunk, #chunk_pred: %d %d %d %d' %
          (count, c.token_counter, c.found_correct, c.found_guessed))
    print('TP, FP, FN: %d %d %d' % (overall.tp, overall.fp, overall.fn))
    print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' % 
          (100.*acc, 100.*overall.prec, 100.*overall.rec, 100.*overall.fscore))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--cudnn', dest='use_cudnn', action='store_true')
    parser.add_argument('--eval', action='store_true', help='Evaluate on given model using input date without training')
    parser.add_argument('--limit', type=int, default=-1, help='Limit the number of samples for quick tests')
    parser.add_argument('--batchsize', '-b', type=int, default=20)
    # parser.add_argument('--bproplen', type=int, default=35, help='length of truncated BPTT')
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--resume', default='', help='Resume the training from snapshot')
    parser.add_argument('--resume_epoch', type=int, default=1, help='Resume the training from the epoch')
    parser.add_argument('--iter_to_report', '-i', type=int, default=10000)
    parser.add_argument('--rnn_layer', '-l', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--rnn_unit_type', default='lstm')
    parser.add_argument('--rnn_bidirection', action='store_true')
    parser.add_argument('--crf', action='store_true')
    parser.add_argument('--linear_activation', '-a', default='')
    parser.add_argument('--rnn_hidden_unit', '-u', type=int, default=800)
    parser.add_argument('--embed_dim', '-d', type=int, default=300)
    parser.add_argument('--embed_path', default='')
    parser.add_argument('--lc', type=int, default=0, help='Left context size')
    parser.add_argument('--rc', type=int, default=0, help='Right context size')
    parser.add_argument('--optimizer', '-o', default='sgd')
    parser.add_argument('--lr', '-r', type=float, default=1, help='Value of initial learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0, help='Momentum ratio')
    parser.add_argument('--lrdecay', default='', help='\'start:width:decay rate\'')
    parser.add_argument('--weightdecay', '-w', type=float, default=0, help='Weight decay ratio')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--gradclip', '-c', type=float, default=5,)
    parser.add_argument('--format', '-f', help='Format of input data')
    parser.add_argument('--lowercase',  action='store_true')
    parser.add_argument('--normalize_digits',  action='store_true')
    parser.add_argument('--subpos_depth', '-s', type=int, default=-1)
    parser.add_argument('--dir_path', '-p', help='Directory path of input data (train, valid and test)')
    parser.add_argument('--trained_data', default='')
    parser.add_argument('--train_data', '-t', help='Filename of training data')
    parser.add_argument('--validation_data', '-v', help='Filename of validation data')
    parser.add_argument('--test_data', help='Filename of test data')
    parser.add_argument('--tag_schema', default='BIES')
    parser.add_argument('--dump_train_data', action='store_true')
    parser.set_defaults(use_cudnn=False)
    parser.set_defaults(crf=False)
    parser.set_defaults(rnn_bidirection=False)
    parser.set_defaults(lowercase=False)
    parser.set_defaults(normalize_digits=False)
    parser.set_defaults(pickle_dump_train_data=False)
    args = parser.parse_args()
    if args.lrdecay:
        parser.ad_argument('lrdecay_start')
        parser.ad_argument('lrdecay_width')
        parser.ad_argument('lrdecay_rate')
        array = args.lrdecay.split(':')
        args.lrdecay_start = int(array[0])
        args.lrdecay_width = int(array[1])
        args.lrdecay_rate = float(array[2])

    # print('# bproplen: {}'.format(args.bproplen))
    print('# No log: {}'.format(args.nolog))
    print('# GPU: {}'.format(args.gpu))
    print('# cudnn: {}'.format(args.use_cudnn))
    print('# eval: {}'.format(args.eval))
    print('# limit: {}'.format(args.limit))
    print('# minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# iteration to report: {}'.format(args.iter_to_report))
    print('# rnn unit type: {}'.format(args.rnn_unit_type))
    print('# rnn bidirection: {}'.format(args.rnn_bidirection))
    print('# crf: {}'.format(args.crf))
    print('# linear_activation: {}'.format(args.linear_activation))
    print('# rnn_layer: {}'.format(args.rnn_layer))
    print('# embedding dimension: {}'.format(args.embed_dim))
    print('# rnn hidden unit: {}'.format(args.rnn_hidden_unit))
    print('# pre-trained embedding model: {}'.format(args.embed_path))
    print('# left context size: {}'.format(args.lc))
    print('# right context size: {}'.format(args.rc))
    print('# optimization algorithm: {}'.format(args.optimizer))
    print('# learning rate: {}'.format(args.lr))
    print('# learning rate decay: {}'.format(args.lrdecay))
    if args.lrdecay:
        print('# learning rate decay start:', lrdecay_start)
        print('# learning rate decay width:', lrdecay_width)
        print('# learning rate decay rate:',  lrdecay_rate)
    print('# momentum: {}'.format(args.momentum))
    print('# wieght decay: {}'.format(args.weightdecay))
    print('# dropout ratio: {}'.format(args.dropout))
    print('# gradient norm threshold to clip: {}'.format(args.gradclip))
    print('# lowercase: {}'.format(args.lowercase))
    print('# normalize digits: {}'.format(args.normalize_digits))
    print('# subpos depth: {}'.format(args.subpos_depth))
    print('# resume: {}'.format(args.resume))
    print('# epoch to resume: {}'.format(args.resume_epoch))
    print('# tag schema: {}'.format(args.tag_schema))
    print(args)
    print('')

    return args


def load_sequence_labeling_data(args, trainer):
    trained_path = args.dir_path + args.trained_data
    train_path = args.dir_path + args.train_data
    val_path = args.dir_path + (args.validation_data if args.validation_data else '')
    test_path = args.dir_path + (args.test_data if args.test_data else '')
    refer_vocab = embed_model.wv if args.embed_path else set()
    params = {}
    token2id = {}
    label2id = {}
    token2id_org = {}

    limit = args.limit if args.limit > 0 else -1

    if args.trained_data:
        if args.trained_data.endswith('pickle'):
            name, ext = os.path.splitext(args.dir_path + args.trained_data)
            trained, trained_t, token2id, label2id, params = util.load_pickled_data(name)
            trainer.log('INFO: params read from pickle: {}'.format(params))
        else:
            trained, trained_t, token2id, label2id = util.read_data(
                args.format, trained_path, subpos_depth=args.subpos_depth, schema=args.tag_schema, 
                limit=limit, lowercase=args.lowercase, normalize_digits=args.normalize_digits)
        token2id_org = token2id.copy()
        print('vocab size:', len(token2id))
    else:
        trained = []
        trained_t = []

    if args.train_data.endswith('pickle'):
        name, ext = os.path.splitext(args.dir_path + args.train_data)
        if len(trained) == 0:
            #TODO label2id が update されるとエラーになるが、未対応
            train, train_t, token2id, label2id, params = util.load_pickled_data(name)
        else:
            train, train_t, _, _, _ = util.load_pickled_data(
                name, load_token2id=False, load_label2id=False, load_params=False)
    else:
        train, train_t, token2id, label2id = util.read_data(
            args.format, train_path, token2id=token2id, label2id=label2id,
            subpos_depth=args.subpos_depth, schema=args.tag_schema, 
            limit=limit, lowercase=args.lowercase, normalize_digits=args.normalize_digits)
    print('vocab size:', len(token2id))
    if not token2id_org:
        token2id_org = token2id

    if args.validation_data:
        if args.validation_data.endswith('pickle'):
            name, ext = os.path.splitext(args.dir_path + args.validation_data)
            val, val_t, _, _, _ = util.load_pickled_data(
                name, load_token2id=False, load_label2id=False, load_params=False)
        else:
            val, val_t, token2id, label2id = util.read_data(
                args.format, val_path, token2id=token2id, label2id=label2id, update_token=False, 
                refer_vocab=refer_vocab, subpos_depth=args.subpos_depth, schema=args.tag_schema, limit=limit, 
                lowercase=args.lowercase, normalize_digits=args.normalize_digits)
        print('vocab size:', len(token2id))        
    else:
        val = []

    if args.test_data:
        if args.test_data.endswith('pickle'):
            name, ext = os.path.splitext(args.dir_path + args.test_data)
            test, test_t, _, _, _ = util.load_pickled_data(
                name, load_token2id=False, load_label2id=False, load_params=False)
        else:
            test, test_t, token2id, label2id = util.read_data(
                args.format, test_path, token2id=token2id, label2id=label2id, update_token=False,
                refer_vocab=refer_vocab, schema=args.tag_schema, limit=limit, 
                lowercase=args.lowercase, normalize_digits=args.normalize_digits)
        print('vocab size:', len(token2id))
    else:
        test = []

    n_train = len(train)
    t2i_tmp = list(token2id.items())
    # id2token = {v:k for k,v in token2id.items()}
    # id2label = {v:k for k,v in label2id.items()}
    print('vocab =', len(token2id))
    print('data length: trained=%d train=%d val=%d test=%d' % (len(trained), n_train, len(val), len(test)))
    print()
    print('train:', train[:3], '...', train[n_train-3:])
    print('train_t:', train_t[:3], '...', train[n_train-3:])
    print()
    print('token2id:', t2i_tmp[:3], '...', t2i_tmp[len(t2i_tmp)-3:])
    print('label2id:', label2id)
    print()
    trainer.log('INFO: vocab = %d\n' % len(token2id))
    trainer.log('INFO: data length: train=%d val=%d\n' % (n_train, len(val)))

    return trained, trained_t, train, train_t, val, val_t, test, test_t, token2id, label2id, token2id_org, params


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.start_time = datetime.now().strftime('%Y%m%d_%H%M')
        self.logger = None

        print('INFO: start: {}\n'.format(self.start_time))
        if not self.args.nolog:
            self.logger = open('log/{}.log'.format(self.start_time), 'a')
            self.logger.write('{}\n'.format(args))
        
    def log(self, message):
        if not self.args.nolog:
            self.logger.write(message)


    def prepare_gpu(self):
        if self.args.gpu >= 0:
            # Make the specified GPU current
            cuda.get_device_from_id(self.args.gpu).use()
            chainer.config.cudnn_deterministic = self.args.use_cudnn


    def set_loaded_data(self, train, train_t, val, val_t, test, test_t, token2id, label2id):
        self.train = train
        self.train_t = train_t
        self.val = val
        self.val_t = val_t
        self.test = test
        self.test_t = test_t
        self.token2id = token2id
        self.label2id = label2id


    def prepare_model_and_parameters(self, embed_model=None, token2id_org=None, params={}):
        # parameter
        if not params:
            params = {'embed_dim' : self.args.embed_dim,
                      'rnn_layer' : self.args.rnn_layer,
                      'rnn_hidden_unit' : self.args.rnn_hidden_unit,
                      'rnn_bidirection' : self.args.rnn_bidirection,
                      'crf' : self.args.crf,
                      'linear_activation' : self.args.linear_activation,
                      'left_contexts' : self.args.lc,
                      'right_contexts' : self.args.rc,
                      'dropout' : self.args.dropout,
                      'tag_schema' : self.args.tag_schema
            }
        if self.args.embed_path:
            params.update({'embed_path' : self.args.embed_path})

        # model
        if self.args.eval or len(token2id) == len(token2id_org):
            t2i_for_model = token2id
            t2iu_for_model = None
        else:
            t2i_for_model = token2id_org
            t2iu_for_model = token2id

        model = util.load_model_from_params(
            params, model_path=self.args.resume, token2id=t2i_for_model, token2id_updated=t2iu_for_model,
            embed_model=embed_model, label2id=label2id, gpu=self.args.gpu)
        model.compute_fscore = True

        self.params = params
        self.model = model

        return self.model


    def setup_optimizer(self):
        if self.args.optimizer == 'sgd':
            if self.args.momentum <= 0:
                optimizer = chainer.optimizers.SGD(lr=self.args.lr)
            else:
                optimizer = chainer.optimizers.MomentumSGD(lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = chainer.optimizers.Adam()
        elif self.args.optimizer == 'adagrad':
            optimizer = chainer.optimizers.AdaGrad(lr=self.args.lr)

        optimizer.setup(self.model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(self.args.gradclip))
        if self.args.weightdecay > 0:
            optimizer.add_hook(chainer.optimizer.WeightDecay(self.args.weightdecay))

        self.optimizer = optimizer

        return self.optimizer


    def write_params_and_indices(self):
        if not self.args.nolog:
            self.params.update({
                'token2id_path' : 'nn_model/vocab_' + self.start_time + '.t2i.txt',
                'label2id_path' : 'nn_model/vocab_' + self.start_time + '.l2i.txt',
                'model_date': self.start_time
            })
            util.write_param_file(self.params, 'nn_model/param_' + self.start_time + '.txt')
            util.write_map(self.token2id, self.params['token2id_path'])
            util.write_map(self.label2id, self.params['label2id_path'])


    def dump_train_data_and_params(self):
        if self.args.dump_train_data and not self.args.train_data.endswith('pickle'):
            name, ext = os.path.splitext(self.args.dir_path + self.args.train_data)
            util.dump_pickled_data(name, self.train, self.train_t, self.token2id, self.label2id, self.params)
            print('pickled training data')
    

    def run(self):
        if self.args.eval:
            self.run_evaluation()
        else:
            self.run_training()


    def run_evaluation(self):
        #    if args.eval:
        print('<test result>')
        xp = cuda.cupy if args.gpu >= 0 else np
        te_stat, te_res = evaluate(self.model, self.test, self.test_t, self.args.batchsize, xp=xp)
        print()

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.log('finish: %s\n' % time)
        print('finish: %s\n' % time)


    def run_training(self):
        n_train = len(self.train)

        n_iter_report = self.args.iter_to_report
        n_iter = 0
        for e in range(max(1, self.args.resume_epoch), self.args.epoch+1):
            time = datetime.now().strftime('%Y%m%d_%H%M')
            self.log('INFO: start epoch %d at %s\n' % (e, time))
            print('start epoch %d: %s' % (e, time))

            # learning rate decay

            if self.args.lrdecay and self.args.optimizer == 'sgd':
                if (e - lrdecay_start) % lrdecay_width == 0:
                    lr_tmp = optimizer.lr
                    optimizer.lr = optimizer.lr * lrdecay_rate
                    if not self.args.nolog:
                        logger.write('learning decay: %f -> %f' % (lr_tmp, optimizer.lr))
                    print('learning decay: %f -> %f' % (lr_tmp, optimizer.lr))

            count = 0
            total_loss = 0
            num_tokens = 0
            total_ecounts = conlleval.EvalCounts()

            i = 0
            xp = cuda.cupy if args.gpu >= 0 else np
            for xs, ts in batch_generator(train, train_t, self.args.batchsize, shuffle=True, xp=xp):
                loss, ecounts = self.model(xs, ts, train=True)
                num_tokens += sum([len(x) for x in xs])
                count += len(xs)
                total_loss += loss.data
                total_ecounts = conlleval.merge_counts(total_ecounts, ecounts)
                i_max = min(i + self.args.batchsize, n_train)
                print('* batch %d-%d loss: %.4f' % ((i+1), i_max, loss.data))
                i = i_max
                n_iter += 1

                optimizer.target.cleargrads() # Clear the parameter gradients
                loss.backward()               # Backprop
                loss.unchain_backward()       # Truncate the graph
                optimizer.update()            # Update the parameters

                # Evaluation
                if (n_iter * self.args.batchsize) % n_iter_report == 0: # or i == n_train:

                    now_e = '%.2f' % (n_iter * self.args.batchsize / n_train)
                    time = datetime.now().strftime('%Y%m%d_%H%M')
                    print()
                    print('### iteration %s (epoch %s)' % ((n_iter * self.args.batchsize), now_e))
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
                        logger.write('INFO: train - %s\n' % t_stat)
                        if n_iter == 1:
                            self.logger.write('data\titer\ep\tacc\tprec\trec\tfb1\tTP\tFP\tFN\ttloss\taloss\n')
                        self.logger.write('train\t%d\t%s\t%s\n' % (n_iter, now_e, t_res))

                    if self.args.validation_data:
                        print('<validation result>')
                        v_stat, v_res = evaluate(self.model, val, val_t, self.args.batchsize, xp=xp)
                        print()

                        if not self.args.nolog:
                            self.logger.write('INFO: valid - %s\n' % v_stat)
                            if n_iter == 1:
                                self.logger.write('data\titer\ep\tacc\tprec\trec\tfb1\tTP\tFP\tFN\ttloss\taloss\n')
                            self.logger.write('valid\t%d\t%s\t%s\n' % (n_iter, now_e, v_res))

                    # Save the model
                    mdl_path = 'nn_model/rnn_%s_i%s.mdl' % (self.start_time, now_e)
                    print('save the model: %s\n' % mdl_path)
                    if not self.args.nolog:
                        self.logger.write('INFO: save the model: %s\n' % mdl_path)
                        serializers.save_npz(mdl_path, self.model)
        
                    # Reset counters
                    count = 0
                    total_loss = 0
                    num_tokens = 0
                    total_ecounts = conlleval.EvalCounts()

                    if not self.args.nolog:
                        self.logger.close() # 一度保存しておく
                        self.logger = open('log/' + self.start_time + '.log', 'a')

        time = datetime.now().strftime('%Y%m%d_%H%M')
        log('finish: %s\n' % time)


if __name__ == '__main__':
    if chainer.__version__[0] != '2':
        print("chainer version>=2.0.0 is required.")
        sys.exit()

    # get arguments and set up trainer

    args = parse_arguments()
    trainer = Trainer(args)
    trainer.prepare_gpu()

    # Load word embedding model, dataset

    embed_model = emb.read_model(args.embed_path) if args.embed_path else None

    trained, trained_t, train, train_t, val, val_t, test, test_t, token2id, label2id, token2id_org, params = load_sequence_labeling_data(trainer.args, trainer)
    trainer.set_loaded_data(train, train_t, val, val_t, test, test_t, token2id, label2id)

    # Set up model and optimizer

    model = trainer.prepare_model_and_parameters(
        embed_model=embed_model, token2id_org=token2id_org, params=params)

    optimizer = trainer.setup_optimizer()

    # Dump objects

    trainer.write_params_and_indices()
    trainer.dump_train_data_and_params()

    # Run

    trainer.run()
    trainer.logger.close()
