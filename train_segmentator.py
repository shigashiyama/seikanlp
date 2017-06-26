"""
- (Zaremba 2015) の著者らによる torch 実装
  - (Zaremba 2015) RECURRENT NEURAL NETWORK REGULARIZATION, ICLR
  - https://github.com/tomsercu/lstm
-> 同等のプログラムの PFN による chainer 実装 -> 改造

"""

from __future__ import division
from __future__ import print_function
from datetime import datetime
import sys
import logging
import argparse
import copy
import numpy as np

import chainer
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L
from chainer import link
from chainer import serializers
from chainer import training
from chainer import reporter as reporter_module
from chainer import cuda
from chainer.training import extensions
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer.links.connection.n_step_rnn import argsort_list_descent, permutate_list

import util
from conlleval import conlleval


class RNN_CRF(chainer.Chain):
    def __init__(self, n_lstm_layers, n_vocab, n_lt_units, n_lstm_units, n_labels, dropout, use_bi_direction):
        super(RNN_CRF, self).__init__()
        with self.init_scope():
            self.lookup = L.EmbedID(n_vocab, n_lt_units)
            self.lstm = L.NStepBiLSTM(n_lstm_layers, n_lt_units, n_lstm_units, dropout) if use_bi_direction else L.NStepLSTM(n_lstm_layers, n_lt_units, n_lstm_units, dropout)
            self.linear = L.Linear(n_lstm_units * (2 if use_bi_direction else 1), n_labels)
            self.crf = L.CRF1d(n_labels)

    def __call__(self, xs, ts, train=True):
        xs = [self.lookup(x) for x in xs]

        with chainer.using_config('train', train):
            # lstm layers
            hy, cy, hs = self.lstm(None, None, xs)

            # linear layer
            hs = [self.linear(h) for h in hs]

            # crf layer
            indices = argsort_list_descent(hs)
            trans_hs = F.transpose_sequence(permutate_list(hs, indices, inv=False))
            trans_ts = F.transpose_sequence(permutate_list(ts, indices, inv=False))
            loss = self.crf(trans_hs, trans_ts)
            score, trans_ys = self.crf.argmax(trans_hs)
            ys = permutate_list(F.transpose_sequence(trans_ys), indices, inv=True)

        return loss, ys


# class RNN_v2(chainer.Chain):
#     def __init__(self, n_lstm_layers, n_vocab, n_lt_units, n_lstm_units, n_labels, dropout, use_bi_direction):
#         l1 = L.NStepBiLSTM(n_lstm_layers, n_lt_units, n_lstm_units, dropout) if use_bi_direction else L.NStepLSTM(n_lstm_layers, n_lt_units, n_lstm_units, dropout)
#         super(RNN_v2, self).__init__(
#             embed = L.EmbedID(n_vocab, n_lt_units),
#             l1 = l1,
#             l2 = L.Linear(n_lstm_units * (2 if use_bi_direction else 1), n_labels),
#         )
        
#     def __call__(self, hx, cx, xs, train=True):
#         xs = [self.embed(x) for x in xs]
#         with chainer.using_config('train', train):
#             hy, cy, ys = self.l1(hx, cx, xs)
#         ys = [self.l2(y) for y in ys]

#         return hy, cy, ys


class SequenceLabelingClassifier(link.Chain):
    compute_fscore = True

    def __init__(self, predictor, id2label):
        super(SequenceLabelingClassifier, self).__init__(predictor=predictor)
        self.id2label = id2label
        
    def __call__(self, *args, **kwargs):
        assert len(args) >= 2
        xs = args[0]
        ts = args[1]
        loss, ys = self.predictor(*args, **kwargs)

        if self.compute_fscore:
            eval_counts = None
            for x, t, y in zip(xs, ts, ys):
                generator = self.generate_lines(x, t, y.data)
                eval_counts = conlleval.merge_counts(eval_counts, conlleval.evaluate(generator))

        return loss, eval_counts

    def generate_lines(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                raise StopIteration
            x_str = str(x[i])
            #print(x_str, t[i], y[i])
            t_str = self.id2label[int(t[i])]
            y_str = self.id2label[int(y[i])]

            yield [x_str, t_str, y_str]

            i += 1
        

def batch_generator(instances, labels, batchsize, shuffle=True, xp=np):

    len_data = len(instances)
    perm = np.random.permutation(len_data) if shuffle else range(0, len_data)
    for i in range(0, len_data, batchsize):
        i_max = min(i + batchsize, len_data)
        xs = [xp.asarray(instances[perm[i]], dtype=np.int32) for i in range(i, i_max)]
        ts = [xp.asarray(labels[perm[i]], dtype=np.int32) for i in range(i, i_max)]

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
        if chainer.__version__[0] == '2':
            with chainer.no_backprop_mode():
                loss, ecounts = evaluator(xs, ts, train=False)
        else:
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


def main():

    # get arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--cudnn', dest='use_cudnn', action='store_true')
    parser.add_argument('--batchsize', '-b', type=int, default=20)
    # parser.add_argument('--bproplen', type=int, default=35, help='length of truncated BPTT')
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--resume', default='', help='Resume the training from snapshot')
    parser.add_argument('--resume_epoch', type=int, default=1, help='Resume the training from the epoch')
    parser.add_argument('--iter_to_report', '-i', type=int, default=10000)
    parser.add_argument('--layer', '-l', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--bidirection', action='store_true')
    parser.add_argument('--hidden_unit', '-u', type=int, default=650)    
    parser.add_argument('--lookup_dim', '-d', type=int, default=650)
    parser.add_argument('--optimizer', '-o', default='sgd')
    parser.add_argument('--lr', '-r', type=float, default=1, help='Value of initial learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum ratio')
    parser.add_argument('--lrdecay', type=float, default=0.05, help='Coefficient for learning rate decay')
    parser.add_argument('--weightdecay', '-w', type=float, default=0.1, help='Weight decay ratio')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--gradclip', '-c', type=float, default=5,)
    parser.add_argument('--test', type=int, default=-1, help='Use tiny datasets for quick tests')
    parser.add_argument('--format', '-f', help='Format of input data')
    parser.add_argument('--subpos_depth', '-s', type=int, default=-1)
    parser.add_argument('--dir_path', '-p', help='Directory path of input data (train, valid and test)')
    parser.add_argument('--train_data', '-t', help='Filename of training data')
    parser.add_argument('--validation_data', '-v', help='Filename of validation data')
    parser.add_argument('--tag_schema', default='BI')
    parser.set_defaults(use_cudnn=False)
    parser.set_defaults(bidirection=False)
    args = parser.parse_args()

    print('# GPU: {}'.format(args.gpu))
    print('# cudnn: {}'.format(args.use_cudnn))
    print('# minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# iteration to report: {}'.format(args.iter_to_report))
    # print('# bproplen: {}'.format(args.bproplen))
    print('# bidirection: {}'.format(args.bidirection))
    print('# layer: {}'.format(args.layer))
    print('# lookup table dimension: {}'.format(args.lookup_dim))
    print('# hiddlen unit: {}'.format(args.hidden_unit))
    print('# optimization algorithm: {}'.format(args.optimizer))
    print('# learning rate: {}'.format(args.lr))
    print('# learning rate decay: {}'.format(args.lrdecay))
    print('# momentum: {}'.format(args.momentum))
    print('# wieght decay: {}'.format(args.weightdecay))
    print('# dropout ratio: {}'.format(args.dropout))
    print('# gradient norm threshold to clip: {}'.format(args.gradclip))
    print('# subpos depth: {}'.format(args.subpos_depth))
    print('# resume: {}'.format(args.resume))
    print('# epoch to resume: {}'.format(args.resume_epoch))
    print('# tag schema: {}'.format(args.tag_schema))
    print('# test: {}'.format(args.test))
    print('')
    print(args)

    # Prepare logger

    time0 = datetime.now().strftime('%Y%m%d_%H%M')
    print('INFO: start: %s\n' % time0)
    if args.test == -1:
        logger = open('log/' + time0 + '.log', 'a')
        logger.write(str(args))

    # Load dataset

    train_path = args.dir_path + args.train_data
    val_path = args.dir_path + args.validation_data
    test_path = args.dir_path + 'LBb_test.tsv'

    limit = args.test if args.test > 0 else -1
    if args.format == 'bccwj':
        train, train_t, token2id, label2id = util.create_data_wordseg(
            train_path, scheme=args.tag_schema, limit=limit)
        val, val_t, token2id, label2id = util.create_data_wordseg(
            val_path, token2id=token2id, label2id=label2id, label_update=False, 
            scheme=args.tag_schema, limit=limit)
        # test, test_t, token2id, label2id = util.create_data_wordseg(
        #     test_path, token2id=token2id, label2id=label2id, token_update=False, label_update=False, limit=limit)
    elif args.format == 'bccwj_pos':
        train, train_t, token2id, label2id = util.create_data_for_pos_tagging(
            train_path, subpos_depth=args.subpos_depth, limit=limit)
        val, val_t, token2id, label2id = util.create_data_for_pos_tagging(
            val_path, token2id=token2id, label2id=label2id, subpos_depth=args.subpos_depth, 
            label_update=False, limit=limit)
    elif args.format == 'wsj':
        train, train_t, token2id, label2id = util.create_data_for_wsj(train_path, limit=limit)
        val, val_t, token2id, label2id = util.create_data_for_wsj(
            val_path, token2id=token2id, label2id=label2id, label_update=False, limit=limit)
    elif args.format == 'conll2003':
        train, train_t, token2id, label2id = util.create_data(train_path, limit=limit)
        val, val_t, token2id, label2id = util.create_data_for_conll2003(
            val_path, token2id=token2id, label2id=label2id, label_update=False, limit=limit)
    else:
        pass

    test = []
    n_train = len(train)
    n_val = len(val)
    n_test = len(test)
    n_vocab = len(token2id)
    n_labels = len(label2id)

    print('vocab =', n_vocab)
    print('data length: train=%d val=%d test=%d' % (n_train, n_val, n_test))
    print()
    print('train:', train[:3], '...', train[n_train-3:])
    print('train_t:', train_t[:3], '...', train[n_train-3:])
    print()
    t2i_tmp = list(token2id.items())
    id2label = {v:k for k,v in label2id.items()}
    print('token2id:', t2i_tmp[:3], '...', t2i_tmp[len(t2i_tmp)-3:])
    print('label2id:', label2id)
    print()
    if args.test == -1:
        logger.write('INFO: vocab = %d\n' % n_vocab)
        logger.write('INFO: data length: train=%d val=%d\n' % (n_train, n_val))

    # Prepare model

    if chainer.__version__[0] == '2':
        chainer.config.cudnn_deterministic = args.use_cudnn
        rnn = RNN_CRF(args.layer, n_vocab, args.lookup_dim, args.hidden_unit, n_labels, args.dropout, args.bidirection)
    else:
        print("chainer version>=2.0.0 is required.")
        sys.exit()

    model = SequenceLabelingClassifier(rnn, id2label)
    model.compute_fscore = True

    if args.resume:
        print('resume training from the model: %s' % args.resume)
        chainer.serializers.load_npz(args.resume, model)

    if args.gpu >= 0:
        # Make the specified GPU current
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = cuda.cupy
    else:
        xp = np

    # Set up optimizer

    if args.optimizer == 'sgd':
        if args.momentum <= 0:
            optimizer = chainer.optimizers.SGD(lr=args.lr)
        else:
            optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = chainer.optimizers.Adam()
    elif args.optimizer == 'adagrad':
        optimizer = chainer.optimizers.AdaGrad(lr=args.lr)

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weightdecay))


    # Run

    n_iter_report = args.iter_to_report
    n_iter = 0
    for e in range(max(1, args.resume_epoch), args.epoch+1):
        time = datetime.now().strftime('%Y%m%d_%H%M')
        if args.test == -1:
            logger.write('INFO: start epoch %d at %s\n' % (e, time))
        print('start epoch %d: %s' % (e, time))

        count = 0
        total_loss = 0
        num_tokens = 0
        total_ecounts = conlleval.EvalCounts()

        i = 0
        for xs, ts in batch_generator(train, train_t, args.batchsize, shuffle=True, xp=xp):
            loss, ecounts = model(xs, ts, train=True)
            num_tokens += sum([len(x) for x in xs])
            count += len(xs)
            total_loss += loss.data
            total_ecounts = conlleval.merge_counts(total_ecounts, ecounts)
            i_max = min(i + args.batchsize, n_train)
            print('* batch %d-%d loss: %.4f' % ((i+1), i_max, loss.data))
            i = i_max
            n_iter += 1

            optimizer.target.cleargrads() # Clear the parameter gradients
            loss.backward()               # Backprop
            loss.unchain_backward()       # Truncate the graph
            optimizer.update()            # Update the parameters

            # Evaluation
            if (n_iter * args.batchsize) % n_iter_report == 0: # or i == n_train:

                now_e = '%.2f' % (n_iter * args.batchsize / n_train)
                time = datetime.now().strftime('%Y%m%d_%H%M')
                print()
                print('### iteration %s (epoch %s)' % ((n_iter * args.batchsize), now_e))
                print('<training result for previous iterations>')

                ave_loss = total_loss / num_tokens
                c = total_ecounts
                acc = conlleval.calculate_accuracy(c.correct_tags, c.token_counter)
                overall = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)
                print_results(total_loss, ave_loss, count, c, acc, overall)
                print()

                if args.test == -1:
                    t_stat = 'n_sen: %d, n_token: %d, n_chunk: %d, n_chunk_p: %d' % (
                        count, c.token_counter, c.found_correct, c.found_guessed)
                    t_res = '%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%d\t%.4f\t%.4f' % (
                        (100.*acc), (100.*overall.prec), (100.*overall.rec), (100.*overall.fscore), 
                        overall.tp, overall.fp, overall.fn, total_loss, ave_loss)
                    logger.write('INFO: train - %s\n' % t_stat)
                    if n_iter == 1:
                        logger.write('data\titer\ep\tacc\tprec\trec\tfb1\tTP\tFP\tFN\ttloss\taloss\n')
                    logger.write('train\t%d\t%s\t%s\n' % (n_iter, now_e, t_res))

                print('<validation result>')
                v_stat, v_res = evaluate(model, val, val_t, args.batchsize, xp=xp)
                print()

                if args.test == -1:
                    logger.write('INFO: valid - %s\n' % v_stat)
                    if n_iter == 1:
                        logger.write('data\titer\ep\tacc\tprec\trec\tfb1\tTP\tFP\tFN\ttloss\taloss\n')
                    logger.write('valid\t%d\t%s\t%s\n' % (n_iter, now_e, v_res))

                # Save the model
                #mdl_path = 'model/rnn_%s_i%.2fk.mdl' % (time0, (1.0 * n_iter / 1000))
                mdl_path = 'model/rnn_%s_i%s.mdl' % (time0, now_e)
                print('save the model: %s\n' % mdl_path)
                if args.test == -1:
                    logger.write('INFO: save the model: %s\n' % mdl_path)
                    serializers.save_npz(mdl_path, model)
        
                # Reset counters
                count = 0
                total_loss = 0
                num_tokens = 0
                total_ecounts = conlleval.EvalCounts()

                if args.test == -1:
                    logger.close()
                    logger = open('log/' + time0 + '.log', 'a')

        # # learning rate decay
        # if args.lrdecay > 0 and args.optimizer == 'sgd':
        #     optimizer.lr = args.lr / (1 + e * args.lrdecay) # (Ma 2016)
        #     print('lr:', optimizer.lr)
        #     print()

    time = datetime.now().strftime('%Y%m%d_%H%M')
    if args.test == -1:
        logger.write('finish: %s\n' % time)
        logger.close()
   
if __name__ == '__main__':
    main()
