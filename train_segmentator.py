"""
- (Zaremba 2015) の著者らによる torch 実装
  - (Zaremba 2015) RECURRENT NEURAL NETWORK REGULARIZATION, ICLR
  - https://github.com/tomsercu/lstm
-> 同等のプログラムの PFN による chainer 実装
-> 東山が改造

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

import util
from conlleval import conlleval


class RNN(chainer.Chain):
    def __init__(self, n_layers, n_vocab, n_labels, n_units, dropout, use_bi_direction, use_cudnn):
        super(RNN, self).__init__(
            embed = L.EmbedID(n_vocab, n_units),
            l1 = L.NStepBiLSTM(n_layers, n_units, n_units, dropout, use_cudnn=use_cudnn) 
            if use_bi_direction else L.NStepLSTM(n_layers, n_units, n_units, dropout, use_cudnn=use_cudnn),
            l2 = L.Linear(n_units * (2 if use_bi_direction else 1), n_labels),
        )
        
    def __call__(self, hx, cx, xs, train=True):
        #print('xs', len(xs), [len(x) for x in xs])

        xs = [self.embed(x) for x in xs]
        # print('xs-embed', len(xs), [x.data.shape for x in xs])
        # print()
        # for i in range(4):
        #     print('l1-'+str(i), [self.l1._children[i].__dict__['w'+str(j)].shape for j in range(8)])
        # print()

        hy, cy, ys = self.l1(hx, cx, xs, train=train)
        # print('ys', len(ys), [y.data.shape for y in ys])
        # print()

        ys = [self.l2(y) for y in ys]
        return hy, cy, ys


class SequenceLabelingClassifier(link.Chain):
    compute_fscore = True

    def __init__(self, predictor, id2label,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(SequenceLabelingClassifier, self).__init__(predictor=predictor)
        self.id2label = id2label
        self.lossfun = lossfun
        self.accfun = accfun
        self.ys = None
        self.loss = None
        self.eval_counts = None
        #self.result = {}
        
    def __call__(self, *args):
        assert len(args) >= 2
        inputs = args[:-1]      # hx, cx, xs
        ts = args[-1]
        self.ys = None
        self.loss = None
        self.eval_counts = None

        hy, cy, self.ys = self.predictor(*inputs)

        # loss
        for y, t in zip(self.ys, ts):
            if self.loss is not None:
                self.loss += self.lossfun(y, t)
            else:
                self.loss = self.lossfun(y, t)
                
        # reporter_module.report({'loss': self.loss}, self)
        # self.result.update({'loss': self.loss})

        if self.compute_fscore:
            xs = inputs[2]
            tmp = [len(x) for x in xs]

            for x, t, y in zip(xs, ts, self.ys):
                p = [np.argmax(yi.data) for yi in y]
                generator = self.generate_lines(x, t, p)
                self.eval_counts = conlleval.merge_counts(self.eval_counts, 
                                                          conlleval.evaluate(generator))

        return self.loss, self.eval_counts

    def generate_lines(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                raise StopIteration
            x_str = str(x[i])
            t_str = self.id2label[int(t[i])]
            y_str = self.id2label[int(y[i])]

            yield [x_str, t_str, y_str]

            i += 1


# does not work
# class SequentialIterator(chainer.dataset.Iterator):

#     def __init__(self, instances, labels, batch_size, train=True, xp=np):
#         self.instances = instances
#         self.labels = labels
#         self.batch_size = batch_size
#         self.size = len(instances)
#         self.perm = np.random.permutation(self.size) if train else range(0, self.size)
#         self.xp = xp
#         self.ni = 0             # next index

#     def __next__(self):
#         if self.ni >= self.size:
#             raise StopIteration

#         i_max = min(self.ni + self.batch_size, self.size)
#         print(self.ni, i_max)
#         xs = [self.xp.asarray(self.instances[self.perm[i]], dtype=np.int32) 
#               for i in range(self.ni, i_max)]
#         hx = None
#         cx = None
#         ts = [self.xp.asarray(self.instances[self.perm[i]], dtype=np.int32)
#               for i in range(self.ni, i_max)]

#         self.ni = i_max

#         return hx, cx, xs, ts
        

def run_epoch(model, instances, labels, batchsize, train=True, optimizer=None, xp=np):
    stat = ''
    res = ''
    count = 0
    total_loss = 0
    total_ecounts = conlleval.EvalCounts()

    len_data = len(instances)
    perm = np.random.permutation(len_data) if train else range(0, len_data)
    for i in range(0, len_data, batchsize):
        i_max = min(i + batchsize, len_data)
        xs = [xp.asarray(instances[perm[i]], dtype=np.int32) for i in range(i, i_max)]
        hx = None
        cx = None
        ts = [xp.asarray(labels[perm[i]], dtype=np.int32) for i in range(i, i_max)]

        loss, ecounts = model(hx, cx, xs, ts)
        if train:
            print('* batch', str(i+1)+'-'+str(i_max), 'loss:', loss.data)
        count += len(xs)
        total_loss += loss.data

        if model.compute_fscore:
            total_ecounts = conlleval.merge_counts(total_ecounts, ecounts)
            c = total_ecounts
            acc = conlleval.calculate_accuracy(c.correct_tags, c.token_counter)
            overall = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)

        if train:
            optimizer.target.cleargrads() # Clear the parameter gradients
            loss.backward()               # Backprop
            loss.unchain_backward()       # Truncate the graph
            optimizer.update()            # Update the parameters

    if model.compute_fscore:
        print('loss: ', total_loss)
        print('#sen, #token, #chunk, #chunk_pred: %d %d %d %d' %
              (count, c.token_counter, c.found_correct, c.found_guessed))
        print('TP, FP, FN: %d %d %d' % (overall.tp, overall.fp, overall.fn))
        print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' % 
              (100.*acc, 100.*overall.prec, 100.*overall.rec, 100.*overall.fscore))
        
        stat = 'n_sen: %d, n_token: %d, n_chunk: %d, n_chunk_p: %d' % (count, c.token_counter, c.found_correct, c.found_guessed)
        res = '%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%d\t%.4f' % ((100.*acc), (100.*overall.prec), (100.*overall.rec), (100.*overall.fscore), overall.tp, overall.fp, overall.fn, total_loss)
            
    return stat, res

        
def main():

    # get arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--cudnn', dest='use_cudnn', action='store_true')
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Number of LSTM layers')
    parser.add_argument('--bidirection', action='store_true',
                        help='Use bi-direction LSTM')
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--lr', '-r', type=float, default=1,
                        help='Value of initial learning rate')
    parser.add_argument('--optimizer', '-o', default='sgd')
    parser.add_argument('--momentum', '-m', type=float, default=0.9,
                        help='Momentum ratio')
    parser.add_argument('--lrdecay', type=float, default=0.05,
                        help='Coefficient for learning rate decay')
    parser.add_argument('--weightdecay', '-w', type=float, default=0.1,
                        help='Weight decay ratio')
    parser.add_argument('--dropout', '-d', type=float, default=0.5,
                        help='Dropout ratio')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--resume', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--resume_epoch', type=int, default=1,
                        help='Resume the training from the epoch')
    parser.add_argument('--test', type=int, default=-1,
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--format', '-f',
                        help='Format of input data')
    parser.add_argument('--dir_path', '-p',
                        help='Directory path of input data (training, validation and test)')
    parser.add_argument('--train_data', '-t',
                        help='Filename of training data')
    parser.add_argument('--validation_data', '-v',
                        help='Filename of validation data')
    # parser.add_argument('--out', '-o', default='out',
    #                     help='Directory to output the result')
    parser.set_defaults(optimizer='sgd')
    parser.set_defaults(lrdecay=False)
    parser.set_defaults(test=False)
    parser.set_defaults(use_cudnn=False)
    parser.set_defaults(bidirection=False)
    args = parser.parse_args()

    print('# GPU: {}'.format(args.gpu))
    print('# cudnn: {}'.format(args.use_cudnn))
    # print('# output directory: {}'.format(args.out))
    print('# minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# bidirection: {}'.format(args.bidirection))
    print('# layer: {}'.format(args.layer))
    print('# unit: {}'.format(args.unit))
    print('# optimization algorithm: {}'.format(args.optimizer))
    print('# learning rate: {}'.format(args.lr))
    print('# learning rate decay: {}'.format(args.lrdecay))
    print('# momentum: {}'.format(args.momentum))
    print('# wieght decay: {}'.format(args.weightdecay))
    print('# dropout ratio: {}'.format(args.dropout))
    print('# gradient norm threshold to clip: {}'.format(args.gradclip))
    print('# resume: {}'.format(args.resume))
    print('# epoch to resume: {}'.format(args.resume_epoch))
    print('# test: {}'.format(args.test))
    print('')

    # Prepare logger

    time = datetime.now().strftime('%Y%m%d_%H%M')
    logger = open('log/' + time + '.log', 'w')
    logger.write('INFO: %s\n' % str(args))
    logger.write('INFO: start: %s\n' % time)

    # Load dataset

    train_path = args.dir_path + args.train_data
    val_path = args.dir_path + args.validation_data
    test_path = args.dir_path + 'LBb_test.tsv'

    limit=args.test if args.test > 0 else -1
    if args.format == 'bccwj':
        train, train_t, token2id, label2id = util.create_data_per_char(train_path, limit=limit)
        val, val_t, token2id, label2id = util.create_data_per_char(
            val_path, token2id=token2id, label2id=label2id, label_update=False, limit=limit)
        # test, test_t, token2id, label2id = util.create_data_per_char(
        #     test_path, token2id=token2id, label2id=label2id, token_update=False, label_update=False, limit=limit)
    elif args.format == 'conll2003':
        train, train_t, token2id, label2id = util.create_data(train_path, limit=limit)
        val, val_t, token2id, label2id = util.create_data(
            val_path, token2id=token2id, label2id=label2id, label_update=False, limit=limit)
    else:
        pass

    n_train = len(train)
    n_val = len(val)
    n_test = 0 #len(test)
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
    logger.write('INFO: vocab = %d\n' % n_vocab)
    logger.write('INFO: data length: train=%d val=%d\n' % (n_train, n_val))

    # Prepare model

    rnn = RNN(args.layer, n_vocab, n_labels, args.unit, args.dropout, args.bidirection, args.use_cudnn)
    model = SequenceLabelingClassifier(rnn, id2label)
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

    for e in range(max(1, args.resume_epoch), args.epoch+1):
        print('epoch:', e)
        t_stat, t_res = run_epoch(model, train, train_t, args.batchsize, optimizer=optimizer, xp=xp)

        print('<training result>')
        if e == 1:
            logger.write('INFO: train - %s\n' % t_stat)
            logger.write('data\tep\tacc\tprec\trec\tfb1\tTP\tFP\tFN\tloss\n')
        logger.write('train\t%d\t%s\n' % (e, t_res))
        print()

        # Evaluation
        evaluator = model.copy()          # to use different state
        evaluator.predictor.train = False # dropout does nothing
        evaluator.compute_fscore = True

        print('<validation result>')
        v_stat, v_res = run_epoch(evaluator, val, val_t, args.batchsize, train=False, xp=xp)
        if e == 1:
            logger.write('INFO: valid - %s\n' % v_stat)
        logger.write('valid\t%d\t%s\n' % (e, v_res))
        print()

        # Save the model and the optimizer

        mdl_path = 'model/rnn_%s_e%d.mdl' % (time, e)
        opt_path = 'model/rnn_%s_e%d.opt' % (time, e)

        logger.write('INFO: save the model: %s\n' % mdl_path)
        logger.write('INFO: save the optimizer: %s\n' % opt_path)
        serializers.save_npz(mdl_path, model)
        serializers.save_npz(opt_path, optimizer)

        
        # learning rate decay (
        if args.lrdecay > 0 and args.optimizer == 'sgd':
            optimizer.lr = args.lr / (1 + e * args.lrdecay) # (Ma 2016)
            print('lr:', optimizer.lr)
            print()

    # Evaluate on test dataset

    time = datetime.now().strftime('%Y%m%d_%H%M')
    logger.write('finish: %s\n' % time)
    logger.close()
   
if __name__ == '__main__':
    main()
