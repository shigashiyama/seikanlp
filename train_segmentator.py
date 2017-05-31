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


class RNNWithNStepLSTM(chainer.Chain):
    def __init__(self, n_layer, n_vocab, n_labels, n_units, dropout, use_cudnn):
        super(RNNWithNStepLSTM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.NStepLSTM(n_layer, n_units, n_units, dropout, use_cudnn=use_cudnn),
            l2=L.Linear(n_units, n_labels),
        )
        
    def __call__(self, hx, cx, xs, train=True):
        xs = [self.embed(x) for x in xs]
        hy, cy, ys = self.l1(hx, cx, xs, train=train)
        ys = [self.l2(y) for y in ys]
        return hy, cy, ys


class SequenceLabelingClassifier(link.Chain):
    compute_accuracy = True
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
        

def run_epoch(model, instances, labels, batchsize, train=True, optimizer=None, xp=np, logging=None):
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

        total_ecounts = conlleval.merge_counts(total_ecounts, ecounts)
        c = total_ecounts
        acc = conlleval.calculate_accuracy(c.correct_tags, c.token_counter)
        overall = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)

        if train:
            optimizer.target.cleargrads() # Clear the parameter gradients
            loss.backward()               # Backprop
            loss.unchain_backward()       # Truncate the graph
            optimizer.update()            # Update the parameters

    print('loss: ', total_loss)
    print('#sen, #token, #chunk, #chunk_pred: %d %d %d %d' %
          (count, c.token_counter, c.found_correct, c.found_guessed))
    print('TP, FP, FN: %d %d %d' % (overall.tp, overall.fp, overall.fn))
    print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' % 
          (100.*acc, 100.*overall.prec, 100.*overall.rec, 100.*overall.fscore))

    if logging is not None:
        logging.info('n_sen    : %d' % count)
        logging.info('n_token  : %d' % c.token_counter)
        logging.info('n_chunk  : %d' % c.found_correct)
        logging.info('n_chunk_p: %d' % c.found_guessed)
        logging.info('TP: %d' % overall.tp)
        logging.info('FP: %d' % overall.fp)
        logging.info('FN: %d' % overall.fn)
        logging.info('loss: %f' % total_loss)
        logging.info('acc :%6.2f' % (100.*acc))
        logging.info('pre :%6.2f' % (100.*overall.prec))
        logging.info('rec :%6.2f' % (100.*overall.rec))
        logging.info('fb1 :%6.2f' % (100.*overall.fscore))
        
def main():

    # get arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--cudnn', dest='use_cudnn', action='store_true')
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--dropout', '-d', type=float, default=0.5,
                        help='Dropout ratio')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='out',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--dir_path', '-p',
                        help='Directory path of input data (training, validation and test)')
    parser.add_argument('--train_data', '-t',
                        help='Filename of training data')
    parser.add_argument('--validation_data', '-v',
                        help='Filename of validation data')
    parser.set_defaults(test=False)
    parser.set_defaults(use_cudnn=False)
    args = parser.parse_args()

    print('# GPU: {}'.format(args.gpu))
    print('# cudnn: {}'.format(args.use_cudnn))
    print('# output directory: {}'.format(args.out))
    print('# minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# unit: {}'.format(args.unit))
    print('# dropout ratio: {}'.format(args.dropout))
    print('# gradient norm threshold to clip: {}'.format(args.gradclip))
    print('# length of truncated BPTT: {}'.format(args.bproplen))
    print('# test: {}'.format(args.test))
    print('')

    # Prepare logger

    time = datetime.now().strftime('%Y%m%d_%H%M')
    logging.basicConfig(filename='log/'+time+'.log', level=logging.INFO)
    logging.info('start: ' + time)
    logging.info(args)

    # Load dataset

    train_path = args.dir_path + args.train_data
    val_path = args.dir_path + args.validation_data
    test_path = args.dir_path + 'LBb_test.tsv'

    limit=21 if args.test else -1
    train, train_t, token2id, label2id = util.create_data_per_char(train_path, limit=limit)
    val, val_t, token2id, label2id = util.create_data_per_char(
        val_path, token2id=token2id, label2id=label2id, label_update=False, limit=limit)
    # test, test_t, token2id, label2id = util.create_data_per_char(
    #     test_path, token2id=token2id, label2id=label2id, token_update=False, label_update=False, limit=limit)

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
    logging.info('vocab = %d' % n_vocab)
    logging.info('data length: train=%d val=%d' % (n_train, n_val))

    # Prepare model

    rnn = RNNWithNStepLSTM(1, n_vocab, n_labels, args.unit, args.dropout, args.use_cudnn)
    model = SequenceLabelingClassifier(rnn, id2label)

    if args.gpu >= 0:
        # Make the specified GPU current
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = cuda.cupy
    else:
        xp = np

    # Set up optimizer

    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    # Run

    for e in range(args.epoch):
        print('epoch:', e+1)
        print('<training result>')
        logging.info('epoch:'+str(e+1))
        logging.info('<training result>')
        run_epoch(model, train, train_t, args.batchsize, optimizer=optimizer, xp=xp, logging=logging)
        print()

        # Evaluation on validation data
        evaluator = model.copy()          # to use different state
        evaluator.predictor.train = False # dropout does nothing
        #evaluator.predictor.reset_state() # initialize state
        print('<validation result>')
        logging.info('<validation result>')
        run_epoch(evaluator, val, val_t, args.batchsize, train=False, xp=xp, logging=logging)
        print()

        # Save the model and the optimizer

        mdl_path = 'model/rnn_%s_e%d.mdl' % (time, e)
        opt_path = 'model/rnn_%s_e%d.opt' % (time, e)

        print('save the model:', mdl_path)
        print('save the optimizer:', opt_path)
        print()
        logging.info('save the model: ' + mdl_path)
        logging.info('save the optimizer: ' + opt_path)
        serializers.save_npz(mdl_path, model)
        serializers.save_npz(opt_path, optimizer)

    # Evaluate on test dataset

    time = datetime.now().strftime('%Y%m%d_%H%M')
    logging.info('finish: ' + time)

   
if __name__ == '__main__':
    main()
