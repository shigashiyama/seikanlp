"""
- (Zaremba 2015) の著者らによる torch 実装
  - (Zaremba 2015) RECURRENT NEURAL NETWORK REGULARIZATION, ICLR
  - https://github.com/tomsercu/lstm
-> 同等のプログラムの PFN による chainer 実装 -> 改造

"""

import copy
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.functions.loss import softmax_cross_entropy
from chainer.links.connection.n_step_rnn import argsort_list_descent, permutate_list

from conlleval import conlleval


class RNN_CRF(chainer.Chain):
    def __init__(self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', n_left_contexts=0, n_right_contexts=0, init_embed=None, gpu=-1):
        super(RNN_CRF, self).__init__()

        with self.init_scope():
            self.act = get_activation(linear_activation)
            if not self.act:
                print('unsupported activation function.')
                sys.exit()

            if init_embed != None:
                n_vocab = init_embed.W.shape[0]
                embed_dim = init_embed.W.shape[1]

            # padding indices for context window
            self.left_padding_ids = self.get_id_array(n_vocab, n_left_contexts, gpu)
            self.right_padding_ids = self.get_id_array(n_vocab + n_left_contexts, n_right_contexts, gpu)
            self.empty_array = cuda.cupy.array([], dtype=np.float32) if gpu >= 0 else np.array([], dtype=np.float32)
            # init fields
            self.embed_dim = embed_dim
            self.context_size = 1 + n_left_contexts + n_right_contexts
            self.input_vec_size = self.embed_dim * self.context_size
            vocab_size = n_vocab + n_left_contexts + n_right_contexts
            rnn_in = embed_dim * self.context_size

            # init layers
            self.lookup = L.EmbedID(vocab_size, self.embed_dim) if init_embed == None else init_embed

            self.rnn_unit_type = rnn_unit_type
            if rnn_unit_type == 'lstm':
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiLSTM(n_rnn_layers, rnn_in, n_rnn_units, dropout)
                else:
                    self.rnn_unit = L.NStepLSTM(n_rnn_layers, embed_dim, n_rnn_units, dropout)
            elif rnn_unit_type == 'gru':
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiGRU(n_rnn_layers, rnn_in, n_rnn_units, dropout) 
                else:
                    self.rnn_unit = L.NStepGRU(n_rnn_layers, embed_dim, n_rnn_units, dropout)
            else:
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiRNNTanh(n_rnn_layers, rnn_in, n_rnn_units, dropout) 
                else:
                    self.rnn_unit = L.NStepRNNTanh(n_rnn_layers, embed_dim, n_rnn_units, dropout)

            self.linear = L.Linear(n_rnn_units * (2 if rnn_bidirection else 1), n_labels)
            self.crf = L.CRF1d(n_labels)

            #tmp
            self.vocab_size = vocab_size
            self.n_rnn_layers = n_rnn_layers
            self.rnn_unit_in = rnn_in
            self.n_rnn_units = n_rnn_units
            self.dropout = dropout
            self.rnn_bidirection = rnn_bidirection
            self.n_labels = n_labels

            print('## parameters')
            print('# lookup:', self.lookup.W.shape)
            print('# rnn unit:', self.rnn_unit)
            # i = 0
            # for c in self.rnn_unit._children:
            #     print('#   param', i)
            #     print('#      0 -', c.w0.shape, '+', c.b0.shape)
            #     print('#      1 -', c.w1.shape, '+', c.b1.shape)
            #     print('#      2 -', c.w2.shape, '+', c.b2.shape)
            #     print('#      3 -', c.w3.shape, '+', c.b3.shape)
            #     print('#      4 -', c.w4.shape, '+', c.b4.shape)
            #     print('#      5 -', c.w5.shape, '+', c.b5.shape)
            #     print('#      6 -', c.w6.shape, '+', c.b6.shape)
            #     print('#      7 -', c.w7.shape, '+', c.b7.shape)
            #     i += 1
            print('# linear:', self.linear.W.shape, '+', self.linear.b.shape)
            print('# linear_activation:', self.act)
            print('# crf:', self.crf.cost.shape)
            print()

    def __call__(self, xs, ts, train=True):
        # create input vector considering context window
        exs = []
        for x in xs:
            if self.context_size > 1:
                embeddings = F.concat((self.lookup(self.left_padding_ids),
                                       self.lookup(x),
                                       self.lookup(self.right_padding_ids)), 0)
                embeddings = F.reshape(embeddings, (len(x) + self.context_size - 1, self.embed_dim))

                ex = self.empty_array.copy()
                for i in range(len(x)):
                    for j in range(i, i + self.context_size):
                        ex = F.concat((ex, embeddings[j]), 0)
                ex = F.reshape(ex, (len(x), self.input_vec_size))
            else:
                ex = self.lookup(x)
                
            exs.append(ex)
        xs = exs

        with chainer.using_config('train', train):
            # rnn layers
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, xs)
            else:
                hy, hs = self.rnn_unit(None, xs)

            # linear layer
            
            if not self.act or self.act == 'identity':
                hs = [self.linear(h) for h in hs]                
            else:
                hs = [self.act(self.linear(h)) for h in hs]

            # crf layer
            indices = argsort_list_descent(hs)
            trans_hs = F.transpose_sequence(permutate_list(hs, indices, inv=False))
            trans_ts = F.transpose_sequence(permutate_list(ts, indices, inv=False))
            loss = self.crf(trans_hs, trans_ts)
            score, trans_ys = self.crf.argmax(trans_hs)
            ys = permutate_list(F.transpose_sequence(trans_ys), indices, inv=True)
            ys = [y.data for y in ys]

        return loss, ys


    def get_id_array(self, start, width, gpu):
        ids = np.array([], dtype=np.int32)
        for i in range(start, start + width):
            ids = np.append(ids, np.int32(i))
        return cuda.to_gpu(ids) if gpu >= 0 else ids


    # temporaly code for memory checking
    def reset(self):
        print('reset')
        del self.lookup, self.rnn_unit, self.linear, self.crf
        gc.collect()

        self.lookup = L.EmbedID(self.vocab_size, self.embed_dim)
        self.rnn_unit = L.NStepBiLSTM(self.n_rnn_layers, self.rnn_unit_in, self.n_rnn_units, self.dropout) if self.rnn_bidirection else L.NStepLSTM(self.n_rnn_layers, self.embed_dim, self.n_rnn_units, self.dropout)
        self.linear = L.Linear(self.n_rnn_units * (2 if self.rnn_bidirection else 1), self.n_labels)
        self.crf = L.CRF1d(self.n_labels)

        self.lookup = self.lookup.to_gpu()
        self.rnn_unit = self.rnn_unit.to_gpu()
        self.linear = self.linear.to_gpu()
        self.crf = self.crf.to_gpu()

        # print('## parameters')
        # print('# lookup:', self.lookup.W.shape)
        # print('# lstm:')
        # i = 0
        # for c in self.rnn_unit._children:
        #     print('#   param', i)
        #     print('#      0 -', c.w0.shape, '+', c.b0.shape)
        #     print('#      1 -', c.w1.shape, '+', c.b1.shape)
        #     print('#      2 -', c.w2.shape, '+', c.b2.shape)
        #     print('#      3 -', c.w3.shape, '+', c.b3.shape)
        #     print('#      4 -', c.w4.shape, '+', c.b4.shape)
        #     print('#      5 -', c.w5.shape, '+', c.b5.shape)
        #     print('#      6 -', c.w6.shape, '+', c.b6.shape)
        #     print('#      7 -', c.w7.shape, '+', c.b7.shape)
        #     i += 1
        # print('# linear:', self.linear.W.shape, '+', self.linear.b.shape)
        # print('# linear_activation:', self.act)
        # print('# crf:', self.crf.cost.shape)
        # print()
        

class RNN(chainer.Chain):
    def __init__(self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity'):
        super(RNN, self).__init__()
        with self.init_scope():
            self.act = get_activation(linear_activation)
            if not self.act:
                print('unsupported activation function.')
                sys.exit()

            self.embed = L.EmbedID(n_vocab, embed_dim)

            self.rnn_unit_type = rnn_unit_type
            if rnn_unit_type == 'lstm':
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiLSTM(n_rnn_layers, rnn_in, n_rnn_units, dropout)
                else:
                    self.rnn_unit = L.NStepLSTM(n_rnn_layers, embed_dim, n_rnn_units, dropout)
            elif rnn_unit_type == 'gru':
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiGRU(n_rnn_layers, rnn_in, n_rnn_units, dropout) 
                else:
                    self.rnn_unit = L.NStepGRU(n_rnn_layers, embed_dim, n_rnn_units, dropout)
            else:
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiRNNTanh(n_rnn_layers, rnn_in, n_rnn_units, dropout) 
                else:
                    self.rnn_unit = L.NStepRNNTanh(n_rnn_layers, embed_dim, n_rnn_units, dropout)

            self.linear = L.Linear(n_rnn_units * (2 if rnn_bidirection else 1), n_labels)
            self.loss_fun = softmax_cross_entropy.softmax_cross_entropy
        
    def __call__(self, xs, ts, train=True):
        xs = [self.embed(x) for x in xs]
        with chainer.using_config('train', train):
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, xs)
            else:
                hy, hs = self.rnn_unit(None, xs)
            ys = [self.act(self.linear(h)) for h in hs]

        loss = None
        ps = []
        for y, t in zip(ys, ts):
            if loss is not None:
                loss += self.loss_fun(y, t)
            else:
                loss = self.loss_fun(y, t)
                ps.append([np.argmax(yi.data) for yi in y])

        return loss, ps


class SequenceTagger(chainer.link.Chain):
    compute_fscore = True

    def __init__(self, predictor, id2label):
        super(SequenceTagger, self).__init__(predictor=predictor)
        self.id2label = id2label
        
    def __call__(self, *args, **kwargs):
        assert len(args) >= 2
        xs = args[0]
        ts = args[1]
        loss, ys = self.predictor(*args, **kwargs)

        if self.compute_fscore:
            eval_counts = None
            for x, t, y in zip(xs, ts, ys):
                generator = self.generate_lines(x, t, y)
                eval_counts = conlleval.merge_counts(eval_counts, conlleval.evaluate(generator))

        # self.predictor.reset()
        # loss = None
        # eval_counts = None

        return loss, eval_counts

    def decode(self, *args, **kwargs):
        _, ps = self.pridictor(*args, **kwargs)
        return ps

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


def get_activation(activation):
    if not activation  or activation == 'identity':
        return F.identity
    elif activation == 'relu':
        return F.relu
    elif activation == 'tanh':
        return F.tanh
    elif activation == 'sigmoid':
        return F.sigmoid
    else:
        return


