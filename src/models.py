"""
This code is implemented by remodeling the implementation of (Zaremba 2015) by developers at PFN.

(Zaremba 2015) RECURRENT NEURAL NETWORK REGULARIZATION, ICLR, 2015, https://github.com/tomsercu/lstm

"""

import copy
import enum
from collections import Counter
from datetime import datetime

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.functions.loss import softmax_cross_entropy
from chainer.links.connection.n_step_rnn import argsort_list_descent, permutate_list
from chainer.functions.math import minmax, exponential, logsumexp
from chainer import initializers
from chainer import Variable


from tools import conlleval
import lattice
import features
from util import Timer


"""
Base model that consists of embedding layer, recurrent network (RNN) layers and linear layer.

Args:
    n_rnn_layers:
        the number of (vertical) layers of recurrent network
    n_vocab:
        size of vocabulary
    n_embed_dim:
        dimention of word embedding
    n_rnn_units:
        the number of units of RNN
    n_labels:
        the number of labels that input instances will be classified into
    dropout:
        dropout ratio of RNN
    rnn_unit_type:
        unit type of RNN: lstm, gru or plain
    rnn_bidirection:
        use bi-directional RNN or not
    linear_activation:
        activation function of linear layer: identity, relu, tanh, sigmoid
    init_embed:
        pre-trained embedding matrix
    feat_extractor:
        FeatureExtractor object to extract additional features 
    gpu:
        gpu device id
"""
class RNNBase(chainer.Chain):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            init_embed=None, feat_extractor=None, gpu=-1):
        super(RNNBase, self).__init__()

        with self.init_scope():
            self.act = get_activation(linear_activation)
            if not self.act:
                print('unsupported activation function.')
                sys.exit()

            if init_embed:
                n_vocab = init_embed.W.shape[0]
                embed_dim = init_embed.W.shape[1]

            # init fields
            self.embed_dim = embed_dim
            if feat_extractor:
                self.feat_extractor = feat_extractor
                self.input_vec_size = self.embed_dim + self.feat_extractor.dim
            else:
                self.feat_extractor = None
                self.input_vec_size = self.embed_dim

            # init layers
            self.embed = L.EmbedID(n_vocab, self.embed_dim) if init_embed == None else init_embed

            self.rnn_unit_type = rnn_unit_type
            if rnn_unit_type == 'lstm':
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiLSTM(n_rnn_layers, self.input_vec_size, n_rnn_units, dropout)
                else:
                    self.rnn_unit = L.NStepLSTM(n_rnn_layers, embed_dim, n_rnn_units, dropout)
            elif rnn_unit_type == 'gru':
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiGRU(n_rnn_layers, self.input_vec_size, n_rnn_units, dropout) 
                else:
                    self.rnn_unit = L.NStepGRU(n_rnn_layers, embed_dim, n_rnn_units, dropout)
            else:
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiRNNTanh(n_rnn_layers, self.input_vec_size, n_rnn_units, dropout) 
                else:
                    self.rnn_unit = L.NStepRNNTanh(n_rnn_layers, embed_dim, n_rnn_units, dropout)

            self.linear = L.Linear(n_rnn_units * (2 if rnn_bidirection else 1), n_labels)

            print('## parameters')
            print('# embed:', self.embed.W.shape)
            print('# rnn unit:', self.rnn_unit)
            if self.rnn_unit_type == 'lstm':
                i = 0
                for c in self.rnn_unit._children:
                    print('#   param', i)
                    print('#      0 -', c.w0.shape, '+', c.b0.shape)
                    print('#      1 -', c.w1.shape, '+', c.b1.shape)
                    print('#      2 -', c.w2.shape, '+', c.b2.shape)
                    print('#      3 -', c.w3.shape, '+', c.b3.shape)
                    print('#      4 -', c.w4.shape, '+', c.b4.shape)
                    print('#      5 -', c.w5.shape, '+', c.b5.shape)
                    print('#      6 -', c.w6.shape, '+', c.b6.shape)
                    print('#      7 -', c.w7.shape, '+', c.b7.shape)
                    i += 1
            print('# linear:', self.linear.W.shape, '+', self.linear.b.shape)
            print('# linear_activation:', self.act)


    # create input vector
    def create_features(self, xs):
        exs = []
        for x in xs:
            if self.feat_extractor:
                emb = self.embed(x)
                feat = self.feat_extractor.extract_features(x)
                ex = F.concat((emb, feat), 1)
            else:
                ex = self.embed(x)
                
            exs.append(ex)
        xs = exs
        return xs


    ## unused
    def get_id_array(self, start, width, gpu=-1):
        ids = np.array([], dtype=np.int32)
        for i in range(start, start + width):
            ids = np.append(ids, np.int32(i))
        return cuda.to_gpu(ids) if gpu >= 0 else ids


class RNN(RNNBase):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            init_embed=None, feat_extractor=None, gpu=-1):
        super(RNN, self).__init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            init_embed=None, feat_extractor=None, gpu=-1)

        self.loss_fun = softmax_cross_entropy.softmax_cross_entropy


    def __call__(self, xs, ts, train=True):
        with chainer.using_config('train', train):
            fs = self.create_features(xs)
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, fs)
            else:
                hy, hs = self.rnn_unit(None, fs)
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


    def decode(self, xs):
        with chainer.no_backprop_mode():
            fs = self.create_features(xs)
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, fs)
            else:
                hy, hs = self.rnn_unit(None, fs)
            ys = [self.act(self.linear(h)) for h in hs]

        return ys


class RNN_CRF(RNNBase):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            init_embed=None, feat_extractor=None, gpu=-1):
        super(RNN_CRF, self).__init__(
            n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout, rnn_unit_type, 
            rnn_bidirection, linear_activation, init_embed, feat_extractor, gpu)

        with self.init_scope():
            self.crf = L.CRF1d(n_labels)

            print('# crf cost:', self.crf.cost.shape)
            print()


    def __call__(self, xs, ts, train=True):
        with chainer.using_config('train', train):
            fs = self.create_features(xs)

            # rnn layers
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, fs)
            else:
                hy, hs = self.rnn_unit(None, fs)

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

            ################
            # loss = chainer.Variable(cuda.cupy.array(0, dtype=np.float32))
            # trans_ys = []
            # for i in range(len(hs)):
            #     hs_tmp = hs[i:i+1]
            #     ts_tmp = ts[i:i+1]
            #     t0 = datetime.now()
            #     indices = argsort_list_descent(hs_tmp)
            #     trans_hs = F.transpose_sequence(permutate_list(hs_tmp, indices, inv=False))
            #     trans_ts = F.transpose_sequence(permutate_list(ts_tmp, indices, inv=False))
            #     t1 = datetime.now()
            #     loss += self.crf(trans_hs, trans_ts)
            #     t2 = datetime.now()
            #     score, trans_y = self.crf.argmax(trans_hs)
            #     t3 = datetime.now()
            #     #trans_ys.append(trans_y)
            #     # print('  transpose     : {}'.format((t1-t0).seconds+(t1-t0).microseconds/10**6))
            #     # print('  crf forward   : {}'.format((t2-t1).seconds+(t2-t1).microseconds/10**6))
            #     # print('  crf argmax    : {}'.format((t3-t2).seconds+(t3-t2).microseconds/10**6))
            # ys = [None]
            ################

        return loss, ys
        

    def decode(self, xs):
        with chainer.no_backprop_mode():
            fs = self.create_features(xs)

            # rnn layers
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, fs)
            else:
                hy, hs = self.rnn_unit(None, fs)

            # linear layer
            if not self.act or self.act == 'identity':
                hs = [self.linear(h) for h in hs]                
            else:
                hs = [self.act(self.linear(h)) for h in hs]

            # crf layer
            indices = argsort_list_descent(hs)
            trans_hs = F.transpose_sequence(permutate_list(hs, indices, inv=False))
            score, trans_ys = self.crf.argmax(trans_hs)
            ys = permutate_list(F.transpose_sequence(trans_ys), indices, inv=True)
            ys = [y.data for y in ys]

        return ys


class SequenceTagger(chainer.link.Chain):
    compute_fscore = True

    def __init__(self, predictor, indices):
        super(SequenceTagger, self).__init__(predictor=predictor)
        self.indices = indices
        
    def __call__(self, *args, **kwargs):
        assert len(args) >= 2
        xs = args[0]
        ts = args[1]
        loss, ys = self.predictor(*args, **kwargs)

        eval_counts = None
        if self.compute_fscore:
            for x, t, y in zip(xs, ts, ys):
                generator = self.generate_lines(x, t, y)
                eval_counts = conlleval.merge_counts(eval_counts, conlleval.evaluate(generator))

        return loss, eval_counts


    def generate_lines(self, x, t, y, is_str=False):
        i = 0
        while True:
            if i == len(x):
                raise StopIteration
            x_str = str(x[i])
            t_str = t[i] if is_str else self.indices.get_label(int(t[i]))
            y_str = y[i] if is_str else self.indices.get_label(int(y[i]))

            yield [x_str, t_str, y_str]

            i += 1


    def grow_lookup_table(self, token2id_new, gpu=-1):
        weight1 = self.predictor.embed.W
        diff = len(token2id_new) - len(weight1)
        weight2 = chainer.variable.Parameter(initializers.normal.Normal(1.0), (diff, weight1.shape[1]))
        weight = F.concat((weight1, weight2), 0)
        embed = L.EmbedID(0, 0)
        embed.W = chainer.Parameter(initializer=weight.data)
        self.predictor.embed = embed

        print('# grow vocab size: %d -> %d' % (weight1.shape[0], weight.shape[0]))
        print('# embed:', self.predictor.embed.W.shape)


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


def add(var_list1, var_list2):
    len_list = len(var_list1)
    ret = [None] * len_list
    for i, var1, var2 in zip(range(len_list), var_list1, var_list2):
        ret[i] = var1 + var2
    return ret
