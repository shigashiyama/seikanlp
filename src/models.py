import sys
import copy
import enum
from collections import Counter, deque
from datetime import datetime

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import initializers
from chainer import variable
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.math import minmax
from chainer.links.connection.n_step_rnn import argsort_list_descent, permutate_list

#from chainer import cuda
#from chainer import Variable
#from util import Timer


# Base model that consists of embedding layer, recurrent network (RNN) layers and affine layer.
class RNNTaggerBase(chainer.Chain):
    def __init__(
            self, n_vocab, embed_dim, rnn_unit_type, rnn_bidirection, n_rnn_layers, n_rnn_units, 
            n_labels, feat_dim=0, dropout=0, initial_embed=None, stream=sys.stderr):
        super(RNNTaggerBase, self).__init__()

        with self.init_scope():
            if initial_embed:
                self.embed = initial_embed
                #n_vocab = initial_embed.W.shape[0]
                embed_dim = initial_embed.W.shape[1]
            else:
                self.embed = L.EmbedID(n_vocab, self.embed_dim)
            input_vec_size = embed_dim + feat_dim

            self.rnn_unit_type = rnn_unit_type
            self.rnn = construct_RNN(
                rnn_unit_type, rnn_bidirection, n_rnn_layers, input_vec_size, n_rnn_units, dropout, 
                stream=stream)
            rnn_output_dim = n_rnn_units * (2 if rnn_bidirection else 1)

            self.affine = L.Linear(rnn_output_dim, n_labels)

            print('### Parameters', file=stream)
            print('# Embedding layer: {}'.format(self.embed.W.shape), file=stream)
            print('# Additional features dimension: {}'.format(feat_dim), file=stream)
            print('# RNN unit: {}'.format(self.rnn), file=stream)
            if rnn_unit_type == 'lstm':
                i = 0
                for c in self.rnn._children:
                    print('#   LSTM {}-th param'.format(i), file=stream)
                    print('#      0 - {}, {}'.format(c.w0.shape, c.b0.shape), file=stream) 
                    print('#      1 - {}, {}'.format(c.w1.shape, c.b1.shape), file=stream) 
                    print('#      2 - {}, {}'.format(c.w2.shape, c.b2.shape), file=stream) 
                    print('#      3 - {}, {}'.format(c.w3.shape, c.b3.shape), file=stream) 
                    print('#      4 - {}, {}'.format(c.w4.shape, c.b4.shape), file=stream) 
                    print('#      5 - {}, {}'.format(c.w5.shape, c.b5.shape), file=stream) 
                    print('#      6 - {}, {}'.format(c.w6.shape, c.b6.shape), file=stream) 
                    print('#      7 - {}, {}'.format(c.w7.shape, c.b7.shape), file=stream) 
                    i += 1
            print('# Affine layer: {}, {}'.format(self.affine.W.shape, self.affine.b.shape), file=stream)


    def rnn_output(self, xs):
        if self.rnn_unit_type == 'lstm':
            hy, cy, hs = self.rnn(None, None, xs)
        else:
            hy, hs = self.rnn(None, xs)
        return hs


class RNNTagger(RNNTaggerBase):
    def __init__(
            self, n_vocab, embed_dim, rnn_unit_type, rnn_bidirection, n_rnn_layers, n_rnn_units, 
            n_labels, feat_dim=0, dropout=0, initial_embed=None, stream=sys.stderr):
        super(RNNTagger, self).__init__(
            n_vocab, embed_dim, rnn_unit_type, rnn_bidirection, n_rnn_layers, n_rnn_units, 
            n_labels, feat_dim, dropout, initial_embed, stream)

        self.loss_fun = softmax_cross_entropy.softmax_cross_entropy


    def __call__(self, xs, ls):
        rs = self.rnn_output(xs)
        ys = [self.affine(r) for r in rs]

        loss = None
        ps = []
        for y, l in zip(ys, ls):
            if loss is not None:
                loss += self.loss_fun(y, l)
            else:
                loss = self.loss_fun(y, l)
                ps.append([np.argmax(yi.data) for yi in y])

        return loss, ps


    def decode(self, xs):
        with chainer.no_backprop_mode():
            rs = self.rnn_output(xs)
            ys = [self.affine(h) for r in rs]
        return ys


class RNNCRFTagger(RNNTaggerBase):
    def __init__(
            self, n_vocab, embed_dim, rnn_unit_type, rnn_bidirection, n_rnn_layers, n_rnn_units, 
            n_labels, feat_dim=0, dropout=0, initial_embed=None, stream=sys.stderr):
        super(RNNTagger, self).__init__(
            n_vocab, embed_dim, rnn_unit_type, rnn_bidirection, n_rnn_layers, n_rnn_units, 
            n_labels, feat_dim, dropout, initial_embed, stream)

        with self.init_scope():
            self.crf = L.CRF1d(n_labels)

            print('# CRF cost: {}\n'.format(self.crf.cost.shape), file=stream)


    # train is unused
    def __call__(self, xs, ls=None, train=True, calculate_loss=True):
        # rnn layers
        rs = self.rnn_output(xs)

        # affine layer
        hs = [self.affine(r) for r in rs]

        # crf layer
        indices = argsort_list_descent(hs)
        trans_hs = F.transpose_sequence(permutate_list(hs, indices, inv=False))
        score, trans_ys = self.crf.argmax(trans_hs)
        ys = permutate_list(F.transpose_sequence(trans_ys), indices, inv=True)
        ys = [y.data for y in ys]

        if calculate_loss:
            trans_ls = F.transpose_sequence(permutate_list(ls, indices, inv=False))
            loss = self.crf(trans_hs, trans_ls)
        else:
            loss = chainer.Variable(xp.array(0, dtype='f'))

        return loss, ys
        

    def decode(self, xs):
        with chainer.no_backprop_mode():
            _, ys = self.__call__(xs, calculate_loss=False)

        return ys


class BiaffineCombination(chainer.Chain):
    # e1 * W * e2 + U * e1 + V * e2 + b
    def __init__(self, left_size, right_size, out_size, 
                 use_U=False, use_V=False, use_b=False):
        super(BiaffineCombination, self).__init__()
        # self.left_size = left_size
        # self.right_size = right_size 
        self.out_size = out_size
        
        initialW = None
        initialU = None
        initialV = None
        initialb = 0

        if self.out_size > 1:
            w_shape = (out_size, left_size, right_size)
        else:
            w_shape = (left_size, right_size)
            
        with self.init_scope():
            self.W = variable.Parameter(
                initializers._get_initializer(initialW), w_shape)
            self.b = variable.Parameter(initialb, out_size)


    # TODO chainer 3 -> F.matmul
    def __call__(self, x1, x2):
        # inputs: x1 = [x1_1 ... x1_n1]; dim(x1_i)=d1=left_size
        #         x2 = [x1_1 ... x2_n2]; dim(x2_i)=d2=right_size
        # output: o[l] = gi * W^(l) * hi (l = 1 ... L=out_size)

        if self.out_size > 1:                  # not validated
            X1 = F.broadcast_to(self.out_size) # X1 = [x1 ... x1]
            X2 = F.transpose(
                F.broadcast_to(self.out_size)) # X2 = [x2 ... x2]^T
            X1_W = F.batch_matmul(X1, self.W)  # (L, n1, d1) * (L, d1, d2) -> (L, n1, d2)
            res = F.batch_matmul(X1_W, X2)     # (L, n1, d2) * (L, d2, n2) -> (L, n1, n2)
        else:
            x2t = F.transpose(x2)
            x1_W = F.matmul(x1, self.W)        # (n1, d1) * (d1, d2) -> (n1, d2)
            res = F.matmul(x1_W, x2t)           # (n1, d2) * (d2, n2) -> (n1, n2)

        return res


class RNNBiaffineParser(chainer.Chain):
    def __init__(
            self, n_words, word_embed_dim, n_pos, pos_embed_dim, 
            rnn_unit_type, rnn_bidirection, n_rnn_layers, n_rnn_units, affine_units, n_labels=0,
            dropout=0, initial_word_embed=None, initial_pos_embed=None,
            stream=sys.stderr):
        super(RNNBiaffineParser, self).__init__()

        with self.init_scope():
            if initial_word_embed:
                self.word_embed = initial_word_embed
                word_embed_dim = initial_word_embed.W.shape[1]
            else:
                self.word_embed = L.EmbedID(n_words, word_embed_dim)

            if initial_pos_embed:
                self.pos_embed = initial_pos_embed
                pos_embed_dim = initial_word_embed.W.shape[1]
            else:
                self.pos_embed = L.EmbedID(n_pos, pos_embed_dim)
            embed_dim = word_embed_dim + pos_embed_dim

            self.rnn_unit_type = rnn_unit_type
            self.rnn = construct_RNN(
                rnn_unit_type, rnn_bidirection, n_rnn_layers, embed_dim, n_rnn_units, dropout, 
                stream=stream)
            rnn_output_dim = n_rnn_units * (2 if rnn_bidirection else 1)

            self.affine_head = L.Linear(rnn_output_dim, affine_units)
            self.affine_mod = L.Linear(rnn_output_dim, affine_units)

            # self.biaffine_arc = L.Bilinear(affine_units, affine_units, 1, nobias=True)
            self.biaffine_arc = BiaffineCombination(affine_units, affine_units, 1)

            self.label_prediction = n_labels > 0
            if self.label_prediction > 0:
                self.biaffine_label = L.Bilinear(affine_units, affine_units, n_labels, nobias=True)

            self.softmax_cross_entropy = softmax_cross_entropy.softmax_cross_entropy

            print('### Parameters', file=stream)
            print('# Word embedding matrix: {}'.format(self.word_embed.W.shape), file=stream)
            print('# POS embedding matrix: {}'.format(self.pos_embed.W.shape), file=stream)
            print('# RNN unit: {}'.format(self.rnn), file=stream)
            if rnn_unit_type == 'lstm':
                i = 0
                for c in self.rnn._children:
                    print('#   LSTM {}-th param'.format(i), file=stream)
                    print('#      0 - {}, {}'.format(c.w0.shape, c.b0.shape), file=stream) 
                    print('#      1 - {}, {}'.format(c.w1.shape, c.b1.shape), file=stream) 
                    print('#      2 - {}, {}'.format(c.w2.shape, c.b2.shape), file=stream) 
                    print('#      3 - {}, {}'.format(c.w3.shape, c.b3.shape), file=stream) 
                    print('#      4 - {}, {}'.format(c.w4.shape, c.b4.shape), file=stream) 
                    print('#      5 - {}, {}'.format(c.w5.shape, c.b5.shape), file=stream) 
                    print('#      6 - {}, {}'.format(c.w6.shape, c.b6.shape), file=stream) 
                    print('#      7 - {}, {}'.format(c.w7.shape, c.b7.shape), file=stream) 
                    i += 1
            print('# Affine layer for heads: {}, {}'.format(
                self.affine_head.W.shape, self.affine_head.b.shape), file=stream)
            print('# Affine layer for modifiers: {}, {}'.format(
                self.affine_mod.W.shape, self.affine_mod.b.shape), file=stream)
            print('# Biaffine layer for arc prediction: {}'.format(
                self.biaffine_arc.W.shape), file=stream)
            print('# Biaffine layer for label prediction: {}\n'.format(
                self.biaffine_label.W.shape), file=stream)


    def embed(self, ws, ps):
        xp = cuda.get_array_module(ws[0])
        xs = []
        for w, p in zip(ws, ps):
            wemb = self.word_embed(w)
            pemb = self.pos_embed(p)
            xemb = F.concat((wemb, pemb), 1)
            xs.append(xemb)
        return xs


    def rnn_output(self, xs):
        if self.rnn_unit_type == 'lstm':
            hy, cy, hs = self.rnn(None, None, xs)
        else:
            hy, hs = self.rnn(None, xs)
        return hs


    def predict_arcs(self, m, h, xp=np):
        scores = self.biaffine_arc(m, h) + gen_masking_matrix(len(m), xp=xp)
        yh = minmax.argmax(scores, axis=1).data
        if xp is cuda.cupy:
            yh = cuda.to_cpu(yh)
        yh = np.insert(yh, 0, np.int32(-1))

        return scores, yh
        

    def predict_labels(self, m, h, xp=np):
        scores = self.biaffine_label(m, h)
        yl = minmax.argmax(scores, axis=1).data
        if xp is cuda.cupy:
            yl = cuda.to_cpu(yl)
        yl = np.insert(yl, 0, np.int32(-1))

        return scores, yl


    def __call__(self, ws, ps, ths=None, tls=None, train=True, calculate_loss=True):
        if train:
            calclulate_loss = True
        if not ths:
            ths = [None] * len(ws)
        if not tls:
            tls = [None] * len(ws)

        # embed
        xs = self.embed(ws, ps)

        # rnn layers
        rs = self.rnn_output(xs)

        # affine layers
        hs = [F.relu(self.affine_head(r)) for r in rs] # head representations
        ms = [F.relu(self.affine_mod(r[1:])) for r in rs] # modifier representations

        xp = cuda.get_array_module(xs[0])
        dim = self.affine_head.W.shape[0]
        loss = chainer.Variable(xp.array(0, dtype='f'))
        yhs = []                # predicted head
        yls = []                # predicted arc label

        # biaffine layers
        for h, m, th, tl in zip(hs, ms, ths, tls): # for each sentence in mini-batch
            scores_a, yh = self.predict_arcs(m, h, xp)
            yhs.append(yh)

            if self.label_prediction:
                n = len(m)      # the number of words except root
                heads = th if train else yh
                h_m = F.reshape(F.concat([h[heads[i]] for i in range(1, n+1)], axis=0), (n, dim))
                scores_l, yl = self.predict_labels(m, h_m, xp)
                yls.append(yl)

            if calculate_loss:
                loss += self.softmax_cross_entropy(scores_a, th[1:])
                if self.label_prediction:
                    loss += self.softmax_cross_entropy(scores_l, tl[1:])

        if self.label_prediction:
            return loss, yhs, yls
        else:
            return loss, yhs


    def decode(self, ws, ps):
        with chainer.no_backprop_mode():
            ret = self.__call__(ws, ps, train=False, calculate_loss=False)

        return ret[1:]


def gen_masking_matrix(sen_len, xp=np):
    mat = xp.array([[get_mask_value(i, j) for j in range(sen_len+1)] for i in range(sen_len+1)])
    return chainer.Variable(mat[1:])


def get_mask_value(i, j):
    return -np.float32(np.infty) if i == j else np.float32(0)


def construct_RNN(unit_type, bidirection, n_layers, n_input, n_units, dropout, stream=sys.stderr):
    rnn = None
    if unit_type == 'lstm':
        if bidirection:
            rnn = L.NStepBiLSTM(n_layers, n_input, n_units, dropout)
        else:
            rnn = L.NStepLSTM(n_layers, n_input, n_units, dropout)
    elif unit_type == 'gru':
        if bidirection:
            rnn = L.NStepBiGRU(n_layers, n_input, n_units, dropout) 
        else:
            rnn = L.NStepGRU(n_layers, n_input, n_units, dropout)
    else:
        if bidirection:
            rnn = L.NStepBiRNNTanh(n_layers, n_input, n_units, dropout) 
        else:
            rnn = L.NStepRNNTanh(n_layers, n_input, n_units, dropout)

    return rnn


# def get_activation(activation):
#     if not activation  or activation == 'identity':
#         return F.identity
#     elif activation == 'relu':
#         return F.relu
#     elif activation == 'tanh':
#         return F.tanh
#     elif activation == 'sigmoid':
#         return F.sigmoid
#     else:
#         print('Unsupported activation function', file=stream)
#         sys.exit()
#         return


# def add(var_list1, var_list2):
#     len_list = len(var_list1)
#     ret = [None] * len_list

#     for i, var1, var2 in zip(range(len_list), var_list1, var_list2):
#         ret[i] = var1 + var2
#     return ret
