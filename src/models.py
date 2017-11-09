import sys
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


    def __call__(self, xs, ls):
        # rnn layers
        rs = self.rnn_output(xs)

        # affine layer
        hs = [self.affine(r) for r in rs]

        # crf layer
        indices = argsort_list_descent(hs)
        trans_hs = F.transpose_sequence(permutate_list(hs, indices, inv=False))
        trans_ls = F.transpose_sequence(permutate_list(ls, indices, inv=False))
        loss = self.crf(trans_hs, trans_ls)
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

        return ys


class RNNBiaffineParser(chainer.Chain):
    def __init__(
            self, n_words, word_embed_dim, n_pos, pos_embed_dim, 
            rnn_unit_type, rnn_bidirection, n_rnn_layers, n_rnn_units, affine_dim, n_labels=0,
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

            self.affine_head = L.Linear(rnn_output_dim, affine_dim)
            self.affine_mod = L.Linear(rnn_output_dim, affine_dim)

            self.biaffine_arc = L.Bilinear(affine_dim, affine_dim, 1, nobias=True)
            self.n_labels = n_labels
            if n_labels > 0:
                self.biaffine_label = L.Bilinear(affine_dim, affine_dim, n_labels, nobias=True)

            self.loss_fun = softmax_cross_entropy.softmax_cross_entropy

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
        xs = []
        # TODO use assarray
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


    # arguments: mini-batch of sequences for word, pos, head label and arc label, respectively
    def __call__(self, ws, ps, ths, tls):
        phs = []                # predicted head
        pls = []                # predicted arc label
        loss = None

        # embed
        xs = self.embed(ws, ps)

        # rnn layers
        rs = self.rnn_output(xs)

        # affine layers
        hs = [self.affine_head(r) for r in rs] # head representations
        ms = [self.affine_mod(r) for r in rs]  # modifier representations

        xp = cuda.get_array_module(xs[0])
        dim = self.affine_head.W.shape[0]

        # biaffine layers
        for h, m, th, tl in zip(hs, ms, ths, tls): # for each sentence in mini-batch
            n = len(h) - 1      # actual number of words except root
            arc_scores = [None] * n
            #chainer.Variable(xp.zeros((n,n), dtype='f'))

            # predict arcs
            for i in range(1, n+1):  # for each head in the sentence
                mis = F.reshape(
                    F.concat([m[i] for j in range(n+1)], axis=0),
                    (n+1, dim))
                # print('h', h.shape, h.__dict__)
                # print('mi', mis.shape, type(mis))

                # Biaffine transformation: [h(0), ..., h(n-1)] * W * [m(i), ..., m(i)]
                arc_scores_i = F.reshape(self.biaffine_arc(h, mis), (n+1,))
                arc_scores[i-1] = F.expand_dims(arc_scores_i, axis=0)
            
            arc_scores = F.concat(arc_scores, axis=0)
            mask = gen_masking_matrix(n, xp=xp)
            arc_scores = arc_scores + mask

            # predict labels. th does not contain ROOT's label
            heads = F.reshape(
                F.concat([h[th[i-1]] for i in range(1, n+1)], axis=0), 
                (n, dim))
            
            # Biaffine transformation: [head(m(0)), ..., head(m(n-1))] * W * [m(0), ..., m(n-1)]
            label_scores = self.biaffine_label(heads, m[1:])
            # print('as',arc_scores)
            # print('ls',label_scores)

            # calculate loss
            loss_arc = self.loss_fun(arc_scores, th[1:])
            loss_label = self.loss_fun(label_scores, tl[1:])
            if loss is not None:
                loss += loss_arc + loss_label
            else:
                loss = loss_arc + loss_label

            ph = minmax.argmax(arc_scores, axis=1).data
            pl = minmax.argmax(label_scores, axis=1).data
            if xp is cuda.cupy:
                ph = cuda.to_cpu(ph)
                pl = cuda.to_cpu(pl)
            ph = np.insert(ph, 0, np.int32(-1))
            pl = np.insert(pl, 0, np.int32(-1))
            phs.append(ph)
            pls.append(pl)
            # print('ph',ph)
            # print('th',th)
            # print('pl',pl)
            # print('tl',tl)
            # print()

        return loss, phs, pls

    
    def decode(self, ws, ps):
        xs = self.embed(ws, ps)
        xp = cuda.get_array_module(xs)

        with chainer.no_backprop_mode():
            # predict labels
            # if xp is numpy:
            #     heads = xp.asarray([h[head_ids[i]].data for i in range(n)], dtype='f')
            # else:
            #     heads = xp.array([h[head_ids[i]] for i in range(n)], dtype='f')

            #head_ids = [None] * n
            pass


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
