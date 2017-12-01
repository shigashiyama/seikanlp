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

import tools.edmonds.edmonds as edmonds


class MLP(chainer.Chain):
    def __init__(self, n_input, n_units, n_middle_units=0, n_layers=1, output_activation=F.relu, dropout=0, 
                 file=sys.stderr):
        super().__init__()
        with self.init_scope():
            layers = [None] * n_layers
            self.acts = [None] * n_layers
            n_middle_units = n_middle_units if n_middle_units > 0 else n_units

            for i in range(n_layers):
                if i == 0:
                    n_left = n_input
                    n_right = n_units if n_layers == 1 else n_middle_units
                    act = output_activation if n_layers == 1 else F.relu

                elif i == n_layers - 1:
                    n_left = n_middle_units
                    n_right = n_units
                    act = output_activation

                else:
                    n_left = n_right = n_middle_units
                    act = F.relu

                layers[i] = L.Linear(n_left, n_right)
                print(i, n_left, n_right, layers[i].W.shape)
                self.acts[i] = act
            
            self.layers = chainer.ChainList(*layers)
            self.dropout = dropout

        for i in range(n_layers):
            print('# Affine {}-th layer:                   W={}, b={}, dropout={}, act={}'.format(
                i, self.layers[i].W.shape, self.layers[i].b.shape, self.dropout, self.acts[i]), file=file)


    def __call__(self, xs, start_index=0, per_element=True):
        hs_ = xs
        hs = None

        if per_element:
            for i in range(len(self.layers)):
                hs = [self.acts[i](
                    self.layers[i](
                        F.dropout(h_, self.dropout))) for h_ in hs_]
                hs_ = hs

        else:
            for i in range(len(self.layers)):
                hs = self.acts[i](
                    self.layers[i](
                        F.dropout(hs_, self.dropout)))
                hs_ = hs

        return hs
    

# Base model that consists of embedding layer, recurrent network (RNN) layers and affine layer.

# TODO: remove initial embedding, add fixed pretrained embedding
# TODO: use MLP
class RNNTaggerBase(chainer.Chain):
    def __init__(
            self, n_vocab, embed_dim, rnn_unit_type, rnn_bidirection, n_rnn_layers, n_rnn_units, 
            n_labels, feat_dim=0, dropout=0, initial_embed=None, file=sys.stderr):
        super(RNNTaggerBase, self).__init__()

        with self.init_scope():
            #self.dropout = dropout

            if initial_embed:
                self.embed = initial_embed
                embed_dim = initial_embed.W.shape[1]
            else:
                self.embed = L.EmbedID(n_vocab, self.embed_dim)
            input_vec_size = embed_dim + feat_dim

            self.rnn_unit_type = rnn_unit_type
            self.rnn = construct_RNN(
                rnn_unit_type, rnn_bidirection, n_rnn_layers, input_vec_size, n_rnn_units, dropout)
            rnn_output_dim = n_rnn_units * (2 if rnn_bidirection else 1)

            self.affine = L.Linear(rnn_output_dim, n_labels)

            print('### Parameters', file=file)
            print('# Embedding layer: W={}'.format(self.embed.W.shape), file=file)
            print('# Additional features dimension: {}'.format(feat_dim), file=file)
            print('# RNN unit: {}, dropout={}'.format(self.rnn, self.rnn.__dict__['dropout']), file=file)
            if rnn_unit_type == 'lstm':
                i = 0
                for c in self.rnn._children:
                    print('#   LSTM {}-th param'.format(i), file=file)
                    print('#      0 - W={}, b={}'.format(c.w0.shape, c.b0.shape), file=file) 
                    print('#      1 - W={}, b={}'.format(c.w1.shape, c.b1.shape), file=file) 
                    print('#      2 - W={}, b={}'.format(c.w2.shape, c.b2.shape), file=file) 
                    print('#      3 - W={}, b={}'.format(c.w3.shape, c.b3.shape), file=file) 
                    print('#      4 - W={}, b={}'.format(c.w4.shape, c.b4.shape), file=file) 
                    print('#      5 - W={}, b={}'.format(c.w5.shape, c.b5.shape), file=file) 
                    print('#      6 - W={}, b={}'.format(c.w6.shape, c.b6.shape), file=file) 
                    print('#      7 - W={}, b={}'.format(c.w7.shape, c.b7.shape), file=file) 
                    i += 1
            print('# Affine layer: W={}, b={}'.format(self.affine.W.shape, self.affine.b.shape), file=file)


    def rnn_output(self, xs):
        if self.rnn_unit_type == 'lstm':
            hy, cy, hs = self.rnn(None, None, xs)
        else:
            hy, hs = self.rnn(None, xs)
        return hs


class RNNTagger(RNNTaggerBase):
    def __init__(
            self, n_vocab, embed_dim, rnn_unit_type, rnn_bidirection, n_rnn_layers, n_rnn_units, 
            n_labels, feat_dim=0, dropout=0, initial_embed=None, file=sys.stderr):
        super(RNNTagger, self).__init__(
            n_vocab, embed_dim, rnn_unit_type, rnn_bidirection, n_rnn_layers, n_rnn_units, 
            n_labels, feat_dim, dropout, initial_embed, file)

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
            n_labels, feat_dim=0, dropout=0, initial_embed=None, file=sys.stderr):
        super(RNNTagger, self).__init__(
            n_vocab, embed_dim, rnn_unit_type, rnn_bidirection, n_rnn_layers, n_rnn_units, 
            n_labels, feat_dim, dropout, initial_embed, file)

        with self.init_scope():
            self.crf = L.CRF1d(n_labels)

            print('# CRF cost: {}\n'.format(self.crf.cost.shape), file=file)


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


# unused
class Biaffine(chainer.Chain):
    def __init__(self, left_size, right_size, out_size, use_b=False):
        super(Biaffine, self).__init__()
        self.out_size = out_size
        
        with self.init_scope():
            initialW = None
            w_shape = (out_size, left_size, right_size)
            self.W = variable.Parameter(initializers._get_initializer(initialW), w_shape)

            if use_b:
                initialb = np.array([[0]])
                b_shape = (out_size, 1)
                self.b = variable.Parameter(initialb, b_shape)
            else:
                self.b = None

    # TODO chainer 3 -> F.matmul
    def __call__(self, x1, x2):
        # inputs: x1 = [x1_1 ... x1_i ... x1_n]; dim(x1_i)=d1=left_size
        #         x2 = [x1_2 ... x2_i ... x2_n]; dim(x2_j)=d2=right_size
        # output: o_ik = x1_i * W^(k) * x2_i (k = 1 ... K=out_size)

        n = x1.shape[0]
        d1 = x1.shape[1]
        d2 = x2.shape[1]
        K = self.out_size

        x1b = F.broadcast_to(x1, (K, n, d1)) # (n, d1) -> (K, n, d1) | x1b = [x1 ... x1]
        x1_W = F.batch_matmul(x1b, self.W)   # (K, n, d1) * (K, d1, d2) => (L, n, d2)
        x1_Wr = F.swapaxes(x1_W, 0, 1)       # (K, n, d2) -> (n, K, d2)

        x2r = F.reshape(x2, (n, d2, 1))      # (n, d2) -> (n, d2, 1)
        x1_W_x2 = F.batch_matmul(x1_Wr, x2r) # (n, K, d2) * (n, d2, 1) -> (n, K, 1)
        res = x1_W_x2
        if self.b is not None:
            b = F.broadcast_to(self.b, (n, K, 1))
            res = x1_W_x2 + b

        res = F.reshape(res, (n, K))

        return res


class BiaffineCombination(chainer.Chain):
    def __init__(self, left_size, right_size, use_b=False):
        super(BiaffineCombination, self).__init__()
        
        with self.init_scope():
            initialW = None
            w_shape = (left_size, right_size)
            self.W = variable.Parameter(initializers._get_initializer(initialW), w_shape)

            initialU = None
            u_shape = (left_size, 1)
            self.U = variable.Parameter(initializers._get_initializer(initialU), u_shape)

            # initialV = None
            # v_shape = (right_size, 1)
            # self.V = variable.Parameter(initializers._get_initializer(initialV), v_shape)            

            if use_b:
                initialb = 0
                b_shape = 1
                self.b = variable.Parameter(initialb, b_shape)
            else:
                self.b = None

    def __call__(self, x1, x2):
        # inputs: x1 = [x1_1 ... x1_i ... x1_n1]; dim(x1_i)=d1=left_size
        #         x2 = [x2_1 ... x2_j ... x2_n2]; dim(x2_j)=d2=right_size
        # output: o_ij = x1_i * W * x2_j + x2_j * U + b

        n1 = x1.shape[0]
        n2 = x2.shape[0]

        x2T = F.transpose(x2)
        x1_W = F.matmul(x1, self.W)                           # (n1, d1) * (d1, d2) => (n1, d2)
        x1_W_x2 = F.matmul(x1_W, x2T)                         # (n1, d2) * (d2, n2) => (n1, n2)
        x1_U = F.broadcast_to(F.matmul(x1, self.U), (n1, n2)) # (n1, d1) * (d1, 1)  => (n1, 1) -> (n1, n2)
        res = x1_W_x2 + x1_U

        # TODO fix
        # x2_V = F.broadcast_to(
        #     F.transpose(F.matmul(x2, self.V)), (n1, n2)) # (n2, d2) * (d2, 1)  => (n2, 1) -> (n1, n2)
        #res = x1_W_x2 + x2_V

        if self.b is not None:
            b = F.broadcast_to(self.b, (n1, n2))
            res = res + b

        return res


class RNNBiaffineParser(chainer.Chain):
    def __init__(
            self, n_words, word_embed_dim, n_pos, pos_embed_dim,
            rnn_unit_type, rnn_bidirection, n_rnn_layers, n_rnn_units, 
            affine_layers_arc, affine_units_arc, affine_layers_label, affine_units_label, 
            n_labels=0, rnn_dropout=0, mlp_dropout=0, biaffine_dropout=0,
            trained_word_embed_dim=0, use_mlp_for_label_pred=False,
            file=sys.stderr):
        super(RNNBiaffineParser, self).__init__()

        self.concat_pretraind_embeddings = True # unused

        with self.init_scope():
            self.use_mlp_for_label_pred = use_mlp_for_label_pred
            self.biaffine_dropout = biaffine_dropout
            self.softmax_cross_entropy = softmax_cross_entropy.softmax_cross_entropy

            print('### Parameters', file=file)

            # word embedding layer(s)

            self.word_embed = L.EmbedID(n_words, word_embed_dim)
            print('# Word embedding matrix: W={}'.format(self.word_embed.W.shape), file=file)

            if trained_word_embed_dim > 0:
                self.trained_word_embed = L.EmbedID(n_words, trained_word_embed_dim)
                print('# Pretrained word embedding matrix: W={}'.format(
                    self.trained_word_embed.W.shape), file=file)
            else:
                self.trained_word_embed = None
            self.trained_word_embed_dim = trained_word_embed_dim

            # pos embedding layer

            if pos_embed_dim > 0:
                self.pos_embed = L.EmbedID(n_pos, pos_embed_dim)
                print('# POS embedding matrix: W={}'.format(self.pos_embed.W.shape), file=file)
            self.pos_embed_dim = pos_embed_dim;

            # recurrent layers
            embed_dim = word_embed_dim + trained_word_embed_dim + pos_embed_dim
            self.rnn_unit_type = rnn_unit_type
            self.rnn = construct_RNN(
                rnn_unit_type, rnn_bidirection, n_rnn_layers, embed_dim, n_rnn_units, rnn_dropout,
                file=file)
            rnn_output_dim = n_rnn_units * (2 if rnn_bidirection else 1)

            # affine layers for arc prediction
            
            print('# MLP for arc heads', file=file)
            self.affine_arc_head = MLP(
                rnn_output_dim, affine_units_arc, n_layers=affine_layers_arc, dropout=mlp_dropout, file=file)

            print('# MLP for arc modifiers', file=file)
            self.affine_arc_mod = MLP(
                rnn_output_dim, affine_units_arc, n_layers=affine_layers_arc, dropout=mlp_dropout, file=file)

            self.biaffine_arc = BiaffineCombination(affine_units_arc, affine_units_arc)
            print('# Biaffine layer for arc prediction:   W={}, U={}, dropout={}'.format(
                self.biaffine_arc.W.shape, self.biaffine_arc.U.shape, self.biaffine_dropout), file=file)

            # affine layers for label prediction

            self.label_prediction = (n_labels > 0)
            if self.label_prediction:
                print('# MLP for label heads', file=file)
                self.affine_label_head = MLP(
                    rnn_output_dim, affine_units_label, n_layers=affine_layers_label, dropout=mlp_dropout, 
                    file=file)

                print('# MLP for label modifiers', file=file)
                self.affine_label_mod = MLP(
                    rnn_output_dim, affine_units_label, n_layers=affine_layers_label, dropout=mlp_dropout, 
                    file=file)

                if self.use_mlp_for_label_pred: # tmp
                    print('# MLP for label prediction:', file=file)
                    self.affine_label = MLP(
                        affine_units_label * 2, n_labels, output_activation=F.identity,
                        dropout=biaffine_dropout, file=file)
                else:
                    self.biaffine_label = L.Bilinear(
                        affine_units_label, affine_units_label, n_labels, nobias=True)
                    print('# Biaffine layer for label prediction: W={}, b={}, dropout={}\n'.format(
                        self.biaffine_label.W.shape, 
                        self.biaffine_label.b.shape if 'b' in self.biaffine_label.__dict__ else None,
                        self.biaffine_dropout), file=file)


    def embed(self, ws, ps=None):
        xs = []

        if self.pos_embed_dim == 0:
            for w in ws:
                we = self.word_embed(w)
                if self.trained_word_embed_dim > 0:
                    twe = self.trained_word_embed(w)
                    xe = F.concat((we, pe), 1)
                else:
                    xe = we
                xs.append(xe)

        else:
            for w, p in zip(ws, ps):
                we = self.word_embed(w)
                pe = self.pos_embed(p)
                if self.trained_word_embed_dim == 0:
                    xe = F.concat((we, pe), 1)
                else:
                    twe = self.trained_word_embed(w)
                    xe = F.concat((we, twe, pe), 1)
                xs.append(xe)
        return xs


    def rnn_output(self, xs):
        if self.rnn_unit_type == 'lstm':
            hy, cy, hs = self.rnn(None, None, xs)
        else:
            hy, hs = self.rnn(None, xs)
        return hs


    def predict_arcs(self, m, h, train=True, xp=np):
        scores = self.biaffine_arc(
            F.dropout(m, self.biaffine_dropout),
            F.dropout(h, self.biaffine_dropout)
        ) + gen_masking_matrix(len(m), xp=xp)

        yh = minmax.argmax(scores, axis=1).data
        if xp is cuda.cupy:
            yh = cuda.to_cpu(yh)

        # if not train:
        #     not_tree = detect_cycle(yh)

        #     if not_tree:
        #         yh_mst = mst(scores)
        #         yh = yh_mst

            # conflict = False
            # for yi, ymi in zip(yh, yh_mst):
            #     if yi != ymi:
            #         conflict = True
            #         break
            # print('\n{} {}'.format(not_tree, conflict))
            
            # print(yh)
            # print(yh_mst)

            # print(scores.data)
            # p = np.zeros((len(yh), len(yh)+1))
            # for i, yi in enumerate(yh):
            #     p[i][yi] = 1
            #     print(p)

        yh = np.insert(yh, 0, np.int32(-1))

        return scores, yh
        

    def predict_labels(self, m, h, xp=np):
        if self.use_mlp_for_label_pred:
            mh = F.concat((m, h), 1)
            scores = self.affine_label(mh, per_element=False)

        else:
            scores = self.biaffine_label(
                F.dropout(m, self.biaffine_dropout),
                F.dropout(h, self.biaffine_dropout))

        yl = minmax.argmax(scores, axis=1).data
        if xp is cuda.cupy:
            yl = cuda.to_cpu(yl)
        yl = np.insert(yl, 0, np.int32(-1))

        return scores, yl


    # batch of words, pos tags, gold head labels, gold arc labels
    def __call__(self, ws, ps=None, ghs=None, gls=None, train=True, calculate_loss=True):
        data_size = len(ws)

        if train:
            calclulate_loss = True
        if not ghs:
            ghs = [None] * data_size
        if not gls:
            gls = [None] * data_size

        # embed
        xs = self.embed(ws, ps)

        # rnn layers
        rs = self.rnn_output(xs)

        # affine layers for arc
        # head representations

        hs_arc = self.affine_arc_head(rs)
        # hs_arc = [F.dropout(F.relu(self.affine_arc_head(r)), self.dropout) for r in rs] 
        # hs_arc = [F.dropout(
        #     F.relu(self.affine_arc_head(F.dropout(r, self.dropout))), self.dropout) for r in rs] 

        # modifier representations
        rs_noroot = [r[1:] for r in rs]
        ms_arc = self.affine_arc_mod(rs_noroot)
        # ms_arc = [F.dropout(F.relu(self.affine_arc_mod(r[1:])), self.dropout) for r in rs]
        # ms_arc = [F.dropout(
        #     F.relu(self.affine_arc_mod(F.dropout(r[1:], self.dropout))), self.dropout) for r in rs]

        # affine layers for label
        if self.label_prediction:
            # head representations
            hs_label = self.affine_label_head(rs)
            # hs_label = [F.dropout(F.relu(self.affine_label_head(r)), self.dropout) for r in rs]

            # modifier representations
            ms_label = self.affine_label_mod(rs_noroot)
            # ms_label = [F.dropout(F.relu(self.affine_label_mod(r[1:])), self.dropout) for r in rs]
        else:
            hs_label = [None] * data_size
            ms_label = [None] * data_size            
            
        xp = cuda.get_array_module(xs[0])
        if self.label_prediction:
            ldim = self.affine_label_head.layers[-1].W.shape[0]
            #ldim = self.affine_label_head.W.shape[0]
        loss = chainer.Variable(xp.array(0, dtype='f'))
        yhs = []                # predicted head
        yls = []                # predicted arc label

        # biaffine layers
        for h_arc, m_arc, h_label, m_label, gh, gl in zip(
                hs_arc, ms_arc, hs_label, ms_label, ghs, gls): # for each sentence in mini-batch
            scores_a, yh = self.predict_arcs(m_arc, h_arc, train, xp)
            yhs.append(yh)

            if self.label_prediction:
                n = len(m_label)      # the number of words except root
                heads = gh if train else yh
                hm_label = F.reshape(F.concat([h_label[heads[i]] for i in range(1, n+1)], axis=0), (n, ldim))
                scores_l, yl = self.predict_labels(m_label, hm_label, xp)
                yls.append(yl)

            if calculate_loss:
                loss += self.softmax_cross_entropy(scores_a, gh[1:])
                if self.label_prediction:
                    loss += self.softmax_cross_entropy(scores_l, gl[1:])

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
    return -np.float32(sys.maxsize) if i == j else np.float32(0)


def mst(scores):
    n1 = scores.shape[0]
    n2 = scores.shape[1]
    prices = {}
    y = np.zeros(n1, dtype='i')

    for i in range(n1):
        for j in range(n2):
            prices[(j,i+1)] = -scores.data[i][j]
    res = edmonds.run(prices)

    for j in res:
        for i in res[j]:
            y[i-1] = j

    return y


def detect_cycle(y):
    whole = list(range(1, len(y)+1))
    buff = []
    stack = []
    pointer = 1

    while True:
        child = pointer
        if not child in buff:
            buff.append(child)

        else:
            return True         # there is a cycle

        parent = y[pointer-1]
        if parent == 0 or parent in stack:
            stack.extend(buff)
            buff = []
            unvisited = set(whole) - set(stack)
            if unvisited:
                pointer = min(unvisited)

            else:
                return False

        else:
            pointer = parent


def construct_RNN(unit_type, bidirection, n_layers, n_input, n_units, dropout, file=sys.stderr):
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

    print('# RNN unit: {}, dropout={}'.format(rnn, rnn.__dict__['dropout']), file=file)
    if unit_type == 'lstm':
        i = 0
        for c in rnn._children:
            print('#   LSTM {}-th param'.format(i), file=file)
            print('#      0 - W={}, b={}'.format(c.w0.shape, c.b0.shape), file=file) 
            print('#      1 - W={}, b={}'.format(c.w1.shape, c.b1.shape), file=file) 
            print('#      2 - W={}, b={}'.format(c.w2.shape, c.b2.shape), file=file) 
            print('#      3 - W={}, b={}'.format(c.w3.shape, c.b3.shape), file=file) 
            print('#      4 - W={}, b={}'.format(c.w4.shape, c.b4.shape), file=file) 
            print('#      5 - W={}, b={}'.format(c.w5.shape, c.b5.shape), file=file) 
            print('#      6 - W={}, b={}'.format(c.w6.shape, c.b6.shape), file=file) 
            print('#      7 - W={}, b={}'.format(c.w7.shape, c.b7.shape), file=file) 
            i += 1

    return rnn


def load_and_update_embedding_layer(embed, id2token, external_model, finetuning=False):
    n_vocab = len(id2token)
    dim = external_model.wv.syn0[0].shape[0]
    initialW = initializers.normal.Normal(1.0)

    weight = []
    count = 0
    for i in range(n_vocab):
        key = id2token[i]
        if key in external_model.wv.vocab:
            vec = external_model.wv[key]
            count += 1
        else:
            if finetuning:
                vec = initializers.generate_array(initialW, (dim, ), np)
            else:
                vec = np.zeros(dim, dtype='f')
        weight.append(vec)

    weight = np.reshape(weight, (n_vocab, dim))
    embed = L.EmbedID(n_vocab, dim)
    embed.W = chainer.Parameter(initializer=weight)

    if count >= 1:
        print('Use {} pretrained embedding vectors\n'.format(count), file=sys.stderr)


def grow_embedding_layers(id2token_org, id2token_grown, rand_embed, 
                          trained_embed=None, external_model=None, train=False, file=sys.stderr):
    n_vocab = rand_embed.W.shape[0]
    d_rand = rand_embed.W.shape[1]
    d_trained = external_model.wv.syn0[0].shape[0] if external_model else 0

    initialW = initializers.normal.Normal(1.0)
    w2_rand = []
    w2_trained = []

    count = 0
    for i in range(len(id2token_org), len(id2token_grown)):
        if train:               # resume training
            vec_rand = initializers.generate_array(initialW, (d_rand, ), np)
        else:                   # test
            vec_rand = rand_embed.W[0].data # use trained vector of unknown word
        w2_rand.append(vec_rand)

        if external_model:
            key = id2token_grown[i]
            if key in external_model.wv.vocab:
                vec_trained = external_model.wv[key]
                count += 1
            else:
                vec_trained = np.zeros(d_trained, dtype='f')
            w2_trained.append(vec_trained)

    diff = len(id2token_grown) - len(id2token_org)
    w2_rand = np.reshape(w2_rand, (diff, d_rand))
    w_rand = F.concat((rand_embed.W, w2_rand), 0)
    rand_embed.W = chainer.Parameter(initializer=w_rand.data)

    if external_model:
        w2_trained = np.reshape(w2_trained, (diff, d_trained))
        w_trained = F.concat((trained_embed.W, w2_trained), 0)
        trained_embed = L.EmbedID(0, 0)
        trained_embed.W = chainer.Parameter(initializer=w_trained.data)

    print('Grow embedding matrix: {} -> {}'.format(n_vocab, rand_embed.W.shape[0]), file=file)
    if count >= 1:
        print('Add {} pretrained embedding vectors'.format(count), file=file)
    print('', file=file)


def grow_affine_layer(id2label_org, id2label_grown, affine, file=sys.stderr):
    org_len = len(id2label_org)
    new_len = len(id2label_grown)
    diff = new_len - org_len

    w_org = predictor.affine.W
    b_org = predictor.affine.b
    w_org_shape = w_org.shape
    b_org_shape = b_org.shape

    dim = w_org.shape[1]
    initialW = initializers.normal.Normal(1.0)

    w_diff = chainer.variable.Parameter(initialW, (diff, dim))
    w_new = F.concat((w_org, w_diff), 0)

    b_diff = chainer.variable.Parameter(initialW, (diff,))
    b_new = F.concat((b_org, b_diff), 0)
            
    affine.W = chainer.Parameter(initializer=w_new.data)
    affine.b = chainer.Parameter(initializer=b_new.data)

    print('Grow affine layer: {}, {} -> {}, {}'.format(
        w_org_shape, b_org_shape, affine.W.shape, affine.b.shape), file=file)
    

def grow_crf_layer(id2label_org, id2label_grown, crf, file=sys.stderr):
    org_len = len(id2label_org)
    new_len = len(id2label_grown)
    diff = new_len - org_len

    c_org = crf.cost
    c_diff1 = chainer.variable.Parameter(0, (org_len, diff))
    c_diff2 = chainer.variable.Parameter(0, (diff, new_len))
    c_tmp = F.concat((c_org, c_diff1), 1)
    c_new = F.concat((c_tmp, c_diff2), 0)
    crf.cost = chainer.Parameter(initializer=c_new.data)
    
    print('Grow CRF layer: {} -> {}'.format(c_org.shape, crf.cost.shape, file=file))


def grow_biaffine_layer(id2label_org, id2label_grown, biaffine, file=sys.stderr):
    org_len = len(id2label_org)
    new_len = len(id2label_grown)
    diff = new_len - org_len

    w_org = affine.W
    w_org_shape = w_org.shape

    dim = w_org.shape[1]
    initialW = initializers.normal.Normal(1.0)

    w_diff = chainer.variable.Parameter(initialW, (diff, dim))
    w_new = F.concat((w_org, w_diff), 0)
    affine.W = chainer.Parameter(initializer=w_new.data)
    w_shape = affine.W.shape

    if 'b' in affine.__dict__:
        b_org = affine.b
        b_org_shape = b_org.shape
        b_diff = chainer.variable.Parameter(initialW, (diff,))
        b_new = F.concat((b_org, b_diff), 0)
        affine.b = chainer.Parameter(initializer=b_new.data)
        b_shape = affine.b.shape
    else:
        b_org_shape = b_shape = None

    print('Grow biaffine layer: {}, {} -> {}, {}'.format(
        w_org_shape, b_org_shape, w_shape, b_shape), file=file)
