import sys

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import initializers
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.math import minmax
#from chainer.functions.array import select_item

sys.path.append('src')
import constants
import util
from models.common import MLP
import models.util

#import tools.edmonds.edmonds as edmonds
#from functions import logdet


class BiaffineCombination(chainer.Chain):
    def __init__(self, left_size, right_size, use_b=False):
        super().__init__()
        
        with self.init_scope():
            initialW = None
            w_shape = (left_size, right_size)
            self.W = chainer.variable.Parameter(initializers._get_initializer(initialW), w_shape)

            initialU = None
            u_shape = (left_size, 1)
            self.U = chainer.variable.Parameter(initializers._get_initializer(initialU), u_shape)

            # initialV = None
            # v_shape = (right_size, 1)
            # self.V = chainer.variable.Parameter(initializers._get_initializer(initialV), v_shape)            

            if use_b:
                initialb = 0
                b_shape = 1
                self.b = chainer.variable.Parameter(initialb, b_shape)
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
            self, n_vocab, unigram_embed_dim, n_pos, pos_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp4arcrep_n_layers, mlp4arcrep_n_units,
            mlp4labelrep_n_layers, mlp4labelrep_n_units, 
            mlp4labelpred_n_layers, mlp4labelpred_n_units,
            mlp4pospred_n_layers=0, mlp4pospred_n_units=0,
            n_labels=0, rnn_dropout=0, hidden_mlp_dropout=0, pred_layers_dropout=0,
            pretrained_unigram_embed_dim=0,
            file=sys.stderr):
        super().__init__()

        self.concat_pretraind_embeddings = False
        self.common_arc_label = False
        self.common_head_mod = False
        self.n_dummy = 1

        with self.init_scope():
            self.pred_layers_dropout = pred_layers_dropout
            self.softmax_cross_entropy = softmax_cross_entropy.softmax_cross_entropy

            print('### Parameters', file=file)
            print('acl=label:', self.common_arc_label, file=file)
            print('head=mod:', self.common_head_mod, file=file)

            # unigram embedding layer(s)

            if pretrained_unigram_embed_dim > 0:
                self.pretrained_unigram_embed = L.EmbedID(n_vocab, pretrained_unigram_embed_dim)
                print('# Pretrained unigram embedding matrix: W={}'.format(
                    self.pretrained_unigram_embed.W.shape), file=file)
                if not self.concat_pretraind_embeddings:
                    unigram_embed_dim = pretrained_unigram_embed_dim

            else:
                self.pretrained_unigram_embed = None
            self.pretrained_unigram_embed_dim = pretrained_unigram_embed_dim

            self.unigram_embed = L.EmbedID(n_vocab, unigram_embed_dim)
            print('# Unigram embedding matrix: W={}'.format(self.unigram_embed.W.shape), file=file)

            # subtoken embedding layer

            if n_subtokens > 0 and subtoken_embed_dim > 0:
                self.subtoken_embed = L.EmbedID(n_subtokens, subtoken_embed_dim)
                print('# Subtoken embedding matrix: W={}'.format(self.subtoken_embed.W.shape), file=file)
            self.subtoken_embed_dim = subtoken_embed_dim

            # pos embedding layer

            if n_pos > 0 and pos_embed_dim > 0:
                self.pos_embed = L.EmbedID(n_pos, pos_embed_dim)
                print('# POS embedding matrix: W={}'.format(self.pos_embed.W.shape), file=file)
            self.pos_embed_dim = pos_embed_dim;

            # recurrent layers

            embed_dim = unigram_embed_dim + subtoken_embed_dim + pos_embed_dim
            if self.concat_pretraind_embeddings:
                embed_dim += pretrained_unigram_embed_dim

            self.rnn_unit_type = rnn_unit_type
            self.rnn = models.util.construct_RNN(
                rnn_unit_type, rnn_bidirection, rnn_n_layers, embed_dim, rnn_n_units, rnn_dropout,
                file=file)
            rnn_output_dim = rnn_n_units * (2 if rnn_bidirection else 1)

            # MLP for pos prediction (tentative)
            
            self.pos_prediction = (mlp4pospred_n_layers > 0 and mlp4pospred_n_units > 0)
            if self.pos_prediction:
                print('# MLP for POS prediction', file=file)
                input_dim = unigram_embed_dim + pretrained_unigram_embed_dim + subtoken_embed_dim
                self.mlp_pos = MLP(
                    input_dim, n_pos, n_hidden_units=mlp4pospred_n_units, n_layers=mlp4pospred_n_layers,
                    output_activation=F.identity, dropout=pred_layers_dropout, file=file)

            # MLPs and biaffine layer for arc prediction
            
            print('# MLP for arc heads', file=file)
            self.mlp_arc_head = MLP(
                rnn_output_dim, mlp4arcrep_n_units, n_layers=mlp4arcrep_n_layers, 
                dropout=hidden_mlp_dropout, file=file)

            print('# MLP for arc modifiers', file=file)
            if not self.common_head_mod:
                self.mlp_arc_mod = MLP(
                    rnn_output_dim, mlp4arcrep_n_units, n_layers=mlp4arcrep_n_layers, 
                    dropout=hidden_mlp_dropout, file=file)
            else:
                self.mlp_arc_mod = None
                print('use common reps for heads and modifiers', file=file)
                
            self.biaffine_arc = BiaffineCombination(mlp4arcrep_n_units, mlp4arcrep_n_units)
            print('# Biaffine layer for arc prediction:   W={}, U={}, dropout={}'.format(
                self.biaffine_arc.W.shape, self.biaffine_arc.U.shape, self.pred_layers_dropout), file=file)

            # MLPs for label prediction

            self.label_prediction = (n_labels > 0)
            if self.label_prediction:
                if not self.common_arc_label:
                    print('# MLP for label heads', file=file)
                    self.mlp_label_head = MLP(
                        rnn_output_dim, mlp4labelrep_n_units, n_layers=mlp4labelrep_n_layers, 
                        dropout=hidden_mlp_dropout, 
                        file=file)
                else:
                    self.mlp_label_head = None
                    print('use common reps for arc and label', file=file)

                if not self.common_arc_label:
                    if not self.common_head_mod:
                        print('# MLP for label modifiers', file=file)
                        self.mlp_label_mod = MLP(
                            rnn_output_dim, mlp4labelrep_n_units, n_layers=mlp4labelrep_n_layers, 
                            dropout=hidden_mlp_dropout, 
                            file=file)
                    else:
                        self.mlp_label_mod = None
                        print('use common reps for head and mod', file=file)
                else:
                    self.mlp_label_mod = None
                    print('use common reps for arc and label', file=file)

                print('# MLP for label prediction:', file=file)
                self.mlp_label = MLP(
                    mlp4labelrep_n_units * 2, n_labels, n_hidden_units=mlp4labelpred_n_units,
                    n_layers=mlp4labelpred_n_layers, 
                    output_activation=F.identity, dropout=pred_layers_dropout, file=file)


    def get_features(self, ws, cs=None, ps=None):
        xs = []

        size = len(ws)
        if ps is None:
            ps = [None] * size
        if cs is None:
            cs = [None] * size

        for w, c, p in zip(ws, cs, ps):
            xe = self.unigram_embed(w)
            if self.pretrained_unigram_embed_dim > 0:
                twe = self.pretrained_unigram_embed(w)
                if self.concat_pretraind_embeddings:
                    xe = F.concat((xe, twe), 1)
                else:
                    xe = xe + twe

            if self.subtoken_embed_dim > 0:
                ce = F.concat(
                    [F.mean(self.subtoken_embed(ci), axis=0, keepdims=True) for ci in c],
                    axis=0)
                # sums = F.sum(self.subtoken_embed(c), axis=1)
                # divider = np.sum(c != 0, axis=1, keepdims=True) # num of non-zero (non-padding) elements
                # divider[divider==0] = 1 # avoid 0 division
                # ce = sums / divider
                xe = F.concat((xe, ce), 1)

            if self.pos_embed_dim > 0:
                pe = self.pos_embed(p)
                xe = F.concat((xe, pe), 1)

            xs.append(xe)

        return xs


    def rnn_output(self, xs):
        if self.rnn_unit_type == 'lstm':
            hy, cy, hs = self.rnn(None, None, xs)
        else:
            hy, hs = self.rnn(None, xs)
        return hs


    def predict_pos(self, w, xp=np):
        x = self.unigram_embed(w)
        scores = self.mlp_pos(x, per_element=False)

        yp = minmax.argmax(scores, axis=1).data
        if xp is cuda.cupy:
            yp = cuda.to_cpu(yp)
        yp = np.insert(yp, 0, np.int32(-1))

        return scores, yp


    def predict_arcs(self, m, h, train=True, xp=np):
        scores = self.biaffine_arc(
            F.dropout(m, self.pred_layers_dropout),
            F.dropout(h, self.pred_layers_dropout)
        ) + masking_matrix(len(m), self.n_dummy, xp=xp)

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

        for i in range(self.n_dummy):
            yh = np.insert(yh, 0, np.int32(constants.NO_PARENTS_ID))

        return scores, yh
        

    def predict_labels(self, m, h, xp=np):
        mh = F.concat((m, h), 1)
        scores = self.mlp_label(mh, per_element=False)

        yl = minmax.argmax(scores, axis=1).data
        if xp is cuda.cupy:
            yl = cuda.to_cpu(yl)
        yl = np.insert(yl, 0, np.int32(-1))

        return scores, yl


    # batch of unigrams, pos tags, gold head labels, gold arc labels
    def __call__(
            self, ws, cs=None, ps=None, ghs=None, gls=None, train=True, calculate_loss=True):
        data_size = len(ws)

        if train:
            calclulate_loss = True
        if not ghs:
            ghs = [None] * data_size
        if not gls:
            gls = [None] * data_size

        # embed
        xs = self.get_features(ws, cs, ps)

        # rnn layers
        rs = self.rnn_output(xs)

        # MLP for arc
        # head representations
        hs_arc = self.mlp_arc_head(rs)

        # modifier representations
        if self.common_head_mod:
            ms_arc = [h_arc[self.n_dummy:] for h_arc in hs_arc]
        else:
            rs_noroot = [r[self.n_dummy:] for r in rs]
            ms_arc = self.mlp_arc_mod(rs_noroot)

        # MLP for label
        if self.label_prediction:
            # head representations
            if self.common_arc_label:
                hs_label = hs_arc
            else:
                hs_label = self.mlp_label_head(rs)

            # modifier representations
            if self.common_head_mod:
                ms_label = [h_label[self.n_dummy:] for h_label in hs_label]
            else:
                if self.common_arc_label:
                    ms_label = ms_arc
                else:
                    ms_label = self.mlp_label_mod(rs_noroot)

        else:
            hs_label = [None] * data_size
            ms_label = [None] * data_size            
            
        xp = cuda.get_array_module(xs[0])
        if self.label_prediction:
            if self.common_arc_label:
                ldim = self.mlp_arc_head.layers[-1].W.shape[0]
            else:
                ldim = self.mlp_label_head.layers[-1].W.shape[0]

        loss = chainer.Variable(xp.array(0, dtype='f'))
        yps = []                # predicted label
        yhs = []                # predicted head
        yls = []                # predicted arc label

        # MLP for pos prediction
        if self.pos_prediction: # tentative
            for w, p in zip(ws, ps):
                scores_p, yp = self.predict_pos(w, xp)
                if calculate_loss:
                    loss += 0.01 * self.softmax_cross_entropy(scores_p, p)
                yps.append(yp)

        # (bi)affine layers for arc and label prediction
        cnt = 0
        for h_arc, m_arc, h_label, m_label, gh, gl in zip(
                hs_arc, ms_arc, hs_label, ms_label, ghs, gls): # for each sentence in mini-batch
            scores_a, yh = self.predict_arcs(m_arc, h_arc, train, xp)
            yhs.append(yh)

            if self.label_prediction:
                n = len(m_label)      # the number of unigrams except root
                heads = gh if train else yh
                hm_label = F.reshape(F.concat([h_label[heads[i]] for i in range(1, n+1)], axis=0), (n, ldim))
                scores_l, yl = self.predict_labels(m_label, hm_label, xp)
                yls.append(yl)

            if calculate_loss:
                # print(scores_a.shape)
                # print(yh[self.n_dummy:])
                # print(gh[self.n_dummy:])
                # print(scores_l.shape)
                # print(ylself.n_dummy:])
                # print(gl[self.n_dummy:])
                # print()

                # print('cnt:',cnt)
                loss += self.softmax_cross_entropy(scores_a, gh[self.n_dummy:], 
                                                   ignore_label=constants.UNK_PARENT_ID)
                #loss += global_likelihood_loss(scores_a, gh[self.n_dummy:])
                #loss += structured_margin_loss(scores_a, gh[self.n_dummy:])
                if self.label_prediction:
                    loss += self.softmax_cross_entropy(scores_l, gl[self.n_dummy:],
                                                       ignore_label=constants.UNK_PARENT_ID)
            cnt += 1

        if self.label_prediction:
            if self.pos_prediction:
                return loss, yps, yhs, yls
            else:
                return loss, yhs, yls
        else:
            if self.pos_prediction:
                return loss, yps, yhs
            else:
                return loss, yhs


    def decode(self, ws, cs=None, ps=None, label_prediction=False):
        self.label_prediction=label_prediction
        with chainer.no_backprop_mode():
            ret = self.__call__(
                ws, cs=cs, ps=ps, train=False, calculate_loss=False)
        return ret[1:]


class RNNBiaffineFlexibleParser(RNNBiaffineParser):
    def __init__(
            self, n_vocab, unigram_embed_dim, n_pos, pos_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp4arcrep_n_layers, mlp4arcrep_n_units,
            mlp4labelrep_n_layers, mlp4labelrep_n_units, 
            mlp4labelpred_n_layers, mlp4labelpred_n_units,
            mlp4pospred_n_layers=0, mlp4pospred_n_units=0,
            n_labels=0, rnn_dropout=0, hidden_mlp_dropout=0, pred_layers_dropout=0,
            pretrained_unigram_embed_dim=0, file=sys.stderr):
        super().__init__(
            n_vocab, unigram_embed_dim, n_pos, pos_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp4arcrep_n_layers, mlp4arcrep_n_units,
            mlp4labelrep_n_layers, mlp4labelrep_n_units, 
            mlp4labelpred_n_layers, mlp4labelpred_n_units,
            mlp4pospred_n_layers, mlp4pospred_n_units,
            n_labels, rnn_dropout, hidden_mlp_dropout, pred_layers_dropout,
            pretrained_unigram_embed_dim, file=sys.stderr)

        self.n_dummy = 2

    # batch of unigrams, pos tags, gold head labels, gold arc labels
    # def __call__(self, ws, cs=None, ps=None, ghs=None, gls=None, train=True, calculate_loss=True):
    #     data_size = len(ws)

    #     if train:
    #         calclulate_loss = True
    #     if not ghs:
    #         ghs = [None] * data_size
    #     if not gls:
    #         gls = [None] * data_size

    #     # embed
    #     xs = self.get_features(ws, cs, ps)

    #     # rnn layers
    #     rs = self.rnn_output(xs)

    #     # MLP for arc
    #     # head representations
    #     hs_arc = self.mlp_arc_head(rs)

    #     # modifier representations
    #     if self.common_head_mod:
    #         ms_arc = [h_arc[2:] for h_arc in hs_arc]
    #     else:
    #         rs_words = [r[2:] for r in rs]
    #         ms_arc = self.mlp_arc_mod(rs_words)

    #     # MLP for label
    #     if self.label_prediction:
    #         # head representations
    #         if self.common_arc_label:
    #             hs_label = hs_arc
    #         else:
    #             hs_label = self.mlp_label_head(rs)

    #         # modifier representations
    #         if self.common_head_mod:
    #             ms_label = [h_label[2:] for h_label in hs_label]
    #         else:
    #             if self.common_arc_label:
    #                 ms_label = ms_arc
    #             else:
    #                 ms_label = self.mlp_label_mod(rs_words)

    #     else:
    #         hs_label = [None] * data_size
    #         ms_label = [None] * data_size            
            
    #     xp = cuda.get_array_module(xs[0])
    #     if self.label_prediction:
    #         if self.common_arc_label:
    #             ldim = self.mlp_arc_head.layers[-1].W.shape[0]
    #         else:
    #             ldim = self.mlp_label_head.layers[-1].W.shape[0]

    #     loss = chainer.Variable(xp.array(0, dtype='f'))
    #     yps = []                # predicted label
    #     yhs = []                # predicted head
    #     yls = []                # predicted arc label

    #     # (bi)affine layers for arc and label prediction
    #     cnt = 0
    #     for h_arc, m_arc, h_label, m_label, gh, gl in zip(
    #             hs_arc, ms_arc, hs_label, ms_label, ghs, gls): # for each sentence in mini-batch
    #         scores_a, yh = self.predict_arcs(m_arc, h_arc, train, xp)
    #         yhs.append(yh)

    #         if self.label_prediction:
    #             n = len(m_label)      # the number of unigrams except root
    #             heads = gh if train else yh
    #             hm_label = F.reshape(F.concat([h_label[heads[i]] for i in range(1, n+1)], axis=0), (n, ldim))
    #             scores_l, yl = self.predict_labels(m_label, hm_label, xp)
    #             yls.append(yl)

    #         if calculate_loss:
    #             # print('cnt:',cnt)
    #             loss += self.softmax_cross_entropy(scores_a, gh[self.n_dummy:])
    #             #loss += global_likelihood_loss(scores_a, gh[self.n_dummy:])
    #             #loss += structured_margin_loss(scores_a, gh[self.n_dummy:])
    #             if self.label_prediction:
    #                 loss += self.softmax_cross_entropy(scores_l, gl[self.n_dummy:])
    #         cnt += 1

    #     if self.label_prediction:
    #         return loss, yhs, yls
    #     else:
    #         return loss, yhs


def masking_matrix(sen_len, n_dummy=0, xp=np):
    mat = xp.array([[get_mask_value(i, j) for j in range(sen_len+n_dummy)] for i in range(sen_len+n_dummy)])
    mat = chainer.Variable(mat[n_dummy:])
    return mat


def get_mask_value(i, j):
    return -np.float32(sys.maxsize) if i == j else np.float32(0)


# deprecated
def mst(scores):
    n1 = scores.shape[0]
    n2 = scores.shape[1]
    prices = {}
    y = np.zeros(n1, dtype='i')

    for i in range(n1):
        for j in range(n2):
            prices[(j,i+1)] = -scores.data[i][j]
    res = None #edmonds.run(prices)

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


# def structured_margin_loss(score_matrix, gold, kappa=1):
#     pred = minmax.argmax(score_matrix, axis=1)
#     score_p = sum(select_item.select_item(score_matrix, pred))
#     score_g = sum(select_item.select_item(score_matrix, gold))
#     margin = sum(pred.data != gold)
#     loss = score_p + margin - score_g
#     print(loss.data, '=', score_p.data, '-', score_g.data, '+', margin)
#     print('(', pred.data, ')')
#     return loss


# (Koo, 2007)
# def global_likelihood_loss(score_matrix, gold):
#     xp = cuda.get_array_module(score_matrix)
#     n = len(gold)               # sentence length (except root)
#     timer = util.Timer()

#     # score for gold tree
#     logg = chainer.Variable(xp.array(0, dtype='f'))
#     for m, h in enumerate(gold):
#         logg += score_matrix[m][h]
#     # print('time: logg:', timer.elapsed)
    
#     ## tmp
#     print(n, gold)
#     A_ = F.reshape(F.concat(
#         [chainer.Variable(xp.array([[0]], dtype='f')) if h+1 == m else score_matrix[h:h+1,m:m+1]
#          for m in range(n+1) for h in range(n)], 
#         axis=0), (n+1, n))
#     print('S',score_matrix.data)
#     print('logA',A_.data)

#     # adjacency matrix - A[h,m]
#     A = F.reshape(F.concat(
#         [chainer.Variable(xp.array([[0]], dtype='f')) if h+1 == m else F.exp(score_matrix[h:h+1,m:m+1])
#          for m in range(n+1) for h in range(n)], 
#         axis=0), (n+1, n))
#     # print('time: A:', timer.elapsed)
#     print('A',A.data)

#     # minor of laplacian matrix L^(0,0)
#     L00 = F.reshape(F.concat(
#         [F.sum(A[:,m:m+1], keepdims=True) if h == m else -A[h+1:h+2,m:m+1]
#          for h in range(n) for m in range(n)],
#         axis=0), (n,n))

#     print('L00',L00.data)
#     logz = logdet.logdet(L00)
#     loss = logz - logg
#     print('[a] {} = {} - {}'.format(loss.data, logz.data, logg.data))
#     # if logz.data != np.infty:
#     # else:                       # logz is actually nan because z <= 0
#     #     score_y = F.exp(logg)
#     #     detL = F.det(L00)
#     #     ratio = detL / score_y
#     #     loss = F.log(ratio)
#     #     print('[b] {} = log({}) = log({} / {})'.format(loss.data, ratio.data, detL.data, logg.data))
        
#     sign, logz_np = np.linalg.slogdet(cuda.to_cpu(L00.data))
#     print('[n] {} = ({}) {} - {}'.format(logz_np - logg.data, sign, logz_np, logg.data))
#     print()

#     # print('time: det:', timer.elapsed)
#     return loss


if __name__ == '__main__':
    
    gold = [2, 0, 4, 2]
    score = chainer.Variable(np.array(
        [[1, 0, 2, 0, 0],
         [3, 1, 0, 0, 1],
         [0, 0, 1, 0, 4],
         [0, 1, 2, 0, 0],
        ], dtype='f'))
    
    global_likelihood_loss(score, gold)
