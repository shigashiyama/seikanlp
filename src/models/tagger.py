import sys
import traceback

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.math import minmax
from chainer.links.connection.n_step_rnn import argsort_list_descent, permutate_list

import constants
import models.util
import models.tagger_v0_0_2
from models.util import ModelUsage
from models.common import MLP
from models.parser import BiaffineCombination

FLOAT_MIN = -100000.0

class RNNTagger(chainer.Chain):
    def __init__chain__(self):  # tmp
        super().__init__()


    def __init__(
            self, n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_tokentypes, tokentype_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, 
            use_crf=True, feat_dim=0, mlp_n_additional_units=0,
            embed_dropout=0, rnn_dropout=0, mlp_dropout=0, 
            pretrained_unigram_embed_dim=0, pretrained_bigram_embed_dim=0, 
            pretrained_embed_usage=ModelUsage.NONE,
            file=sys.stderr):
        super().__init__()

        with self.init_scope():
            print('### Parameters', file=file)

            # embedding layer(s)

            self.pretrained_embed_usage = pretrained_embed_usage

            self.embed_dropout = embed_dropout
            print('# Embedding dropout ratio={}'.format(self.embed_dropout), file=file)
            self.unigram_embed, self.pretrained_unigram_embed = models.util.construct_embeddings(
                n_vocab, unigram_embed_dim, pretrained_unigram_embed_dim, pretrained_embed_usage)
            if self.pretrained_embed_usage != ModelUsage.NONE:
                print('# Pretrained embedding usage: {}'.format(self.pretrained_embed_usage), file=file)
            print('# Unigram embedding matrix: W={}'.format(self.unigram_embed.W.shape), file=file)
            embed_dim = self.unigram_embed.W.shape[1]
            if self.pretrained_unigram_embed is not None:
                if self.pretrained_embed_usage == ModelUsage.CONCAT:
                    embed_dim += self.pretrained_unigram_embed.W.shape[1]
                print('# Pretrained unigram embedding matrix: W={}'.format(
                    self.pretrained_unigram_embed.W.shape), file=file)

            if n_bigrams > 0 and bigram_embed_dim > 0:
                self.bigram_embed, self.pretrained_bigram_embed = models.util.construct_embeddings(
                    n_bigrams, bigram_embed_dim, pretrained_bigram_embed_dim, pretrained_embed_usage)
                if self.pretrained_embed_usage != ModelUsage.NONE:
                    print('# Pretrained embedding usage: {}'.format(self.pretrained_embed_usage), file=file)
                print('# Bigram embedding matrix: W={}'.format(self.bigram_embed.W.shape), file=file)
                embed_dim += self.bigram_embed.W.shape[1]
                if self.pretrained_bigram_embed is not None:
                    if self.pretrained_embed_usage == ModelUsage.CONCAT:
                        embed_dim += self.pretrained_bigram_embed.W.shape[1]
                    print('# Pretrained bigram embedding matrix: W={}'.format(
                        self.pretrained_bigram_embed.W.shape), file=file)
                    
            if n_tokentypes > 0 and tokentype_embed_dim > 0:
                self.type_embed = L.EmbedID(n_tokentypes, tokentype_embed_dim)
                embed_dim += tokentype_embed_dim
                print('# Token type embedding matrix: W={}'.format(self.type_embed.W.shape), file=file)

            if n_subtokens > 0 and subtoken_embed_dim > 0:
                self.subtoken_embed = L.EmbedID(n_subtokens, subtoken_embed_dim)
                embed_dim += subtoken_embed_dim
                print('# Subtoken embedding matrix: W={}'.format(self.subtoken_embed.W.shape), file=file)
            self.subtoken_embed_dim = subtoken_embed_dim;

            self.additional_feat_dim = feat_dim
            if feat_dim > 0:
                embed_dim += feat_dim
                print('# Additional features dimension: {}'.format(feat_dim), file=file)

            # recurrent layers

            self.rnn_unit_type = rnn_unit_type
            self.rnn = models.util.construct_RNN(
                rnn_unit_type, rnn_bidirection, rnn_n_layers, embed_dim, rnn_n_units, rnn_dropout)
            rnn_output_dim = rnn_n_units * (2 if rnn_bidirection else 1)

            # MLP

            print('# MLP', file=file)
            mlp_in = rnn_output_dim + mlp_n_additional_units
            self.mlp = MLP(mlp_in, n_labels, n_hidden_units=mlp_n_units, n_layers=mlp_n_layers,
                           output_activation=F.identity, dropout=mlp_dropout, file=file)

            # CRF or softmax

            self.use_crf = use_crf
            if self.use_crf:
                self.crf = L.CRF1d(n_labels)
                print('# CRF cost: {}'.format(self.crf.cost.shape), file=file)
            else:
                self.softmax_cross_entropy = softmax_cross_entropy.softmax_cross_entropy


    # unigram, bigram, type, subtoken, feature, label
    def __call__(self, us, bs=None, ts=None, ss=None, fs=None, ls=None, calculate_loss=True):
        xs = self.extract_features(us, bs, ts, ss, fs)
        rs = self.rnn_output(xs)
        loss, ps = self.predict(rs, ls=ls, calculate_loss=calculate_loss)

        return loss, ps
        

    def extract_features(self, us, bs, ts, ss, fs):
        xs = []
        if bs is None:
            bs = [None] * len(us)
        if ts is None:
            ts = [None] * len(us)
        if ss is None:
            ss = [None] * len(us)
        if fs is None:
            fs = [None] * len(us)

        for u, b, t, s, f in zip(us, bs, ts, ss, fs):
            ue = self.unigram_embed(u)
            if self.pretrained_unigram_embed is not None:
                if self.pretrained_embed_usage == ModelUsage.ADD:
                    pe = self.pretrained_unigram_embed(u)
                    ue = ue + pe
                elif self.pretrained_embed_usage == ModelUsage.CONCAT:
                    pe = self.pretrained_unigram_embed(u)
                    ue = F.concat((ue, pe), 1)
            ue = F.dropout(ue, self.embed_dropout)
            xe = ue

            if b is not None:
                be = self.bigram_embed(b)
                if self.pretrained_unigram_embed is not None:
                    if self.pretrained_embed_usage == ModelUsage.ADD:
                        pe = self.pretrained_bigram_embed(b)
                        be = be + pe
                    elif self.pretrained_embed_usage == ModelUsage.CONCAT:
                        pe = self.pretrained_bigram_embed(b)
                        be = F.concat((be, pe), 1)
                be = F.dropout(ue, self.embed_dropout)
                xe = F.concat((xe, be), 1)

            if t is not None:
                te = self.type_embed(t)
                te = F.dropout(ue, self.embed_dropout)
                xe = F.concat((xe, te), 1)

            if s is not None:   # TODO fix
                se = self.subtoken_embed(s)
                se = F.dropout(ue, self.embed_dropout)
                xe = F.concat((xe, se), 1)

            if f is not None:
                xe = F.concat((xe, f), 1)

            xs.append(xe)

        return xs


    def rnn_output(self, xs):
        if self.rnn_unit_type == 'lstm':
            hy, cy, hs = self.rnn(None, None, xs)
        else:
            hy, hs = self.rnn(None, xs)
        return hs


    def predict(self, rs, ls=None, calculate_loss=True):
        if self.use_crf:
            return self.predict_crf(rs, ls, calculate_loss)
        else:
            return self.predict_softmax(rs, ls, calculate_loss)


    def predict_softmax(self, rs, ls=None, calculate_loss=True):
        ys = self.mlp(rs)
        ps = []

        xp = cuda.get_array_module(rs[0])
        loss = chainer.Variable(xp.array(0, dtype='f'))
        if not ls:
            ls = [None] * len(rs)
        for y, l in zip(ys, ls):
            if calculate_loss:
                loss += self.softmax_cross_entropy(y, l)
            ps.append([np.argmax(yi.data) for yi in y])
        
        return loss, ps


    def predict_crf(self, rs, ls=None, calculate_loss=True):
        hs = self.mlp(rs)

        indices = argsort_list_descent(hs)
        trans_hs = F.transpose_sequence(permutate_list(hs, indices, inv=False))
        score, trans_ys = self.crf.argmax(trans_hs)
        ys = permutate_list(F.transpose_sequence(trans_ys), indices, inv=True)
        ps = [y.data for y in ys]

        xp = cuda.get_array_module(hs[0])
        if calculate_loss:
            tmp = permutate_list(ls, indices, inv=False)
            trans_ls = F.transpose_sequence(permutate_list(ls, indices, inv=False)) # sort by length
            loss = self.crf(trans_hs, trans_ls)
        else:
            loss = chainer.Variable(xp.array(0, dtype='f'))

        return loss, ps


    def decode(self, us, bs=None, ts=None, ss=None, fs=None):
        with chainer.no_backprop_mode():
            _, ps = self.__call__(us, bs, ts, ss, fs, calculate_loss=False)
        return ps


class RNNTaggerWithChunk(RNNTagger):
    def __init__(
            self, n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_tokentypes, tokentype_embed_dim, n_subtokens, subtoken_embed_dim,
            n_chunks, chunk_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers1, rnn_n_units1, rnn_n_layers2, rnn_n_units2,
            mlp_n_layers, mlp_n_units, n_labels, use_crf=True,
            feat_dim=0, embed_dropout=0, rnn_dropout=0, biaffine_dropout=0, mlp_dropout=0, chunk_vector_dropout=0, 
            pretrained_unigram_embed_dim=0, pretrained_bigram_embed_dim=0, pretrained_chunk_embed_dim=0, 
            pretrained_embed_usage=ModelUsage.NONE, chunk_pooling_type='wave', max_chunk_len=0,
            chunk_loss_ratio=0, biaffine_type='',
            file=sys.stderr):
        self.__init__chain__()

        # tmp
        self.id2token = None
        self.id2chunk = None
        self.id2label = None

        # TODO refactor with super class
        self.chunk_loss_ratio = chunk_loss_ratio
        self.use_attention = (chunk_pooling_type == 'wave' or chunk_pooling_type == 'wcon')
        self.use_concat = (chunk_pooling_type == 'con' or chunk_pooling_type == 'wcon') or chunk_pooling_type == 'concat' # tmp
        self.use_rnn2 = rnn_n_layers2 > 0 and rnn_n_units2 > 0
        chunk_embed_dim_merged = (
            chunk_embed_dim +
            (pretrained_chunk_embed_dim if pretrained_embed_usage == ModelUsage.CONCAT else 0))
        if self.use_concat:
            self.max_chunk_len = max_chunk_len
            self.chunk_concat_num = sum([i for i in range(self.max_chunk_len+1)])
            self.chunk_embed_out_dim = chunk_embed_dim_merged * self.chunk_concat_num
        else:
            self.chunk_embed_out_dim = chunk_embed_dim_merged
        print(self.use_concat, self.chunk_embed_out_dim)

        with self.init_scope():
            print('### Parameters', file=file)
            print('# Chunk loss ratio: {}'.format(self.chunk_loss_ratio), file=file)

            # embedding layers

            self.pretrained_embed_usage = pretrained_embed_usage

            self.embed_dropout = embed_dropout
            print('# Embedding dropout ratio={}'.format(self.embed_dropout), file=file)
            self.unigram_embed, self.pretrained_unigram_embed = models.util.construct_embeddings(
                n_vocab, unigram_embed_dim, pretrained_unigram_embed_dim, pretrained_embed_usage)
            if self.pretrained_embed_usage != ModelUsage.NONE:
                print('# Pretrained embedding usage: {}'.format(self.pretrained_embed_usage), file=file)
            print('# Unigram embedding matrix: W={}'.format(self.unigram_embed.W.shape), file=file)
            embed_dim = self.unigram_embed.W.shape[1]
            if self.pretrained_unigram_embed is not None:
                if self.pretrained_embed_usage == ModelUsage.CONCAT:
                    embed_dim += self.pretrained_unigram_embed.W.shape[1]
                print('# Pretrained unigram embedding matrix: W={}'.format(
                    self.pretrained_unigram_embed.W.shape), file=file)

            if n_bigrams > 0 and bigram_embed_dim > 0:
                self.bigram_embed, self.pretrained_bigram_embed = models.util.construct_embeddings(
                    n_bigrams, bigram_embed_dim, pretrained_bigram_embed_dim, pretrained_embed_usage)
                if self.pretrained_embed_usage != ModelUsage.NONE:
                    print('# Pretrained embedding usage: {}'.format(self.pretrained_embed_usage), file=file)
                print('# Bigram embedding matrix: W={}'.format(self.bigram_embed.W.shape), file=file)
                embed_dim += self.bigram_embed.W.shape[1]
                if self.pretrained_bigram_embed is not None:
                    if self.pretrained_embed_usage == ModelUsage.CONCAT:
                        embed_dim += self.pretrained_bigram_embed.W.shape[1]
                    print('# Pretrained bigram embedding matrix: W={}'.format(
                        self.pretrained_bigram_embed.W.shape), file=file)

            if n_tokentypes > 0 and tokentype_embed_dim > 0:
                self.type_embed = L.EmbedID(n_tokentypes, tokentype_embed_dim)
                embed_dim += tokentype_embed_dim
                print('# Token type embedding matrix: W={}'.format(self.type_embed.W.shape), file=file)

            if n_subtokens > 0 and subtoken_embed_dim > 0:
                self.subtoken_embed = L.EmbedID(n_subtokens, subtoken_embed_dim)
                embed_dim += subtoken_embed_dim
                print('# Subtoken embedding matrix: W={}'.format(self.subtoken_embed.W.shape), file=file)
            self.subtoken_embed_dim = subtoken_embed_dim;

            self.additional_feat_dim = feat_dim
            if feat_dim > 0:
                embed_dim += feat_dim
                print('# Additional features dimension: {}'.format(feat_dim), file=file)

            self.chunk_embed, self.pretrained_chunk_embed = models.util.construct_embeddings(
                n_chunks, chunk_embed_dim, pretrained_chunk_embed_dim, pretrained_embed_usage)
            print('# Chunk embedding matrix: W={}'.format(self.chunk_embed.W.shape), file=file)
            if self.pretrained_chunk_embed is not None:
                print('# Pretrained chunk embedding matrix: W={}'.format(
                    self.pretrained_chunk_embed.W.shape), file=file)

            self.rnn_unit_type = rnn_unit_type
            self.rnn = models.util.construct_RNN(
                rnn_unit_type, rnn_bidirection, rnn_n_layers1, embed_dim, rnn_n_units1, rnn_dropout)
            rnn_output_dim1 = rnn_n_units1 * (2 if rnn_bidirection else 1)

            # biaffine b/w token and chunk
            if self.use_attention:
                use_U = 'u' in biaffine_type or 'U' in biaffine_type
                use_V = 'v' in biaffine_type or 'V' in biaffine_type
                use_b = 'b' in biaffine_type or 'B' in biaffine_type

                biaffine_left_dim = rnn_output_dim1
                self.biaffine = BiaffineCombination(biaffine_left_dim, chunk_embed_dim_merged,
                                                    use_U=use_U, use_V=use_V, use_b=use_b)
                self.biaffine_dropout = biaffine_dropout
                print('# Biaffine layer for attention:   W={}, U={}, V={}, b={}, dropout={}'.format(
                    self.biaffine.W.shape, 
                    self.biaffine.U.shape if self.biaffine.U is not None else None, 
                    self.biaffine.V.shape if self.biaffine.V is not None else None, 
                    self.biaffine.b.shape if self.biaffine.b is not None else None, 
                    self.biaffine_dropout), file=file)

            # chunk vector dropout

            self.chunk_vector_dropout = chunk_vector_dropout
            print('# Chunk vector dropout={}'.format(self.chunk_vector_dropout), file=file)

            # recurrent layers2
         
            if self.use_rnn2:
                rnn_input_dim2 = rnn_output_dim1 + self.chunk_embed_out_dim
                self.rnn2 = models.util.construct_RNN(
                    rnn_unit_type, rnn_bidirection, rnn_n_layers2, rnn_input_dim2, rnn_n_units2, rnn_dropout)
                rnn_output_dim2 = rnn_n_units2 * (2 if rnn_bidirection else 1)
                mlp_input_dim = rnn_output_dim2
            else:
                mlp_input_dim = rnn_output_dim1 + self.chunk_embed_out_dim

            # MLP

            print('# MLP', file=file)
            self.mlp = MLP(mlp_input_dim, n_labels, n_hidden_units=mlp_n_units, n_layers=mlp_n_layers,
                           output_activation=F.identity, dropout=mlp_dropout, file=file)

            # CRF or softmax

            self.use_crf = use_crf
            if self.use_crf:
                self.crf = L.CRF1d(n_labels)
                print('# CRF cost: {}'.format(self.crf.cost.shape), file=file)
            else:
                self.softmax_cross_entropy = softmax_cross_entropy.softmax_cross_entropy


    """
    ts: mini-batch of token (char) sequences 
    cs: mini-batch of chunk (word) sequences
    ms: mini-batch of masking matrix
    fs: mini-batch of additional features
    ls: mini-batch of label sequences  
    """
    def __call__(
            self, us, cs, ds, ms, bs=None, ts=None, ss=None, fs=None, gls=None, gcs=None, calculate_loss=True):
        closs = None
        pcs = None

        xs = self.extract_token_features(us, bs, ts, ss, fs)          # token unigram etc. -[Embed]-> x
        rs = self.rnn_output(xs)                                      # x -[RNN]-> r

        if cs is not None:
            ws = self.extract_chunk_features(cs) # chunk -[Embed]-> w (chunk sequence)
        else:
            ws = [None] * len(us)

        if ds is not None:
            vs = self.extract_chunk_features(ds) # chunk -[Embed]-> w (concatenated chunk matrix)
        else:
            vs = [None] * len(us)

        # r @ r$w -> h
        closs, pcs, hs = self.act_and_merge_features(rs, ws, vs, ms, gcs, calculate_loss=calculate_loss)

        if self.use_rnn2:
            hs = self.rnn2_output(hs)                # h -[RNN]-> h'
        sloss, pls = self.predict(hs, ls=gls, calculate_loss=calculate_loss)

        if closs is not None:
            loss = (1 - self.chunk_loss_ratio) * sloss + self.chunk_loss_ratio * closs
        else:
            loss = sloss

        return loss, pls, pcs


    def decode(
            self, us, cs, ds, ms, bs=None, ts=None, ss=None, fs=None):
        with chainer.no_backprop_mode():
            _, ps, _ = self.__call__(us, cs, ds, ms, calculate_loss=False)
        return ps


    def do_analysis(
            self, us, cs, ds, ms, bs=None, ts=None, ss=None, fs=None, gls=None, gcs=None):
        closs = None
        pcs = None

        xs = self.extract_token_features(us, bs, ts, ss, fs)          # token unigram etc. -[Embed]-> x
        rs = self.rnn_output(xs)                                      # x -[RNN]-> r

        if cs is not None:
            ws = self.extract_chunk_features(cs) # chunk -[Embed]-> w (chunk sequence)
        else:
            ws = [None] * len(us)

        if ds is not None:
            vs = self.extract_chunk_features(ds) # chunk -[Embed]-> w (concatenated chunk matrix)
        else:
            vs = [None] * len(us)

        _, pcs, hs = self.act_and_merge_features(rs, ws, vs, ms, gcs) # r @ r$w -> h

        if self.use_rnn2:
            hs = self.rnn2_output(hs)                # h -[RNN]-> h'
        _, pls = self.predict(hs, ls=gls, calculate_loss=False)

        # print('pl', pls[0])

        # return gcs, pcs, ncands, gls, pls
        return gcs, pcs, gls, pls


    def rnn2_output(self, xs):
        if self.rnn_unit_type == 'lstm':
            hy, cy, hs = self.rnn2(None, None, xs)
        else:
            hy, hs = self.rnn2(None, xs)
        return hs


    # tmp
    def set_dics(self, id2token, id2chunk, id2label):
        self.id2token = id2token
        self.id2chunk = id2chunk
        self.id2label = id2label


    def extract_token_features(self, us, bs, ts, ss, fs):
        return super().extract_features(us, bs, ts, ss, fs)


    def extract_chunk_features(self, cs):
        xs = []
        i = 0 # tmp
        for c in cs:
            xe = self.chunk_embed(c) if c.any() else None
            if i == 0:
                # print('word', c, xe.shape, xe)
                i = 1

            if c is not None and self.pretrained_chunk_embed is not None:
                if self.pretrained_embed_usage == ModelUsage.ADD:
                    pce = self.pretrained_chunk_embed(c)
                    xe = xe + pce
                elif self.pretrained_embed_usage == ModelUsage.CONCAT:
                    pce = self.pretrained_chunk_embed(c)
                    xe = F.concat((xe, pce), 1)

            xs.append(xe)
        return xs


    def act_and_merge_features(self, xs, ws, vs, ms, gcs=None, calculate_loss=True):
        hs = []
        pcs = []
        # ncands = []

        xp = cuda.get_array_module(xs[0])
        closs = chainer.Variable(xp.array(0, dtype='f'))

        if gcs is None:
            gcs = [None] * len(xs)
        for x, w, v, gc, mask in zip(xs, ws, vs, gcs, ms):
            # ave   w, m1
            # wave  w, m1, m2
            # con   v, m0
            # wcon  w, v, m0, m1, m2

            if w is None and v is None: # no words were found for devel/test data
                a = xp.zeros((len(x), self.chunk_embed_out_dim), dtype='f')
                pc = np.zeros(len(x), 'i')
                pcs.append(pc)
                continue

            elif w is not None:
                w = F.dropout(w, self.embed_dropout)

            # print('x', x.shape)
            if self.use_attention or not self.use_concat:
                if self.use_attention: # wave or wcon
                    # print('w', w.shape)

                    w_scores = self.biaffine(
                        F.dropout(x, self.biaffine_dropout),
                        F.dropout(w, self.biaffine_dropout)) # (n, m)
                    w_scores = w_scores + mask[1] # a masked element becomes 0 after softmax operation
                    w_weight = F.softmax(w_scores)
                    w_weight = w_weight * mask[2] # raw of char w/o no candidate words become a 0 vector
                    # print('ww', w_weight.shape, w_weight)
                    _mask = mask[1]

                elif not self.use_concat: # ave
                    w_weight = self.normalize(mask[1], xp=xp)
                    
                if self.chunk_vector_dropout > 0:
                    mask1_shape = mask[1].shape
                    mask_drop = xp.ones(mask1_shape, dtype='f')
                    for i in range(mask1_shape[0]):
                        if self.chunk_vector_dropout > np.random.rand():
                            mask_drop[i] = xp.zeros(mask1_shape[1], dtype='f')
                    w_weight = w_weight * mask_drop

                if self.use_concat:
                    n = x.shape[0]
                    wd = w.shape[1]

                    indexes = [char_index2feat_indexes(i, n, self.max_chunk_len) for i in range(n)]
                    mask3 = xp.array([[0 if val >= 0 else FLOAT_MIN for val in iraw] for iraw in indexes], 'f')
                    v_weight0 = F.concat([F.expand_dims( # (n, m) -> (n, k)
                        F.get_item(w_weight[i], indexes[i]), axis=0) for i in range(n)], axis=0)
                    v_scores = v_weight0 + mask3
                    # print('vw', v_weight0.shape, v_weight0)
                    # print(mask3.shape, mask3)
                    # print(v_scores.shape, v_scores)

                    v_weight = F.transpose(v_weight0)
                    v_weight = F.expand_dims(v_weight, 2)
                    v_weight = F.broadcast_to(v_weight, (self.chunk_concat_num, n, wd))

                else:                         # ave or wave
                    a = F.matmul(w_weight, w) # (n, m) * (m, dc)  => (n, dc)
                    
            else:
                v_weight = None

            if self.use_concat: # con or wcon
                # print('v', v.shape)
                # print('m0', mask[0].shape)
                if v_weight is None:
                    v_weight = mask[0]
                else:
                    v_weight *= mask[0]

                v_weight = F.concat(v_weight, axis=1)
                # print('vw', v_weight.shape, v_weight)

                if self.chunk_vector_dropout > 0:
                    mask_drop = xp.ones(v_weight.shape, dtype='f')
                    for i in range(v_weight.shape[0]):
                        if self.chunk_vector_dropout > np.random.rand():
                            mask_drop[i] = xp.zeros(v_weight.shape[1], dtype='f')
                    v_weight *= mask_drop

                v = F.concat(v, axis=1)
                a = v * v_weight
                # print('a', a.shape, a)

            if self.use_attention: # wave or wcon
                weight = v_scores if self.use_concat else w_weight
                pc = minmax.argmax(weight, axis=1).data
                if xp is cuda.cupy:
                    pc = cuda.to_cpu(pc)
                pcs.append(pc)

                # ncand = [sum([1 if val >= 0 else 0 for val in raw]) for raw in _mask]
                # print('pred', pc)
                # print('gold', gc)
                # print('ncand', ncand)
                # print('weight', weight.shape, weight.data)
                # print('weight')
                # for i, e in enumerate(weight.data):
                #     print(i, e)

                if self.chunk_loss_ratio > 0 and calculate_loss:
                    scores = v_scores if self.use_concat else w_scores
                    closs += softmax_cross_entropy.softmax_cross_entropy(scores, gc)

            h = F.concat((x, a), axis=1) # (n, dt) @ (n, dc) => (n, dt+dc)
            hs.append(h)

        if closs.data == 0:
            closs = None
        else:
            closs /= len(xs)

        return closs, pcs, hs


    def act_and_merge_features_old(self, xs, ws, ms, gcs=None):
        hs = []
        pcs = []

        xp = cuda.get_array_module(xs[0])
        closs = chainer.Variable(xp.array(0, dtype='f'))

        for x, w, gc, mask in zip(xs, ws, gcs, ms):
            if w is None:
                a = xp.zeros((len(x), self.chunk_embed_out_dim), dtype='f')
                pc = np.zeros(len(x), 'i')
                pcs.append(pc)

            else:
                w = F.dropout(w, self.embed_dropout)

                if self.use_concat:
                    # m = mask[0]
                    m = F.concat(mask[0], axis=1)
                    if self.chunk_vector_dropout > 0:
                        mask_drop = xp.ones(m.shape, dtype='f')
                        for i in range(m.shape[0]):
                            if self.chunk_vector_dropout > np.random.rand():
                                mask_drop[i] = xp.zeros(m.shape[1], dtype='f')
                        m = m * mask_drop

                    w = F.concat(w, axis=1)
                    a = w * m
            
                else:
                    if self.use_attention:
                        scores = self.biaffine(
                            F.dropout(x, self.biaffine_dropout),
                            F.dropout(w, self.biaffine_dropout)) # (n, m)
                        scores = scores + mask[1] # a masked element becomes 0 after softmax operation
                        weight = F.softmax(scores)
                        weight = weight * mask[2] # a raw for character w/o no candidate words become 0 vector

                    else:
                        weight = self.normalize(mask[1], xp=xp)
                    
                    if self.chunk_vector_dropout > 0:
                        mask_drop = xp.ones(mask[1].shape, dtype='f')
                        for i in range(mask[1].shape[0]):
                            if self.chunk_vector_dropout > np.random.rand():
                                mask_drop[i] = xp.zeros(mask[1].shape[1], dtype='f')
                        weight = weight * mask_drop

                    a = F.matmul(weight, w)    # (n, m) * (m, dc)  => (n, dc)

                    if self.use_attention:
                        pc = minmax.argmax(weight, axis=1).data
                        if xp is cuda.cupy:
                            pc = cuda.to_cpu(pc)
                        pcs.append(pc)

                        if self.chunk_loss_ratio > 0:
                            closs += softmax_cross_entropy.softmax_cross_entropy(scores, gc)

            h = F.concat((x, a), axis=1) # (n, dt) @ (n, dc) => (n, dt+dc)
            hs.append(h)

        if closs.data == 0:
            closs = None
        else:
            closs /= len(xs)

        return closs, pcs, hs


    def normalize(self, array, xp=np):
        denom = F.sum(array, axis=1, keepdims=True)
        adjust = xp.asarray(
            [[np.float32(1) if denom.data[i][0] == 0 else np.float32(0)] for i in range(denom.shape[0])])
        denom = denom + adjust      # avoid zero division
        denom = F.broadcast_to(denom, array.shape)
        return array / denom


def construct_RNNTagger(
        n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
        n_subtokens, subtoken_embed_dim, n_chunks, chunk_embed_dim,
        rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, rnn_n_layers2, rnn_n_units2, 
        mlp_n_layers, mlp_n_units, n_labels, use_crf=True,
        feat_dim=0, mlp_n_additional_units=0,
        embed_dropout=0, rnn_dropout=0, biaffine_dropout=0, mlp_dropout=0, chunk_vector_dropout=0, 
        pretrained_unigram_embed_dim=0, pretrained_bigram_embed_dim=0, pretrained_chunk_embed_dim=0, 
        pretrained_embed_usage=ModelUsage.NONE, chunk_pooling_type='wave', 
        max_chunk_len=0, chunk_loss_ratio=0, biaffine_type='',
        file=sys.stderr):

    tagger = None
    if n_chunks > 0 and chunk_embed_dim > 0:
        tagger = models.tagger.RNNTaggerWithChunk(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
            n_subtokens, subtoken_embed_dim, n_chunks, chunk_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, rnn_n_layers2, rnn_n_units2, 
            mlp_n_layers, mlp_n_units, n_labels, use_crf=use_crf, feat_dim=feat_dim, 
            embed_dropout=embed_dropout, rnn_dropout=rnn_dropout, biaffine_dropout=biaffine_dropout, 
            mlp_dropout=mlp_dropout, chunk_vector_dropout=chunk_vector_dropout, 
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim,
            pretrained_bigram_embed_dim=pretrained_bigram_embed_dim, 
            pretrained_chunk_embed_dim=pretrained_chunk_embed_dim, 
            pretrained_embed_usage=pretrained_embed_usage, 
            chunk_pooling_type=chunk_pooling_type, max_chunk_len=max_chunk_len, 
            chunk_loss_ratio=chunk_loss_ratio, biaffine_type=biaffine_type,
            file=file)

    else:
        if constants.__version__ == 'v0.0.3':
            tagger = RNNTagger(
                n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
                n_subtokens, subtoken_embed_dim,
                rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
                mlp_n_layers, mlp_n_units, n_labels, use_crf=use_crf,
                feat_dim=feat_dim, 
                mlp_n_additional_units=mlp_n_additional_units,
                embed_dropout=embed_dropout, rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
                pretrained_unigram_embed_dim=pretrained_unigram_embed_dim, 
                pretrained_bigram_embed_dim=pretrained_bigram_embed_dim, 
                pretrained_embed_usage=pretrained_embed_usage, file=file)

        elif constants.__version__ == 'v0.0.2':
            if use_crf:
                tagger = models.tagger_v0_0_2.RNNCRFTagger(
                    n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
                    n_subtokens, subtoken_embed_dim,
                    rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
                    mlp_n_layers, mlp_n_units, n_labels, feat_dim=feat_dim, 
                    mlp_n_additional_units=mlp_n_additional_units,
                    rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
                    pretrained_unigram_embed_dim=pretrained_unigram_embed_dim, 
                    pretrained_embed_usage=pretrained_embed_usage, file=file)
            else:
                tagger = models.tagger_v0_0_2.RNNTagger(
                    n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
                    n_subtokens, subtoken_embed_dim,
                    rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
                    mlp_n_layers, mlp_n_units, n_labels, feat_dim=feat_dim, 
                    mlp_n_additional_units=mlp_n_additional_units,
                    rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
                    pretrained_unigram_embed_dim=pretrained_unigram_embed_dim,
                    pretrained_embed_usage=pretrained_embed_usage, file=file)

        else:
            print('Invalid seikanlp version: {}'.format(constants.__version__))
            sys.exit()

    return tagger


def char_index2feat_indexes(ti, sen_len, max_chunk_len):
    indexes = []

    for l in range(max_chunk_len):
        begin = sum([sen_len - k for k in range(l)])
        for k in range(l, -1, -1):
            offset = ti - k
            if offset < 0 or sen_len - l - 1 < offset:
                index = -1
            else:
                index = begin + offset
            indexes.append(index)
            # print(' i={} l={} k={} offset={} res={}'.format(ti, l, k, offset, index))

    return indexes
