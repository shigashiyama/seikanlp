import sys

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
from models.util import ModelUsage
from models.common import MLP
from models.parser import BiaffineCombination

FLOAT_MIN = -100000.0


class RNNTagger(chainer.Chain):
    def __init__chain__(self):  # tmp
        super().__init__()


    def __init__(
            self, n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_tokentypes, tokentype_embed_dim, n_attrs, attr_embed_dim,
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

            if n_attrs > 0 and attr_embed_dim > 0: # tmp subtoken=attr
                self.subtoken_embed = L.EmbedID(n_attrs, attr_embed_dim)
                embed_dim += attr_embed_dim
                print('# Attribute embedding matrix: W={}'.format(self.subtoken_embed.W.shape), file=file)
            self.subtoken_embed_dim = attr_embed_dim

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


    # unigram, bigram, type, attr, feature, label
    def __call__(self, us, bs=None, ts=None, es=None, fs=None, ls=None, calculate_loss=True):
        xs = self.extract_features(us, bs, ts, es, fs)
        rs = self.rnn_output(xs)
        ys = self.mlp(rs)
        loss, ps = self.predict(ys, ls=ls, calculate_loss=calculate_loss)

        return loss, ps
        

    def extract_features(self, us, bs, ts, es, fs):
        xs = []
        if bs is None:
            bs = [None] * len(us)
        if ts is None:
            ts = [None] * len(us)
        if es is None:
            es = [None] * len(us)
        if fs is None:
            fs = [None] * len(us)

        for u, b, t, e, f in zip(us, bs, ts, es, fs):
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
                be = F.dropout(be, self.embed_dropout)
                xe = F.concat((xe, be), 1)

            if t is not None:
                te = self.type_embed(t)
                te = F.dropout(te, self.embed_dropout)
                xe = F.concat((xe, te), 1)

            if e is not None:
                ee = self.subtoken_embed(e)
                ee = F.dropout(ee, self.embed_dropout)
                xe = F.concat((xe, ee), 1)

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


    def predict_softmax(self, ys, ls=None, calculate_loss=True):
        xp = cuda.get_array_module(ys[0])

        ps = []
        loss = chainer.Variable(xp.array(0, dtype='f'))
        if not ls:
            ls = [None] * len(rs)
        for y, l in zip(ys, ls):
            if calculate_loss:
                loss += self.softmax_cross_entropy(y, l)
            ps.append([np.argmax(yi.data) for yi in y])
        
        return loss, ps


    def predict_crf(self, hs, ls=None, calculate_loss=True):
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


    def do_analysis(self, us, bs=None, ts=None, es=None, fs=None, gls=None, gcs=None):
        closs = None
        pcs = None

        xs = self.extract_features(us, bs, ts, es, fs)
        rs = self.rnn_output(xs)
        _, pls = self.predict(rs, ls=gls, calculate_loss=False)

        return gls, pls


class RNNTaggerWithChunk(RNNTagger):
    def __init__(
            self, n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_tokentypes, tokentype_embed_dim, n_subtokens, subtoken_embed_dim,
            n_chunks, chunk_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers1, rnn_n_units1, rnn_n_layers2, rnn_n_units2,
            mlp_n_layers, mlp_n_units, n_labels, use_crf=True,
            feat_dim=0, embed_dropout=0, rnn_dropout=0, biaffine_dropout=0, mlp_dropout=0, chunk_vector_dropout=0, 
            pretrained_unigram_embed_dim=0, pretrained_bigram_embed_dim=0, pretrained_chunk_embed_dim=0, 
            pretrained_embed_usage=ModelUsage.NONE, chunk_pooling_type=constants.AVG, 
            min_chunk_len=1, max_chunk_len=0, chunk_loss_ratio=0, biaffine_type='',
            file=sys.stderr):
        self.__init__chain__()

        self.rnn1_compression = False # tmp

        # TODO refactor with super class
        self.chunk_loss_ratio = chunk_loss_ratio
        self.chunk_pooling_type = chunk_pooling_type
        self.use_attention = (chunk_pooling_type == constants.WAVG or chunk_pooling_type == constants.WCON)
        self.use_concat = (chunk_pooling_type == constants.CON or chunk_pooling_type == constants.WCON)
        self.use_average = not self.use_concat
        self.use_rnn2 = rnn_n_layers2 > 0 and rnn_n_units2 > 0
        self.chunk_embed_dim_merged = (
            chunk_embed_dim +
            (pretrained_chunk_embed_dim if pretrained_embed_usage == ModelUsage.CONCAT else 0))
        if self.use_concat:
            self.chunk_concat_num = sum([i for i in range(min_chunk_len, max_chunk_len+1)])
            self.chunk_embed_out_dim = self.chunk_embed_dim_merged * self.chunk_concat_num
        else:
            self.chunk_embed_out_dim = self.chunk_embed_dim_merged

        with self.init_scope():
            print('### Parameters', file=file)
            print('# Chunk pooling type: {}'.format(self.chunk_pooling_type), file=file)
            print('# Chunk loss ratio: {}'.format(self.chunk_loss_ratio))

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

            # tmp
            if self.rnn1_compression:
                print('# RNN1 Comp', file=file)
                self.w_rnn1comp = MLP(rnn_output_dim1, self.chunk_embed_out_dim,
                                      output_activation=F.tanh, file=file)
                rnn_output_dim1 = self.chunk_embed_out_dim

            # biaffine b/w token and chunk
            if self.use_attention:
                use_U = 'u' in biaffine_type or 'U' in biaffine_type
                use_V = 'v' in biaffine_type or 'V' in biaffine_type
                use_b = 'b' in biaffine_type or 'B' in biaffine_type

                biaffine_left_dim = rnn_output_dim1 
                self.biaffine = BiaffineCombination(biaffine_left_dim, self.chunk_embed_dim_merged,
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

        if self.rnn1_compression: # tmp
            rs = self.w_rnn1comp(rs)

        if cs is not None:
            ws = self.extract_chunk_features(cs) # chunk -[Embed]-> w (chunk sequence)
        else:
            ws = [None] * len(us)

        if ds is not None:
            vs = self.extract_chunk_features(ds) # chunk -[Embed]-> w (concatenated chunk matrix)
        else:
            vs = [None] * len(us)

        closs, pcs, hs = self.act_and_merge_features(rs, ws, vs, ms, gcs) # r @ r$w -> h
        # for analysis
        # _, pcs, hs, ass = self.act_and_merge_features(rs, ws, vs, ms, gcs, get_att_score=True) # r @ r$w -> h

        if self.use_rnn2:
            hs = self.rnn2_output(hs)                # h -[RNN]-> h'
        ys = self.mlp(hs)
        sloss, pls = self.predict(ys, ls=gls, calculate_loss=calculate_loss)

        if closs is not None:
            loss = (1 - self.chunk_loss_ratio) * sloss + self.chunk_loss_ratio * closs
        else:
            loss = sloss

        return loss, pls, pcs


    def decode(self, us, cs, ds, ms, bs=None, ts=None, ss=None, fs=None):
        with chainer.no_backprop_mode():
            _, ps, _ = self.__call__(us, cs, ds, ms, calculate_loss=False)
        return ps


    def rnn2_output(self, xs):
        if self.rnn_unit_type == 'lstm':
            hy, cy, hs = self.rnn2(None, None, xs)
        else:
            hy, hs = self.rnn2(None, xs)
        return hs


    # def set_dics(self, id2token, id2chunk, id2label):
    #     self.id2token = id2token
    #     self.id2chunk = id2chunk
    #     self.id2label = id2label


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


    def act_and_merge_features(self, xs, ws, vs, ms, gcs=None, get_att_score=False):
        hs = []
        pcs = []
        ass = []                # attention scores
        # ncands = []

        xp = cuda.get_array_module(xs[0])
        closs = chainer.Variable(xp.array(0, dtype='f'))

        if gcs is None:
            gcs = [None] * len(xs)
        for x, w, v, gc, mask in zip(xs, ws, vs, gcs, ms):
            # print('x', x.shape)
            if w is None and v is None: # no words were found for devel/test data
                a = xp.zeros((len(x), self.chunk_embed_out_dim), dtype='f')
                pc = np.zeros(len(x), 'i')
                pcs.append(pc)
                h = F.concat((x, a), axis=1) # (n, dt) @ (n, dc) => (n, dt+dc)
                hs.append(h)
                continue

            if w is not None:
                w = F.dropout(w, self.embed_dropout)

            ## calculate weight for w

            mask_ij = mask[0]
            if self.use_attention: # wavg or wcon
                mask_i = mask[1]
                # print('w', w.shape)

                w_scores = self.biaffine(
                    F.dropout(x, self.biaffine_dropout),
                    F.dropout(w, self.biaffine_dropout)) # (n, m)
                w_scores = w_scores + mask_ij # a masked element becomes 0 after softmax operation
                w_weight = F.softmax(w_scores)
                w_weight = w_weight * mask_i # raw of char w/o no candidate words become a 0 vector

                # print('ww', w_weight.shape, '\n', w_weight)
                
            elif self.chunk_pooling_type == constants.AVG:
                w_weight = self.normalize(mask_ij, xp=xp)

            if not self.use_concat and self.chunk_vector_dropout > 0:
                mask_drop = xp.ones(w_weight.shape, dtype='f')
                for i in range(w_weight.shape[0]):
                    if self.chunk_vector_dropout > np.random.rand():
                        mask_drop[i] = xp.zeros(w_weight.shape[1], dtype='f')
                w_weight = w_weight * mask_drop

            ## calculate weight for v

            if self.use_concat:
                mask_ik = mask[2]
                n = x.shape[0]
                wd = self.chunk_embed_dim_merged #w.shape[1]
                if self.chunk_pooling_type == constants.WCON:
                    ikj_table = mask[3]
                    v_weight0 = F.concat([F.expand_dims( # (n, m) -> (n, k)
                        F.get_item(w_weight[i], ikj_table[i]), axis=0) for i in range(n)], axis=0)
                    # print('mask_ik', mask_ik.shape, '\n', mask_ik)
                    # print('v_weight0', v_weight0.shape, '\n', v_weight0)
                    v_weight0 *= mask_ik
                    # print('ikj_table', ikj_table)

                else:
                    v_weight0 = mask_ik

                v_weight = F.transpose(v_weight0)                                   # (n,k)
                v_weight = F.expand_dims(v_weight, 2)                               # (k,n)
                v_weight = F.broadcast_to(v_weight, (self.chunk_concat_num, n, wd)) # (k,n,wd)
                v_weight = F.concat(v_weight, axis=1)                               # (k,n*wd)

                if self.chunk_vector_dropout > 0:
                    mask_drop = xp.ones(v_weight.shape, dtype='f')
                    for i in range(v_weight.shape[0]):
                        if self.chunk_vector_dropout > np.random.rand():
                            mask_drop[i] = xp.zeros(v_weight.shape[1], dtype='f')
                    v_weight *= mask_drop

            ## calculate summary vector a
            if self.use_average:          # avg or wavg
                a = F.matmul(w_weight, w) # (n, m) * (m, dc)  => (n, dc)

            else: # con or wcon
                v = F.concat(v, axis=1)
                a = v * v_weight
                # print('a', a.shape, a)

            ## get predicted (attended) chunks
            if self.use_attention: # wavg or wcon
                if self.chunk_pooling_type == constants.WAVG:
                    weight = w_weight
                else:
                    weight = v_weight0
                pc = minmax.argmax(weight, axis=1).data
                if xp is cuda.cupy:
                    pc = cuda.to_cpu(pc)
                pcs.append(pc)

            #     if get_att_score:
            #         ascore = minmax.max(weight, axis=1).data
            #         ass.append(ascore)

            #     ncand = [sum([1 if val >= 0 else 0 for val in raw]) for raw in _mask]
            #     print('pred', pc)
            #     print('gold', gc)
            #     print('ncand', ncand)
            #     print('weight', weight.shape, weight.data)
            #     print('weight')
            #     for i, e in enumerate(weight.data):
            #         print(i, e)

            h = F.concat((x, a), axis=1) # (n, dt) @ (n, dc) => (n, dt+dc)

            hs.append(h)

        if closs.data == 0:
            closs = None
        else:
            closs /= len(xs)

        if get_att_score:
            return closs, pcs, hs, ass
        else:
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
        n_attrs, attr_embed_dim, n_chunks, chunk_embed_dim,
        rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, rnn_n_layers2, rnn_n_units2, 
        mlp_n_layers, mlp_n_units, n_labels, use_crf=True,
        feat_dim=0, mlp_n_additional_units=0,
        embed_dropout=0, rnn_dropout=0, biaffine_dropout=0, mlp_dropout=0, chunk_vector_dropout=0, 
        pretrained_unigram_embed_dim=0, pretrained_bigram_embed_dim=0, pretrained_chunk_embed_dim=0, 
        pretrained_embed_usage=ModelUsage.NONE, chunk_pooling_type=constants.AVG, 
        min_chunk_len=1, max_chunk_len=0, chunk_loss_ratio=0, biaffine_type='',
        file=sys.stderr):

    tagger = None
    if n_chunks > 0 and chunk_embed_dim > 0:
        tagger = models.tagger.RNNTaggerWithChunk(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
            n_attrs, attr_embed_dim, n_chunks, chunk_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, rnn_n_layers2, rnn_n_units2, 
            mlp_n_layers, mlp_n_units, n_labels, use_crf=use_crf, feat_dim=feat_dim, 
            embed_dropout=embed_dropout, rnn_dropout=rnn_dropout, biaffine_dropout=biaffine_dropout, 
            mlp_dropout=mlp_dropout, chunk_vector_dropout=chunk_vector_dropout, 
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim,
            pretrained_bigram_embed_dim=pretrained_bigram_embed_dim, 
            pretrained_chunk_embed_dim=pretrained_chunk_embed_dim, 
            pretrained_embed_usage=pretrained_embed_usage, 
            chunk_pooling_type=chunk_pooling_type, min_chunk_len=min_chunk_len, max_chunk_len=max_chunk_len, 
            chunk_loss_ratio=chunk_loss_ratio, biaffine_type=biaffine_type,
            file=file)

    else:
        tagger = RNNTagger(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
            n_attrs, attr_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, use_crf=use_crf,
            feat_dim=feat_dim, 
            mlp_n_additional_units=mlp_n_additional_units,
            embed_dropout=embed_dropout, rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim, 
            pretrained_bigram_embed_dim=pretrained_bigram_embed_dim, 
            pretrained_embed_usage=pretrained_embed_usage, file=file)

    return tagger
