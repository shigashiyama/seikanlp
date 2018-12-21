import sys

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.math import minmax

import constants
import models.util
from models.util import ModelUsage
from models.common import MLP
from models.parser import BiaffineCombination
from models.tagger import RNNTagger

FLOAT_MIN = -100000.0


class RNNTaggerWithChunk0413(RNNTagger):
    def __init__(
            self, n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_tokentypes, tokentype_embed_dim, n_subtokens, subtoken_embed_dim,
            n_chunks, chunk_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers1, rnn_n_units1, rnn_n_layers2, rnn_n_units2,
            mlp_n_layers, mlp_n_units, n_labels, use_crf=True,
            feat_dim=0, embed_dropout=0, rnn_dropout=0, biaffine_dropout=0, mlp_dropout=0, chunk_vector_dropout=0, 
            pretrained_unigram_embed_dim=0, pretrained_bigram_embed_dim=0, pretrained_chunk_embed_dim=0, 
            pretrained_embed_usage=ModelUsage.NONE, chunk_pooling_type=constants.AVG, 
            # min_chunk_len=1,
            max_chunk_len=0, chunk_loss_ratio=0, biaffine_type='',
            file=sys.stderr):
        self.__init__chain__()

        # TODO refactor with super class
        self.chunk_loss_ratio = chunk_loss_ratio
        # chunk_pooling_type = chunk_pooling_type.lower()
        self.chunk_pooling_type = chunk_pooling_type
        self.use_attention = (chunk_pooling_type == constants.WAVG or chunk_pooling_type == constants.WCON)
        self.use_concat = (chunk_pooling_type == constants.CON or chunk_pooling_type == constants.WCON)
        self.use_average = not self.use_concat
        self.use_rnn2 = rnn_n_layers2 > 0 and rnn_n_units2 > 0
        self.chunk_embed_dim_merged = (
            chunk_embed_dim +
            (pretrained_chunk_embed_dim if pretrained_embed_usage == ModelUsage.CONCAT else 0))
        if self.use_concat:
            self.max_chunk_len = max_chunk_len
            self.chunk_concat_num = sum([i for i in range(self.max_chunk_len+1)])
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

        if cs is not None:
            ws = self.extract_chunk_features(cs) # chunk -[Embed]-> w (chunk sequence)
        else:
            ws = [None] * len(us)

        if ds is not None:
            vs = self.extract_chunk_features(ds) # chunk -[Embed]-> w (concatenated chunk matrix)
        else:
            vs = [None] * len(us)

        closs, pcs, hs = self.act_and_merge_features(rs, ws, vs, ms, gcs) # r @ r$w -> h

        if self.use_rnn2:
            hs = self.rnn2_output(hs)                # h -[RNN]-> h'
        ys = self.mlp(hs)
        sloss, pls = self.predict(ys, ls=gls, calculate_loss=calculate_loss)

        if closs is not None:
            loss = (1 - self.chunk_loss_ratio) * sloss + self.chunk_loss_ratio * closs
        else:
            loss = sloss

        return loss, pls, pcs


    def decode(self, us, cs, ms, bs=None, ts=None, ss=None, fs=None):
        with chainer.no_backprop_mode():
            _, ps = self.__call__(us, cs, ms, bs, ts, ss, fs, calculate_loss=False)
        return ps


    def rnn2_output(self, xs):
        if self.rnn_unit_type == 'lstm':
            hy, cy, hs = self.rnn2(None, None, xs)
        else:
            hy, hs = self.rnn2(None, xs)
        return hs


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


    def act_and_merge_features(self, xs, ws, vs, ms, gcs=None):
        hs = []
        pcs = []

        xp = cuda.get_array_module(xs[0])
        closs = chainer.Variable(xp.array(0, dtype='f'))

        #RM
        # if gcs is None: # used when chunk_loss_ratio > 0
        #     gcs = [None] * len(xs)
        for x, w, v, gc, mask in zip(xs, ws, vs, gcs, ms):
            # avg   w, m1
            # wavg  w, m1, m2
            # con   v, m0
            # wcon  w, v, m0, m1, m2

            # print('x', x.shape)
            if w is None and v is None: # no words were found for devel/test data
                a = xp.zeros((len(x), self.chunk_embed_out_dim), dtype='f')
                pc = np.zeros(len(x), 'i')
                pcs.append(pc)
                #RM 
                # h = F.concat((x, a), axis=1) # (n, dt) @ (n, dc) => (n, dt+dc)
                # hs.append(h)
                continue

            elif w is not None:
                w = F.dropout(w, self.embed_dropout)

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

                if self.chunk_loss_ratio > 0:
                    scores = v_scores if self.use_concat else w_scores
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
