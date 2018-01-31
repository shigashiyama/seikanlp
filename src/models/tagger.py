import sys

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.functions.loss import softmax_cross_entropy
from chainer.links.connection.n_step_rnn import argsort_list_descent, permutate_list

import models.util
from models.util import ModelUsage
from models.common import MLP
from models.parser import BiaffineCombination


# Base model that consists of embedding layer, recurrent network (RNN) layers and affine layer.
class RNNTaggerBase(chainer.Chain):
    def __init__(
            self, n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_tokentypes, tokentype_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=0, mlp_n_additional_units=0,
            rnn_dropout=0, mlp_dropout=0, 
            pretrained_unigram_embed_dim=0, pretrained_embed_usage=ModelUsage.NONE,
            file=sys.stderr):
        super().__init__()

        with self.init_scope():
            print('### Parameters', file=file)

            # embedding layer(s)

            self.pretrained_embed_usage = pretrained_embed_usage

            self.unigram_embed, self.pretrained_unigram_embed = models.util.construct_embeddings(
                n_vocab, unigram_embed_dim, pretrained_unigram_embed_dim, pretrained_embed_usage)
            if self.pretrained_embed_usage != ModelUsage.NONE:
                print('# Pretrained embedding usage: {}'.format(self.pretrained_embed_usage))
            print('# Unigram embedding matrix: W={}'.format(self.unigram_embed.W.shape), file=file)
            embed_dim = self.unigram_embed.W.shape[1]
            if self.pretrained_unigram_embed is not None:
                if self.pretrained_embed_usage == ModelUsage.CONCAT:
                    embed_dim += self.pretrained_unigram_embed.W.shape[1]
                print('# Pretrained unigram embedding matrix: W={}'.format(
                    self.pretrained_unigram_embed.W.shape), file=file)

            if n_bigrams > 0 and bigram_embed_dim > 0:
                self.bigram_embed = L.EmbedID(n_bigrams, bigram_embed_dim)
                embed_dim += bigram_embed_dim
                print('# Bigram embedding matrix: W={}'.format(self.bigram_embed.W.shape), file=file)
                    
            if n_tokentypes > 0 and tokentype_embed_dim > 0:
                self.type_embed = L.EmbedID(n_tokentypes, tokentype_embed_dim)
                embed_dim += tokentype_embed_dim
                print('# Token type embedding matrix: W={}'.format(self.type_embed.W.shape), file=file)

            self.additional_feat_dim = feat_dim
            if feat_dim > 0:
                embed_dim += feat_dim
                print('# Additional features dimension: {}'.format(feat_dim), file=file)

            #embed_dim += mlp_n_additional_units

            # subtoken embedding layer

            if n_subtokens > 0 and subtoken_embed_dim > 0:
                self.subtoken_embed = L.EmbedID(n_subtokens, subtoken_embed_dim)
                print('# Subtoken embedding matrix: W={}'.format(self.subtoken_embed.W.shape), file=file)
            self.subtoken_embed_dim = subtoken_embed_dim;

            # recurrent layers

            self.rnn_unit_type = rnn_unit_type
            self.rnn = models.util.construct_RNN(
                rnn_unit_type, rnn_bidirection, rnn_n_layers, embed_dim, rnn_n_units, rnn_dropout)
            rnn_output_dim = rnn_n_units * (2 if rnn_bidirection else 1)

            # MLP

            print('# MLP', file=file)
            mlp_in = rnn_output_dim + mlp_n_additional_units
            #mlp_in = rnn_output_dim
            self.mlp = MLP(mlp_in, n_labels, n_hidden_units=mlp_n_units, n_layers=mlp_n_layers,
                           output_activation=F.identity, dropout=mlp_dropout, file=file)


    # TODO fixed bug: subtoken embed is not used
    # unigram, bigram, type, subtoken, feature, label
    def __call__(self, us, bs=None, ts=None, ss=None, fs=None, ls=None, calculate_loss=True):
        # print((len(us), us[0].shape))
        # print((len(bs), bs[0].shape) if bs is not None else None)
        # print((len(ts), ts[0].shape) if ts is not None else None)
        # print((len(ss), ss[0].shape) if ss is not None else None)
        # print((len(fs), fs[0].shape) if fs is not None else None)
        # print((len(ls), ls[0]) if ls is not None else None)
        xs = self.get_features(us, bs, ts, ss, fs)
        # print(len(xs), xs[0].shape)
        rs = self.rnn_output(xs)
        loss, ps = self.predict(rs, ls=ls, calculate_loss=calculate_loss)

        return loss, ps
        

    def get_features(self, us, bs, ts, ss, fs):
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
            xe = self.unigram_embed(u)

            if self.pretrained_unigram_embed is not None:
                if self.pretrained_embed_usage == ModelUsage.ADD:
                    pe = self.pretrained_unigram_embed(u)
                    xe = xe + pe
                elif self.pretrained_embed_usage == ModelUsage.CONCAT:
                    pe = self.pretrained_unigram_embed(u)
                    xe = F.concat((xe, pe), 1)

            if b is not None:
                be = self.bigram_embed(b)
                xe = F.concat((xe, be), 1)

            if t is not None:
                te = self.type_embed(t)
                xe = F.concat((xe, te), 1)

            if s is not None:
                se = self.subtoken_embed(s)
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
        # to be implemented in sub-class
        pass


    def decode(self, us, fs):
        with chainer.no_backprop_mode():
            _, ps = self.__call__(us, fs, calculate_loss=False)

        return ps


class RNNTagger(RNNTaggerBase):
    def __init__(
            self, n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_tokentypes, tokentype_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=0, mlp_n_additional_units=0,
            rnn_dropout=0, mlp_dropout=0, 
            pretrained_unigram_embed_dim=0, pretrained_embed_usage=ModelUsage.NONE,
            file=sys.stderr):
        super().__init__(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_tokentypes, tokentype_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=feat_dim, 
            mlp_n_additional_units=mlp_n_additional_units,
            rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim, 
            pretrained_embed_usage=pretrained_embed_usage,
            file=file)

        self.softmax_cross_entropy = softmax_cross_entropy.softmax_cross_entropy


    def predict(self, rs, ls=None, calculate_loss=True):
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


class RNNCRFTagger(RNNTaggerBase):
    def __init__(
            self, n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_tokentypes, tokentype_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=0, mlp_n_additional_units=0,
            rnn_dropout=0, mlp_dropout=0, 
            pretrained_unigram_embed_dim=0, pretrained_embed_usage=ModelUsage.NONE,
            file=sys.stderr):
        super().__init__(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_tokentypes, tokentype_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=feat_dim, 
            mlp_n_additional_units=mlp_n_additional_units,
            rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim, 
            pretrained_embed_usage=pretrained_embed_usage,
            file=file)

        with self.init_scope():
            self.crf = L.CRF1d(n_labels)

            print('# CRF cost: {}'.format(self.crf.cost.shape), file=file)


    def predict(self, rs, ls=None, calculate_loss=True):
        hs = self.mlp(rs)

        indices = argsort_list_descent(hs)
        trans_hs = F.transpose_sequence(permutate_list(hs, indices, inv=False))
        score, trans_ys = self.crf.argmax(trans_hs)
        ys = permutate_list(F.transpose_sequence(trans_ys), indices, inv=True)
        ps = [y.data for y in ys]

        xp = cuda.get_array_module(hs[0])
        if calculate_loss:
            trans_ls = F.transpose_sequence(permutate_list(ls, indices, inv=False))
            loss = self.crf(trans_hs, trans_ls)
        else:
            loss = chainer.Variable(xp.array(0, dtype='f'))

        return loss, ps


class RNNCRFTaggerWithAttention(RNNCRFTagger):
    def __init__(
            self, n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_tokentypes, tokentype_embed_dim, n_subtokens, subtoken_embed_dim,
            n_chunks, chunk_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=0, mlp_n_additional_units=0,
            rnn_dropout=0, mlp_dropout=0, 
            pretrained_unigram_embed_dim=0, pretrained_chunk_embed_dim=0, 
            pretrained_embed_usage=ModelUsage.NONE, use_attention=False,
            file=sys.stderr):
        super().__init__(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_tokentypes, tokentype_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=feat_dim, 
            mlp_n_additional_units=chunk_embed_dim+(
                pretrained_chunk_embed_dim if pretrained_embed_usage == ModelUsage.CONCAT else 0),  
            rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
            pretrained_embed_usage=pretrained_embed_usage,
            file=file)

        with self.init_scope():
            self.chunk_embed, self.pretrained_chunk_embed = models.util.construct_embeddings(
                n_chunks, chunk_embed_dim, pretrained_chunk_embed_dim, pretrained_embed_usage)
            self.cembed_out_dim = chunk_embed_dim
            print('# Chunk embedding matrix: W={}'.format(self.chunk_embed.W.shape), file=file)
            if self.pretrained_chunk_embed is not None:
                if pretrained_embed_usage == ModelUsage.CONCAT:
                    self.cembed_out_dim += self.pretrained_chunk_embed.W.shape[1]
                print('# Pretrained chunk embedding matrix: W={}'.format(
                    self.pretrained_chunk_embed.W.shape), file=file)


            self.use_attention = use_attention
            if self.use_attention:
                rnn_output_dim = rnn_n_units * (2 if rnn_bidirection else 1)
                self.biaffine = BiaffineCombination(rnn_output_dim, self.cembed_out_dim)
                print('# Biaffine layer for attention:   W={}, U={}, b={}, dropout={}\n'.format(
                    self.biaffine.W.shape, 
                    self.biaffine.U.shape if self.biaffine.U is not None else None, 
                    self.biaffine.b.shape if self.biaffine.b is not None else None, 
                    0.0), file=file)


    """
    ts: mini-batch of token (char) sequences 
    cs: mini-batch of chunk (word) sequences
    ms: mini-batch of masking matrix
    fs: mini-batch of additional features
    ls: mini-batch of label sequences
  
    """
    def __call__(self, us, cs, ms, bs=None, ts=None, ss=None, fs=None, ls=None, calculate_loss=True):
        xs = self.get_token_features(us, bs, ts, ss, fs)
        rs = self.rnn_output(xs)
        ws = self.get_chunk_features(cs)
        hs = self.get_hidden_vectors(rs, ws, ms)

        loss, ps = self.predict(hs, ls=ls, calculate_loss=calculate_loss)

        return loss, ps
        

    def get_token_features(self, us, bs, ts, ss, fs):
        return super().get_features(us, bs, ts, ss, fs)


    def get_chunk_features(self, cs):
        xs = []
        for c in cs:
            xe = self.chunk_embed(c) if c.any() else None

            if c is not None and self.pretrained_chunk_embed is not None:
                if self.pretrained_embed_usage == ModelUsage.ADD:
                    pce = self.pretrained_chunk_embed(c)
                    xe = xe + pce
                elif self.pretrained_embed_usage == ModelUsage.CONCAT:
                    pce = self.pretrained_chunk_embed(c)
                    xe = F.concat((xe, pce), 1)

            xs.append(xe)
        return xs


    def get_hidden_vectors(self, rs, ws, ms):
        hs = []
        xp = cuda.get_array_module(rs[0])

        for r, w, mask in zip(rs, ws, ms):
            if w is None:       # no words are found for devel/test data
                a = xp.zeros((len(r), self.cembed_out_dim), dtype='f')  

            else:
                # TODO dropout
                if self.use_attention:
                    scores = self.biaffine(r, w) # (n, m)
                    max_elems = F.broadcast_to(F.max(scores, axis=1, keepdims=True), scores.shape)
                    masked_scores = F.exp(scores - max_elems) * mask
                else:
                    masked_scores = mask
                weight = self.normalize(masked_scores, xp=xp)
                a = F.matmul(weight, w)      # (n, m) * (m, dc)  => (n, dc)

                # print('r:', r.shape, 'w:', w.shape)
                # print('r', r)
                # print('w', w)
                # print('W', self.biaffine.W)
                # print('s', scores)
                # print('m', mask)
                # print('sm', scores_m)
                # print('weight', weight)
                # print('a', a)
                # print('w', w)
                # print()

            h = F.concat((r, a), axis=1) # (n, dt) @ (n, dc) => (n, dt+dc)
            hs.append(h)

        return hs


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
        rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
        mlp_n_layers, mlp_n_units, n_labels, use_crf=True,
        feat_dim=0, mlp_n_additional_units=0,
        rnn_dropout=0, mlp_dropout=0, 
        pretrained_unigram_embed_dim=0, pretrained_chunk_embed_dim=0, 
        pretrained_embed_usage=ModelUsage.NONE, use_attention=False,
        file=sys.stderr):

    tagger = None
    if n_chunks > 0 and chunk_embed_dim > 0:
        tagger = models.tagger.RNNCRFTaggerWithAttention(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
            n_subtokens, subtoken_embed_dim, n_chunks, chunk_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=feat_dim, 
            mlp_n_additional_units=mlp_n_additional_units,
            rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim, 
            pretrained_chunk_embed_dim=pretrained_chunk_embed_dim, 
            pretrained_embed_usage=pretrained_embed_usage,
            use_attention=use_attention,
            file=file)

    elif use_crf:
        tagger = models.tagger.RNNCRFTagger(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
            n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=feat_dim, 
            mlp_n_additional_units=mlp_n_additional_units,
            rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim, 
            pretrained_embed_usage=pretrained_embed_usage, file=file)
    else:
        tagger = models.tagger.RNNTagger(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
            n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=feat_dim, 
            mlp_n_additional_units=mlp_n_additional_units,
            rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim,
            pretrained_embed_usage=pretrained_embed_usage, file=file)

    return tagger
