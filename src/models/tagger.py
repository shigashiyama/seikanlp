import sys

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.links.connection.n_step_rnn import argsort_list_descent, permutate_list

from models.common import MLP
import models.util


# Base model that consists of embedding layer, recurrent network (RNN) layers and affine layer.
class RNNTaggerBase(chainer.Chain):
    def __init__(
            self, n_vocab, unigram_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=0, mlp_n_additional_units=0,
            rnn_dropout=0, mlp_dropout=0, pretrained_unigram_embed_dim=0, file=sys.stderr):
        super().__init__()

        with self.init_scope():
            print('### Parameters', file=file)

            # embedding layer(s)

            self.unigram_embed = L.EmbedID(n_vocab, unigram_embed_dim)
            print('# Unigram embedding matrix: W={}'.format(self.unigram_embed.W.shape), file=file)

            if pretrained_unigram_embed_dim > 0:
                self.pretrained_unigram_embed = L.EmbedID(n_vocab, pretrained_unigram_embed_dim)
                print('# Pretrained unigram embedding matrix: W={}'.format(
                    self.pretrained_unigram_embed.W.shape), file=file)
            else:
                self.pretrained_unigram_embed = None
            self.pretrained_unigram_embed_dim = pretrained_unigram_embed_dim

            self.additional_feat_dim = feat_dim
            if feat_dim > 0:
                print('# Additional features dimension: {}'.format(feat_dim), file=file)

            embed_dim = unigram_embed_dim + pretrained_unigram_embed_dim + feat_dim
            #embed_dim = unigram_embed_dim + pretrained_unigram_embed_dim + feat_dim + mlp_n_additional_units

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


    def __call__(self, ws, fs=None, ls=None, calculate_loss=True):
        xs = self.get_features(ws, fs)
        rs = self.rnn_output(xs)
        loss, ps = self.predict(rs, ls=ls, calculate_loss=calculate_loss)

        return loss, ps
        

    def get_features(self, ws, fs=None):
        xs = []

        if self.additional_feat_dim > 0:
            for w, f in zip(ws, fs):
                we = self.unigram_embed(w)
                if self.pretrained_unigram_embed_dim > 0:
                    twe = self.pretrained_unigram_embed(w)
                    xe = F.concat((we, twe, f), 1)
                else:
                    xe = F.concat((we, f), 1)
                xs.append(xe)

        else:
            for w in ws:
                we = self.unigram_embed(w)
                if self.pretrained_unigram_embed_dim > 0:
                    twe = self.pretrained_unigram_embed(w)
                    xe = F.concat((we, twe), 1)
                else:
                    xe = we
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


    def decode(self, ws, fs):
        with chainer.no_backprop_mode():
            _, ps = self.__call__(ws, fs, calculate_loss=False)

        return ps


class RNNTagger(RNNTaggerBase):
    def __init__(
            self, n_vocab, unigram_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=0, mlp_n_additional_units=0,
            rnn_dropout=0, mlp_dropout=0, pretrained_unigram_embed_dim=0, file=sys.stderr):
        super().__init__(
            n_vocab, unigram_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=feat_dim, 
            mlp_n_additional_units=mlp_n_additional_units,
            rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim, file=file)

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
            self, n_vocab, unigram_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=0, mlp_n_additional_units=0,
            rnn_dropout=0, mlp_dropout=0, pretrained_unigram_embed_dim=0, file=sys.stderr):
        super().__init__(
            n_vocab, unigram_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, feat_dim=feat_dim, 
            mlp_n_additional_units=mlp_n_additional_units,
            rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim, file=file)

        with self.init_scope():
            self.crf = L.CRF1d(n_labels)

            print('# CRF cost: {}\n'.format(self.crf.cost.shape), file=file)


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
