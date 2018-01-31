import sys

import chainer
import chainer.functions as F
import chainer.links as L

import models.tagger
from models.common import MLP
from models.tagger import RNNTagger, RNNCRFTagger


class DualRNNTagger(chainer.Chain):
    def __init__(
            self, n_vocab, unigram_embed_dim, n_subtokens, subtoken_embed_dim,
            rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
            mlp_n_layers, mlp_n_units, n_labels, use_crf=True, feat_dim=0, 
            rnn_dropout=0, mlp_dropout=0, pretrained_unigram_embed_dim=0, short_model_usage='independent',
            file=sys.stderr):
        super().__init__()
        self.short_model_usage = short_model_usage
        if self.short_model_usage == 'concat':
            sm_mlp_n_additional_units = rnn_n_units * (2 if rnn_bidirection else 1)
        else:
            sm_mlp_n_additional_units = 0

        with self.init_scope():
            print('### Short unit model', file=file)
            self.su_tagger = models.tagger.construct_RNNTagger(
                n_vocab, unigram_embed_dim, n_subtokens, subtoken_embed_dim,
                rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
                mlp_n_layers, mlp_n_units, n_labels, use_crf=use_crf, feat_dim=feat_dim, 
                rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
                pretrained_unigram_embed_dim=pretrained_unigram_embed_dim, file=file)
            print('### Long unit model', file=file)
            self.lu_tagger = models.tagger.construct_RNNTagger(
                n_vocab, unigram_embed_dim, n_subtokens, subtoken_embed_dim,
                rnn_unit_type, rnn_bidirection, rnn_n_layers, rnn_n_units, 
                mlp_n_layers, mlp_n_units, n_labels, use_crf=use_crf, feat_dim=feat_dim, 
                mlp_n_additional_units=sm_mlp_n_additional_units,
                rnn_dropout=rnn_dropout, mlp_dropout=mlp_dropout, 
                pretrained_unigram_embed_dim=pretrained_unigram_embed_dim, file=file)


    def __call__(self, ws, fs=None, ls=None, calculate_loss=True):
        if self.short_model_usage == 'concat' or self.short_model_usage == 'add':
            with chainer.no_backprop_mode():
                sxs = self.su_tagger.get_features(ws, fs)
                srs = self.su_tagger.rnn_output(sxs)

        lxs = self.lu_tagger.get_features(ws, fs)
        lrs = self.lu_tagger.rnn_output(lxs)

        if self.short_model_usage == 'concat':
            rs = [F.concat((sr, lr), 1) for sr, lr in zip(srs, lrs)]
        elif self.short_model_usage == 'add':
            rs = [sr + lr for sr, lr in zip(srs, lrs)]
        else:
            rs = lrs

        loss, ps = self.lu_tagger.predict(rs, ls=ls, calculate_loss=calculate_loss)

        return loss, ps


    def decode(self, ws, fs=None):
        with chainer.no_backprop_mode():
            sxs = self.su_tagger.get_features(ws, None)
            srs = self.su_tagger.rnn_output(sxs)
            _, sps = self.su_tagger.predict(srs, calculate_loss=False)

            lxs = self.lu_tagger.get_features(ws, fs)
            lrs = self.lu_tagger.rnn_output(lxs)
            if self.short_model_usage == 'concat':
                rs = [F.concat((sr, lr), 1) for sr, lr in zip(srs, lrs)]
            elif self.short_model_usage == 'add':
                rs = [sr + lr for sr, lr in zip(srs, lrs)]
            else:
                rs = lrs

            _, lps = self.lu_tagger.predict(rs, calculate_loss=False)

        return sps, lps
