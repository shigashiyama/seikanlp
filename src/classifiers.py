import sys

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

import evaluators
import models.util
from models.tagger import RNNCRFTagger
from models.dual_tagger import DualRNNTagger
from models.parser import RNNBiaffineParser, RNNBiaffineFlexibleParser
from models.attribute_annotator import AttributeAnnotator


class Classifier(chainer.link.Chain):
    def __init__(self, predictor):
        super().__init__(predictor=predictor)
    

    def load_pretrained_embedding_layer(self, dic, external_model, finetuning=False):
        id2unigram = dic.tables['unigram'].id2str
        models.util.load_pretrained_embedding_layer(
            id2unigram, self.predictor.pretrained_unigram_embed, external_model, finetuning=finetuning)

        
class PatternMatcher(object):
    def __init__(self, predictor):
        self.predictor = predictor


    def train(self, ws, ps, ls):
        if not ps:
            ps = [None] * len(ws)

        for w, p, l in zip(ws, ps, ls):
            if not p:
                p = [None] * len(w)

            for wi, pi, li in zip(w, p, l):
                self.predictor.update(wi, pi, li)


    def decode(self, ws, ps):
        ys = []

        if not ps:
            ps = [None] * len(ws)

        for w, p in zip(ws, ps):
            if not p:
                p = [None] * len(w)

            y = []
            for wi, pi in zip(w, p):
                yi = self.predictor.predict(wi, pi)
                y.append(yi)

            ys.append(y)

        return ys


class SequenceTagger(Classifier):
    def __init__(self, predictor, task='seg'):
        super().__init__(predictor=predictor)
        self.task = task

        
    def __call__(self, ws, fs, ls, train=False):
        ret = self.predictor(ws, fs, ls)
        return ret


    def decode(self, ws, fs):
        ys = self.predictor.decode(ws, fs)
        return ys


    def change_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.change_rnn_dropout_ratio(dropout_ratio, file=file)
        self.change_hidden_mlp_dropout_ratio(dropout_ratio, file=file)
        print('', file=file)


    def change_rnn_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.rnn.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('RNN', self.predictor.rnn.dropout), file=file)


    def change_hidden_mlp_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.mlp.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('MLP', dropout_ratio), file=file)


    def grow_embedding_layers(self, dic_grown, external_model=None, train=True):
        id2unigram_grown = dic_grown.tables['unigram'].id2str
        n_vocab_org = self.predictor.unigram_embed.W.shape[0]
        n_vocab_grown = len(id2unigram_grown)
        models.util.grow_embedding_layers(
            n_vocab_org, n_vocab_grown, self.predictor.unigram_embed, 
            self.predictor.pretrained_unigram_embed, external_model, id2unigram_grown, train=train)

        if 'subtoken' in dic_grown.tables:
            id2subtoken_grown = dic_grown.tables['subtoken'].id2str
            n_subtokens_org = self.predictor.subtoken_embed.W.shape[0]
            n_subtokens_grown = len(id2subtoken_grown)
            models.util.grow_embedding_layers(
                n_subtokens_org, n_subtokens_grown, self.predictor.subtoken_embed, train=train)

        
    def grow_inference_layers(self, dic_grown):
        n_labels_org = self.predictor.mlp.layers[-1].W.shape[0]
        if self.task == 'seg' or self.task == 'seg_tag':
            n_labels_grown = len(dic_grown.tables['seg_label'].id2str)
        else:
            n_labels_grown = len(dic_grown.tables['pos_label'].id2str)

        models.util.grow_MLP(n_labels_org, n_labels_grown, self.predictor.mlp.layers[-1])
        if isinstance(self.predictor, RNNCRFTagger):
            models.util.grow_crf_layer(n_labels_org, n_labels_grown, self.predictor.crf)
                

class DualSequenceTagger(Classifier):
    def __init__(self, predictor, task='seg'):
        super().__init__(predictor=predictor)
        self.task = task


    # argument train is unused
    def __call__(self, ws, fs, ls, train=False):
        ret = self.predictor(ws, fs, ls)
        return ret


    def decode(self, ws, fs):
        sys, lys = self.predictor.decode(ws, fs)
        return sys, lys


    def change_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.change_rnn_dropout_ratio(dropout_ratio, file=file)
        self.change_hidden_mlp_dropout_ratio(dropout_ratio, file=file)
        print('', file=file)


    def change_rnn_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.lu_tagger.rnn.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('RNN', self.predictor.lu_tagger.rnn.dropout), file=file)


    def change_hidden_mlp_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.lu_tagger.mlp.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('MLP', dropout_ratio), file=file)


    def integrate_submodel(self, short_unit_model, file=sys.stderr):
        self.predictor.su_tagger = short_unit_model
        sut = self.predictor.su_tagger
        lut = self.predictor.lu_tagger

        n_vocab_org = sut.unigram_embed.W.shape[0]
        n_vocab_grown = lut.unigram_embed.W.shape[0]
        models.util.grow_embedding_layers(n_vocab_org, n_vocab_grown, sut.unigram_embed)

        n_labels_org = sut.mlp.layers[-1].W.shape[0]
        n_labels_grown = lut.mlp.layers[-1].W.shape[0]
        models.util.grow_MLP(n_labels_org, n_labels_grown, sut.mlp.layers[-1])        
        if isinstance(sut, RNNCRFTagger):
            models.util.grow_crf_layer(n_labels_org, n_labels_grown, sut.crf)
        print('Copied parameters from loaded short unit model', file=file)


    def load_pretrained_embedding_layer(self, dic, external_model, finetuning=False):
        id2unigram = dic.tables['unigram'].id2str
        models.util.load_pretrained_embedding_layer(
            id2unigram, self.predictor.lu_tagger.pretrained_unigram_embed, external_model, finetuning=finetuning)


    def grow_embedding_layers(self, dic_grown, external_model=None, train=True):
        # id2unigram_org = dic_org.tables['unigram'].id2str
        id2unigram_grown = dic_grown.tables['unigram'].id2str
        n_vocab_org = self.predictor.su_tagger.unigram_embed.W.shape[0]
        n_vocab_grown = len(id2unigram_grown)

        models.util.grow_embedding_layers(
            n_vocab_org, n_vocab_grown, self.predictor.su_tagger.unigram_embed, 
            self.predictor.su_tagger.pretrained_unigram_embed, external_model, id2unigram_grown, train=train)
        models.util.grow_embedding_layers(
            n_vocab_org, n_vocab_grown, self.predictor.lu_tagger.unigram_embed, 
            self.predictor.lu_tagger.pretrained_unigram_embed, external_model, id2unigram_grown, train=train)
            
        if 'subtoken' in dic_grown.tables:
            id2subtoken_grown = dic_grown.tables['subtoken'].id2str
            n_subtokens_org = self.predictor.subtoken_embed.W.shape[0]
            n_subtokens_grown = len(id2subtoken_grown)
            models.util.grow_embedding_layers(
                n_subtokens_org, n_subtokens_grown, self.predictor.subtoken_embed, train=train)

        
    def grow_inference_layers(self, dic_grown):
        n_labels_org = self.predictor.su_tagger.mlp.layers[-1].W.shape[0]
        if self.task == 'dual_seg' or self.task == 'dual_seg_tag':
            n_labels_grown = len(dic_grown.tables['seg_label'].id2str)
        else:
            n_labels_grown = len(dic_grown.tables['pos_label'].id2str)

        models.util.grow_MLP(n_labels_org, n_labels_grown, self.predictor.su_tagger.mlp.layers[-1])
        models.util.grow_MLP(n_labels_org, n_labels_grown, self.predictor.lu_tagger.mlp.layers[-1])
        if isinstance(self.predictor.su_tagger, RNNCRFTagger):
            models.util.grow_crf_layer(n_labels_org, n_labels_grown, self.predictor.su_tagger.crf)
            models.util.grow_crf_layer(n_labels_org, n_labels_grown, self.predictor.lu_tagger.crf)


class DependencyParser(Classifier):
    def __init__(self, predictor):
        super(DependencyParser, self).__init__(predictor=predictor)

        
    def __call__(self, ws, cs, ps, ths=None, tls=None, train=False):
        ret = self.predictor(ws, cs, ps, ths, tls, train=train)
        return ret


    def decode(self, ws, cs, ps, label_prediction=False):
        ret = self.predictor.decode(ws, cs, ps, label_prediction)
        return ret


    def change_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.change_rnn_dropout_ratio(dropout_ratio, file=file)
        self.change_hidden_mlp_dropout_ratio(dropout_ratio, file=file)
        self.change_pred_layers_dropout_ratio(dropout_ratio, file=file)
        print('', file=file)


    def change_rnn_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.rnn.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('RNN', self.predictor.rnn.dropout), file=file)


    def change_hidden_mlp_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.mlp_arc_head.dropout = dropout_ratio
        if self.predictor.mlp_arc_mod is not None:
            self.predictor.mlp_arc_mod.dropout = dropout_ratio
        if self.predictor.label_prediction:
            if self.predictor.mlp_label_head is not None:
                self.predictor.mlp_label_head.dropout = dropout_ratio
            if self.predictor.mlp_label_mod is not None:
                self.predictor.mlp_label_mod.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('MLP', dropout_ratio), file=file)


    def change_pred_layers_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.pred_layers_dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('biaffine',dropout_ratio), file=file)


    def grow_embedding_layers(self, dic_grown, external_model=None, train=True):
        id2unigram_grown = dic_grown.tables['unigram'].id2str
        n_vocab_org = self.predictor.unigram_embed.W.shape[0]
        n_vocab_grown = len(id2unigram_grown)
        models.util.grow_embedding_layers(
            n_vocab_org, n_vocab_grown, self.predictor.unigram_embed, 
            self.predictor.pretrained_unigram_embed, external_model, id2unigram_grown, train=train)

        if 'pos_label' in dic_grown.tables:
            id2pos_grown = dic_grown.tables['pos_label'].id2str
            n_pos_org = self.predictor.pos_embed.W.shape[0]
            n_pos_grown = len(id2pos_grown)
            models.util.grow_embedding_layers(
                n_pos_org, n_pos_grown, self.predictor.pos_embed, train=train)

        if 'subtoken' in dic_grown.tables:
            id2subtoken_grown = dic_grown.tables['subtoken'].id2str
            n_subtokens_org = self.predictor.subtoken_embed.W.shape[0]
            n_subtokens_grown = len(id2subtoken_grown)
            models.util.grow_embedding_layers(
                n_subtokens_org, n_subtokens_grown, self.predictor.subtoken_embed, train=train)


    def grow_inference_layers(self, dic_grown):
        if self.predictor.label_prediction:
            id2label_grown = dic_grown.tables['arc_label'].id2str
            n_labels_org = self.predictor.mlp_label.layers[-1].W.shape[0]
            n_labels_grown = len(id2label_grown)
            models.util.grow_MLP(n_labels_org, n_labels_grown, self.predictor.mlp_label.layers[-1])


def init_classifier(task, hparams, dic):
    n_vocab = len(dic.tables['unigram'])
    if 'subtoken_embed_dim' in hparams and hparams['subtoken_embed_dim'] > 0:
        n_subtokens = len(dic.tables['subtoken'])
    else:
        n_subtokens = 0

    # single tagger
    if task == 'seg' or task == 'seg_tag' or task == 'tag':
        if 'seg' in task:
            n_labels = len(dic.tables['seg_label'])
        else:
            n_labels = len(dic.tables['pos_label'])

        use_crf = hparams['inference_layer'] == 'crf'

        predictor = models.util.construct_RNNTagger(
            n_vocab, hparams['unigram_embed_dim'], 
            n_subtokens, hparams['subtoken_embed_dim'], hparams['rnn_unit_type'], 
            hparams['rnn_bidirection'], hparams['rnn_n_layers'], hparams['rnn_n_units'], 
            hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels, use_crf=use_crf,
            feat_dim=hparams['additional_feat_dim'], mlp_n_additional_units=0,
            rnn_dropout=hparams['rnn_dropout'], mlp_dropout=hparams['mlp_dropout'],
            pretrained_unigram_embed_dim=hparams['pretrained_unigram_embed_dim'])

        classifier = SequenceTagger(predictor, task=task)

    # dual tagger
    elif task == 'dual_seg' or task == 'dual_seg_tag' or task == 'dual_tag':
        if 'seg' in task:
            n_labels = len(dic.tables['seg_label'])
        else:
            n_labels = len(dic.tables['pos_label'])
            
        predictor = DualRNNTagger(
            n_vocab, hparams['unigram_embed_dim'], 
            n_subtokens, hparams['subtoken_embed_dim'], hparams['rnn_unit_type'], 
            hparams['rnn_bidirection'], hparams['rnn_n_layers'], hparams['rnn_n_units'], 
            hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels, 
            feat_dim=hparams['additional_feat_dim'],
            rnn_dropout=hparams['rnn_dropout'], mlp_dropout=hparams['mlp_dropout'],
            pretrained_unigram_embed_dim=hparams['pretrained_unigram_embed_dim'],
            short_model_usage=hparams['submodel_usage'])

        classifier = DualSequenceTagger(predictor, task=task)

    # parser
    elif task == 'dep' or task == 'tdep':
        n_pos = len(dic.tables['pos_label']) if hparams['pos_embed_dim'] > 0 else 0
        n_labels = len(dic.tables['arc_label']) if task == 'tdep' else 0
        mlp4pospred_n_layers = 0
        mlp4pospred_n_units = 0

        predictor = RNNBiaffineParser(
            n_vocab, hparams['unigram_embed_dim'], n_pos, hparams['pos_embed_dim'],
            n_subtokens, hparams['subtoken_embed_dim'],
            hparams['rnn_unit_type'], hparams['rnn_bidirection'], hparams['rnn_n_layers'], 
            hparams['rnn_n_units'], 
            hparams['mlp4arcrep_n_layers'], hparams['mlp4arcrep_n_units'],
            hparams['mlp4labelrep_n_layers'], hparams['mlp4labelrep_n_units'],
            mlp4labelpred_n_layers=hparams['mlp4labelpred_n_layers'], 
            mlp4labelpred_n_units=hparams['mlp4labelpred_n_units'],
            mlp4pospred_n_layers=mlp4pospred_n_layers, mlp4pospred_n_units=mlp4pospred_n_units,
            n_labels=n_labels, rnn_dropout=hparams['rnn_dropout'], 
            hidden_mlp_dropout=hparams['hidden_mlp_dropout'], 
            pred_layers_dropout=hparams['pred_layers_dropout'],
            pretrained_unigram_embed_dim=hparams['pretrained_unigram_embed_dim'])

        classifier = DependencyParser(predictor)

    elif task == 'attr':
        predictor = AttributeAnnotator()
        classifier = PatternMatcher(predictor)

    # tagger and parser
    # elif task == 'tag_dep' or task == 'tag_tdep':
    #     n_pos = len(dic.tables['pos_label']) if hparams['pos_embed_dim'] > 0 else 0
    #     n_labels = len(dic.tables['arc_label']) if task == 'tag_tdep' else 0
    #     mlp4pospred_n_layers = hparams['mlp4pospred_n_layers']
    #     mlp4pospred_n_units = hparams['mlp4pospred_n_units']

    #     predictor = RNNBiaffineParser(
    #         n_vocab, hparams['unigram_embed_dim'], n_pos, hparams['pos_embed_dim'],
    #         n_subtokens, hparams['subtoken_embed_dim'],
    #         hparams['rnn_unit_type'], hparams['rnn_bidirection'], hparams['rnn_n_layers'], 
    #         hparams['rnn_n_units'], 
    #         hparams['mlp4arcrep_n_layers'], hparams['mlp4arcrep_n_units'],
    #         hparams['mlp4labelrep_n_layers'], hparams['mlp4labelrep_n_units'],
    #         mlp4labelpred_n_layers=hparams['mlp4labelpred_n_layers'], 
    #         mlp4labelpred_n_units=hparams['mlp4labelpred_n_units'],
    #         mlp4pospred_n_layers=mlp4pospred_n_layers, mlp4pospred_n_units=mlp4pospred_n_units,
    #         n_labels=n_labels, rnn_dropout=hparams['rnn_dropout'], 
    #         hidden_mlp_dropout=hparams['hidden_mlp_dropout'], 
    #         pred_layers_dropout=hparams['pred_layers_dropout'],
    #         pretrained_unigram_embed_dim=pretrained_unigram_embed_dim)

    #     classifier = DependencyParser(predictor)

    else:
        print('Invalid task type: {}'.format(task), file=sys.stderr)
        sys.exit()
        
    return classifier
    

def init_evaluator(task, dic, ignore_labels):
    if task == 'seg' or task == 'dual_seg':
        return evaluators.SegmenterEvaluator(dic.tables['seg_label'].id2str)

    elif task == 'seg_tag' or task == 'dual_seg_tag':
        return evaluators.JointSegmenterEvaluator(dic.tables['seg_label'].id2str)

    elif task == 'tag' or task == 'dual_tag':
        return evaluators.TaggerEvaluator(ignore_head=False, ignore_labels=ignore_labels)

    elif task == 'dep':
        return evaluators.ParserEvaluator(ignore_head=True, ignore_labels=ignore_labels)

    elif task == 'tdep':
        return evaluators.TypedParserEvaluator(ignore_head=True, ignore_labels=ignore_labels)

    elif task == 'tag_dep':
        return evaluators.TagerParserEvaluator(ignore_head=True, ignore_labels=ignore_labels)

    elif task == 'tag_tdep':
        return evaluators.TaggerTypedParserEvaluator(ignore_head=True, ignore_labels=ignore_labels)

    elif task == 'attr':
        return evaluators.AccuracyEvaluator(ignore_head=False, ignore_labels=ignore_labels)

    else:
        return None

