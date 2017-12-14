import sys

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

import models
import evaluators


class Classifier(chainer.link.Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)
    

    # def merge_features(self, xs, fs=None):
    #     exs = []
    #     if fs:
    #         for x, feat in zip(xs, fs):
    #             emb = self.predictor.embed(x)
    #             ex = F.concat((emb, feat), 1)
    #             exs.append(ex)

    #     else:
    #         for x in xs:
    #             ex = self.predictor.embed(x)
    #             exs.append(ex)

    #     return exs

        
class SequenceTagger(Classifier):
    def __init__(self, predictor, task='seg'):
        super(SequenceTagger, self).__init__(predictor=predictor)
        self.task = task

        
    # argument train is unused
    def __call__(self, ws, fs, ls, train=False):
        ret = self.predictor(ws, fs, ls)
        return ret


    def decode(self, xs, fs):
        ys = self.predictor.decode(ws, fs)
        return ys


    def change_dropout_ratio(self, dropout_ratio, stream=sys.stderr):
        self.predictor.rnn.dropout = dropout_ratio
        self.predictor.mlp.dropout = dropout_ratio
        print('Set dropout ratio to {}\n'.format(self.predictor.rnn.dropout), file=stream)


    def grow_embedding_layers(self, dic_org, dic_grown, external_model=None, train=True):
        id2token_grown = dic_grown.token_indices.id2str
        id2token_org = dic_org.token_indices.id2str
        if len(id2token_grown) > len(id2token_org):
            models.grow_embedding_layers(
                id2token_org, id2token_grown, 
                self.predictor.token_embed, self.predictor.pretrained_token_embed, external_model, train=train)

        
    def grow_inference_layers(self, dic_org, dic_grown):
        if self.task == 'seg' or self.task == 'seg_tag':
            id2lab_grown = dic_grown.seg_label_indices.id2str
            id2lab_org = dic_org.seg_label_indices.id2str
        else:
            id2lab_grown = dic_grown.pos_label_indices.id2str
            id2lab_org = dic_org.pos_label_indices.id2str

        if len(id2lab_grown) > len(id2lab_org):
            models.grow_MLP(id2lab_org, id2lab_grown, self.predictor.mlp.layers[-1])

            if isinstance(self.predictor, models.RNNCRFTagger):
                models.grow_crf_layer(id2lab_org, id2lab_grown, self.predictor.crf)
                

class DependencyParser(Classifier):
    def __init__(self, predictor):
        super(DependencyParser, self).__init__(predictor=predictor)

        
    def __call__(self, ws, cs, ps, ths=None, tls=None, train=True):
        ret = self.predictor(ws, cs, ps, ths, tls, train=train)
        return ret


    def decode(self, ws, cs, ps):
        ret = self.predictor.decode(ws, cs, ps)
        return ret


    def change_dropout_ratio(self, dropout_ratio, stream=sys.stderr):
        self.change_rnn_dropout_ratio(dropout_ratio, stream=stream)
        self.change_hidden_mlp_dropout_ratio(dropout_ratio, stream=stream)
        self.change_pred_layers_dropout_ratio(dropout_ratio, stream=stream)
        print('', file=stream)


    def change_rnn_dropout_ratio(self, dropout_ratio, stream=sys.stderr):
        self.predictor.rnn.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('RNN', self.predictor.rnn.dropout), file=stream)


    def change_hidden_mlp_dropout_ratio(self, dropout_ratio, stream=sys.stderr):
        self.predictor.mlp_arc_head.dropout = dropout_ratio
        if self.predictor.mlp_arc_mod is not None:
            self.predictor.mlp_arc_mod.dropout = dropout_ratio
        if self.predictor.label_prediction:
            if self.predictor.mlp_label_head is not None:
                self.predictor.mlp_label_head.dropout = dropout_ratio
            if self.predictor.mlp_label_mod is not None:
                self.predictor.mlp_label_mod.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('MLP', dropout_ratio), file=stream)


    def change_pred_layers_dropout_ratio(self, dropout_ratio, stream=sys.stderr):
        self.predictor.pred_layers_dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('biaffine',dropout_ratio), file=stream)


    def grow_embedding_layers(self, dic_org, dic_grown, external_model=None, train=True):
        id2token_grown = dic_grown.token_indices.id2str
        id2token_org = dic_org.token_indices.id2str
        if len(id2token_grown) > len(id2token_org):
            models.grow_embedding_layers(
                id2token_org, id2token_grown, 
                self.predictor.token_embed, self.predictor.pretrained_token_embed, external_model, train=train)

        id2pos_grown = dic_grown.pos_label_indices.id2str
        id2pos_org = dic_org.pos_label_indices.id2str
        if len(id2pos_grown) > len(id2pos_org):
            models.grow_embedding_layers(
                id2pos_org, id2pos_grown, self.predictor.pos_embed,
                pretrained_embed=None, external_model=None, train=train)


    def grow_inference_layers(self, dic_org, dic_grown):
        id2alab_grown = dic_grown.arc_label_indices.id2str
        id2alab_org = dic_org.arc_label_indices.id2str

        if (self.predictor.label_prediction and
            len(id2alab_grown) > len(id2alab_org)):
            models.grow_MLP(
                id2alab_org, id2alab_grown, self.predictor.mlp_label.layers[-1])


def init_classifier(classifier_type, hparams, dic, pretrained_token_embed_dim=0):
    #, finetune_external_embed=False):
    n_vocab = len(dic.token_indices)
    n_subtokens = len(dic.subtoken_indices) if hparams['subtoken_embed_dim'] > 0 else 0

    # tagger
    if classifier_type == 'seg' or classifier_type == 'seg_tag' or classifier_type == 'tag':
        if 'seg' in classifier_type:
            n_labels = len(dic.seg_label_indices)
        else:
            n_labels = len(dic.pos_label_indices)

        if hparams['inference_layer'] == 'crf':
            predictor = models.RNNCRFTagger(
                n_vocab, hparams['token_embed_dim'], 
                n_subtokens, hparams['subtoken_embed_dim'], hparams['rnn_unit_type'], 
                hparams['rnn_bidirection'], hparams['rnn_n_layers'], hparams['rnn_n_units'], 
                hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels, 
                feat_dim=hparams['additional_feat_dim'], 
                rnn_dropout=hparams['rnn_dropout'], mlp_dropout=hparams['mlp_dropout'],
                pretrained_embed_dim=pretrained_token_embed_dim)
                
        else:
            predictor = models.RNNTagger(
                n_vocab, hparams['token_embed_dim'], 
                n_subtokens, hparams['subtoken_embed_dim'], hparams['rnn_unit_type'], 
                hparams['rnn_bidirection'], hparams['rnn_n_layers'], hparams['rnn_n_units'], 
                hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels,
                feat_dim=hparams['additional_feat_dim'], 
                rnn_dropout=hparams['rnn_dropout'], mlp_dropout=hparams['mlp_dropout'],
                pretrained_embed_dim=pretrained_token_embed_dim)
            
        classifier = SequenceTagger(predictor, task=classifier_type)
    
    # parser
    elif (classifier_type == 'dep' or classifier_type == 'tdep' or
          classifier_type == 'tag_dep' or classifier_type == 'tag_tdep'):
        n_pos = len(dic.pos_label_indices) if hparams['pos_embed_dim'] > 0 else 0

        if classifier_type == 'tdep' or classifier_type == 'tag_tdep':
            n_labels = len(dic.arc_label_indices)
        else:
            n_labels = 0

        if classifier_type == 'tag_dep' or classifier_type == 'tag_tdep':
            mlp4pospred_n_layers = hparams['mlp4pospred_n_layers']
            mlp4pospred_n_units = hparams['mlp4pospred_n_units']
        else:
            mlp4pospred_n_layers = 0
            mlp4pospred_n_units = 0

        predictor = models.RNNBiaffineParser(
            n_vocab, hparams['token_embed_dim'], n_pos, hparams['pos_embed_dim'],
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
            pretrained_token_embed_dim=pretrained_token_embed_dim)

        classifier = DependencyParser(predictor)

    else:
        print('Invalid type: {}'.format(classifier_type), file=sys.stderr)
        sys.exit()
        
    return classifier
    

def init_evaluator(classifier_type, dic, ignore_labels):
    if classifier_type == 'seg':
        return evaluators.SegmenterEvaluator(dic.seg_label_indices.id2str)

    elif classifier_type == 'seg_tag':
        return evaluators.JointSegmenterEvaluator(dic.seg_label_indices.id2str)

    elif classifier_type == 'tag':
        return evaluators.TaggerEvaluator(ignore_head=False, ignore_labels=ignore_labels)

    elif classifier_type == 'dep':
        return evaluators.ParserEvaluator(ignore_head=True, ignore_labels=ignore_labels)

    elif classifier_type == 'tdep':
        return evaluators.TypedParserEvaluator(ignore_head=True, ignore_labels=ignore_labels)

    elif classifier_type == 'tag_dep':
        return evaluators.TagerParserEvaluator(ignore_head=True, ignore_labels=ignore_labels)

    elif classifier_type == 'tag_tdep':
        return evaluators.TaggerTypedParserEvaluator(ignore_head=True, ignore_labels=ignore_labels)

    else:
        return None

