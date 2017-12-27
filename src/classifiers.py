import sys

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

import evaluators
import models.util
from models.tagging import RNNCRFTagger
from models.dual_tagging import DualRNNTagger
from models.parsing import RNNBiaffineParser, RNNBiaffineFlexibleParser


class Classifier(chainer.link.Chain):
    def __init__(self, predictor):
        super().__init__(predictor=predictor)
    

    def load_pretrained_embedding_layer(self, dic, external_model, finetuning=False):
        id2token = dic.token_indices.id2str
        models.util.load_pretrained_embedding_layer(
            id2token, self.predictor.pretrained_token_embed, external_model, finetuning=finetuning)

        
class SequenceTagger(Classifier):
    def __init__(self, predictor, task='seg'):
        super().__init__(predictor=predictor)
        self.task = task

        
    # argument train is unused
    def __call__(self, ws, fs, ls, train=False):
        ret = self.predictor(ws, fs, ls)
        return ret


    def decode(self, xs, fs):
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


    def grow_embedding_layers(self, dic_org, dic_grown, external_model=None, train=True):
        id2token_grown = dic_grown.token_indices.id2str
        id2token_org = dic_org.token_indices.id2str
        if len(id2token_grown) > len(id2token_org):
            models.util.grow_embedding_layers(
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
            models.util.grow_MLP(id2lab_org, id2lab_grown, self.predictor.mlp.layers[-1])

            if isinstance(self.predictor, RNNCRFTagger):
                models.util.grow_crf_layer(id2lab_org, id2lab_grown, self.predictor.crf)
                

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
        semb = self.predictor.su_tagger.token_embed
        lemb = self.predictor.lu_tagger.token_embed
        dim = semb.W.shape[1]
        n_vocab_org = semb.W.shape[0]
        n_vocab_grown = lemb.W.shape[0]
        if n_vocab_grown > n_vocab_org:
            diff = n_vocab_grown - n_vocab_org
            vecs = np.zeros((diff, dim), dtype='f')
            new_W = F.concat((semb.W, vecs), axis=0)
            semb.W = chainer.Parameter(initializer=new_W.data, name='W')
        print('Copied parameters from loaded short unit model', file=file)

        # tagger_from = su_tagger
        # tagger_to = self.predictor.su_tagger

        # models.util.copy_embed_parameters(tagger_from.token_embed, tagger_to.token_embed)
        # if tagger_to.pretrained_token_embed_dim > 0:
        #     models.util.copy_embed_parameters(tagger_from.pretrained_token_embed, tagger_to.pretrained_token_embed)
        # if tagger_to.subtoken_embed_dim > 0:
        #     models.util.copy_embed_parameters(tagger_from.subtoken_embed, tagger_to.subtoken_embed)

        # models.util.copy_rnn_parameters(tagger_from.rnn, tagger_to.rnn)

        # models.util.copy_mlp_parameters(tagger_from.mlp, tagger_to.mlp)

        # if isinstance(tagger_to, RNNCRFTagger):
        #     models.util.copy_crf_parameters(tagger_from.crf, tagger_to.crf)


    # TODO confirm
    def load_pretrained_embedding_layer(self, dic, external_model, finetuning=False):
        id2token = dic.token_indices.id2str
        models.util.load_pretrained_embedding_layer(
            id2token, self.predictor.su_tagger.pretrained_token_embed, external_model, finetuning=False)
        models.util.load_pretrained_embedding_layer(
            id2token, self.predictor.lu_tagger.pretrained_token_embed, external_model, finetuning=finetuning)


    # TODO confirm
    def grow_embedding_layers(self, dic_org, dic_grown, external_model=None, train=True):
        id2token_grown = dic_grown.token_indices.id2str
        id2token_org = dic_org.token_indices.id2str
        if len(id2token_grown) > len(id2token_org):
            models.util.grow_embedding_layers(
                id2token_org, id2token_grown, self.predictor.su_tagger.token_embed, 
                self.predictor.su_tagger.pretrained_token_embed, external_model, train=train)

            models.util.grow_embedding_layers(
                id2token_org, id2token_grown, self.predictor.lu_tagger.token_embed, 
                self.predictor.lu_tagger.pretrained_token_embed, external_model, train=train)

        
    # TODO confirm
    def grow_inference_layers(self, dic_org, dic_grown):
        if self.task == 'dual_seg' or self.task == 'dual_seg_tag':
            id2lab_grown = dic_grown.seg_label_indices.id2str
            id2lab_org = dic_org.seg_label_indices.id2str
        else:
            id2lab_grown = dic_grown.pos_label_indices.id2str
            id2lab_org = dic_org.pos_label_indices.id2str

        if len(id2lab_grown) > len(id2lab_org):
            models.util.grow_MLP(id2lab_org, id2lab_grown, self.predictor.su_tagger.mlp.layers[-1])
            models.util.grow_MLP(id2lab_org, id2lab_grown, self.predictor.lu_tagger.mlp.layers[-1])

            if isinstance(self.predictor, RNNCRFTagger):
                models.util.grow_crf_layer(id2lab_org, id2lab_grown, self.predictor.su_tagger.crf)
                models.util.grow_crf_layer(id2lab_org, id2lab_grown, self.predictor.lu_tagger.crf)


class DependencyParser(Classifier):
    def __init__(self, predictor):
        super(DependencyParser, self).__init__(predictor=predictor)

        
    def __call__(self, ws, cs, ps, ths=None, tls=None, train=True):
        ret = self.predictor(ws, cs, ps, ths, tls, train=train)
        return ret


    def decode(self, ws, cs, ps):
        ret = self.predictor.decode(ws, cs, ps)
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


    def grow_embedding_layers(self, dic_org, dic_grown, external_model=None, train=True):
        id2token_grown = dic_grown.token_indices.id2str
        id2token_org = dic_org.token_indices.id2str
        if len(id2token_grown) > len(id2token_org):
            models.util.grow_embedding_layers(
                id2token_org, id2token_grown, 
                self.predictor.token_embed, self.predictor.pretrained_token_embed, external_model, train=train)

        id2pos_grown = dic_grown.pos_label_indices.id2str
        id2pos_org = dic_org.pos_label_indices.id2str
        if len(id2pos_grown) > len(id2pos_org):
            models.util.grow_embedding_layers(
                id2pos_org, id2pos_grown, self.predictor.pos_embed,
                pretrained_embed=None, external_model=None, train=train)


    def grow_inference_layers(self, dic_org, dic_grown):
        id2alab_grown = dic_grown.arc_label_indices.id2str
        id2alab_org = dic_org.arc_label_indices.id2str

        if (self.predictor.label_prediction and
            len(id2alab_grown) > len(id2alab_org)):
            models.util.grow_MLP(
                id2alab_org, id2alab_grown, self.predictor.mlp_label.layers[-1])


def init_classifier(
        classifier_type, hparams, dic, 
        pretrained_token_embed_dim=0, predict_parent_existence=False):
    n_vocab = len(dic.token_indices)
    n_subtokens = len(dic.subtoken_indices) if hparams['subtoken_embed_dim'] > 0 else 0

    # single tagger
    if (classifier_type == 'seg' or classifier_type == 'seg_tag' or classifier_type == 'tag'):
        if 'seg' in classifier_type:
            n_labels = len(dic.seg_label_indices)
        else:
            n_labels = len(dic.pos_label_indices)

        use_crf = hparams['inference_layer'] == 'crf'
        mlp_n_additional_units=0

        predictor = models.util.construct_RNNTagger(
            n_vocab, hparams['token_embed_dim'], 
            n_subtokens, hparams['subtoken_embed_dim'], hparams['rnn_unit_type'], 
            hparams['rnn_bidirection'], hparams['rnn_n_layers'], hparams['rnn_n_units'], 
            hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels, use_crf=use_crf,
            feat_dim=hparams['additional_feat_dim'], mlp_n_additional_units=mlp_n_additional_units,
            rnn_dropout=hparams['rnn_dropout'], mlp_dropout=hparams['mlp_dropout'],
            pretrained_token_embed_dim=pretrained_token_embed_dim)

        classifier = SequenceTagger(predictor, task=classifier_type)

    # dual tagger
    elif (classifier_type == 'dual_seg' or classifier_type == 'dual_seg_tag' or classifier_type == 'dual_tag'):
        if 'seg' in classifier_type:
            n_labels = len(dic.seg_label_indices)
        else:
            n_labels = len(dic.pos_label_indices)
            
        mlp_n_additional_units = 0

        predictor = DualRNNTagger(
            n_vocab, hparams['token_embed_dim'], 
            n_subtokens, hparams['subtoken_embed_dim'], hparams['rnn_unit_type'], 
            hparams['rnn_bidirection'], hparams['rnn_n_layers'], hparams['rnn_n_units'], 
            hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels, 
            feat_dim=hparams['additional_feat_dim'], mlp_n_additional_units=mlp_n_additional_units,
            rnn_dropout=hparams['rnn_dropout'], mlp_dropout=hparams['mlp_dropout'],
            pretrained_token_embed_dim=pretrained_token_embed_dim)

        classifier = DualSequenceTagger(predictor, task=classifier_type)

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

        if predict_parent_existence:
            predictor = RNNBiaffineFlexibleParser(
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
            
        else:
            predictor = RNNBiaffineParser(
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
    if classifier_type == 'seg' or classifier_type == 'dual_seg':
        return evaluators.SegmenterEvaluator(dic.seg_label_indices.id2str)

    elif classifier_type == 'seg_tag' or classifier_type == 'dual_seg_tag':
        return evaluators.JointSegmenterEvaluator(dic.seg_label_indices.id2str)

    elif classifier_type == 'tag' or classifier_type == 'dual_tag':
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

