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
        id2unigram = dic.tables['unigram'].id2str
        models.util.load_pretrained_embedding_layer(
            id2unigram, self.predictor.pretrained_unigram_embed, external_model, finetuning=finetuning)

        
class SequenceTagger(Classifier):
    def __init__(self, predictor, task='seg'):
        super().__init__(predictor=predictor)
        self.task = task

        
    # argument train is unused
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


    def grow_embedding_layers(self, dic_org, dic_grown, external_model=None, train=True):
        id2unigram_grown = dic_grown.tables['unigram'].id2str
        id2unigram_org = dic_org.tables['unigram'].id2str
        if len(id2unigram_grown) > len(id2unigram_org):
            models.util.grow_embedding_layers(
                id2unigram_org, id2unigram_grown, 
                self.predictor.unigram_embed, self.predictor.pretrained_unigram_embed, external_model, train=train)

        
    def grow_inference_layers(self, dic_org, dic_grown):
        if self.task == 'seg' or self.task == 'seg_tag':
            id2lab_grown = dic_grown.tables['seg_label'].id2str
            id2lab_org = dic_org.tables['seg_label'].id2str
        else:
            id2lab_grown = dic_grown.tables['pos_label'].id2str
            id2lab_org = dic_org.tables['pos_label'].id2str

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
        semb = self.predictor.su_tagger.unigram_embed
        lemb = self.predictor.lu_tagger.unigram_embed
        dim = semb.W.shape[1]
        n_vocab_org = semb.W.shape[0]
        n_vocab_grown = lemb.W.shape[0]
        if n_vocab_grown > n_vocab_org:
            diff = n_vocab_grown - n_vocab_org
            vecs = np.zeros((diff, dim), dtype='f')
            new_W = F.concat((semb.W, vecs), axis=0)
            semb.W = chainer.Parameter(initializer=new_W.data, name='W')
        print('Copied parameters from loaded short unit model', file=file)


    # TODO confirm
    def load_pretrained_embedding_layer(self, dic, external_model, finetuning=False):
        id2unigram = dic.tables['unigram'].id2str
        models.util.load_pretrained_embedding_layer(
            id2unigram, self.predictor.su_tagger.pretrained_unigram_embed, external_model, finetuning=False)
        models.util.load_pretrained_embedding_layer(
            id2unigram, self.predictor.lu_tagger.pretrained_unigram_embed, external_model, finetuning=finetuning)


    # TODO confirm
    def grow_embedding_layers(self, dic_org, dic_grown, external_model=None, train=True):
        id2unigram_grown = dic_grown.tables['unigram'].id2str
        id2unigram_org = dic_org.tables['unigram'].id2str
        if len(id2unigram_grown) > len(id2unigram_org):
            models.util.grow_embedding_layers(
                id2unigram_org, id2unigram_grown, self.predictor.su_tagger.unigram_embed, 
                self.predictor.su_tagger.pretrained_unigram_embed, external_model, train=train)

            models.util.grow_embedding_layers(
                id2unigram_org, id2unigram_grown, self.predictor.lu_tagger.unigram_embed, 
                self.predictor.lu_tagger.pretrained_unigram_embed, external_model, train=train)

        
    # TODO confirm
    def grow_inference_layers(self, dic_org, dic_grown):
        if self.task == 'dual_seg' or self.task == 'dual_seg_tag':
            id2lab_grown = dic_grown.tables['seg_label'].id2str
            id2lab_org = dic_org.tables['seg_label'].id2str
        else:
            id2lab_grown = dic_grown.tables['pos_label'].id2str
            id2lab_org = dic_org.tables['pos_label'].id2str

        if len(id2lab_grown) > len(id2lab_org):
            models.util.grow_MLP(id2lab_org, id2lab_grown, self.predictor.su_tagger.mlp.layers[-1])
            models.util.grow_MLP(id2lab_org, id2lab_grown, self.predictor.lu_tagger.mlp.layers[-1])

            if isinstance(self.predictor, RNNCRFTagger):
                models.util.grow_crf_layer(id2lab_org, id2lab_grown, self.predictor.su_tagger.crf)
                models.util.grow_crf_layer(id2lab_org, id2lab_grown, self.predictor.lu_tagger.crf)


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


    def grow_embedding_layers(self, dic_org, dic_grown, external_model=None, train=True):
        id2unigram_grown = dic_grown.tables['unigram'].id2str
        id2unigram_org = dic_org.tables['unigram'].id2str
        if len(id2unigram_grown) > len(id2unigram_org):
            models.util.grow_embedding_layers(
                id2unigram_org, id2unigram_grown, 
                self.predictor.unigram_embed, self.predictor.pretrained_unigram_embed, external_model, train=train)

        id2pos_grown = dic_grown.tables['pos_label'].id2str
        id2pos_org = dic_org.tables['pos_label'].id2str
        if len(id2pos_grown) > len(id2pos_org):
            models.util.grow_embedding_layers(
                id2pos_org, id2pos_grown, self.predictor.pos_embed,
                pretrained_embed=None, external_model=None, train=train)


    def grow_inference_layers(self, dic_org, dic_grown):
        id2alab_grown = dic_grown.tables['arc_label'].id2str
        id2alab_org = dic_org.tables['arc_label'].id2str

        if (self.predictor.label_prediction and
            len(id2alab_grown) > len(id2alab_org)):
            models.util.grow_MLP(
                id2alab_org, id2alab_grown, self.predictor.mlp_label.layers[-1])


def init_classifier(
        classifier_type, hparams, dic, 
        pretrained_unigram_embed_dim=0, predict_parent_existence=False):
    n_vocab = len(dic.tables['unigram'])
    n_subtokens = len(dic.tables['subtoken']) if hparams['subtoken_embed_dim'] > 0 else 0

    # single tagger
    if (classifier_type == 'seg' or classifier_type == 'seg_tag' or classifier_type == 'tag'):
        if 'seg' in classifier_type:
            n_labels = len(dic.tables['seg_label'])
        else:
            n_labels = len(dic.tables['pos_label'])

        use_crf = hparams['inference_layer'] == 'crf'
        mlp_n_additional_units=0

        predictor = models.util.construct_RNNTagger(
            n_vocab, hparams['unigram_embed_dim'], 
            n_subtokens, hparams['subtoken_embed_dim'], hparams['rnn_unit_type'], 
            hparams['rnn_bidirection'], hparams['rnn_n_layers'], hparams['rnn_n_units'], 
            hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels, use_crf=use_crf,
            feat_dim=hparams['additional_feat_dim'], mlp_n_additional_units=mlp_n_additional_units,
            rnn_dropout=hparams['rnn_dropout'], mlp_dropout=hparams['mlp_dropout'],
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim)

        classifier = SequenceTagger(predictor, task=classifier_type)

    # dual tagger
    elif (classifier_type == 'dual_seg' or classifier_type == 'dual_seg_tag' or classifier_type == 'dual_tag'):
        if 'seg' in classifier_type:
            n_labels = len(dic.tables['seg_label'])
        else:
            n_labels = len(dic.tables['pos_label'])
            
        mlp_n_additional_units = 0

        predictor = DualRNNTagger(
            n_vocab, hparams['unigram_embed_dim'], 
            n_subtokens, hparams['subtoken_embed_dim'], hparams['rnn_unit_type'], 
            hparams['rnn_bidirection'], hparams['rnn_n_layers'], hparams['rnn_n_units'], 
            hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels, 
            feat_dim=hparams['additional_feat_dim'], mlp_n_additional_units=mlp_n_additional_units,
            rnn_dropout=hparams['rnn_dropout'], mlp_dropout=hparams['mlp_dropout'],
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim)

        classifier = DualSequenceTagger(predictor, task=classifier_type)

    # parser
    elif (classifier_type == 'dep' or classifier_type == 'tdep' or
          classifier_type == 'tag_dep' or classifier_type == 'tag_tdep'):
        n_pos = len(dic.tables['pos_label']) if hparams['pos_embed_dim'] > 0 else 0

        if classifier_type == 'tdep' or classifier_type == 'tag_tdep':
            n_labels = len(dic.tables['arc_label'])
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
                pretrained_unigram_embed_dim=pretrained_unigram_embed_dim)
            
        else:
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
                pretrained_unigram_embed_dim=pretrained_unigram_embed_dim)

        classifier = DependencyParser(predictor)

    else:
        print('Invalid type: {}'.format(classifier_type), file=sys.stderr)
        sys.exit()
        
    return classifier
    

def init_evaluator(classifier_type, dic, ignore_labels):
    if classifier_type == 'seg' or classifier_type == 'dual_seg':
        return evaluators.SegmenterEvaluator(dic.tables['seg_label'].id2str)

    elif classifier_type == 'seg_tag' or classifier_type == 'dual_seg_tag':
        return evaluators.JointSegmenterEvaluator(dic.tables['seg_label'].id2str)

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

