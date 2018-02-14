import sys

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

import common
import constants
import evaluators
import models.util
import models.tagger
import models.tagger_v0_0_2
from models.dual_tagger import DualRNNTagger
from models.parser import RNNBiaffineParser
from models.attribute_annotator import AttributeAnnotator


class Classifier(chainer.link.Chain):
    def __init__(self, predictor):
        super().__init__(predictor=predictor)
    

    def load_pretrained_embedding_layer(self, dic, external_model, finetuning=False):
        id2unigram = dic.tables[constants.UNIGRAM].id2str
        usage = self.predictor.pretrained_embed_usage

        if usage == models.util.ModelUsage.INIT:
            models.util.load_pretrained_embedding_layer(
                id2unigram, self.predictor.unigram_embed, external_model, finetuning=finetuning)

        elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
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
    def __init__(self, predictor, task=constants.TASK_SEG):
        super().__init__(predictor=predictor)
        self.task = task

        
    def __call__(self, us, bs, ts, ss, fs, ls, train=False):
        ret = self.predictor(us, bs, ts, ss, fs, ls)
        return ret


    def decode(self, *inputs):
        ys = self.predictor.decode(*inputs)
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
        id2unigram_grown = dic_grown.tables[constants.UNIGRAM].id2str
        n_vocab_org = self.predictor.unigram_embed.W.shape[0]
        n_vocab_grown = len(id2unigram_grown)
        if (self.predictor.pretrained_embed_usage == models.util.ModelUsage.ADD or
            self.predictor.pretrained_embed_usage == models.util.ModelUsage.CONCAT):
            pretrained_unigram_embed = self.predictor.pretrained_unigram_embed
        else:
            pretrained_unigram_embed = None
        models.util.grow_embedding_layers(
            n_vocab_org, n_vocab_grown, self.predictor.unigram_embed, 
            pretrained_unigram_embed, external_model, id2unigram_grown,
            self.predictor.pretrained_embed_usage, train=train)

        if constants.SUBTOKEN in dic_grown.tables:
            id2subtoken_grown = dic_grown.tables[constants.SUBTOKEN].id2str
            n_subtokens_org = self.predictor.subtoken_embed.W.shape[0]
            n_subtokens_grown = len(id2subtoken_grown)
            models.util.grow_embedding_layers(
                n_subtokens_org, n_subtokens_grown, self.predictor.subtoken_embed, train=train)

        
    def grow_inference_layers(self, dic_grown):
        n_labels_org = self.predictor.mlp.layers[-1].W.shape[0]
        if common.is_segmentation_task(self.task):
            n_labels_grown = len(dic_grown.tables[constants.SEG_LABEL].id2str)
        else:
            n_labels_grown = len(dic_grown.tables[constants.POS_LABEL].id2str)

        models.util.grow_MLP(n_labels_org, n_labels_grown, self.predictor.mlp.layers[-1])
        if constants.__version__ == 'v0.0.3':
            if self.predictor.use_crf:
                models.util.grow_crf_layer(n_labels_org, n_labels_grown, self.predictor.crf)
        elif constants.__version__ == 'v0.0.2':
            if isinstance(self.predictor, models.tagger_v0_0_2.RNNCRFTagger):
                models.util.grow_crf_layer(n_labels_org, n_labels_grown, self.predictor.crf)
                

class HybridSequenceTagger(SequenceTagger):
    def __init__(self, predictor, task=constants.TASK_HSEG):
        super().__init__(predictor=predictor)
        self.task = task

        
    def change_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.change_rnn_dropout_ratio(dropout_ratio, file=file)
        self.change_biaffine_dropout_ratio(dropout_ratio, file=file)
        self.change_hidden_mlp_dropout_ratio(dropout_ratio, file=file)
        print('', file=file)


    def change_biaffine_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.biaffine_dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('Biaffine', self.predictor.biaffine_dropout), file=file)


    def load_pretrained_embedding_layer(
            self, dic, external_unigram_model, external_chunk_model, finetuning=False):
        id2unigram = dic.tables[constants.UNIGRAM].id2str
        id2chunk = dic.tries[constants.CHUNK].id2chunk
        usage = self.predictor.pretrained_embed_usage

        if external_unigram_model:
            if usage == models.util.ModelUsage.INIT:
                models.util.load_pretrained_embedding_layer(
                    id2unigram, self.predictor.unigram_embed, external_unigram_model, finetuning=finetuning)
            elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
                models.util.load_pretrained_embedding_layer(
                    id2unigram, self.predictor.pretrained_unigram_embed, external_unigram_model, 
                    finetuning=finetuning)

        if external_chunk_model:
            if usage == models.util.ModelUsage.INIT:
                models.util.load_pretrained_embedding_layer(
                    id2chunk, self.predictor.chunk_embed, external_chunk_model, finetuning=finetuning)
            elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
                models.util.load_pretrained_embedding_layer(
                    id2chunk, self.predictor.pretrained_chunk_embed, external_chunk_model, finetuning=finetuning)


    def grow_embedding_layers(
            self, dic_grown, external_unigram_model=None, external_chunk_model=None, train=True):
        if (self.predictor.pretrained_embed_usage == models.util.ModelUsage.ADD or
            self.predictor.pretrained_embed_usage == models.util.ModelUsage.CONCAT):
            pretrained_unigram_embed = self.predictor.pretrained_unigram_embed
            pretrained_chunk_embed = self.predictor.pretrained_chunk_embed
        else:
            pretrained_unigram_embed = None
            pretrained_chunk_embed = None

        id2unigram_grown = dic_grown.tables[constants.UNIGRAM].id2str
        n_vocab_org = self.predictor.unigram_embed.W.shape[0]
        n_vocab_grown = len(id2unigram_grown)
        models.util.grow_embedding_layers(
            n_vocab_org, n_vocab_grown, self.predictor.unigram_embed, 
            pretrained_unigram_embed, external_unigram_model, id2unigram_grown,
            self.predictor.pretrained_embed_usage, train=train)

        id2chunk_grown = dic_grown.tries[constants.CHUNK].id2chunk
        n_vocab_org = self.predictor.chunk_embed.W.shape[0]
        n_vocab_grown = len(id2chunk_grown)
        models.util.grow_embedding_layers(
            n_vocab_org, n_vocab_grown, self.predictor.chunk_embed, 
            pretrained_chunk_embed, external_chunk_model, id2chunk_grown,
            self.predictor.pretrained_embed_usage, train=train)

        if constants.SUBTOKEN in dic_grown.tables:
            id2subtoken_grown = dic_grown.tables[constants.SUBTOKEN].id2str
            n_subtokens_org = self.predictor.subtoken_embed.W.shape[0]
            n_subtokens_grown = len(id2subtoken_grown)
            models.util.grow_embedding_layers(
                n_subtokens_org, n_subtokens_grown, self.predictor.subtoken_embed, train=train)

        
    def __call__(self, us, cs, ms, bs, ts, ss, fs, ls, train=False):
        ret = self.predictor(us, cs, ms, bs, ts, None, fs, ls)
        return ret


class DualSequenceTagger(Classifier):
    def __init__(self, predictor, task=constants.TASK_SEG):
        super().__init__(predictor=predictor)
        self.task = task


    # argument train is unused
    def __call__(self, ws, fs, ls, train=False):
        ret = self.predictor(ws, fs, ls)
        return ret


    def decode(self, *inputs):
        sys, lys = self.predictor.decode(*inputs)
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
        if constants.__version__ == 'v0.0.3':
            pass
        elif constants.__version__ == 'v0.0.2':
            if isinstance(sut, models.tagger_v0_0_2.RNNCRFTagger):
                models.util.grow_crf_layer(n_labels_org, n_labels_grown, sut.crf)
        print('Copied parameters from loaded short unit model', file=file)


    def load_pretrained_embedding_layer(self, dic, external_model, finetuning=False):
        pass
        # id2unigram = dic.tables[constants.UNIGRAM].id2str
        # models.util.load_pretrained_embedding_layer(
        #     id2unigram, self.predictor.lu_tagger.pretrained_unigram_embed, external_model, finetuning=finetuning)


    def grow_embedding_layers(self, dic_grown, external_model=None, train=True):
        id2unigram_grown = dic_grown.tables[constants.UNIGRAM].id2str
        n_vocab_org = self.predictor.su_tagger.unigram_embed.W.shape[0]
        n_vocab_grown = len(id2unigram_grown)

        models.util.grow_embedding_layers(
            n_vocab_org, n_vocab_grown, self.predictor.su_tagger.unigram_embed, 
            self.predictor.su_tagger.pretrained_unigram_embed, external_model, id2unigram_grown, train=train)
        models.util.grow_embedding_layers(
            n_vocab_org, n_vocab_grown, self.predictor.lu_tagger.unigram_embed, 
            self.predictor.lu_tagger.pretrained_unigram_embed, external_model, id2unigram_grown, train=train)
            
        if constants.SUBTOKEN in dic_grown.tables:
            id2subtoken_grown = dic_grown.tables[constants.SUBTOKEN].id2str
            n_subtokens_org = self.predictor.subtoken_embed.W.shape[0]
            n_subtokens_grown = len(id2subtoken_grown)
            models.util.grow_embedding_layers(
                n_subtokens_org, n_subtokens_grown, self.predictor.subtoken_embed, train=train)

        
    def grow_inference_layers(self, dic_grown):
        n_labels_org = self.predictor.su_tagger.mlp.layers[-1].W.shape[0]
        if common.is_dual_st_task(self.task):
            n_labels_grown = len(dic_grown.tables[constants.SEG_LABEL].id2str)
        else:
            n_labels_grown = len(dic_grown.tables[constants.POS_LABEL].id2str)

        models.util.grow_MLP(n_labels_org, n_labels_grown, self.predictor.su_tagger.mlp.layers[-1])
        models.util.grow_MLP(n_labels_org, n_labels_grown, self.predictor.lu_tagger.mlp.layers[-1])
        if constants.__version__ == 'v0.0.3':
            pass
        elif constants.__version__ == 'v0.0.2':
            if isinstance(self.predictor.su_tagger, models.tagger_v0_0_2.RNNCRFTagger):
                models.util.grow_crf_layer(n_labels_org, n_labels_grown, self.predictor.su_tagger.crf)
                models.util.grow_crf_layer(n_labels_org, n_labels_grown, self.predictor.lu_tagger.crf)


class DependencyParser(Classifier):
    def __init__(self, predictor):
        super(DependencyParser, self).__init__(predictor=predictor)

        
    def __call__(self, ws, cs, ps, ths=None, tls=None, train=False):
        ret = self.predictor(ws, cs, ps, ths, tls, train=train)
        return ret


    def decode(self, *inputs, label_prediction=False):
        ret = self.predictor.decode(*inputs, label_prediction)
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
        id2unigram_grown = dic_grown.tables[constants.UNIGRAM].id2str
        n_vocab_org = self.predictor.unigram_embed.W.shape[0]
        n_vocab_grown = len(id2unigram_grown)
        if (self.predictor.pretrained_embed_usage == models.util.ModelUsage.ADD or
            self.predictor.pretrained_embed_usage == models.util.ModelUsage.CONCAT):
            pretrained_unigram_embed = self.predictor.pretrained_unigram_embed
        else:
            pretrained_unigram_embed = None
        models.util.grow_embedding_layers(
            n_vocab_org, n_vocab_grown, self.predictor.unigram_embed, 
            pretrained_unigram_embed, external_model, id2unigram_grown,
            self.predictor.pretrained_embed_usage, train=train)

        if constants.POS_LABEL in dic_grown.tables:
            id2pos_grown = dic_grown.tables[constants.POS_LABEL].id2str
            n_pos_org = self.predictor.pos_embed.W.shape[0]
            n_pos_grown = len(id2pos_grown)
            models.util.grow_embedding_layers(
                n_pos_org, n_pos_grown, self.predictor.pos_embed, train=train)

        if constants.SUBTOKEN in dic_grown.tables:
            id2subtoken_grown = dic_grown.tables[constants.SUBTOKEN].id2str
            n_subtokens_org = self.predictor.subtoken_embed.W.shape[0]
            n_subtokens_grown = len(id2subtoken_grown)
            models.util.grow_embedding_layers(
                n_subtokens_org, n_subtokens_grown, self.predictor.subtoken_embed, train=train)


    def grow_inference_layers(self, dic_grown):
        if self.predictor.label_prediction:
            id2label_grown = dic_grown.tables[constants.ARC_LABEL].id2str
            n_labels_org = self.predictor.mlp_label.layers[-1].W.shape[0]
            n_labels_grown = len(id2label_grown)
            models.util.grow_MLP(n_labels_org, n_labels_grown, self.predictor.mlp_label.layers[-1])


def init_classifier(task, hparams, dic):
    n_vocab = len(dic.tables['unigram'])

    if 'unigram_embed_dim' in hparams and hparams['unigram_embed_dim'] > 0:
        unigram_embed_dim = hparams['unigram_embed_dim']
    else:
        unigram_embed_dim = 0

    if 'chunk_embed_dim' in hparams and hparams['chunk_embed_dim'] > 0:
        chunk_embed_dim = hparams['chunk_embed_dim']
        n_chunks = len(dic.tries[constants.CHUNK])
    else:
        chunk_embed_dim = n_chunks = 0

    if 'bigram_embed_dim' in hparams and hparams['bigram_embed_dim'] > 0:
        bigram_embed_dim = hparams['bigram_embed_dim']
        n_bigrams = len(dic.tables[constants.BIGRAM])
    else:
        bigram_embed_dim = n_bigrams = 0

    if 'tokentype_embed_dim' in hparams and hparams['tokentype_embed_dim'] > 0:
        tokentype_embed_dim = hparams['tokentype_embed_dim']
        n_tokentypes = len(dic.tables[constants.TOKEN_TYPE])
    else:
        tokentype_embed_dim = n_tokentypes = 0

    if 'subtoken_embed_dim' in hparams and hparams['subtoken_embed_dim'] > 0:
        subtoken_embed_dim = hparams['subtoken_embed_dim']
        n_subtokens = len(dic.tables[constants.SUBTOKEN])
    else:
        subtoken_embed_dim = n_subtokens = 0

    if 'pretrained_unigram_embed_dim' in hparams and hparams['pretrained_unigram_embed_dim'] > 0:
        pretrained_unigram_embed_dim = hparams['pretrained_unigram_embed_dim']
    else:
        pretrained_unigram_embed_dim = 0

    if 'pretrained_chunk_embed_dim' in hparams and hparams['pretrained_chunk_embed_dim'] > 0:
        pretrained_chunk_embed_dim = hparams['pretrained_chunk_embed_dim']
    else:
        pretrained_chunk_embed_dim = 0

    if 'pretrained_embed_usage' in hparams:
        pretrained_embed_usage = models.util.ModelUsage.get_instance(hparams['pretrained_embed_usage'])
    else:
        pretrained_embed_usage = models.util.ModelUsage.NONE

    use_attention = 'use_attention' in hparams and hparams['use_attention'] == True
    use_chunk_first = 'use_chunk_first' in hparams and hparams['use_chunk_first'] == True

    if (pretrained_embed_usage == models.util.ModelUsage.ADD or
        pretrained_embed_usage == models.util.ModelUsage.INIT):
        if pretrained_unigram_embed_dim > 0 and pretrained_unigram_embed_dim != unigram_embed_dim:
            print('Error: pre-trained and random initialized unigram embedding vectors '
                  + 'must be the same dimension for {} operation'.format(hparams['pretrained_embed_usage'])
                  + ': d1={}, d2={}'.format(pretrained_unigram_embed_dim, unigram_embed_dim),
                  file=sys.stderr)
            sys.exit()

        if pretrained_chunk_embed_dim > 0 and pretrained_chunk_embed_dim != chunk_embed_dim:
            print('Error: pre-trained and random initialized chunk embedding vectors '
                  + 'must be the same dimension for {} operation'.format(hparams['pretrained_embed_usage'])
                  + ': d1={}, d2={}'.format(pretrained_chunk_embed_dim, chunk_embed_dim),
                  file=sys.stderr)
            sys.exit()

    # single tagger
    if common.is_single_st_task(task):
        if common.is_segmentation_task(task):
            n_labels = len(dic.tables[constants.SEG_LABEL])
        else:
            n_labels = len(dic.tables[constants.POS_LABEL])

        use_crf = hparams['inference_layer'] == 'crf'

        rnn_n_layers2 = hparams['rnn_n_layers2'] if 'rnn_n_layers2' in hparams else 0
        rnn_n_units2 = hparams['rnn_n_units2'] if 'rnn_n_units2' in hparams else 0
        predictor = models.tagger.construct_RNNTagger(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
            n_subtokens, subtoken_embed_dim, n_chunks, chunk_embed_dim, 
            hparams['rnn_unit_type'], hparams['rnn_bidirection'], 
            hparams['rnn_n_layers'], hparams['rnn_n_units'], rnn_n_layers2, rnn_n_units2,
            hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels, use_crf=use_crf,
            feat_dim=hparams['additional_feat_dim'], mlp_n_additional_units=0,
            rnn_dropout=hparams['rnn_dropout'],
            biaffine_dropout=hparams['biaffine_dropout'] if 'biaffine_dropout' in hparams else 0.0,
            mlp_dropout=hparams['mlp_dropout'],
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim,
            pretrained_chunk_embed_dim=pretrained_chunk_embed_dim,
            pretrained_embed_usage=pretrained_embed_usage,
            use_attention=use_attention, use_chunk_first=use_chunk_first)

        if task == constants.TASK_HSEG:
            classifier = HybridSequenceTagger(predictor, task=task)            
        else:
            classifier = SequenceTagger(predictor, task=task)

    # dual tagger
    elif common.is_dual_st_task(task):
        if common.is_segmentation_task(task):
            n_labels = len(dic.tables[constants.SEG_LABEL])
        else:
            n_labels = len(dic.tables[constants.POS_LABEL])
            
        predictor = DualRNNTagger(
            n_vocab, unigram_embed_dim, n_subtokens, subtoken_embed_dim, 
            hparams['rnn_unit_type'], 
            hparams['rnn_bidirection'], hparams['rnn_n_layers'], hparams['rnn_n_units'], 
            hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels, 
            feat_dim=hparams['additional_feat_dim'],
            rnn_dropout=hparams['rnn_dropout'], mlp_dropout=hparams['mlp_dropout'],
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim,
            short_model_usage=hparams['submodel_usage'])

        classifier = DualSequenceTagger(predictor, task=task)

    # parser
    elif common.is_parsing_task(task):
        n_pos = len(dic.tables[constants.POS_LABEL]) if hparams['pos_embed_dim'] > 0 else 0
        n_labels = len(dic.tables[constants.ARC_LABEL]) if common.is_typed_parsing_task(task) else 0
        mlp4pospred_n_layers = 0
        mlp4pospred_n_units = 0

        predictor = RNNBiaffineParser(
            n_vocab, unigram_embed_dim, n_pos, hparams['pos_embed_dim'],
            n_subtokens, subtoken_embed_dim, 
            hparams['rnn_unit_type'], hparams['rnn_bidirection'], hparams['rnn_n_layers'], 
            hparams['rnn_n_units'], 
            hparams['mlp4arcrep_n_layers'], hparams['mlp4arcrep_n_units'],
            hparams['mlp4labelrep_n_layers'], hparams['mlp4labelrep_n_units'],
            mlp4labelpred_n_layers=hparams['mlp4labelpred_n_layers'], 
            mlp4labelpred_n_units=hparams['mlp4labelpred_n_units'],
            # mlp4pospred_n_layers=mlp4pospred_n_layers, mlp4pospred_n_units=mlp4pospred_n_units,
            n_labels=n_labels, rnn_dropout=hparams['rnn_dropout'], 
            hidden_mlp_dropout=hparams['hidden_mlp_dropout'], 
            pred_layers_dropout=hparams['pred_layers_dropout'],
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim,
            pretrained_embed_usage=pretrained_embed_usage)

        classifier = DependencyParser(predictor)

    elif common.is_attribute_annotation_task(task):
        predictor = AttributeAnnotator()
        classifier = PatternMatcher(predictor)

    # tagger and parser
    # elif task == 'tag_dep' or task == 'tag_tdep':
    #     n_pos = len(dic.tables[constants.POS_LABEL]) if hparams['pos_embed_dim'] > 0 else 0
    #     n_labels = len(dic.tables[constants.ARC_LABEL]) if task == 'tag_tdep' else 0
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
    

def init_evaluator(task, dic, ignored_labels):
    if task == constants.TASK_SEG or task == constants.TASK_DUAL_SEG or constants.TASK_HSEG:
        return evaluators.SegmenterEvaluator(dic.tables[constants.SEG_LABEL].id2str)

    elif task == constants.TASK_SEGTAG or task == constants.TASK_DUAL_SEGTAG:
        return evaluators.JointSegmenterEvaluator(dic.tables[constants.SEG_LABEL].id2str)

    elif task == constants.TASK_TAG or task == constants.TASK_DUAL_TAG:
        return evaluators.TaggerEvaluator(ignore_head=False, ignored_labels=ignored_labels)

    elif task == constants.TASK_DEP:
        return evaluators.ParserEvaluator(ignore_head=True)

    elif task == constants.TASK_TDEP:
        return evaluators.TypedParserEvaluator(ignore_head=True, ignored_labels=ignored_labels)

    # elif task == 'tag_dep':
    #     return evaluators.TagerParserEvaluator(ignore_head=True, ignored_labels=ignored_labels)

    # elif task == 'tag_tdep':
    #     return evaluators.TaggerTypedParserEvaluator(ignore_head=True, ignored_labels=ignored_labels)

    elif task == constants.TASK_ATTR:
        return evaluators.AccuracyEvaluator(ignore_head=False, ignored_labels=ignored_labels)

    else:
        return None

