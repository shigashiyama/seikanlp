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
from models.parser import RNNBiaffineParser


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

        
    def __call__(self, us, bs, ts, es, fs, ls, train=False):
        ret = self.predictor(us, bs, ts, es, fs, ls)
        return ret


    def decode(self, *inputs):
        ys = self.predictor.decode(*inputs)
        return ys


    def change_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.change_rnn_dropout_ratio(dropout_ratio, file=file)
        self.change_hidden_mlp_dropout_ratio(dropout_ratio, file=file)
        print('', file=file)


    def change_embed_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.embed_dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('embedding', self.predictor.embed_dropout), file=file)


    def change_rnn_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.rnn.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('RNN', self.predictor.rnn.dropout), file=file)


    def change_hidden_mlp_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.mlp.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('MLP', dropout_ratio), file=file)


    def load_pretrained_embedding_layer(
            self, dic, external_unigram_model, external_bigram_model, finetuning=False):
        usage = self.predictor.pretrained_embed_usage

        if external_unigram_model:
            id2unigram = dic.tables[constants.UNIGRAM].id2str
            if usage == models.util.ModelUsage.INIT:
                models.util.load_pretrained_embedding_layer(
                    id2unigram, self.predictor.unigram_embed, external_unigram_model, finetuning=finetuning)
            elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
                models.util.load_pretrained_embedding_layer(
                    id2unigram, self.predictor.pretrained_unigram_embed, external_unigram_model, 
                    finetuning=finetuning)

        if external_bigram_model: #and dic.has_table(constants.BIGRAM):
            id2bigram = dic.tables[constants.BIGRAM].id2str
            if usage == models.util.ModelUsage.INIT:
                models.util.load_pretrained_embedding_layer(
                    id2bigram, self.predictor.bigram_embed, external_bigram_model, finetuning=finetuning)
            elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
                models.util.load_pretrained_embedding_layer(
                    id2bigram, self.predictor.pretrained_bigram_embed, external_bigram_model, 
                    finetuning=finetuning)


    def grow_embedding_layers(self, dic_grown, external_unigram_model=None, external_bigram_model=None, 
                              train=True):
        if (self.predictor.pretrained_embed_usage == models.util.ModelUsage.ADD or
            self.predictor.pretrained_embed_usage == models.util.ModelUsage.CONCAT):
            pretrained_unigram_embed = self.predictor.pretrained_unigram_embed
            pretrained_bigram_embed = self.predictor.pretrained_bigram_embed
        else:
            pretrained_unigram_embed = None
            pretrained_bigram_embed = None

        id2unigram_grown = dic_grown.tables[constants.UNIGRAM].id2str
        n_unigrams_org = self.predictor.unigram_embed.W.shape[0]
        n_unigrams_grown = len(id2unigram_grown)
        models.util.grow_embedding_layers(
            n_unigrams_org, n_unigrams_grown, self.predictor.unigram_embed, 
            pretrained_unigram_embed, external_unigram_model, id2unigram_grown,
            self.predictor.pretrained_embed_usage, train=train)

        if dic_grown.has_table(constants.BIGRAM):
            id2bigram_grown = dic_grown.tables[constants.BIGRAM].id2str
            n_bigrams_org = self.predictor.bigram_embed.W.shape[0]
            n_bigrams_grown = len(id2bigram_grown)
            models.util.grow_embedding_layers(
                n_bigrams_org, n_bigrams_grown, self.predictor.bigram_embed, 
                pretrained_bigram_embed, external_bigram_model, id2bigram_grown,
                self.predictor.pretrained_embed_usage, train=train)

        if dic_grown.has_table(constants.SUBTOKEN):
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
            n_labels_grown = len(dic_grown.tables[constants.ATTR_LABEL(0)].id2str)

        models.util.grow_MLP(n_labels_org, n_labels_grown, self.predictor.mlp.layers[-1])
        if self.predictor.use_crf:
            models.util.grow_crf_layer(n_labels_org, n_labels_grown, self.predictor.crf)
                

class HybridSequenceTagger(SequenceTagger):
    def __init__(self, predictor, task=constants.TASK_SEG):
        super().__init__(predictor=predictor)
        self.task = task

        
    def change_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.change_embed_dropout_ratio(dropout_ratio, file=file)
        self.change_rnn_dropout_ratio(dropout_ratio, file=file)
        self.change_biaffine_dropout_ratio(dropout_ratio, file=file)
        self.change_hidden_mlp_dropout_ratio(dropout_ratio, file=file)
        self.change_chunk_vector_dropout_ratio(dropout_ratio, file=file)
        print('', file=file)


    def change_biaffine_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.biaffine_dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('Biaffine', self.predictor.biaffine_dropout), file=file)


    def change_chunk_vector_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.predictor.chunk_vector_dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format(
            'Chunk vector', self.predictor.chunk_vector_dropout), file=file)


    def load_pretrained_embedding_layer(
            self, dic, external_unigram_model, external_bigram_model, external_chunk_model, finetuning=False):
        usage = self.predictor.pretrained_embed_usage

        if external_unigram_model:
            id2unigram = dic.tables[constants.UNIGRAM].id2str
            if usage == models.util.ModelUsage.INIT:
                models.util.load_pretrained_embedding_layer(
                    id2unigram, self.predictor.unigram_embed, external_unigram_model, finetuning=finetuning)
            elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
                models.util.load_pretrained_embedding_layer(
                    id2unigram, self.predictor.pretrained_unigram_embed, external_unigram_model, 
                    finetuning=finetuning)

        if external_bigram_model: #and dic.has_table(constants.BIGRAM):
            id2bigram = dic.tables[constants.BIGRAM].id2str
            if usage == models.util.ModelUsage.INIT:
                models.util.load_pretrained_embedding_layer(
                    id2bigram, self.predictor.bigram_embed, external_bigram_model, finetuning=finetuning)
            elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
                models.util.load_pretrained_embedding_layer(
                    id2bigram, self.predictor.pretrained_bigram_embed, external_bigram_model, 
                    finetuning=finetuning)

        if external_chunk_model:
            id2chunk = dic.tries[constants.CHUNK].id2chunk
            if usage == models.util.ModelUsage.INIT:
                models.util.load_pretrained_embedding_layer(
                    id2chunk, self.predictor.chunk_embed, external_chunk_model, finetuning=finetuning)
            elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
                models.util.load_pretrained_embedding_layer(
                    id2chunk, self.predictor.pretrained_chunk_embed, external_chunk_model, finetuning=finetuning)


    def grow_embedding_layers(
            self, dic_grown, external_unigram_model=None, external_bigram_model=None,
            external_chunk_model=None, train=True):
        if (self.predictor.pretrained_embed_usage == models.util.ModelUsage.ADD or
            self.predictor.pretrained_embed_usage == models.util.ModelUsage.CONCAT):
            pretrained_unigram_embed = self.predictor.pretrained_unigram_embed
            pretrained_bigram_embed = self.predictor.pretrained_bigram_embed
            pretrained_chunk_embed = self.predictor.pretrained_chunk_embed
        else:
            pretrained_unigram_embed = None
            pretrained_bigram_embed = None
            pretrained_chunk_embed = None

        id2unigram_grown = dic_grown.tables[constants.UNIGRAM].id2str
        n_unigrams_org = self.predictor.unigram_embed.W.shape[0]
        n_unigrams_grown = len(id2unigram_grown)
        models.util.grow_embedding_layers(
            n_unigrams_org, n_unigrams_grown, self.predictor.unigram_embed, 
            pretrained_unigram_embed, external_unigram_model, id2unigram_grown,
            self.predictor.pretrained_embed_usage, train=train)

        if external_bigram_model: #and dic.has_table(constants.BIGRAM):
            id2bigram_grown = dic_grown.tables[constants.BIGRAM].id2str
            n_bigrams_org = self.predictor.bigram_embed.W.shape[0]
            n_bigrams_grown = len(id2bigram_grown)
            models.util.grow_embedding_layers(
                n_bigrams_org, n_bigrams_grown, self.predictor.bigram_embed, 
                pretrained_bigram_embed, external_bigram_model, id2bigram_grown,
                self.predictor.pretrained_embed_usage, train=train)

        id2chunk_grown = dic_grown.tries[constants.CHUNK].id2chunk
        n_chunks_org = self.predictor.chunk_embed.W.shape[0]
        n_chunks_grown = len(id2chunk_grown)
        models.util.grow_embedding_layers(
            n_chunks_org, n_chunks_grown, self.predictor.chunk_embed, 
            pretrained_chunk_embed, external_chunk_model, id2chunk_grown,
            self.predictor.pretrained_embed_usage, train=train)

        # if constants.SUBTOKEN in dic_grown.tables:
        #     id2subtoken_grown = dic_grown.tables[constants.SUBTOKEN].id2str
        #     n_subtokens_org = self.predictor.subtoken_embed.W.shape[0]
        #     n_subtokens_grown = len(id2subtoken_grown)
        #     models.util.grow_embedding_layers(
        #         n_subtokens_org, n_subtokens_grown, self.predictor.subtoken_embed, train=train)

        
    # def __call__(self, us, cs, ms, bs, ts, ss, fs, ls, train=False):
    #     ret = self.predictor(us, cs, ms, bs, ts, None, fs, ls)
    #     return ret

    def __call__(self, us, cs, ds, ms, bs, ts, ss, fs, gls, gcs, train=False):
        ret = self.predictor(us, cs, ds, ms, bs, ts, None, fs, gls, gcs)
        return ret


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

        # if constants.SUBTOKEN in dic_grown.tables:
        #     id2subtoken_grown = dic_grown.tables[constants.SUBTOKEN].id2str
        #     n_subtokens_org = self.predictor.subtoken_embed.W.shape[0]
        #     n_subtokens_grown = len(id2subtoken_grown)
        #     models.util.grow_embedding_layers(
        #         n_subtokens_org, n_subtokens_grown, self.predictor.subtoken_embed, train=train)

        if constants.ATTR_LABEL(0) in dic_grown.tables: # POS
            id2pos_grown = dic_grown.tables[constants.ATTR_LABEL(0)].id2str
            n_pos_org = self.predictor.pos_embed.W.shape[0]
            n_pos_grown = len(id2pos_grown)
            models.util.grow_embedding_layers(
                n_pos_org, n_pos_grown, self.predictor.pos_embed, train=train)


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

    # if 'subtoken_embed_dim' in hparams and hparams['subtoken_embed_dim'] > 0:
    #     subtoken_embed_dim = hparams['subtoken_embed_dim']
    #     n_subtokens = len(dic.tables[constants.SUBTOKEN])
    # else:
    subtoken_embed_dim = n_subtokens = 0

    if 'pretrained_unigram_embed_dim' in hparams and hparams['pretrained_unigram_embed_dim'] > 0:
        pretrained_unigram_embed_dim = hparams['pretrained_unigram_embed_dim']
    else:
        pretrained_unigram_embed_dim = 0

    if 'pretrained_bigram_embed_dim' in hparams and hparams['pretrained_bigram_embed_dim'] > 0:
        pretrained_bigram_embed_dim = hparams['pretrained_bigram_embed_dim']
    else:
        pretrained_bigram_embed_dim = 0

    if 'pretrained_chunk_embed_dim' in hparams and hparams['pretrained_chunk_embed_dim'] > 0:
        pretrained_chunk_embed_dim = hparams['pretrained_chunk_embed_dim']
    else:
        pretrained_chunk_embed_dim = 0

    if 'pretrained_embed_usage' in hparams:
        pretrained_embed_usage = models.util.ModelUsage.get_instance(hparams['pretrained_embed_usage'])
    else:
        pretrained_embed_usage = models.util.ModelUsage.NONE

    if (pretrained_embed_usage == models.util.ModelUsage.ADD or
        pretrained_embed_usage == models.util.ModelUsage.INIT):
        if pretrained_unigram_embed_dim > 0 and pretrained_unigram_embed_dim != unigram_embed_dim:
            print('Error: pre-trained and random initialized unigram embedding vectors '
                  + 'must be the same dimension for {} operation'.format(hparams['pretrained_embed_usage'])
                  + ': d1={}, d2={}'.format(pretrained_unigram_embed_dim, unigram_embed_dim),
                  file=sys.stderr)
            sys.exit()

        if pretrained_bigram_embed_dim > 0 and pretrained_bigram_embed_dim != bigram_embed_dim:
            print('Error: pre-trained and random initialized bigram embedding vectors '
                  + 'must be the same dimension for {} operation'.format(hparams['pretrained_embed_usage'])
                  + ': d1={}, d2={}'.format(pretrained_bigram_embed_dim, bigram_embed_dim),
                  file=sys.stderr)
            sys.exit()

        if pretrained_chunk_embed_dim > 0 and pretrained_chunk_embed_dim != chunk_embed_dim:
            print('Error: pre-trained and random initialized chunk embedding vectors '
                  + 'must be the same dimension for {} operation'.format(hparams['pretrained_embed_usage'])
                  + ': d1={}, d2={}'.format(pretrained_chunk_embed_dim, chunk_embed_dim),
                  file=sys.stderr)
            sys.exit()

    # single tagger
    if common.is_sequence_labeling_task(task):
        if common.is_segmentation_task(task):
            n_label = len(dic.tables[constants.SEG_LABEL])
            n_labels = [n_label]
            attr1_embed_dim = n_attr1 = 0

        else:
            n_labels = []
            for i in range(3): # tmp
                if constants.ATTR_LABEL(i) in dic.tables:
                    n_label = len(dic.tables[constants.ATTR_LABEL(i)])
                    n_labels.append(n_label)
                
            if 'attr1_embed_dim' in hparams and hparams['attr1_embed_dim'] > 0:
                attr1_embed_dim = hparams['attr1_embed_dim']
                n_attr1 = n_labels[1] if len(n_labels) > 1 else 0
            else:
                attr1_embed_dim = n_attr1 = 0

            # if hparams['attr_predictions']:
            #     attr_predictions = [int(val) for val in hparams['attr_predictions'].split(',')]

        use_crf = hparams['inference_layer'] == 'crf'

        rnn_n_layers2 = hparams['rnn_n_layers2'] if 'rnn_n_layers2' in hparams else 0
        rnn_n_units2 = hparams['rnn_n_units2'] if 'rnn_n_units2' in hparams else 0
        predictor = models.tagger.construct_RNNTagger(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
            n_attr1, attr1_embed_dim, n_chunks, chunk_embed_dim, 
            hparams['rnn_unit_type'], hparams['rnn_bidirection'], 
            hparams['rnn_n_layers'], hparams['rnn_n_units'], rnn_n_layers2, rnn_n_units2,
            hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels[0], use_crf=use_crf,
            feat_dim=hparams['additional_feat_dim'], mlp_n_additional_units=0,
            rnn_dropout=hparams['rnn_dropout'],
            biaffine_dropout=hparams['biaffine_dropout'] if 'biaffine_dropout' in hparams else 0.0,
            embed_dropout=hparams['embed_dropout'] if 'embed_dropout' in hparams else 0.0,
            mlp_dropout=hparams['mlp_dropout'],
            chunk_vector_dropout=hparams['chunk_vector_dropout'] if 'chunk_vector_dropout' in hparams else 0.0,
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim,
            pretrained_bigram_embed_dim=pretrained_bigram_embed_dim,
            pretrained_chunk_embed_dim=pretrained_chunk_embed_dim,
            pretrained_embed_usage=pretrained_embed_usage,
            chunk_pooling_type=hparams['chunk_pooling_type'] if 'chunk_pooling_type' in hparams else '',
            max_chunk_len=hparams['max_chunk_len'] if 'max_chunk_len' in hparams else 0,
            chunk_loss_ratio=hparams['chunk_loss_ratio'] if 'chunk_loss_ratio' in hparams else 0.0,
            biaffine_type=hparams['biaffine_type'] if 'biaffine_type' in hparams else '')

        if 'tagging_unit' in hparams and hparams['tagging_unit'] == 'hybrid':
            classifier = HybridSequenceTagger(predictor, task=task)
        else:
            classifier = SequenceTagger(predictor, task=task)

    # parser
    elif common.is_parsing_task(task):
        n_attr0 = len(dic.tables[constants.ATTR_LABEL(0)]) if (
            hparams['attr0_embed_dim'] > 0 and constants.ATTR_LABEL(0) in dic.tables) else 0
        n_labels = len(dic.tables[constants.ARC_LABEL]) if common.is_typed_parsing_task(task) else 0
        mlp4pospred_n_layers = 0
        mlp4pospred_n_units = 0
        attr0_embed_dim = hparams['attr0_embed_dim'] if n_attr0 > 0 else 0

        predictor = RNNBiaffineParser(
            n_vocab, unigram_embed_dim, n_attr0, attr0_embed_dim,
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

    else:
        print('Invalid task type: {}'.format(task), file=sys.stderr)
        sys.exit()
        
    return classifier
    

def init_evaluator(task, dic, type='', ignored_labels=set()):
    if task == constants.TASK_SEG:
        if type == 'hybrid':
            return evaluators.HybridSegmenterEvaluator(dic.tables[constants.SEG_LABEL].id2str)
        else:
            return evaluators.FMeasureEvaluator(dic.tables[constants.SEG_LABEL].id2str)

    elif task == constants.TASK_SEGTAG:
        if type == 'hybrid':
            return evaluators.HybridTaggerEvaluator(dic.tables[constants.SEG_LABEL].id2str)
        else:
            return evaluators.DoubleFMeasureEvaluator(dic.tables[constants.SEG_LABEL].id2str)
        
    elif task == constants.TASK_TAG:
        if common.use_fmeasure(dic.tables[constants.ATTR_LABEL(0)].str2id):
            return evaluators.FMeasureEvaluator(dic.tables[constants.ATTR_LABEL(0)].id2str)
        else:
            return evaluators.AccuracyEvaluator(dic.tables[constants.ATTR_LABEL(0)].id2str)

    elif task == constants.TASK_DEP:
        return evaluators.ParserEvaluator(ignore_head=True)

    elif task == constants.TASK_TDEP:
        return evaluators.TypedParserEvaluator(ignore_head=True, ignored_labels=ignored_labels)

    else:
        return None

