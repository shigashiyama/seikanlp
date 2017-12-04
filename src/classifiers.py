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
    

    def merge_features(self, xs, fs=None):
        exs = []
        if fs:
            for x, feat in zip(xs, fs):
                emb = self.predictor.embed(x)
                ex = F.concat((emb, feat), 1)
                exs.append(ex)

        else:
            for x in xs:
                ex = self.predictor.embed(x)
                exs.append(ex)

        return exs


    # deprecated
    def grow_embedding_layer_old(self, embed, id2token_all, id2token_trained={}, embed_external=None, 
                             stream=sys.stderr):
        diff = len(id2token_all) - len(id2token_trained)
        weight1 = embed.W
        weight2 = []

        dim = embed.W.shape[1]
        initialW = initializers.normal.Normal(1.0)

        count = 0
        if embed_external:
            start_i = len(id2token_trained)
            for i in range(start_i, len(id2token_all)):
                key = id2token_all[i]
                if key in embed_external.wv.vocab:
                    vec = embed_external.wv[key]
                    weight2.append(vec)
                    count += 1
                else:
                    vec = initializers.generate_array(initialW, (dim, ), np)
                    weight2.append(vec)

            weight2 = np.reshape(weight2, (diff, dim))

        else:
            weight2 = chainer.variable.Parameter(initialW, (diff, dim))

        if id2token_trained:
            weight = F.concat((weight1, weight2), 0)
        else:
            weight = weight2
        print(type(weight))
        print(weight.data)

        embed = L.EmbedID(0, 0)
        embed.W = chainer.Parameter(initializer=weight)

        print('Grow embedding layer: {} -> {}'.format(weight1.shape, weight.shape), file=stream)
        if count >= 1:
            print('Use {} pretrained embedding vectors\n'.format(count), file=stream)

        
class SequenceTagger(Classifier):
    def __init__(self, predictor):
        super(SequenceTagger, self).__init__(predictor=predictor)

        
    def __call__(self, xs, fs, ls):
        exs = self.merge_features(xs, fs)
        loss, ys = self.predictor(exs, ls)
        return loss, ys


    def decode(self, xs, fs):
        exs = self.merge_features(xs, fs)
        ys = self.predictor.decode(exs)
        return ys


    def change_dropout_ratio(self, dropout_ratio, stream=sys.stderr):
        self.predictor.rnn.dropout = dropout_ratio
        print('Set dropout ratio to {}\n'.format(self.predictor.rnn.dropout), file=stream)


    def grow_embedding_layers(self):
        #TODO implement
        pass


    # def grow_embedding_layer_old(self, id2token_all, id2token_trained={}, embed_model=None, stream=sys.stderr):
    #     super().grow_embedding_layer_old(
    #         self.predictor.embed, id2token_all, id2token_trained, embed_model, stream)

        
class DependencyParser(Classifier):
    def __init__(self, predictor):
        super(DependencyParser, self).__init__(predictor=predictor)

        
    def __call__(self, ws, ps, ths=None, tls=None, train=True):
        ret = self.predictor(ws, ps, ths, tls, train=train)
        return ret


    def decode(self, ws, ps):
        ret = self.predictor.decode(ws, ps)
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
        self.predictor.affine_arc_head.dropout = dropout_ratio
        self.predictor.affine_arc_mod.dropout = dropout_ratio
        if self.predictor.label_prediction:
            self.predictor.affine_label_head.dropout = dropout_ratio
            self.predictor.affine_label_mod.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('MLP', dropout_ratio), file=stream)


    def change_pred_layers_dropout_ratio(self, dropout_ratio, stream=sys.stderr):
        self.predictor.pred_layers_dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('biaffine',dropout_ratio), file=stream)


    def grow_embedding_layers(self, indices_org, indices_grown, external_model=None, train=True):
        id2token_grown = indices_grown.token_indices.id2str
        id2token_org = indices_org.token_indices.id2str
        if len(id2token_grown) > len(id2token_org):
            models.grow_embedding_layers(
                id2token_org, id2token_grown, 
                self.predictor.word_embed, self.predictor.trained_word_embed, external_model, train=train)

        id2pos_grown = indices_grown.pos_label_indices.id2str
        id2pos_org = indices_org.pos_label_indices.id2str
        if len(id2pos_grown) > len(id2pos_org):
            models.grow_embedding_layers(
                id2pos_org, id2pos_grown, self.predictor.pos_embed,
                trained_embed=None, external_model=None, train=train)


    def grow_inference_layers(self, indices_org, indices_grown):
        id2alab_grown = indices_grown.arc_label_indices.id2str
        id2alab_org = indices_org.arc_label_indices.id2str

        if (self.predictor.label_prediction and
            len(id2alab_grown) > len(id2alab_org)):
            models.grow_biaffine_layer(
                id2alab_org, id2alab_grown, self.predictor.biaffine_label)


def init_classifier(classifier_type, hparams, indices, pretrained_unit_embed_dim=0):
    #, finetune_external_embed=False):
    n_vocab = len(indices.token_indices)

    # tagger
    if classifier_type == 'seg' or classifier_type == 'seg_tag' or classifier_type == 'tag':
        if 'seg' in classifier_type:
            n_labels = len(indices.seg_label_indices)
        else:
            n_labels = len(indices.pos_label_indices)

        if hparams['inference_layer'] == 'crf':
            predictor = models.RNNCRFTagger(
                n_vocab, hparams['unit_embed_dim'], hparams['rnn_unit_type'], 
                hparams['rnn_bidirection'], hparams['rnn_n_layers'], hparams['rnn_n_units'], 
                n_labels, feat_dim=hparams['additional_feat_dim'], dropout=hparams['dropout'])
                
        else:
            predictor = models.RNNTagger(
                n_vocab, hparams['unit_embed_dim'], hparams['rnn_unit_type'], 
                hparams['rnn_bidirection'], hparams['rnn_n_layers'], hparams['rnn_n_units'], 
                n_labels, feat_dim=hparams['additional_feat_dim'], dropout=hparams['dropout'])
            
        classifier = SequenceTagger(predictor)
    
    # parser
    elif (classifier_type == 'dep' or classifier_type == 'tdep' or
          classifier_type == 'tag_dep' or classifier_type == 'tag_tdep'):
        n_pos = len(indices.pos_label_indices)

        if classifier_type == 'tdep' or classifier_type == 'tag_tdep':
            n_labels = len(indices.arc_label_indices)
        else:
            n_labels = 0

        if classifier_type == 'tag_dep' or classifier_type == 'tag_tdep':
            mlp4pospred_n_layers = hparams['mlp4pospred_n_layers']
            mlp4pospred_n_units = hparams['mlp4pospred_n_units']
        else:
            mlp4pospred_n_layers = 0
            mlp4pospred_n_units = 0

        predictor = models.RNNBiaffineParser(
            n_vocab, hparams['unit_embed_dim'], n_pos, hparams['pos_embed_dim'],
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
            trained_word_embed_dim=pretrained_unit_embed_dim)

        classifier = DependencyParser(predictor)

    else:
        print('Invalid type: {}'.format(classifier_type), file=sys.stderr)
        sys.exit()
        
    return classifier
    

def init_evaluator(classifier_type, indices, ignore_labels):
    if classifier_type == 'seg':
        return evaluators.SegmenterEvaluator(indices.seg_label_indices)

    elif classifier_type == 'seg_tag':
        return evaluators.JointSegmenterEvaluator(indices.seg_label_indices)

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

