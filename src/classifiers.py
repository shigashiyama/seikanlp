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


    def grow_embedding_layer(self, embed, id2token_all, id2token_trained={}, embed_external=None, 
                             stream=sys.stderr):
        diff = len(id2token_all) - len(id2token_trained)
        weight1 = embed.W
        weight2 = []

        dim = embed.W.shape[0]
        initialW = initializers.normal.Normal(1.0)

        count = 0
        if embed_external:
            start_i = len(id2token_trained)
            for i in range(start_i, len(id2token_all)):
                key = id2token_all[i]
                    
                if key in embed_external.wv.vocab:
                    vec = embed_external.wv[key]
                    weight.append(vec)
                    count += 1
                else:
                    vec = initializers.generate_array(initialW, (dim, ), np)
                    weight.append(vec)

            weight2 = np.reshape(weight2, (diff, dim))
            #TODO type check

        else:
            weight2 = chainer.variable.Parameter(initialW, (diff, dim))

        if id2token_trained:
            weight = F.concat((weight1, weight2), 0)
        else:
            weight = weight2

        embed = L.EmbedID(0, 0)
        embed.W = chainer.Parameter(initializer=weight.data)

        print('Grow embedding layer: {} -> {}'.format(weight1.shape, weight.shape), file=stream)
        if count >= 1:
            print('Use %d pretrained embedding vectors'.format(count), file=stream)
        print('', file=stream)

        
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


    def turn_off_dropout(self, stream=sys.stderr):
        self.rnn.__dict__['dropout'] = 0
        print('Set dropout ratio to 0', file=stream)


    def grow_embedding_layer(self, id2token_all, id2token_trained={}, embed_model=None, stream=sys.stderr):
        super().grow_embedding_layer(
            self.predictor.embed, id2token_all, id2token_trained, embed_model, stream)

        
    def grow_inference_layers(self, id2label_all, id2label_trained, stream=sys.stderr):
        org_len = len(id2label_trained)
        new_len = len(id2label_all)
        diff = new_len - org_len


        # affine layer
        w_org = self.predictor.affine.W
        b_org = self.predictor.affine.b
        w_org_shape = w_org.shape
        b_org_shape = b_org.shape

        dim = w_org.shape[1]
        initialW = initializers.normal.Normal(1.0)

        w_diff = chainer.variable.Parameter(initialW, (diff, dim))
        w_new = F.concat((w_org, w_diff), 0)

        b_diff = chainer.variable.Parameter(initialW, (diff,))
        b_new = F.concat((b_org, b_diff), 0)
            
        self.predictor.affine.W = chainer.Parameter(initializer=w_new.data)
        self.predictor.affine.b = chainer.Parameter(initializer=b_new.data)

        print('Grow affine layer: {}, {} -> {}, {}'.format(
            w_org_shape, b_org_shape, self.predictor.affine.W.shape, 
            self.predictor.affine.b.shape), file=stream)

        # crf layer
        if isinstance(self.predictor, RNN_CRF):
            c_org = self.predictor.crf.cost
            c_diff1 = chainer.variable.Parameter(0, (org_len, diff))
            c_diff2 = chainer.variable.Parameter(0, (diff, new_len))
            c_tmp = F.concat((c_org, c_diff1), 1)
            c_new = F.concat((c_tmp, c_diff2), 0)
            self.predictor.crf.cost = chainer.Parameter(initializer=c_new.data)

            print('Grow crf layer: {} -> {}'.format(
                c_org.shape, self.predictor.crf.cost.shape, file=stream))

        print('', file=stream)
        

class DependencyParser(Classifier):
    def __init__(self, predictor):
        super(DependencyParser, self).__init__(predictor=predictor)

        
    def __call__(self, ws, ps, ths=None, tls=None, train=True):
        ret = self.predictor(ws, ps, ths, tls, train=train)
        return ret


    def decode(self, ws, ps):
        ret = self.predictor.decode(ws, ps)
        return ret


    def turn_off_dropout(self, stream=sys.stderr):
        self.predictor.dropout = 0 # for mlp outputs
        self.predictor.rnn.__dict__['dropout'] = 0 # for rnn layer
        print('Set dropout ratio to 0\n', file=stream)


    def grow_embedding_layer(self, id2token_all, id2token_trained={}, embed_model=None, stream=sys.stderr):
        super().grow_embedding_layer(
            self.predictor.word_embed, id2token_all, id2token_trained, embed_model, stream)


def init_classifier(classifier_type, hparams, indices, train=True):
    n_vocab = len(indices.token_indices)
    dropout = hparams['dropout'] if train else 0
    if classifier_type == 'seg' or classifier_type == 'seg_tag' or classifier_type == 'tag':
        if 'seg' in classifier_type:
            n_labels = len(indices.seg_label_indices)
        else:
            n_labels = len(indices.pos_label_indices)

        if hparams['inference_layer'] == 'crf':
            predictor = models.RNNCRFTagger(
                n_vocab, hparams['unit_embed_dim'], hparams['rnn_unit_type'], 
                hparams['rnn_bidirection'], hparams['rnn_layers'], hparams['rnn_hidden_units'], 
                n_labels, feat_dim=hparams['additional_feat_dim'], dropout=dropout)
                
        else:
            predictor = models.RNNTagger(
                n_vocab, hparams['unit_embed_dim'], hparams['rnn_unit_type'], 
                hparams['rnn_bidirection'], hparams['rnn_layers'], hparams['rnn_hidden_units'], 
                n_labels, feat_dim=hparams['additional_feat_dim'], dropout=dropout)
            
        classifier = SequenceTagger(predictor)
            
    elif classifier_type == 'dep' or classifier_type == 'tdep':
        n_pos = len(indices.pos_label_indices)

        if classifier_type == 'dep':
            n_labels = 0
            predictor = models.RNNBiaffineParser(
                n_vocab, hparams['unit_embed_dim'], n_pos, hparams['pos_embed_dim'],
                hparams['rnn_unit_type'], hparams['rnn_bidirection'], hparams['rnn_layers'], 
                hparams['rnn_hidden_units'], hparams['affine_units_arc'], hparams['affine_units_label'],
                n_labels=n_labels, 
                dropout=dropout)

        elif classifier_type == 'tdep':
            n_labels = len(indices.arc_label_indices)
            predictor = models.RNNBiaffineParser(
                n_vocab, hparams['unit_embed_dim'], n_pos, hparams['pos_embed_dim'],
                hparams['rnn_unit_type'], hparams['rnn_bidirection'], hparams['rnn_layers'], 
                hparams['rnn_hidden_units'], hparams['affine_units_arc'], hparams['affine_units_label'],
                n_labels=n_labels, 
                dropout=dropout)

        else:
            predictor = None

        classifier = DependencyParser(predictor)

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

    else:
        return None

