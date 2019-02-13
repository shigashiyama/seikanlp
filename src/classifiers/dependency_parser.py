import sys

from classifiers.classifier import Classifier
import constants
import models.util


class DependencyParser(Classifier):
    def __init__(self, predictor):
        super(DependencyParser, self).__init__(predictor=predictor)

        
    def __call__(self, *inputs, train=False):
        ret = self.predictor(*inputs, train=train)
        return ret


    def decode(self, *inputs, label_prediction=False):
        ret = self.predictor.decode(*inputs, label_prediction)
        return ret


    def change_dropout_ratio(self, dropout_ratio,):
        self.change_rnn_dropout_ratio(dropout_ratio)
        self.change_hidden_mlp_dropout_ratio(dropout_ratio)
        self.change_pred_layers_dropout_ratio(dropout_ratio)
        print('', file=sys.stderr)


    def change_rnn_dropout_ratio(self, dropout_ratio):
        self.predictor.rnn.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('RNN', self.predictor.rnn.dropout), file=sys.stderr)


    def change_hidden_mlp_dropout_ratio(self, dropout_ratio):
        self.predictor.mlp_arc_head.dropout = dropout_ratio
        if self.predictor.mlp_arc_mod is not None:
            self.predictor.mlp_arc_mod.dropout = dropout_ratio
        if self.predictor.label_prediction:
            if self.predictor.mlp_label_head is not None:
                self.predictor.mlp_label_head.dropout = dropout_ratio
            if self.predictor.mlp_label_mod is not None:
                self.predictor.mlp_label_mod.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('MLP', dropout_ratio), file=sys.stderr)


    def change_pred_layers_dropout_ratio(self, dropout_ratio):
        self.predictor.pred_layers_dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('biaffine',dropout_ratio), file=sys.stderr)


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
