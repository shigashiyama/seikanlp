"""
This code is implemented by (drastically) remodeling the implementation of (Zaremba 2015) by developers at PFN.

(Zaremba 2015) RECURRENT NEURAL NETWORK REGULARIZATION, ICLR, 2015, https://github.com/tomsercu/lstm

"""

import sys
import copy
import enum
from collections import Counter
from datetime import datetime

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.functions.loss import softmax_cross_entropy
from chainer.links.connection.n_step_rnn import argsort_list_descent, permutate_list
from chainer.functions.math import minmax, exponential, logsumexp
from chainer import initializers
from chainer import Variable


from tools import conlleval
import lattice
import features
from util import Timer


"""
Base model that consists of embedding layer, recurrent network (RNN) layers and affine layer.

Args:
    n_rnn_layers:
        the number of (vertical) layers of recurrent network
    n_vocab:
        size of vocabulary
    n_embed_dim:
        dimention of word embedding
    n_rnn_units:
        the number of units of RNN
    n_labels:
        the number of labels that input instances will be classified into
    dropout:
        dropout ratio of RNN
    rnn_unit_type:
        unit type of RNN: lstm, gru or plain
    rnn_bidirection:
        use bi-directional RNN or not
    affine_activation:
        activation function of affine layer: identity, relu, tanh, sigmoid
    init_embed:
        pre-trained embedding matrix
    feat_extractor:
        FeatureExtractor object to extract additional features 
"""
class RNNBase(chainer.Chain):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, affine_activation='identity', 
            init_embed=None, feat_extractor=None, stream=sys.stderr):
        super(RNNBase, self).__init__()

        with self.init_scope():
            self.act = get_activation(affine_activation)
            if not self.act:
                print('Unsupported activation function', file=stream)
                sys.exit()

            if init_embed:
                n_vocab = init_embed.W.shape[0]
                embed_dim = init_embed.W.shape[1]

            # init fields
            self.embed_dim = embed_dim
            if feat_extractor:
                self.feat_extractor = feat_extractor
                self.input_vec_size = self.embed_dim + self.feat_extractor.dim
            else:
                self.feat_extractor = None
                self.input_vec_size = self.embed_dim

            # init layers
            self.embed = L.EmbedID(n_vocab, self.embed_dim) if init_embed == None else init_embed

            self.rnn_unit_type = rnn_unit_type
            if rnn_unit_type == 'lstm':
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiLSTM(n_rnn_layers, self.input_vec_size, n_rnn_units, dropout)
                else:
                    self.rnn_unit = L.NStepLSTM(n_rnn_layers, embed_dim, n_rnn_units, dropout)
            elif rnn_unit_type == 'gru':
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiGRU(n_rnn_layers, self.input_vec_size, n_rnn_units, dropout) 
                else:
                    self.rnn_unit = L.NStepGRU(n_rnn_layers, embed_dim, n_rnn_units, dropout)
            else:
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiRNNTanh(n_rnn_layers, self.input_vec_size, n_rnn_units, dropout) 
                else:
                    self.rnn_unit = L.NStepRNNTanh(n_rnn_layers, embed_dim, n_rnn_units, dropout)

            self.affine = L.Linear(n_rnn_units * (2 if rnn_bidirection else 1), n_labels)

            print('### Parameters', file=stream)
            print('# Embedding layer: {}'.format(self.embed.W.shape), file=stream)
            print('# RNN unit: {}'.format(self.rnn_unit), file=stream)
            if self.rnn_unit_type == 'lstm':
                i = 0
                for c in self.rnn_unit._children:
                    print('#   LSTM {}-th param'.format(i), file=stream)
                    print('#      0 - {}, {}'.format(c.w0.shape, c.b0.shape), file=stream) 
                    print('#      1 - {}, {}'.format(c.w1.shape, c.b1.shape), file=stream) 
                    print('#      2 - {}, {}'.format(c.w2.shape, c.b2.shape), file=stream) 
                    print('#      3 - {}, {}'.format(c.w3.shape, c.b3.shape), file=stream) 
                    print('#      4 - {}, {}'.format(c.w4.shape, c.b4.shape), file=stream) 
                    print('#      5 - {}, {}'.format(c.w5.shape, c.b5.shape), file=stream) 
                    print('#      6 - {}, {}'.format(c.w6.shape, c.b6.shape), file=stream) 
                    print('#      7 - {}, {}'.format(c.w7.shape, c.b7.shape), file=stream) 
                    i += 1
            print('# Affine layer: {}, {}'.format(self.affine.W.shape, self.affine.b.shape), file=stream)


    # create input vector
    def create_features(self, xs):
        exs = []
        for x in xs:
            if self.feat_extractor:
                emb = self.embed(x)
                feat = self.feat_extractor.extract_features(x)
                #print('emb:', type(emb.data))
                #print('feat:', type(feat.data))
                ex = F.concat((emb, feat), 1)
            else:
                #print('embed:', type(self.embed.W.data))
                ex = self.embed(x)
                
            exs.append(ex)
        xs = exs
        return xs


    # def get_id_array(self, start, width, gpu=-1):
    #     ids = np.array([], dtype=np.int32)
    #     for i in range(start, start + width):
    #         ids = np.append(ids, np.int32(i))
    #     return cuda.to_gpu(ids) if gpu >= 0 else ids


class RNN(RNNBase):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, affine_activation='identity', 
            init_embed=None, feat_extractor=None, stream=sys.stderr):
        super(RNN, self).__init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, affine_activation='identity', 
            init_embed=None, feat_extractor=None, stream=sys.stderr)

        self.loss_fun = softmax_cross_entropy.softmax_cross_entropy


    def __call__(self, xs, ts, train=True):
        with chainer.using_config('train', train):
            fs = self.create_features(xs)
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, fs)
            else:
                hy, hs = self.rnn_unit(None, fs)
            ys = [self.act(self.affine(h)) for h in hs]

        loss = None
        ps = []
        for y, t in zip(ys, ts):
            if loss is not None:
                loss += self.loss_fun(y, t)
            else:
                loss = self.loss_fun(y, t)
                ps.append([np.argmax(yi.data) for yi in y])

        return loss, ps


    def decode(self, xs):
        with chainer.no_backprop_mode():
            fs = self.create_features(xs)
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, fs)
            else:
                hy, hs = self.rnn_unit(None, fs)
            ys = [self.act(self.affine(h)) for h in hs]

        return ys


class RNN_CRF(RNNBase):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, affine_activation='identity', 
            init_embed=None, feat_extractor=None, stream=sys.stderr):
        super(RNN_CRF, self).__init__(
            n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout, rnn_unit_type, 
            rnn_bidirection, affine_activation, init_embed, feat_extractor, stream=sys.stderr)

        with self.init_scope():
            self.crf = L.CRF1d(n_labels)

            print('# CRF cost: {}\n'.format(self.crf.cost.shape), file=stream)


    def __call__(self, xs, ts, train=True):
        with chainer.using_config('train', train):
            fs = self.create_features(xs)

            # rnn layers
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, fs)
            else:
                hy, hs = self.rnn_unit(None, fs)

            # affine layer
            if not self.act or self.act == 'identity':
                hs = [self.affine(h) for h in hs]                
            else:
                hs = [self.act(self.affine(h)) for h in hs]

            # crf layer
            indices = argsort_list_descent(hs)
            trans_hs = F.transpose_sequence(permutate_list(hs, indices, inv=False))
            trans_ts = F.transpose_sequence(permutate_list(ts, indices, inv=False))
            loss = self.crf(trans_hs, trans_ts)
            score, trans_ys = self.crf.argmax(trans_hs)
            ys = permutate_list(F.transpose_sequence(trans_ys), indices, inv=True)
            ys = [y.data for y in ys]

            ################
            # loss = chainer.Variable(cuda.cupy.array(0, dtype=np.float32))
            # trans_ys = []
            # for i in range(len(hs)):
            #     hs_tmp = hs[i:i+1]
            #     ts_tmp = ts[i:i+1]
            #     t0 = datetime.now()
            #     indices = argsort_list_descent(hs_tmp)
            #     trans_hs = F.transpose_sequence(permutate_list(hs_tmp, indices, inv=False))
            #     trans_ts = F.transpose_sequence(permutate_list(ts_tmp, indices, inv=False))
            #     t1 = datetime.now()
            #     loss += self.crf(trans_hs, trans_ts)
            #     t2 = datetime.now()
            #     score, trans_y = self.crf.argmax(trans_hs)
            #     t3 = datetime.now()
            #     #trans_ys.append(trans_y)
            #     # print('  transpose     : {}'.format((t1-t0).seconds+(t1-t0).microseconds/10**6))
            #     # print('  crf forward   : {}'.format((t2-t1).seconds+(t2-t1).microseconds/10**6))
            #     # print('  crf argmax    : {}'.format((t3-t2).seconds+(t3-t2).microseconds/10**6))
            # ys = [None]
            ################

        return loss, ys
        

    def decode(self, xs):
        with chainer.no_backprop_mode():
            fs = self.create_features(xs)

            # rnn layers
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, fs)
            else:
                hy, hs = self.rnn_unit(None, fs)

            # affine layer
            if not self.act or self.act == 'identity':
                hs = [self.affine(h) for h in hs]                
            else:
                hs = [self.act(self.affine(h)) for h in hs]

            # crf layer
            indices = argsort_list_descent(hs)
            trans_hs = F.transpose_sequence(permutate_list(hs, indices, inv=False))
            score, trans_ys = self.crf.argmax(trans_hs)
            ys = permutate_list(F.transpose_sequence(trans_ys), indices, inv=True)
            ys = [y.data for y in ys]

        return ys


class SequenceTagger(chainer.link.Chain):
    compute_fscore = True

    def __init__(self, predictor, indices):
        super(SequenceTagger, self).__init__(predictor=predictor)
        self.indices = indices
        
    def __call__(self, *args, **kwargs):
        assert len(args) >= 2
        xs = args[0]
        ts = args[1]
        loss, ys = self.predictor(*args, **kwargs)

        eval_counts = None
        if self.compute_fscore:
            for x, t, y in zip(xs, ts, ys):
                generator = self.generate_lines(x, t, y)
                eval_counts = conlleval.merge_counts(eval_counts, conlleval.evaluate(generator))

        return loss, eval_counts


    def generate_lines(self, x, t, y, is_str=False):
        i = 0
        while True:
            if i == len(x):
                raise StopIteration
            x_str = str(x[i])
            t_str = t[i] if is_str else self.indices.get_label(int(t[i]))
            y_str = y[i] if is_str else self.indices.get_label(int(y[i]))

            yield [x_str, t_str, y_str]

            i += 1


    # def grow_embedding_layer(self, token2id_new):
    #     weight1 = self.predictor.embed.W
    #     diff = len(token2id_new) - len(weight1)
    #     weight2 = chainer.variable.Parameter(initializers.normal.Normal(1.0), (diff, weight1.shape[1]))
    #     weight = F.concat((weight1, weight2), 0)
    #     embed = L.EmbedID(0, 0)
    #     embed.W = chainer.Parameter(initializer=weight.data)
    #     self.predictor.embed = embed


    def grow_embedding_layer(self, id2token_all, id2token_trained={}, embed_model=None, stream=sys.stderr):
        diff = len(id2token_all) - len(id2token_trained)
        weight1 = self.predictor.embed.W
        weight2 = []

        dim = self.predictor.embed_dim
        initialW = initializers.normal.Normal(1.0)

        count = 0
        if embed_model:
            start_i = len(id2token_trained)
            for i in range(start_i, len(id2token_all)):
                key = id2token_all[i]
                    
                if key in embed_model.wv.vocab:
                    vec = embed_model.wv[key]
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
            wieght = weight2

        self.predictor.embed = L.EmbedID(0, 0)
        self.predictor.embed.W = chainer.Parameter(initializer=weight.data)

        print('Grow embedding layer: {} -> {}'.format(weight1.shape, weight.shape), file=stream)
        if count >= 1:
            print('Use %d pretrained embedding vectors'.format(count), file=stream)
        print('', file=stream)

        
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
        


def init_tagger(indices, hparams, use_gpu=False, joint_type=''):
    n_vocab = len(indices.token_indices)
    n_labels = len(indices.label_indices)

    if hparams['use_dict_feature']:
        feat_extractor = features.DictionaryFeatureExtractor(indices, use_gpu=use_gpu)
    else:
        feat_extractor = None

    if joint_type:
        if joint_type == 'lattice':
            rnn = models_joint.RNN_LatticeCRF(
                hparams['rnn_layers'], n_vocab, hparams['embed_dimension'], 
                hparams['rnn_hidden_units'], n_labels, indices, dropout=hparams['dropout'],
                rnn_unit_type=hparams['rnn_unit_type'], rnn_bidirection=hparams['rnn_bidirection'], 
                feat_extractor=feat_extractor) #gpu=gpu

        elif hparams['joint_type'] == 'dual_rnn':
            rnn = None
            print('Not implemented yet', file=sys.stderr)
            sys.exit()

        tagger = models_joint.JointMorphologicalAnalyzer(rnn, indices.id2label)

    else:
        if hparams['inference_layer'] == 'crf':
            rnn = RNN_CRF(
                hparams['rnn_layers'], n_vocab, hparams['embed_dimension'], 
                hparams['rnn_hidden_units'], n_labels, dropout=hparams['dropout'], 
                rnn_unit_type=hparams['rnn_unit_type'], rnn_bidirection=hparams['rnn_bidirection'], 
                feat_extractor=feat_extractor)

        else:
            rnn = RNN(
                hparams['rnn_layers'], n_vocab, hparams['embed_dimension'], 
                hparams['rnn_hidden_units'], n_labels, dropout=hparams['dropout'], 
                rnn_unit_type=hparams['rnn_unit_type'], rnn_bidirection=hparams['rnn_bidirection'], 
                feat_extractor=feat_extractor)

        tagger = SequenceTagger(rnn, indices)

    return tagger
    

def get_activation(activation):
    if not activation  or activation == 'identity':
        return F.identity
    elif activation == 'relu':
        return F.relu
    elif activation == 'tanh':
        return F.tanh
    elif activation == 'sigmoid':
        return F.sigmoid
    else:
        return


def add(var_list1, var_list2):
    len_list = len(var_list1)
    ret = [None] * len_list
    for i, var1, var2 in zip(range(len_list), var_list1, var_list2):
        ret[i] = var1 + var2
    return ret
