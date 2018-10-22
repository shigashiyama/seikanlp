import sys
from enum import Enum, auto

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from chainer import cuda


class ModelUsage(Enum):
    NONE = auto()
    ADD = auto()
    CONCAT = auto()
    INIT = auto()

    def get_instance(key):
        if key.lower() == 'concat':
            return ModelUsage.CONCAT
        elif key.lower() == 'add':
            return ModelUsage.ADD
        elif key.lower() == 'init':
            return ModelUsage.INIT
        else:
            return ModelUsage.NONE


def construct_embeddings(n_vocab, rand_dim, pretrained_dim=0, usage=ModelUsage.ADD):
    if pretrained_dim <= 0 or usage == ModelUsage.NONE:
        rand_model = L.EmbedID(n_vocab, rand_dim)
        pretrained_model = None

    elif usage == ModelUsage.CONCAT or usage == ModelUsage.ADD:
        rand_model = L.EmbedID(n_vocab, rand_dim)
        pretrained_model = L.EmbedID(n_vocab, pretrained_dim)

    elif usage == ModelUsage.INIT:
        rand_model = L.EmbedID(n_vocab, pretrained_dim)
        pretrained_model = None

    return rand_model, pretrained_model


def construct_RNN(unit_type, bidirection, n_layers, n_input, n_units, dropout, file=sys.stderr):
    rnn = None
    if unit_type == 'lstm':
        if bidirection:
            rnn = L.NStepBiLSTM(n_layers, n_input, n_units, dropout)
        else:
            rnn = L.NStepLSTM(n_layers, n_input, n_units, dropout)
    elif unit_type == 'gru':
        if bidirection:
            rnn = L.NStepBiGRU(n_layers, n_input, n_units, dropout) 
        else:
            rnn = L.NStepGRU(n_layers, n_input, n_units, dropout)
    else:
        if bidirection:
            rnn = L.NStepBiRNNTanh(n_layers, n_input, n_units, dropout) 
        else:
            rnn = L.NStepRNNTanh(n_layers, n_input, n_units, dropout)

    print('# RNN unit: {}, dropout={}'.format(rnn, rnn.__dict__['dropout']), file=file)
    for i, c in enumerate(rnn._children):
        print('#   {}-th param'.format(i), file=file)
        print('#      0 - W={}, b={}'.format(c.w0.shape, c.b0.shape), file=file) 
        print('#      1 - W={}, b={}'.format(c.w1.shape, c.b1.shape), file=file) 

        if unit_type == 'gru' or unit_type == 'lstm':
            print('#      2 - W={}, b={}'.format(c.w2.shape, c.b2.shape), file=file) 
            print('#      3 - W={}, b={}'.format(c.w3.shape, c.b3.shape), file=file) 
            print('#      4 - W={}, b={}'.format(c.w4.shape, c.b4.shape), file=file) 
            print('#      5 - W={}, b={}'.format(c.w5.shape, c.b5.shape), file=file) 

        if unit_type == 'lstm':
            print('#      6 - W={}, b={}'.format(c.w6.shape, c.b6.shape), file=file) 
            print('#      7 - W={}, b={}'.format(c.w7.shape, c.b7.shape), file=file) 

    return rnn


def load_pretrained_embedding_layer(id2unigram, embed, external_model, finetuning=False):
    xp = cuda.get_array_module(embed.W)

    n_vocab = len(id2unigram)
    dim = external_model.wv.syn0[0].shape[0]
    initialW = initializers.normal.Normal(1.0)

    weight = []
    count = 0

    for i in range(n_vocab):
        key = id2unigram[i]
        if key in external_model.wv.vocab:
            vec = external_model.wv[key]
            count += 1
        else:
            if finetuning:
                vec = initializers.generate_array(initialW, (dim, ), xp)
            else:
                vec = xp.zeros(dim, dtype='f')
        weight.append(vec)

    weight = xp.reshape(weight, (n_vocab, dim))
    embed.W = chainer.Parameter(initializer=weight)

    if count >= 1:
        print('Use {} pretrained embedding vectors\n'.format(count), file=sys.stderr)


def grow_embedding_layers(
        n_vocab_org, n_vocab_grown, rand_embed, 
        pretrained_embed=None, external_model=None, id2unigram_grown=None, 
        pretrained_model_usage=ModelUsage.NONE, train=False, file=sys.stderr):
    if n_vocab_org == n_vocab_grown:
        return

    if external_model and pretrained_model_usage != ModelUsage.NONE:
        if pretrained_model_usage == ModelUsage.INIT:
            grow_embedding_layer_with_pretrained_model(
                n_vocab_org, n_vocab_grown, rand_embed,
                external_model, id2unigram_grown, train=train, file=file)

        else:
            grow_embedding_layers_with_pretrained_model(
                n_vocab_org, n_vocab_grown, rand_embed, pretrained_embed, 
                external_model, id2unigram_grown, train=train, file=file)

    else:
        grow_embedding_layer_without_pretrained_model(n_vocab_org, n_vocab_grown, rand_embed)
        

def grow_embedding_layer_without_pretrained_model(
        n_vocab_org, n_vocab_grown, rand_embed, train=False, file=sys.stderr):
    xp = cuda.get_array_module(rand_embed.W)
    diff = n_vocab_grown - n_vocab_org
    d_rand = rand_embed.W.shape[1]

    if train:
        w2_rand = initializers.generate_array(initialW, (diff, d_rand), xp)
    else:
        w2_rand = xp.zeros((diff, d_rand), dtype='f')

    w_rand = F.concat((rand_embed.W, w2_rand), axis=0)
    rand_embed.W = chainer.Parameter(initializer=w_rand.data, name='W')
    print('Grow embedding matrix: {} -> {}'.format(n_vocab_org, rand_embed.W.shape[0]), file=file)


# rand model       -> grow using external model
def grow_embedding_layer_with_pretrained_model(
        n_vocab_org, n_vocab_grown, rand_embed, 
        external_model, id2unigram_grown, train=False, file=sys.stderr):
    diff = n_vocab_grown - n_vocab_org
    d_rand = rand_embed.W.shape[1]

    count = 0
    initialW = initializers.normal.Normal(1.0)
    w2_rand = []

    for i in range(n_vocab_org, n_vocab_grown):
        key = id2unigram_grown[i]
        if key in external_model.wv.vocab:
            vec_rand = external_model.wv[key]
            count += 1
        elif train:
            vec_rand = initializers.generate_array(initialW, (d_rand, ), np)
        else:
            vec_rand = rand_embed.W[0].data # use pretrained vector of unknown token
        w2_rand.append(vec_rand)

    w2_rand = np.reshape(w2_rand, (diff, d_rand))
    if cuda.get_array_module(rand_embed.W) == cuda.cupy:
        w2_rand = chainer.Variable(w2_rand)
        w2_rand.to_gpu()

    w_rand = F.concat((rand_embed.W, w2_rand), axis=0)
    rand_embed.W = chainer.Parameter(initializer=w_rand.data, name='W')

    print('Grow embedding matrix: {} -> {}'.format(n_vocab_org, rand_embed.W.shape[0]), file=file)
    if count >= 1:
        print('Add {} pretrained embedding vectors'.format(count), file=file)


# rand model       -> grow 
# pretrained model -> grow using external model
def grow_embedding_layers_with_pretrained_model(
        n_vocab_org, n_vocab_grown, rand_embed, pretrained_embed, 
        external_model, id2unigram_grown, train=False, file=sys.stderr):
    diff = n_vocab_grown - n_vocab_org
    d_rand = rand_embed.W.shape[1]
    d_pretrained = pretrained_embed.W.shape[1] #external_model.wv.syn0[0].shape[0]

    count = 0
    initialW = initializers.normal.Normal(1.0)
    w2_rand = []
    w2_pretrained = []

    for i in range(n_vocab_org, n_vocab_grown):
        if train:               # resume training
            vec_rand = initializers.generate_array(initialW, (d_rand, ), np)
        else:                   # test
            vec_rand = rand_embed.W[0].data # use pretrained vector of unknown token
        w2_rand.append(vec_rand)

        key = id2unigram_grown[i]
        if key in external_model.wv.vocab:
            vec_pretrained = external_model.wv[key]
            count += 1
        else:
            vec_pretrained = np.zeros(d_pretrained, dtype='f')
        w2_pretrained.append(vec_pretrained)

    w2_rand = np.reshape(w2_rand, (diff, d_rand))
    if cuda.get_array_module(rand_embed.W) == cuda.cupy:
        w2_rand = chainer.Variable(w2_rand)
        w2_rand.to_gpu()
    w_rand = F.concat((rand_embed.W, w2_rand), axis=0)
    rand_embed.W = chainer.Parameter(initializer=w_rand.data, name='W')

    w2_pretrained = np.reshape(w2_pretrained, (diff, d_pretrained))
    if cuda.get_array_module(rand_embed.W) == cuda.cupy:
        w2_pretrained = chainer.Variable(w2_pretrained)
        w2_pretrained.to_gpu()
    w_pretrained = F.concat((pretrained_embed.W, w2_pretrained), 0)
    pretrained_embed.W = chainer.Parameter(initializer=w_pretrained.data, name='W')

    print('Grow embedding matrix: {} -> {}'.format(n_vocab_org, rand_embed.W.shape[0]), file=file)
    print('Grow pretrained embedding matrix: {} -> {}'.format(
        n_vocab_org, pretrained_embed.W.shape[0]), file=file)
    if count >= 1:
        print('Add {} pretrained embedding vectors'.format(count), file=file)


# def grow_affine_layer(n_labels_org, n_labels_grown, affine, file=sys.stderr):
#     diff = n_labels_grown - n_labels_org
#     if diff <= 0:
#         return

#     w_org = predictor.affine.W
#     b_org = predictor.affine.b
#     w_org_shape = w_org.shape
#     b_org_shape = b_org.shape

#     dim = w_org.shape[1]
#     initialW = initializers.normal.Normal(1.0)

#     w_diff = chainer.Parameter(initialW, (diff, dim))
#     w_new = F.concat((w_org, w_diff), 0)

#     b_diff = chainer.Parameter(initialW, (diff,))
#     b_new = F.concat((b_org, b_diff), 0)
            
#     affine.W = chainer.Parameter(initializer=w_new.data, name='W')
#     affine.b = chainer.Parameter(initializer=b_new.data, name='b')

#     print('Grow affine layer: {}, {} -> {}, {}'.format(
#         w_org_shape, b_org_shape, affine.W.shape, affine.b.shape), file=file)
    

def grow_crf_layer(n_labels_org, n_labels_grown, crf, file=sys.stderr):
    xp = cuda.get_array_module(crf.cost)
    diff = n_labels_grown - n_labels_org
    if diff <= 0:
        return

    c_org = crf.cost
    c_diff1 = chainer.Parameter(xp.zeros((n_labels_org, diff), dtype=np.float32))
    c_diff2 = chainer.Parameter(xp.zeros((diff, n_labels_grown), dtype=np.float32))
    c_tmp = F.concat((c_org, c_diff1), 1)
    c_new = F.concat((c_tmp, c_diff2), 0)
    crf.cost = chainer.Parameter(initializer=c_new.data, name='cost')
    
    print('Grow CRF layer: {} -> {}'.format(c_org.shape, crf.cost.shape, file=file))


def grow_MLP(n_labels_org, n_labels_grown, out_layer, file=sys.stderr):
    xp = cuda.get_array_module(out_layer.W)
    diff = n_labels_grown - n_labels_org
    if diff <= 0:
        return

    w_org = out_layer.W
    w_org_shape = w_org.shape

    dim = w_org.shape[1]
    w_diff_array = xp.zeros((diff, dim), dtype=np.float32)
    w_diff_array[:] =xp.random.normal(scale=1.0, size=(diff, dim))
    w_diff = chainer.Parameter(w_diff_array)
    w_new = F.concat((w_org, w_diff), 0)
    out_layer.W = chainer.Parameter(initializer=w_new.data)
    w_shape = out_layer.W.shape

    if 'b' in out_layer.__dict__:
        b_org = out_layer.b
        b_org_shape = b_org.shape
        b_diff_array = xp.zeros((diff,), dtype=np.float32)
        b_diff_array[:] = xp.random.normal(scale=1.0, size=(diff,))
        b_diff = chainer.Parameter(b_diff_array)
        b_new = F.concat((b_org, b_diff), 0)
        out_layer.b = chainer.Parameter(initializer=b_new.data)
        b_shape = out_layer.b.shape
    else:
        b_org_shape = b_shape = None

    print('Grow MLP output layer: {}, {} -> {}, {}'.format(
        w_org_shape, b_org_shape, w_shape, b_shape), file=file)


def grow_biaffine_layer(n_labels_org, n_labels_grown, biaffine, file=sys.stderr):
    xp = cuda.get_array_module(biaffinie.W)
    diff = n_labels_grown - n_labels_org
    if diff <= 0:
        return

    w_org = affine.W
    w_org_shape = w_org.shape

    dim = w_org.shape[1]

    w_diff_array = xp.zeros((diff, dim), dtype=np.float32)
    w_diff_array[:] =xp.random.normal(scale=1.0, size=(diff, dim))
    w_diff = chainer.Parameter(w_diff_array)
    w_new = F.concat((w_org, w_diff), 0)
    affine.W = chainer.Parameter(initializer=w_new.data)
    w_shape = affine.W.shape

    if 'b' in affine.__dict__:
        b_org = affine.b
        b_org_shape = b_org.shape
        b_diff_array = xp.zeros((diff,), dtype=np.float32)
        b_diff_array[:] = xp.random.normal(scale=1.0, size=(diff,))
        b_diff = chainer.Parameter(b_diff_array)
        b_new = F.concat((b_org, b_diff), 0)
        affine.b = chainer.Parameter(initializer=b_new.data)
        b_shape = affine.b.shape
    else:
        b_org_shape = b_shape = None

    print('Grow biaffine layer: {}, {} -> {}, {}'.format(
        w_org_shape, b_org_shape, w_shape, b_shape), file=file)
