import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import initializers
from chainer import cuda

import gensim
from gensim.models import keyedvectors
from gensim.models import word2vec

from util import UNK_SYMBOL


def read_model(model_path):
    if model_path.endswith('model'):
        model = word2vec.Word2Vec.load(model_path)        
    elif model_path.endswith('bin'):
        model = keyedvectors.KeyedVectors.load_word2vec_format(model_path, binary=True)
    elif model_path.endswith('txt'):
        model = keyedvectors.KeyedVectors.load_word2vec_format(model_path, binary=False)
    else:
        print("unsuported format of word embedding model.")
        return

    print("load embedding model: vocab=%d" % len(model.wv.vocab))
    return model


def construct_lookup_table(id2token, model, gpu=-1):
    n_vocab = len(id2token)
    dim = model.wv.syn0[0].shape[0] #model.wv.vector_size
    initialW = initializers.normal.Normal(1.0)
    weight = []

    for i in range(len(id2token)):
        key = id2token[i]

        if key in model.wv.vocab:
            vec = model.wv[key]
            weight.append(vec)
        else:
            vec = initializers.generate_array(initialW, (dim, ), np)
            weight.append(vec)

    if gpu >= 0:
        weight = cuda.to_gpu(weight, device=gpu)
    else:
        weight = np.reshape(weight, (n_vocab, dim))
        
    embed = L.EmbedID(0, 0)
    embed.W = chainer.Parameter(initializer=weight)
    
    print("construct embedding matrix:", embed.W.shape)

    return embed


# def create_initial_vocab(model):
#     words = model.wv.index2word
#     token2id = {UNK_SYMBOL: np.int32(0)}
#     token2id.update(
#         {(i+1):words[i] for i in range(len(words))}
#     )
#     print("create vocab: size=%d" % len(model.wv.vocab))
#     return token2id
        

# deprecated
def read_weight_and_dict(model_path):
    model = read_model(model_path)
    token2id = {UNK_SYMBOL: np.int32(0)}
    weight = []
    n_vocab = len(model.wv.vocab)

    index = 1
    for key in model.wv.vocab.keys():
        token2id[key] = index
        vec = model.wv[key]
        weight.append(vec)
        index += 1
    weight = np.reshape(weight, (n_vocab, model.wv.syn0[0].shape))

    print("reshape embedding model:", weight.shape)

    return token2id, weight
        
# deprecated
# add new dimensions with random initialization for new tokens in token2id
def grow_lookup_table(weight, token2id):
    diff = len(token2id) - len(weight)
    initialW = normal.Normal(1.0)
    weight2 = chainer.variable.Parameter(initialW, (diff, weight.shape[1]))
    weight = F.concat((weight, weight2), 0)
    embed = L.EmbedID(0, 0)
    embed.W = chainer.Parameter(initializer=weight)

    print("construct embedding matrix:", embed.W.shape)

    return embed


def save_shrinked_model(size, model, out_path):
    delkeys = model.wv.index2word[size:]
    model.wv.index2word = model.wv.index2word[:size]
    model.wv.syn0 = model.wv.syn0[:size]
    for key in delkeys:
        del(model.wv.vocab[key])
    print(model.wv.syn0.shape)
    out_path = 'embed_model/word2vec/GoogleNews-vectors-negative300_100k.bin'
    model.save_word2vec_format(out_path, binary=True)
    

if __name__ == '__main__':
    # model_path = 'embed_model/word2vec/GoogleNews-vectors-negative300.bin'
    # model_path = 'embed_model/word2vec/GoogleNews-vectors-negative300_100k.bin'
    # model_path = 'embed_model/tohoku_entity_vector/entity_vector.model.bin'
    # model_path = 'embed_model/word2vec_shiroyagi/word2vec.gensim.model'
    model_path = 'embed_model/CWS_mzhang/char.txt'

    model = read_model(model_path)
