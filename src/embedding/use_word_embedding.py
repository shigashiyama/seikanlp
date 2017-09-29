import gensim
from gensim.models import keyedvectors
from gensim.models import word2vec

if __name__ == '__main__':
    model_dir = 'embed_model/public/'
    
    ### load JP model
    # model = word2vec.Word2Vec.load(model_dir + 'word2vec_shiroyagi/word2vec.gensim.model')
    # model = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'tohoku_entity_vector/entity_vector.model.bin', binary=True)
    # model = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'fasttext/wiki.ja.bin', binary=True)
    # model = keyedvectors.KeyedVectors.load_word2vec_format('embed_model/bccwj-lb.txt', binary=False)

    ### load EN model
    model = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    # model = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'fasttext/wiki.en.bin', binary=True)
    # model = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'glove/glove.6B.300d.txt', binary=False)
    print('load model')

    ### load CH model
    # model1 = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'CWE/skipgram.7z.001', binary=True)
    # model2 = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'CWE/skipgram.7z.002', binary=True)

    ### save model
    model.wv.save_word2vec_format(model_dir + 'word2vec/GoogleNews-vectors-negative300.txt', binary=False)
