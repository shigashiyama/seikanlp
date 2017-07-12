import gensim
from gensim.models import keyedvectors
from gensim.models import word2vec

def eval_jp_data(model):
    data_dir = '/home/shigashi/data_shigashi/resources/word_embedding/'
    test_jws0 = data_dir + 'JapaneseWordSimilarityDataset/processed/all.tsv'
    test_jws1 = data_dir + 'JapaneseWordSimilarityDataset/processed/score_noun.tsv'
    test_jws2 = data_dir + 'JapaneseWordSimilarityDataset/processed/score_adj.tsv'
    test_jws3 = data_dir + 'JapaneseWordSimilarityDataset/processed/score_adv.tsv'
    test_jws4 = data_dir + 'JapaneseWordSimilarityDataset/processed/score_verb.tsv'
    tests = [test_jws0, test_jws1, test_jws2, test_jws3, test_jws4]
    
    for test in tests:
        tup = model.wv.evaluate_word_pairs(test)
        print('#', test)
        print(tup[1])

def eval_en_data(model):
    data_dir = '/home/shigashi/data_shigashi/resources/word_embedding/'
    test_ws = data_dir + 'wordsim353/combined.tab'
    test_vsd = data_dir + 'VerbSimilarityDataset/verb_similarity_dataset.tsv'
    test_men = data_dir + 'MEN/natural_form_full.tsv'
    test_sl0 = data_dir + 'SimLex-999/processed/all.tsv'
    test_sl1 = data_dir + 'SimLex-999/processed/n.tsv'
    test_sl2 = data_dir + 'SimLex-999/processed/v.tsv'
    test_sl3 = data_dir + 'SimLex-999/processed/a.tsv'
    test_sv0 = data_dir + 'simverb-3500/processed/all.tsv'
    test_sv1 = data_dir + 'simverb-3500/processed/none.tsv'
    test_sv2 = data_dir + 'simverb-3500/processed/antonyms.tsv'
    test_sv3 = data_dir + 'simverb-3500/processed/cohyponyms.tsv'
    test_sv4 = data_dir + 'simverb-3500/processed/hypernyms.tsv'
    test_sv5 = data_dir + 'simverb-3500/processed/synonyms.tsv'
    tests = [test_ws, test_vsd, test_men, test_sl0, test_sl1, test_sl2, test_sl3, test_sv0, test_sv1, test_sv2, test_sv4, test_sv5]

    for test in tests:
        tup = model.wv.evaluate_word_pairs(test)
        print('#', test)
        print(tup[1])

def eval_zh_data(model):
    test_cws240 = data_dir + 'CWE/wordsim240.txt'
    test_cws297 = data_dir + 'CWE/wordsim297.txt'
    
    tests = [test_cws240, test_cws247]
    for test in tests:
        tup = model.wv.evaluate_word_pairs(test)
        print('#', test)
        print(tup[1])

if __name__ == '__main__':
    model_dir = 'embed_model/public/'
    
    ### load JP model
    model = word2vec.Word2Vec.load(model_dir + 'word2vec_shiroyagi/word2vec.gensim.model')
    # model = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'tohoku_entity_vector/entity_vector.model.bin', binary=True)
    # model = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'fasttext/wiki.ja.bin', binary=True)
    # model = keyedvectors.KeyedVectors.load_word2vec_format('embed_model/bccwj-lb.txt', binary=False)

    ### load EN model
    # model = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    # model = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'fasttext/wiki.en.bin', binary=True)
    # model = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'glove/glove.6B.300d.txt', binary=False)
    print('load model')

    ### load CH model
    # model1 = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'CWE/skipgram.7z.001', binary=True)
    # model2 = keyedvectors.KeyedVectors.load_word2vec_format(model_dir + 'CWE/skipgram.7z.002', binary=True)

    ### save model
    # model.wv.save_word2vec_format(model_dir + 'glove/glove.42B.300d.bin', binary=True)

    ### display
    # model.wv['日本']
    # model.wv.most_similar(positive=['日本', 'パリ'], negative=['東京'])
    # model.wv.most_similar(positive=['神奈川', '前橋'], negative=['横浜'])

    models = [model]
    for model in models:
        eval_jp_data(model)
        # eval_en_data(model)
        print()
        
