import sys
import copy
import logging
import argparse
import itertools

from gensim.models import keyedvectors, Word2Vec, FastText
from gensim.models.word2vec import LineSentence


def train_model():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', required=True, help='directory or file path of input data')
    parser.add_argument('--out_path', '-o', required=True, default='')
    parser.add_argument('--model_type', '-t', default='word2vec')
    parser.add_argument('--num_iter', '-n', type=int, default=5)
    parser.add_argument('--dimension', '-d', type=int, default=300)
    parser.add_argument('--context_size', '-c', type=int, default=5)
    parser.add_argument('--min_count', '-m', type=int, default=5)
    parser.add_argument('--max_vocab_size', '-v', type=int, default=-1)
    parser.add_argument('--min_n', type=int, default=2)
    parser.add_argument('--max_n', type=int, default=6)
    args = parser.parse_args()
    print(args, '\n', file=sys.stderr)
    
    sentence = LineSentence(args.input_path, max_sentence_length=10000)
    args.model_type = args.model_type.lower()
    if args.model_type == 'fasttext':
        model = FastText(
            sentences=sentence,
            sg=1,
            size=args.dimension,
            window=args.context_size,
            min_count=args.min_count,
            max_vocab_size=args.max_vocab_size if args.max_vocab_size > 0 else None,
            min_n=args.min_n,
            max_n=args.max_n,
            # sample=args.th_downsampling, # default 1e-5
            # negative=arg.num_negative,   # default 5
            iter=args.num_iter,     # default 5
        )
    else:
        model = Word2Vec(
            sentences=sentence,
            sg=1,
            size=args.dimension,
            window=args.context_size,
            min_count=args.min_count,
            max_vocab_size=args.max_vocab_size if args.max_vocab_size > 0 else None,
            # sample=args.th_downsampling, # default 1e-5
            # negative=arg.num_negative,   # default 5
            iter=args.num_iter,     # default 5
        )
    save_model(model, args.out_path, args.model_type, binary=True)


def load_model(model_path):
    # Word2Vec
    if model_path.endswith('bin'):
        model = keyedvectors.KeyedVectors.load_word2vec_format(model_path, binary=True)
    elif model_path.endswith('txt') or model_path.endswith('vec'):
        model = keyedvectors.KeyedVectors.load_word2vec_format(model_path, binary=False)

    # FastText
    elif model_path.endswith('ftx'):
        model = FastText.load(model_path)

    else:
        print("unsuported format of word embedding model.", file=sys.stderr)
        return

    print("load embedding model: vocab=%d" % len(model.wv.vocab))
    return model


def save_model(model, out_path, model_type, binary=True):
    if model_type == 'word2vec':
        model.wv.save_word2vec_format(out_path, binary=binary)

    elif model_type == 'fasttext':
        model.save(out_path)

    print('save model to {}\n'.format(out_path), file=sys.stderr)


def shrink_model(model, size=100000): # for word2vec
    smodel = copy.deepcopy(model)
    delkeys = smodel.wv.index2word[size:]
    smodel.wv.index2word = model.wv.index2word[:size]
    smodel.wv.syn0 = smodel.wv.syn0[:size]
    for key in delkeys:
        del(smodel.wv.vocab[key])
    return smodel


if __name__ == '__main__':
    train_model()
