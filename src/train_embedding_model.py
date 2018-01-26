import sys
import copy
import logging
import argparse
import itertools

import gensim


def train_model():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', required=True, help='directory or file path of input data')
    parser.add_argument('--out_path', '-o', required=True, default='')
    parser.add_argument('--num_iter', '-n', type=int, default=5)
    parser.add_argument('--dimension', '-d', type=int, default=300)
    parser.add_argument('--context_size', '-c', type=int, default=5)
    parser.add_argument('--min_count', '-m', type=int, default=5)
    parser.add_argument('--max_vocab_size', '-v', type=int, default=-1)
    args = parser.parse_args()
    print(args, '\n', file=sys.stderr)
    
    #stoplist = ['\u3000']

    sentence = gensim.models.word2vec.LineSentence(args.input_path, max_sentence_length=10000)
    model = gensim.models.Word2Vec(
        sentences=sentence,
        sg=1,
        size=args.dimension,
        window=args.context_size,
        min_count=args.min_count,
        max_vocab_size=args.max_vocab_size if args.max_vocab_size > 0 else None,
        # sample=args.th_downsampling, # default 1e-5
        # negative=arg.num_negative,   # default 5
        iter=args.num_iter,     # default 5
        compute_loss=True,
    )
    print('ave loss:', model.running_training_loss / (args.num_iter * len(model.wv.vocab)),
          file=sys.stderr)
    save_model(model, args.out_path, binary=True)


def load_model(model_path):
    if model_path.endswith('model'):
        model = gensim.models.word2vec.Word2Vec.load(model_path)        
    elif model_path.endswith('bin'):
        model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(model_path, binary=True)
    elif model_path.endswith('txt'):
        model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(model_path, binary=False)
    else:
        print("unsuported format of word embedding model.", file=sys.stderr)
        return

    print("load embedding model: vocab=%d" % len(model.wv.vocab))
    return model


def save_model(model, out_path, binary=True):
    model.wv.save_word2vec_format(out_path, binary=binary)
    print('save model to {}\n'.format(out_path), file=sys.stderr)


def shrink_model(model, size=100000):
    smodel = copy.deepcopy(model)
    delkeys = smodel.wv.index2word[size:]
    smodel.wv.index2word = model.wv.index2word[:size]
    smodel.wv.syn0 = smodel.wv.syn0[:size]
    for key in delkeys:
        del(smodel.wv.vocab[key])
    return smodel


if __name__ == '__main__':
    train_model()

    # in_path = sys.argv[1]
    # out_path = sys.argv[2]
    # model = load_model(in_path)
    # model = shrink_model(model)
    # save_model(model, out_path, binary=False)
