import sys
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

    model.wv.save_word2vec_format(args.out_path, binary=True)
    print('save the model: {}\n'.format(args.out_path), file=sys.stderr)


if __name__ == '__main__':
    train_model()
