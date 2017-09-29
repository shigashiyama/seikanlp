import sys
import logging
import argparse

import gensim

import eval_embedding as eval

def get_sentence_iterator(num_iter, path, file_format, seq_type='w', stoplist=[]):
    if seq_type == 'w':
        pass
    elif seq_type == 'word':
        seq_type = 'w'
    else:
        seq_type = 'c'

    if file_format == 'bccwj':
        return WordSeqIteratorBCCWJ(num_iter, path, seq_type, stoplist)
    else:
        return


class WordSeqIteratorBCCWJ(object):
    def __init__(self, num_iter, path, seq_type, stoplist):
        self.max_num_iter = num_iter
        self.iter = 0
        self.path = path
        self.seq_type = seq_type
        self.stoplist = stoplist
        self.reinit()

    def reinit(self):
        self.f = open(self.path)
        self.bof = True
        self.line = ''
        self.sen = []

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.iter == self.max_num_iter:
                self.f.close()
                raise StopIteration

            while self.line or self.bof:
                self.line = self.f.readline()
                if len(self.line) < 2:
                    continue

                arr = self.line.rstrip('\n').split('\t')
                bos = arr[2]
                word = arr[5]

                if not self.bof and bos == 'B':
                    self.bof = False
                    ret = self.sen
                    self.sen = []
                    add_token(self.sen, word, self.seq_type, self.stoplist)
                    return ret
                else:
                    self.bof = False
                    add_token(self.sen, word, self.seq_type, self.stoplist)

            self.iter += 1
            ret = self.sen
            self.reinit()
            return ret


def add_token(sen, word, seq_type, stoplist):
    if word in stoplist:
        return
    
    if seq_type == 'w':
        sen.append(word)
    else:
        sen.extend([word[i] for i in range(len(word))])


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', help='directory or file path of input data')
    parser.add_argument('--format', '-f', default='')
    parser.add_argument('--sequence_type', '-t', default='word')
    parser.add_argument('--out_path', '-o', default='')
    parser.add_argument('--num_iter', '-n', type=int, default=5)
    parser.add_argument('--dimension', '-d', type=int, default=300)
    parser.add_argument('--context_size', '-c', type=int, default=5)
    parser.add_argument('--min_count', '-m', type=int, default=5)
    parser.add_argument('--max_vocab_size', '-v', type=int, default=-1)
    args = parser.parse_args()
    print(args)

    stoplist = ['\u3000']
    sentences = get_sentence_iterator(args.num_iter+1, args.input_path, args.format, 
                                      args.sequence_type, stoplist)
    
    # for sen in sentences:
    #     print(sen)
    # sys.exit()
    
    model = gensim.models.Word2Vec(sentences,
                                   sg=1,
                                   size=args.dimension,
                                   window=args.context_size,
                                   min_count=args.min_count,
                                   max_vocab_size=args.max_vocab_size if args.max_vocab_size > 0 else None,
                                   # sample=args.th_downsampling, # default 1e-5       
                                   # negative=arg.num_negative,   # default 5
                                   # iter=args.num_epoch,         # default 5
    )
    model.wv.save_word2vec_format(args.out_path, binary=True)

    print('save the model:', args.out_path, '\n')

    eval.eval_jp_data(model)

    # print(model.wv.syn0[0].shape)
    # for i in range(3):
    #     print(model.wv.index2word[i], model.wv.syn0[i])

