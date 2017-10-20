import numpy as np
import chainer
from chainer import cuda


class FeatureExtractor():
    def __init__(self):
        self.dim = 0


class DictionaryFeatureExtractor(FeatureExtractor):
    def __init__(self, dic, maxlen=10, use_gpu=False):
        super(DictionaryFeatureExtractor).__init__()
        self.dic = dic
        self.maxlen = maxlen
        self.dim = 5 * (self.maxlen - 1)
        # self.dim = 5 * self.maxlen
        self.use_gpu = use_gpu

    # r_n - 右隣が長さ n の単語
    # l_n - 左隣が長さ n の単語
    # b_n - 現在の文字が長さ n の単語の先頭
    # e_n - 現在の文字が長さ n の単語の末尾
    # i_n - 現在の文字が長さ n の単語の内部
    def extract_features(self, x):
        sen_len = len(x)
        found = []
        for i in range(sen_len):
            res = self.dic.chunk_trie.common_prefix_search(x, i, i + self.maxlen)
            found.extend(res)

        if self.use_gpu:
            xp = cuda.cupy
        else:
            xp = np
        feat = xp.zeros((len(x), self.dim), dtype='f')

        # print('----')
        # print(len(x), x)
        # print([self.dic.get_token(xi) for xi in x])
        # print(found)
        # print(feat.shape, feat)
            
        for b, e in found:
            length = e - b
            # ほとんどの文字が長さ1の単語として登録されているので、素性に入れない
            if length == 1 or self.maxlen < length:
                continue
            # if self.maxlen < length:
            #     continue

            # wid = self.dic.chunk_trie.get_word_id(x[b:e])
            # w = self.dic.id2chunk[wid]
            # print((b, e), ':', w)

            # 添字に変換
            maxlen_cnv = self.maxlen - 1
            length_cnv = length - 2
            # length_cnv = length - 1
            
            # r_length
            if b > 0:
                feat[b-1][length_cnv] = 1

            # b_length
            feat[b][maxlen_cnv + length_cnv] = 1

            # i_length
            for k in range(b+1, e-1):
                feat[k][2 * maxlen_cnv + length_cnv] = 1

            # e_length
            feat[e-1][3 * maxlen_cnv + length_cnv] = 1

            # l_length
            if e < sen_len:
                feat[e][4 * maxlen_cnv + length_cnv] = 1

        feat = chainer.Variable(feat)
        return feat
