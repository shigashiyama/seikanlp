import numpy as np
import chainer
from chainer import cuda

import constants
    

class FeatureExtractor():
    def __init__(self):
        self.dim = 0

"""
Features for word segmentation
  R_n-m - fires if there is a word whose length is in [n,m] on the right side
  L_n-m - fires if there is a word whose length is in [n,m] on the left side
  B_n-m - fires if current character is the beginning of a word whose length is in [n,m]
  E_n-m - fires if current character is the end of a word whose length is in [n,m]
  I_n-m - fires if current character is inside of a word whose length is in [n,m]

Example: seg:L:2-3,4-5,6-10
  segmentation features of L_2-3, L_4-5, L_6-10 are created  
"""
class DictionaryFeatureExtractor(FeatureExtractor):
    def __init__(self, template, use_gpu=False):
        super(DictionaryFeatureExtractor).__init__()
        self.use_gpu = use_gpu

        if template == 'default':
            self.dim = 15
            self.max_len = 10
            self.seg_lg = {              # length groups for each position
                constants.L: {2:0, 3:0, 4:1, 5:1, 6:2, 7:2, 8:2, 9:2, 10:2},
                constants.R: {2:0, 3:0, 4:1, 5:1, 6:2, 7:2, 8:2, 9:2, 10:2},
                constants.I: {2:0, 3:0, 4:1, 5:1, 6:2, 7:2, 8:2, 9:2, 10:2},
                constants.B: {2:0, 3:0, 4:1, 5:1, 6:2, 7:2, 8:2, 9:2, 10:2},
                constants.E: {2:0, 3:0, 4:1, 5:1, 6:2, 7:2, 8:2, 9:2, 10:2}
            }
            self.seg_starts = {          # start id for each position
                constants.L: 0,
                constants.R: 3,
                constants.I: 6,
                constants.B: 9,
                constants.E: 12
            }

        else:
            self.read_template(template)        


    def read_template(self, template_file):
        valid_positions = [constants.L, constants.R, constants.I,constants.B, constants.E]

        self.max_len = 0
        self.seg_lg = {              # length groups for each position
            constants.L: {},
            constants.R: {},
            constants.I: {},
            constants.B: {},
            constants.E: {}
        }
        self.seg_starts = {          # start id for each position
            constants.L: -1,
            constants.R: -1,
            constants.I: -1,
            constants.B: -1,
            constants.E: -1
        }

        with open(template_file, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) < 1 or line[0] == constants.COMMENT_SYM:
                    continue

                array = line.split(constants.FEAT_TEMP_DELIM)
                task = array[0].lower()
                position = array[1]
                spans = array[2].split(constants.FEAT_TEMP_RANGE_DELIM)

                if len(position) != 1 or not position in valid_positions:
                    continue

                if task == constants.SEG:
                    for k, span in enumerate(spans):
                        if constants.FEAT_TEMP_RANGE_SYM in span:
                            first, last = span.split(constants.FEAT_TEMP_RANGE_SYM)
                            first = int(first)
                            last = int(last)
                        else:
                            first = last = int(span)

                        for i in range(first, last+1):
                            self.seg_lg[position].update({i:k})

                elif task == constants.SEG_TAG:
                    pass

        next_start = 0
        for p in valid_positions:
            self.seg_starts[p] = next_start
            if self.seg_lg[p]:
                keys = set(self.seg_lg[p].keys())
                max_len = max(keys)
            else:
                max_len = 0
            self.max_len = max(self.max_len, max_len)

            vals = set(self.seg_lg[p].values())
            next_start += len(vals)
        self.dim = next_start


    def extract_features(self, instances, trie):
        features = [None] * len(instances)

        for i, x in enumerate(instances):
            feat = self.extract_features_for_sentence(x, trie)
            features[i] = chainer.Variable(feat)

        return features


    def extract_features_for_sentence(self, x, trie):
        xp = cuda.cupy if self.use_gpu else np
        feat = xp.zeros((len(x), self.dim), dtype='f')

        sen_len = len(x)
        found = []
        for i in range(sen_len):
            res = trie.common_prefix_search(x, i, i + self.max_len)
            found.extend(res)

        for b, e in found:
            length = e - b
            if self.max_len < length:
                continue

            # R, length
            if (b > 0 and 
                self.seg_starts[constants.R] >= 0 and length in self.seg_lg[constants.R]):
                index = self.seg_starts[constants.R] + self.seg_lg[constants.R][length]
                feat[b-1][index] = 1

            # B, length
            if (self.seg_starts[constants.B] >= 0 and length in self.seg_lg[constants.B]):
                index = self.seg_starts[constants.B] + self.seg_lg[constants.B][length]
                feat[b][index] = 1

            # I, length
            if self.seg_starts[constants.I] >= 0:
                for k in range(b+1, e-1):
                    if length in self.seg_lg[constants.I]:
                        index = self.seg_starts[constants.I] + self.seg_lg[constants.I][length]
                        feat[k][index] = 1

            # E, length
            if (self.seg_starts[constants.E] >= 0 and length in self.seg_lg[constants.E]):
                index = self.seg_starts[constants.E] + self.seg_lg[constants.E][length]
                feat[e-1][index] = 1

            # L, length
            if (e < sen_len and
                self.seg_starts[constants.L] >= 0 and length in self.seg_lg[constants.L]):
                index = self.seg_starts[constants.L] + self.seg_lg[constants.L][length]
                feat[e][index] = 1

        return feat
