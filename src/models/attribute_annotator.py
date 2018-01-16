import collections


class Key2Counter(object):
    def __init__(self):
        self.key2counter = {}
        
        
    def __len__(self):
        return len(self.key2counter)
        
        
    def __str__(self):
        return str(self.key2counter)
        
        
    def add(self, key, val):
        if not key in self.key2counter:
            self.key2counter[key] = collections.Counter()
        self.key2counter[key][val] += 1


    def get(self, key):
        if key in self.key2counter:
            return self.key2counter[key]
        else:
            return None
            
            
    def keys(self):
        return self.key2counter.keys()


class AttributeAnnotator(object):
    def __init__(self):
        self.word_dic = Key2Counter()
        self.pos_dic = Key2Counter()
        self.tgt_words = collections.Counter()


    def update(self, src, pos, tgt):
        self.word_dic.add(src, tgt)
        self.tgt_words[tgt] += 1
        if pos:
            self.pos_dic.add(pos, tgt)


    def predict(self, src, pos=None):
        most_common_tgt = None

        if src in self.word_dic.key2counter:
            counter = self.word_dic.get(src)
            pred = counter.most_common(1)[0][0]

        else:
            if pos in self.pos_dic.key2counter:
                counter = self.pos_dic.get(pos)
                pred = counter.most_common(1)[0][0]

            else:
                if not most_common_tgt:
                    counter = self.tgt_words
                    most_common_tgt = counter.most_common(1)[0][0]
                pred = most_common_tgt

        return pred
