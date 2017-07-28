import pickle

import numpy as np

#from marisa_trie import Trie


UNK_SYMBOL = '<UNK>'
DUMMY_POS = '-'


class Key2Values(object):
    def __init__(self):
        self.key2values = {}


    def __len__(self):
        return len(self.key2values)


    def __str__(self):
        return str(self.key2values)

        
    def add(self, key, val):
        if key in self.key2values:
            vals = self.key2values[key]
        else:
            vals = set()
            self.key2values[key] = vals
        vals.add(val)


    def get(self, key):
        return self.key2values[key] if key in self.key2values else set()


    def keys(self):
        return self.key2values.keys()


class TokenIndices(object):
    def __init__(self, token2id=None, use_unknown=True):
        if token2id:
            self.token2id = token2id
        else:
            self.token2id = {UNK_SYMBOL : np.int32(0)} if use_unknown else {}


    def __len__(self):
        return len(self.token2id)


    def get_id(self, token, update=False):
        if token in self.token2id:
            return self.token2id[token]
        elif update:
            id = np.int32(len(self.token2id))
            self.token2id[token] = id
            return id
        else:
            return self.token2id[UNK_SYMBOL]


    # def get_id_exp(self, tokens, update=False):
    #     if update:
    #         for token in tokens:
    #             self.get_id(token, True)

    #     return '_'.join([str(self.token2id[t]) for t in tokens])


class IndicesPair(object):
    def __init__(self, token_indices=None, label_indices=None):
        self.token_indices = token_indices if token_indices else TokenIndices()
        self.label_indices = label_indices if label_indices else TokenIndices(use_unknown=False)
        self.id2token = {}
        self.id2label = {}


    def create_id2token(self):
        self.id2token = {v:k for k,v in self.token_indices.token2id.items()}


    def create_id2label(self):
        self.id2label = {v:k for k,v in self.label_indices.token2id.items()}


    def init_label_indices(self, labels):
        self.label_indices = TokenIndices(use_unknown=False)
        for label in labels:
            self.label_indices.get_id(label, update=True)
        

    def set_id2token(self, id2token):
        self.id2token = id2token

    
    def get_token(self, ti):
        return self.id2token[ti]


    def get_token_id(self, token):
        return self.token_indices.get_id(token)


class IndicesTriplet(IndicesPair):
    def __init__(self, token_indices=None, label_indices=None, pos_indices=None):
        super(IndicesTriplet, self).__init__(token_indices, label_indices)
        self.pos_indices = pos_indices if pos_indices else TokenIndices()


class MorphologyDictionary(IndicesTriplet):
    def __init__(self, token_indices, pos_indices, chunk_indices, ti2maxlen, ci2lis):
        super(MorphologyDictionary, self).__init__(token_indices, None, pos_indices)
        self.init_label_indices('BIES')
        self.ti2maxlen = ti2maxlen           #TODO 数値の trie にする?
        self.chunk_indices = chunk_indices
        #self.trie = Trie(chunk_indices.token2id.keys())
        self.ci2lis = ci2lis                 #TODO 文字の trie にする?
        self.pos_indices = pos_indices


    def get_pos_ids(self, ci):
        return self.ci2lis.get(ci)


    # def get_chunk(self, ci):
    #     return self.id2chunk[ci]


    def get_chunk_id(self, chunk):
        if len(chunk) == 1:
            return self.token_indices.get_id(chunk)

        return self.chunk_indices.get_id(chunk)


    # max length of chunk (word) starting with the input token (char)
    def maxlen(self, token):
        return self.ti2maxlen[token] if token in self.ti2maxlen else 1

    
class Node(object):
    def __init__(self, begin, end, pos, score=0.0, prev_best_node=None):
        self.begin = begin
        self.end = end
        self.pos = pos
        self.score = score
        self.prev_best_node = prev_best_node

    def __str__(self):
        return '%d:%d:%s' % (self.begin, self.end, self.pos)

    def __repr__(self):
        return self.__str__()


class LatticeFactory(object):
    def __init__(self, dic):
        self.dic = dic


    def create_lattice(self, sen):
        return Lattice(sen, self.dic)


class Lattice(object):
    def __init__(self, sen, dic):
        self.sen = sen
        self.sid2nodes = Key2Values()        # forward
        self.eid2nodes = Key2Values()        # backward

        for i in range(len(sen)):
            ti = sen[i]
            chunk = '{}'.format(dic.get_token(ti))

            # word consisting of the present character
            lis = dic.get_pos_ids(dic.get_chunk_id(chunk))
            for li in lis:
                node = Node(i, i + 1, li)
                self.sid2nodes.add(i, node)
                self.eid2nodes.add(i + 1, node)

            # word consisting of the present and the succeeding characters
            maxlen = dic.maxlen(ti)
            print(i, ti, chunk, lis, maxlen)
            for j in range(i + 1, min(i + maxlen, len(sen))):
                chunk = '{}{}'.format(chunk, dic.get_token(sen[j]))
                ci = dic.get_chunk_id(chunk)

                for li in dic.get_pos_ids(ci):
                    node = Node(i, j + 1, li)
                    self.sid2nodes.add(i, node)
                    self.eid2nodes.add(j + 1, node)


    def forward(self):
        for i in range(len(self.sen)):
            for node in self.sid2nodes.get(i):
                score = -1       #TODO - * large number にする
                prev_best_node = None

                # prev node + edge score
                for prev_node in self.eid2nodes.get(node.begin):
                    edge_score_tmp = prev_node.score
                    edge_score_tmp += calc_edge_score(prev_node, node)
                    if edge_score_tmp > score:
                        score = edge_score_tmp
                        prev_best_node = prev_node 

                # node score
                if score < 0:
                    score = 0
                    
                score += calc_node_score(node)
                node.score = score
                node.prev_best_node = prev_best_node
                print(i, prev_best_node, node, score)


    def get_optimal_path(self):
        max_score = 0
        last_node = None
        for node in self.eid2nodes.get(len(self.sen)):
            if node.score > max_score:
                last_node = node
                max_score = node.score

        path = [last_node]
        while path[0].prev_best_node:
            path.insert(0, path[0].prev_best_node)

        return path
    

def calc_node_score(node):
    score = 0

    if node.begin == 0 and node.end == 2:
        score = 1
    elif node.begin == 4 and node.end == 8:
        score = 2
    return score
                    

def calc_edge_score(node1, node2):
    score = 0
    return score
                        

def load_dictionary(path, indexing=True, read_pos=True):
    if path.endswith('pickle'):
        with open(path, 'rb') as f:
            dic = pickle.load(f)
        return dic

    token_indices = TokenIndices() if indexing else None
    chunk_indices = TokenIndices() if indexing else None
    pos_indices = TokenIndices(use_unknown=False) if read_pos and indexing else None
    ti2maxlen = {}
    ci2lis = Key2Values()

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            arr = line.split('/')
            if len(arr) < 2 or len(arr[0]) == 0:
                continue

            chunk = arr[0]
            pos = arr[1] if read_pos else DUMMY_POS

            first_token_id = token_indices.get_id(chunk[0], update=True) if indexing else chunk[0]
            if not first_token_id in ti2maxlen or len(chunk) > ti2maxlen[first_token_id]:
                ti2maxlen[first_token_id] = len(chunk)

            if len(chunk) == 1:
                ci = first_token_id
            else:
                ci = chunk_indices.get_id(chunk, update=True) if indexing else chunk

            li = pos_indices.get_id(pos, update=True) if indexing else pos
            ci2lis.add(ci, li)

    return MorphologyDictionary(token_indices, pos_indices, chunk_indices, ti2maxlen, ci2lis)


if __name__ == '__main__':

    dic_path = '../unidic/lex4kytea_zen.txt'
    dic = load_dictionary(dic_path, read_pos=True)

    dic_pic_path = '../unidic/lex4kytea_zen_li.pickle'
    with open(dic_pic_path, 'wb') as f:
        pickle.dump(dic, f)
    # dic_pic_path = '../unidic/lex4kytea_zen_li.pickle'
    # with open(dic_pic_path, 'rb') as f:
    #     dic = pickle.load(f)
        
    org_sen = 'どこかに引っ越す。'
    sen = [dic.get_token_id(c) for c in org_sen]
    print(sen)
    print([dic.maxlen(c) for c in sen])
    print([dic.get_pos_ids(c) for c in sen])

    latticeFactory = LatticeFactory(dic)
    
    lat = latticeFactory.create_lattice(sen)
    print(lat.sid2nodes)
    print(lat.eid2nodes)
    print()

    lat.forward()
    path = lat.get_optimal_path()
    print(path)

    res = []
    for node in path:
        res.append('{}/{}'.format(org_sen[node.begin:node.end], node.pos))
    print(res)
