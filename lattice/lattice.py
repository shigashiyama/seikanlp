import enum
import pickle

import numpy as np
import chainer


UNK_SYMBOL = '<UNK>'
DUMMY_POS = '*'


ValueType = enum.Enum("ValueType", "Set List")


class Key2Values(object):
    def __init__(self, val_type='set'):
        self.val_type = ValueType.Set if val_type == 'set' else ValueType.List
        self.key2values = {}


    def __len__(self):
        return len(self.key2values)


    def __str__(self):
        return str(self.key2values)

        
    def add(self, key, val):
        if key in self.key2values:
            vals = self.key2values[key]
        else:
            vals = set() if self.val_type == ValueType.Set else []
            self.key2values[key] = vals

        if self.val_type == ValueType.Set:
            vals.add(val)
        else:
            vals.append(val)


    def get(self, key):
        if key in self.key2values:
            return self.key2values[key]
        else:
            return set() if self.val_type == ValueType.Set else []


    def keys(self):
        return self.key2values.keys()


class TokenIndices(object):
    def __init__(self, token2id=None, use_unknown=True):
        if token2id:
            self.token2id = token2id
            self.unk_id = np.int32(-1)
        else:
            if use_unknown:
                self.unk_id = np.int32(0)
                self.token2id = {UNK_SYMBOL : self.unk_id}
            else:
                self.token2id = {}


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
            return self.unk_id


class IndicesPair(object):
    def __init__(self, token_indices=None, label_indices=None):
        self.token_indices = token_indices if token_indices else TokenIndices()
        self.label_indices = label_indices if label_indices else TokenIndices(use_unknown=False)
        self.id2token = {}
        self.id2label = {}
        if token_indices:
            self.create_id2token()
        if label_indices:
            self.create_id2label()


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


    def get_label(self, li):
        return self.id2label[li]


    def get_label_id(self, label):
        return self.label_indices.get_id(label)


class IndicesTriplet(IndicesPair):
    def __init__(self, token_indices=None, label_indices=None, pos_indices=None):
        super(IndicesTriplet, self).__init__(token_indices, label_indices)
        self.pos_indices = pos_indices if pos_indices else TokenIndices(use_unknown=False)


class MorphologyDictionary(IndicesTriplet):
    def __init__(self, token_indices=None, pos_indices=None, chunk_indices=None):
        if pos_indices:
            self.pos_indices = pos_indices
        else:
            self.dummy_pos_id = np.int32(0)
            self.pos_indices = TokenIndices(token2id={DUMMY_POS : self.dummy_pos_id})

        self.init_label_indices('BIES')
        super(MorphologyDictionary, self).__init__(token_indices, self.label_indices, self.pos_indices)
        self.chunk_indices = chunk_indices if chunk_indices else TokenIndices()
        self.ti2maxlen = {}           #TODO 数値の trie にする?
        self.ci2lis = Key2Values()    #TODO 文字の trie にする?
        self.ci2lis.add(self.token_indices.unk_id, self.dummy_pos_id)
        #self.trie = Trie(chunk_indices.token2id.keys())


    def create_id2pos(self):
        self.id2pos = {v:k for k,v in self.pos_indices.token2id.items()}


    def get_entries(self, chunk, pos, update=False):
        # fti_tmp = self.token_indices.get_id(chunk[0], update=False)
        # if fti_tmp == 0:
        #     print('an entry with new first char:', chunk, pos, update, self)

        first_token_id = self.token_indices.get_id(chunk[0], update=update)

        tis = [first_token_id]
        if update:
            if not first_token_id in self.ti2maxlen or len(chunk) > self.ti2maxlen[first_token_id]:
                self.ti2maxlen[first_token_id] = len(chunk)

        ci = self.chunk_indices.get_id(chunk, update=update)
        if len(chunk) > 1:
            tis.extend([self.token_indices.get_id(token, update=update) for token in chunk[1:]])

        li = self.pos_indices.get_id(pos, update=update) if pos else self.dummy_pos_id
        if update:
            self.ci2lis.add(ci, li)
        # if fti_tmp == 0:
        #     fti = first_token_id
        #     print('registered fti={}, ci={}, li={}, lis={} | ft='.format(first_token_id, ci, li, self.ci2lis.get(ci), self.get_token(fti)))

        return tis, ci, li


    def get_token(self, ti):
        return self.id2token[ti] if ti in self.id2token else UNK_SYMBOL


    def get_token_id(self, token):
        return self.token_indices.get_id(token)


    def get_pos(self, pi):
        return self.id2pos[pi]


    def get_pos_ids(self, ci):
        return self.ci2lis.get(ci)


    def get_pos_ids(self, ci):
        return self.ci2lis.get(ci)


    # def get_chunk(self, ci):
    #     return self.id2chunk[ci]


    def get_chunk_id(self, chunk):
        return self.chunk_indices.get_id(chunk)


    # max length of chunk (word) starting with the input token (char)
    def maxlen(self, token):
        return self.ti2maxlen[token] if token in self.ti2maxlen else 1


class Lattice(object):
    def __init__(self, sen, dic, debug=False):
        self.sen = sen
        # self.key2node = {}
        self.eid2nodes = Key2Values(val_type='list')
        self.debug = debug

        if self.debug:
            print('\ncreate lattice')
        
        for i in range(len(sen)):
            ti = int(sen[i])
            chunk = '{}'.format(dic.get_token(ti))

            # word consisting of the present character
            lis = dic.get_pos_ids(dic.get_chunk_id(chunk))
            if self.debug:
                print(i, ti, dic.get_token(ti), dic.get_chunk_id(chunk), lis)

            for li in lis:
                node = (i, i + 1, li) # (begin index, end index, pos)
                #self.key2node[str(node)] = node
                self.eid2nodes.add(i + 1, node)
            if self.debug and lis:
                print('  nodes with length {}: {}'.format(1, self.eid2nodes.get(i + 1)))

            # word consisting of the present and the succeeding characters
            maxlen = dic.maxlen(ti)
            # if self.debug:
            #     print(i, ti, chunk, lis, maxlen)
            for j in range(i + 1, min(i + maxlen, len(sen))):
                chunk = '{}{}'.format(chunk, dic.get_token(int(sen[j])))
                ci = dic.get_chunk_id(chunk)

                for li in dic.get_pos_ids(ci):
                    node = (i, j + 1, li)
                    # self.key2node[str(node)] = node
                    self.eid2nodes.add(j + 1, node)
                if self.debug and self.eid2nodes.get(j + 1):
                    print('  nodes with length {}: {}'.format(j+1-i, self.eid2nodes.get(j + 1)))

    # def get_node(self, begin, end, label):
    #     return self.key2node['{}:{}:{}'.format(begin, end, label)]


    def prepare_forward(self, xp=np):
        self.deltas = []
        self.scores = []
        self.prev_pointers = []
        for t in range(0, len(self.sen) + 1):
            nodes = self.eid2nodes.get(t)
            if nodes:
                nlen = len(nodes)
                self.deltas.append(chainer.Variable(xp.zeros(nlen, dtype='f')))
                self.scores.append(chainer.Variable(xp.zeros(nlen, dtype='f')))
                self.prev_pointers.append(-np.ones(nlen).astype('i'))
            else:
                self.deltas.append([])
                self.scores.append([])
                self.prev_pointers.append([])


    # def forward(self):
    #     for i in range(len(self.sen)):
    #         for node in self.sid2nodes.get(i):
    #             score = 0
    #             prev_best_node = None

    #             # prev node + edge score
    #             for prev_node in self.eid2nodes.get(node.begin):
    #                 edge_score_tmp = prev_node.score
    #                 edge_score_tmp += calc_edge_score(prev_node, node)
    #                 if edge_score_tmp > score:
    #                     score = edge_score_tmp
    #                     prev_best_node = prev_node 

    #             # node score
    #             # if score < 0:
    #             #     score = 0
                    
    #             score += calc_node_score(node)
    #             node.score = score
    #             node.prev_best_node = prev_best_node
    #             print(i, prev_best_node, node, score)


    # def argmax(self):
    #     max_score = -np.Infinity
    #     last_node = None
    #     for node in self.eid2nodes.get(len(self.sen)):
    #         if node.score > max_score:
    #             last_node = node
    #             max_score = node.score

    #     path = [last_node]
    #     while path[0].prev_best_node:
    #         path.insert(0, path[0].prev_best_node)

    #     return path
                        

def node_len(node):
    return node[1] - node[0]


def load_dictionary(path, read_pos=True):
    if path.endswith('pickle'):
        with open(path, 'rb') as f:
            dic = pickle.load(f)
            dic.create_id2token()
        return dic

    dic = MorphologyDictionary()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            arr = line.split('/')
            if len(arr) < 2 or len(arr[0]) == 0:
                continue

            chunk = arr[0]
            pos = arr[1] if read_pos else DUMMY_POS
            dic.get_entries(chunk, pos, update=True)

    dic.create_id2token()
    dic.create_id2pos()
    return dic


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
    
    lat = Lattice(sen, dic)
    print(lat.sid2nodes)
    print(lat.eid2nodes)
    print()

    lat.forward()
    path = lat.argmax()
    print(path)

    res = []
    for node in path:
        res.append('{}/{}'.format(org_sen[node[0]:node[1]], node[2]))
    print(res)
