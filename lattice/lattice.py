import enum
import pickle

import numpy as np
import chainer
import chainer.functions as F
#from chainer.cuda import cupy


UNK_TOKEN = '<UNK>'             # common among word and char
UNK_TOKEN_ID = 0                # common among word and char
DUMMY_POS = '*'
DUMMY_POS_ID = 0

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


class MapTrie(object):
    def __init__(self):
        self.tree = TrieNode(-1)
        self.next_id = 1        # 0 is used for <UNK>
        # self.debug = False


    def get_word_id(self, word, update=False):
        len_word = len(word) 
        node = self.tree

        for i, char in enumerate(word):
            child = node.get_child(char)
            # if self.debug:
            #     print(i, char, child.id if child else None)

            if not child:
                if not update or char == UNK_TOKEN_ID:
                    return UNK_TOKEN_ID
                child = TrieNode()
                node.set_child(char, child)

            if i == len_word - 1:
                if child.id == UNK_TOKEN_ID and update:
                    child.id = self.next_id
                    self.next_id += 1
                return child.id

            node = child


    def common_prefix_search(self, word, begin_index=0):
        res = []
        node = self.tree

        append = res.append
        for i, char in enumerate(word):
            child = node.get_child(char)
            if not child:
                # if i == 0:
                #     res = [(1, 0)] 
                break

            if child.id != UNK_TOKEN_ID:
                word_id = child.id
                append((begin_index + i + 1, child.id))
            node = child

        return res


class TrieNode(object):
    def __init__(self, id=UNK_TOKEN_ID):
        self.id = id
        self.children = {}


    def get_child(self, char):
        char = int(char)
        return self.children[char] if char in self.children else None


    def set_child(self, char, child_node):
        self.children[char] = child_node
        

class TokenIndices(object):
    def __init__(self, token2id=None, unk_symbol=None):
        if token2id:
            self.token2id = token2id
        else:
            if unk_symbol:
                self.unk_id = np.int32(0)
                self.token2id = {unk_symbol : self.unk_id}
            else:
                self.unk_id = -1
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
        self.token_indices = token_indices if token_indices else TokenIndices(unk_symbol=UNK_TOKEN)
        self.label_indices = label_indices if label_indices else TokenIndices(unk_symbol=None)
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
        self.label_indices = TokenIndices(unk_symbol=None)
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
        self.pos_indices = pos_indices if pos_indices else TokenIndices(unk_symbol=DUMMY_POS)

class MorphologyDictionary(IndicesTriplet):
    def __init__(self):
        self.pos_indices = TokenIndices(unk_symbol=DUMMY_POS)
        self.dummy_pos_id = self.pos_indices.unk_id
        self.init_label_indices('BIES')
        super(MorphologyDictionary, self).__init__(None, self.label_indices, self.pos_indices)
        self.chunk_trie = MapTrie()
        self.id2chunk = {UNK_TOKEN_ID : UNK_TOKEN}
        self.ci2lis = Key2Values()
        self.ci2lis.add(self.token_indices.unk_id, self.dummy_pos_id)


    def create_id2pos(self):
        self.id2pos = {v:k for k,v in self.pos_indices.token2id.items()}


    def get_entries(self, chunk, pos, update=False):
        tis = [self.token_indices.get_id(token, update=update) for token in chunk]

        # 全ての文字を単語として登録
        if update:
            for i, ti in enumerate(tis):
                ci_of_token = self.chunk_trie.get_word_id([ti], True)
                self.id2chunk[ci_of_token] = chunk[i]
                self.ci2lis.add(ci_of_token, self.dummy_pos_id)

        # 単語登録と ID 取得
        ci = self.chunk_trie.get_word_id(tis, update)
        pi = self.pos_indices.get_id(pos, update=update)
        if update:
            self.id2chunk[ci] = chunk
            self.ci2lis.add(ci, pi)

        return tis, ci, pi


    def get_token(self, ti):
        ti = int(ti)
        return self.id2token[ti] if ti in self.id2token else UNK_TOKEN


    def get_token_id(self, token):
        return self.token_indices.get_id(token)


    def get_pos(self, pi):
        return self.id2pos[pi]


    def get_pos_ids(self, ci):
        return self.ci2lis.get(ci)


    def get_chunk(self, ci):
        if ci in self.id2chunk:
            return self.id2chunk[ci]
        else:
            return UNK_TOKEN


class Lattice(object):
    def __init__(self, sen, dic, debug=False):
        self.sen = sen
        self.eid2nodes = Key2Values(val_type='list')
        self.debug = debug

        if self.debug:
            print('\ncreate lattice')
        
        for i in range(len(sen)):
            sen_rest = sen[i:]
            hit = dic.chunk_trie.common_prefix_search(sen_rest, i)

            if not hit:
                # 先頭文字を未知単語として追加
                self.eid2nodes.add(i, i+1, dic.unk_pos_id)

            for entry in hit:
                j = entry[0]
                wi = entry[1]
                pis = dic.get_pos_ids(wi)
                for pi in pis:
                    node = (i, j, pi) # (begin index, end index, pos)                    
                    self.eid2nodes.add(i + 1, node)
                    if self.debug:
                        print('  node=({}, {}, {}) word={}/{}, pos={}'.format(
                            i, j, pi, wi, dic.get_chunk(wi), dic.get_pos(pi)))

        if self.debug:
            print('\nconstructed lattice:')
            for k, v in self.eid2nodes.key2values.items():
                print('  {}: {}'.format(k, v))


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

            # print('{}-th nodes     {}:'.format(t, self.eid2nodes.get(t)))
            # print('{}-th pointers: {}'.format(t, self.prev_pointers[t]))


class Lattice2(object):
    def __init__(self, sen, dic, debug=False):
        self.sen = sen
        self.end2begins = Key2Values(val_type='set')
        self.debug = debug

        if self.debug:
            print('\ncreate lattice')
        
        # print('sen', sen)
        T = len(self.sen)
        for i in range(T):
            sen_rest = sen[i:]
            hit = dic.chunk_trie.common_prefix_search(sen_rest, i)
            
            # if not hit:
                # 先頭文字を未知単語として追加
                # t = i + 1
                # self.eid2posi[t] =  UNK_TOKEN_ID

            for tup in hit:
                t = tup[0]
                wi = tup[1]
                pis = tuple(sorted(dic.get_pos_ids(wi)))
                entry = (i, wi, pis)
                self.end2begins.add(t, entry)

                if self.debug:
                    for pi in pis:
                        print('  node=({}, {}, {}) word={}/{}, pos={}'.format(
                            i, t, pis, wi, dic.get_chunk(wi), dic.get_pos(pi)))

        if self.debug:
            print('\nconstructed lattice:')
            for k, v in self.end2begins.key2values.items():
                print('  {}: {}'.format(k, v))


    def prepare_forward(self, xp=np):
        self.end2begins.val_type = ValueType.List

        T = len(self.sen)
        self.deltas = [[] for i in range(T+1)]
        self.scores = [[] for i in range(T+1)]
        self.prev_pointers = [[] for i in range(T+1)]

        for t in range(1, T + 1): # t=0 は dummy
            
            # convert set to sorted list
            begins = sorted(self.end2begins.key2values[t], reverse=True)
            self.end2begins.key2values[t] = begins

            if begins:
                num_nodes = sum([len(entry[2]) for entry in begins])
                self.prev_pointers[t] = -np.ones(num_nodes).astype('i')
                #self.deltas[t] = chainer.Variable(xp.zeros(num_nodes, dtype='f'))
                #self.scores[t] = chainer.Variable(xp.zeros(num_nodes, dtype='f'))
                #print(type(self.deltas[t]), self.deltas[t])


class Lattice3(object):
    def __init__(self, sen, dic, debug=False):
        self.sen = sen
        self.end2begins = Key2Values(val_type='set')
        self.debug = debug

        if self.debug:
            print('\ncreate lattice')
        
        # print('sen', sen)
        T = len(self.sen)
        for i in range(T):
            sen_rest = sen[i:]
            hit = dic.chunk_trie.common_prefix_search(sen_rest, i)
            #print(i, hit)
            
            if not hit:
                # 先頭文字を未知単語として追加
                t = i + 1
                #self.eid2posi[t] =  UNK_TOKEN_ID

            for tup in hit:
                t = tup[0]
                wi = tup[1]
                pis = tuple(sorted(dic.get_pos_ids(wi)))
                entry = (i, wi, pis)
                self.end2begins.add(t, entry)

                if self.debug:
                    for pi in pis:
                        print('  node=({}, {}, {}) word={}/{}, pos={}'.format(
                            i, j, pi, wi, dic.get_chunk(wi), dic.get_pos(pi)))

        if self.debug:
            print('\nconstructed lattice:')
            for k, v in self.eid2nodes.key2values.items():
                print('  {}: {}'.format(k, v))


    def prepare_forward(self, xp=np):
        self.end2begins.val_type = ValueType.List

        T = len(self.sen)
        self.deltas = [[] for i in range(T+1)]
        self.scores = [[] for i in range(T+1)]
        self.prev_pointers = [[] for i in range(T+1)]

        for t in range(1, T + 1): # t=0 は dummy
            
            # convert set to sorted list
            begins = sorted(self.end2begins.key2values[t], reverse=True)
            self.end2begins.key2values[t] = begins
            # print(t, begins)

            if begins:
                num_nodes = sum([len(entry[2]) for entry in begins])
                self.deltas[t] = [chainer.Variable(xp.array(0, dtype='f')) for i in range(num_nodes)]
                self.scores[t] = [chainer.Variable(xp.array(0, dtype='f')) for i in range(num_nodes)]
                self.prev_pointers[t] = -np.ones(num_nodes).astype('i')


    # def prepare_backprop(self):
    #     for t in range(1, len(self.sen) + 1): # t=0 は dummy
    #         self.deltas[t] = F.concat([F.expand_dims(var, axis=0) for var in self.deltas[t]], axis=0)
    #         self.scores[t] = F.concat([F.expand_dims(var, axis=0) for var in self.scores[t]], axis=0)


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
