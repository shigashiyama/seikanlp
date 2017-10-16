import enum
import pickle

import numpy as np
import chainer
import chainer.functions as F


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
        #self.debug = True


    # word: list of char IDs
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


    def common_prefix_search(self, word, begin_index=0, last_index=-1):
        res = []
        node = self.tree

        seq = word[begin_index:last_index] if last_index >= 0 else word[begin_index:]
        append = res.append
        for i, char in enumerate(seq):
            child = node.get_child(char)
            if not child:
                break
            #print(' ',child.id)

            if child.id != UNK_TOKEN_ID:
                append((begin_index, begin_index + i + 1))
                #append((begin_index + i + 1, child.id))
            node = child

        return res


class TrieNode(object):
    def __init__(self, id=UNK_TOKEN_ID):
        self.id = id            # id != UNK_TOKEN_ID なら終端
        #self.is_terminal = False
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


# token indices, label indices
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


# token indices, label indices, pos indices
class IndicesTriplet(IndicesPair):
    def __init__(self, token_indices=None, label_indices=None, pos_indices=None):
        super(IndicesTriplet, self).__init__(token_indices, label_indices)
        self.pos_indices = pos_indices if pos_indices else TokenIndices(unk_symbol=DUMMY_POS)


# token indices, label indices, pos indices, word trie
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
        self.end2begins = Key2Values(val_type='set')
        self.debug = debug

        if self.debug:
            print('\ncreate lattice')
        
        T = len(self.sen)
        for i in range(T):
            hit = dic.chunk_trie.common_prefix_search(sen[i:], i)
            if not hit:
                # 文字を未知単語として追加
                t = i + 1
                entry = (i, dic.token_indices.unk_id, (dic.dummy_pos_id,))
                self.end2begins.add(t, entry)

            for tup in hit:
                t = tup[0]
                wi = tup[1]
                pis = tuple(sorted(dic.get_pos_ids(wi)))
                if len(pis) > 0:
                    entry = (i, wi, pis)
                    self.end2begins.add(t, entry)

                    if self.debug:
                        for pi in pis:
                            print('  node=({}, {}, {}) word={}/{}, pos={}'.format(
                                i, t, pis, wi, dic.get_chunk(wi), dic.get_pos(pi)))
                            #print(' ', self.end2begins.get(t))

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
            if not t in self.end2begins.key2values:
                continue
            
            # convert set to sorted list
            begins = sorted(self.end2begins.key2values[t], reverse=True)
            self.end2begins.key2values[t] = begins

            if begins:
                num_nodes = sum([len(entry[2]) for entry in begins])
                self.prev_pointers[t] = -np.ones(num_nodes).astype('i')


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
            tis, _, _ = dic.get_entries(chunk, pos, update=True)

    dic.create_id2token()
    dic.create_id2pos()
    return dic


if __name__ == '__main__':

    dic_path = '/home/shigashi/data_shigashi/work/work_neural/sequence_labeling/unidic/dic_10k.zen'
    dic = load_dictionary(dic_path, read_pos=True)

    # dic_pic_path = '../unidic/lex4kytea_zen_li.pickle'
    # with open(dic_pic_path, 'wb') as f:
    #     pickle.dump(dic, f)
    # dic_pic_path = '../unidic/lex4kytea_zen_li.pickle'
    # with open(dic_pic_path, 'rb') as f:
    #     dic = pickle.load(f)
        
    # org_sen = 'どこかに引っ越す。'
    # sen = [dic.get_token_id(c) for c in org_sen]
    # print(sen)
    # print([dic.maxlen(c) for c in sen])
    # print([dic.get_pos_ids(c) for c in sen])

    # latticeFactory = LatticeFactory(dic) 
    
    # lat = Lattice(sen, dic)
    # print(lat.sid2nodes)
    # print(lat.eid2nodes)
    # print()

    # lat.forward()
    # path = lat.argmax()
    # print(path)

    # res = []
    # for node in path:
    #     res.append('{}/{}'.format(org_sen[node[0]:node[1]], node[2]))
    # print(res)
