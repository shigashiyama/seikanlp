import enum

import numpy as np

import constants


ValueType = enum.Enum("ValueType", "Set List")
UNK_ID = 0

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
                if not update or char == UNK_ID:
                    return UNK_ID
                child = TrieNode()
                node.set_child(char, child)

            if i == len_word - 1:
                if child.id == UNK_ID and update:
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

            if child.id != UNK_ID:
                append((begin_index, begin_index + i + 1))
                #append((begin_index + i + 1, child.id))
            node = child

        return res


class TrieNode(object):
    def __init__(self, id=UNK_ID):
        self.id = id            # id != UNK_ID なら終端
        #self.is_terminal = False
        self.children = {}


    def get_child(self, char):
        char = int(char)
        return self.children[char] if char in self.children else None


    def set_child(self, char, child_node):
        self.children[char] = child_node


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
        

class IndexTable(object):
    def __init__(self, str2id=None, unk_symbol=None):
        self.unk_id = -1

        if str2id:
            self.str2id = str2id
        else:
            self.str2id = {}
            if unk_symbol:
                self.set_unk(unk_symbol)

        self.id2str = {}


    def set_unk(self, unk_symbol):
        if self.unk_id < 0:
            self.unk_id = len(self.str2id)
            self.str2id[unk_symbol] = self.unk_id
            return self.unk_id

        else:
            return -1


    def __len__(self):
        return len(self.str2id)


    def create_id2str(self):
        self.id2str = {v:k for k,v in self.str2id.items()}


    def get_id(self, key, update=False):
        if key in self.str2id:
            return self.str2id[key]
        elif update:
            id = np.int32(len(self.str2id))
            self.str2id[key] = id
            return id
        else:
            return self.unk_id


    def add_entries(self, strs):
        for s in strs:
            self.get_id(s, update=True)


class Dictionary(object):
    def __init__(self, use_seg_label=False, use_pos_label=False, use_arc_label=False,
                 use_word_trie=False, use_root=False):
        self.token_indices = IndexTable()
        self.seg_label_indices = IndexTable() if use_seg_label else None
        self.pos_label_indices = IndexTable() if use_pos_label else None
        self.arc_label_indices = IndexTable() if use_arc_label else None

        if use_root:
            self.root_id = self.token_indices.get_id(constants.ROOT_SYMBOL, update=True)
            self.root_id = self.pos_label_indices.get_id(constants.ROOT_SYMBOL, update=True)
        else:
            self.root_id = -1

        self.token_indices.set_unk(constants.UNK_SYMBOL)

        if use_word_trie:
            word_unk_id = np.int32(0)
            self.word_trie = MapTrie()
            self.id2word = {word_unk_id : constants.UNK_SYMBOL}
            pos_unk_id = self.pos_label_indices.set_unk(constants.UNK_SYMBOL)
            self.wid2pids = Key2Values()
            self.wid2pids.add(word_unk_id, pos_unk_id)
        else:
            self.word_trie = None
            self.id2word = None
            self.wid2pids = None


    def create_id2strs(self):
        if self.token_indices:
            self.token_indices.create_id2str()

        if self.seg_label_indices:
            self.seg_label_indices.create_id2str()

        if self.pos_label_indices:
            self.pos_label_indices.create_id2str()

        if self.arc_label_indices:
            self.arc_label_indices.create_id2str()
        

    def get_token(self, ti):
        ti = int(ti)
        return self.token_indices.id2str[ti]
        # return self.id2token[ti] if ti in self.id2token else UNK_TOKEN


    def get_token_id(self, token):
        return self.token_indices.get_id(token)


    def get_seg_label(self, li):
        return self.seg_label_indices.id2str[li]


    def get_seg_label_id(self, label):
        return self.seg_label_indices.get_id(label)


    def get_pos_label(self, li):
        return self.pos_label_indices.id2str[li]


    def get_pos_label_id(self, label):
        return self.pos_label_indices.get_id(label)


    def get_arc_label(self, li):
        return self.arc_label_indices.id2str[li]


    def get_arc_label_id(self, label):
        return self.arc_label_indices.get_id(label)


    def get_word(self, wi):
        if wi in self.id2word:
            return self.id2word[wi]
        else:
            return constants.UNK_SYMBOL


    def get_entries(self, word, pos, update=False):
        char_ids = [self.token_indices.get_id(char, update=update) for char in word]

        wid = self.word_trie.get_word_id(char_ids, update)
        pid = self.pos_label_indices.get_id(pos, update=update) if pos else None
        if update:
            self.id2word[wid] = word
            if pid:
                self.wid2pids.add(wid, pid)
        return char_ids, wid, pid


    # def init_segmentation_labels(self):
    #     self.slabel_indices.add_entries(constants.SEG_LABELS)


# load vocabulary from external dictionary resource
def load_vocabulary(path, read_pos=True):
    if path.endswith('pickle'):
        with open(path, 'rb') as f:
            dic = pickle.load(f)
            dic.create_id2strs()
        return dic

    dic = Dictionary()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            arr = line.split('/')
            if len(arr) < 2 or len(arr[0]) == 0:
                continue

            word = arr[0]
            pos = arr[1] if read_pos else None
            dic.get_entries(word, pos, update=True)

    dic.create_id2strs()
    return dic


