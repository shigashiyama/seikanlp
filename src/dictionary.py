import numpy as np

import constants


class MapTrie(object):
    UNK_ID = 0
    UNK_SYMBOL = constants.UNK_SYMBOL

    def __init__(self):
        self.tree = TrieNode(-1)
        self.id2chunk = {UNK_ID : UNK_SYMBOL}
        self.next_id = 1
        #self.debug = True


    def get_chunk(self, chunk_id):
        if chunk_id in self.id2chunk:
            return self.id2chunk[chunk_id]
        else:
            return UNK_SYMBOL


    # chunk equals to list of token IDs
    def get_chunk_id(self, chunk, update=False):
        len_chunk = len(chunk) 
        node = self.tree

        for i, token in enumerate(chunk):
            child = node.get_child(token)
            # if self.debug:
            #     print(i, token, child.id if child else None)

            if not child:
                if not update or token == UNK_ID:
                    return UNK_ID
                child = TrieNode()
                node.set_child(token, child)

            if i == len_chunk - 1:
                if child.id == UNK_ID and update:
                    child.id = self.next_id
                    self.id2chunk[child.id] = chunk
                    self.next_id += 1
                return child.id

            node = child


    def common_prefix_search(self, chunk, begin_index=0, last_index=-1):
        res = []
        node = self.tree

        seq = chunk[begin_index:last_index] if last_index >= 0 else chunk[begin_index:]
        append = res.append
        for i, token in enumerate(seq):
            child = node.get_child(token)
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
        self.children = {}


    def get_child(self, token):
        token = int(token)
        return self.children[token] if token in self.children else None


    def set_child(self, token, child_node):
        self.children[token] = child_node


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
        if key in self.key2values:
            return self.key2values[key]
        else:
            return set()


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
                 use_subtoken=False, use_chunk_trie=False, use_root=False, use_dummy=False):
        self.token_indices = IndexTable()
        self.subtoken_indices = IndexTable() if use_subtoken else None
        self.seg_label_indices = IndexTable() if use_seg_label else None
        self.pos_label_indices = IndexTable() if use_pos_label else None
        self.arc_label_indices = IndexTable() if use_arc_label else None

        if use_dummy:
            self.dummy_id = self.token_indices.get_id(constants.DUMMY_SYMBOL, update=True)
            self.pos_label_indices.get_id(constants.DUMMY_SYMBOL, update=True)
            if use_subtoken:
                self.subtoken_indices.get_id(constants.DUMMY_SYMBOL, update=True)
        else:
            self.dummy_id = -1
            
        if use_root:
            self.root_id = self.token_indices.get_id(constants.ROOT_SYMBOL, update=True)
            self.pos_label_indices.get_id(constants.ROOT_SYMBOL, update=True)
            if use_subtoken:
                self.subtoken_indices.get_id(constants.ROOT_SYMBOL, update=True)
        else:
            self.root_id = -1

        # unknown token
        self.token_indices.set_unk(constants.UNK_SYMBOL)
        if use_subtoken:
            self.subtoken_indices.set_unk(constants.UNK_SYMBOL)

        if use_chunk_trie:
            chunk_unk_id = np.int32(0)
            self.chunk_trie = MapTrie()
            self.id2chunk = {chunk_unk_id : constants.UNK_SYMBOL}
            pos_unk_id = self.pos_label_indices.set_unk(constants.UNK_SYMBOL)
            # self.wid2pids = Key2Values()
            # self.wid2pids.add(chunk_unk_id, pos_unk_id)
        else:
            self.chunk_trie = None
            self.id2chunk = None
            # self.wid2pids = None


    def has_seg_label(self):
        return self.seg_label_indices is not None


    def has_pos_label(self):
        return self.pos_label_indices is not None


    def has_arc_label(self):
        return self.arc_label_indices is not None


    def create_id2strs(self):
        if self.token_indices:
            self.token_indices.create_id2str()

        if self.subtoken_indices:
            self.subtoken_indices.create_id2str()

        if self.seg_label_indices:
            self.seg_label_indices.create_id2str()

        if self.pos_label_indices:
            self.pos_label_indices.create_id2str()

        if self.arc_label_indices:
            self.arc_label_indices.create_id2str()
        

    def get_token(self, ti):
        ti = int(ti)
        return self.token_indices.id2str[ti]


    def get_token_id(self, token):
        return self.token_indices.get_id(token)


    def get_subtoken(self, si):
        si = int(si)
        return self.subtoken_indices.id2str[si]


    def get_subtoken_id(self, subtoken):
        return self.subtoken_indices.get_id(subtoken)


    # def get_subtoken_ids(self, ti):
    #     return [self.subtoken_indices.get_id(s) for s in self.get_token(ti)]


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


    def get_chunk(self, wi):
        if wi in self.id2chunk:
            return self.id2chunk[wi]
        else:
            return constants.UNK_SYMBOL


    # token indicates character
    # chunk indicates word
    def get_entries(self, chunk, pos, update=False):
        token_ids = [self.token_indices.get_id(t, update=update) for t in chunk]

        cid = self.chunk_trie.get_chunk_id(token_ids, update)
        pid = self.pos_label_indices.get_id(pos, update=update) if pos else None
        if update:
            self.id2chunk[cid] = chunk
            # if pid:
            #     self.wid2pids.add(cid, pid)
        return char_ids, wid, pid


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


