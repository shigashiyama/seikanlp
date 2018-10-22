import numpy as np

import constants


class MapTrie(object):
    UNK_ID = 0
    UNK_SYMBOL = constants.UNK_SYMBOL

    def __init__(self, UNK_ID=UNK_ID, UNK_SYMBOL=UNK_SYMBOL):
        self.tree = TrieNode(-1)
        self.id2chunk = {UNK_ID : UNK_SYMBOL}
        self.unk_id = UNK_ID
        self.next_id = 1
        #self.debug = True


    def __len__(self):
        return len(self.id2chunk)


    def get_chunk(self, chunk_id):
        if chunk_id in self.id2chunk:
            return self.id2chunk[chunk_id]
        else:
            return UNK_SYMBOL


    # chunk equals to list of token IDs
    def get_chunk_id(self, chunk, word=None, update=False):
        len_chunk = len(chunk) 
        node = self.tree

        for i, token in enumerate(chunk):
            child = node.get_child(token)
            # if self.debug:
            #     print(i, token, child.id if child else None)

            if not child:
                if not update or token == self.unk_id:
                    return self.unk_id
                child = TrieNode(self.unk_id)
                node.set_child(token, child)

            if i == len_chunk - 1:
                if child.id == self.unk_id and update:
                    child.id = self.next_id
                    self.id2chunk[child.id] = word if word else chunk
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

            if child.id != self.unk_id:
                append((begin_index, begin_index + i + 1))
                #append((begin_index + i + 1, child.id))
            node = child

        return res


class TrieNode(object):
    def __init__(self, id):
        self.id = id            # id != unk_id indicates terminal node
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
    def __init__(self):
        self.tables = {}
        self.tries = {}


    def create_table(self, table_name):
        # string to index table
        self.tables[table_name] = IndexTable()


    def create_id2strs(self):
        for table in self.tables.values():
            table.create_id2str()


    def create_trie(self, trie_name):
        self.tries[trie_name] = MapTrie()


    def has_table(self, table_name):
        return table_name in self.tables


    def has_trie(self, trie_name):
        return trie_name in self.tries
