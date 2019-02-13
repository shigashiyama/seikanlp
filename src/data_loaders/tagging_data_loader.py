import sys

import constants
import dictionary
from data_loaders import data_loader
from data_loaders.data_loader import DataLoader, Data, RestorableData


class TaggingDataLoader(DataLoader):
    def __init__(self,
                 token_index=0,
                 attr_indexes=[], 
                 attr_depths=[], 
                 attr_chunking_flags=[],
                 attr_target_labelsets=[],
                 attr_delim=None,
                 lowercasing=False, 
                 normalize_digits=True, 
                 token_max_vocab_size=-1,
                 token_freq_threshold=1, 
                 unigram_vocab=set(), 
    ):
        self.token_index = token_index
        self.attr_indexes = attr_indexes
        self.attr_depths = attr_depths
        self.attr_chunking_flags = attr_chunking_flags
        self.attr_target_labelsets = attr_target_labelsets
        self.attr_delim = attr_delim
        self.lowercasing = lowercasing
        self.normalize_digits = normalize_digits
        self.token_max_vocab_size=token_max_vocab_size
        self.token_freq_threshold=token_freq_threshold
        self.unigram_vocab = unigram_vocab
        self.freq_tokens = set()


    def load_gold_data(self, path, data_format, dic=None, train=True):
        if data_format == constants.WL_FORMAT:
            if self.token_freq_threshold > 1 or self.token_max_vocab_size > 0:
                self.freq_tokens = self.get_frequent_tokens_WL(
                    path, self.token_freq_threshold, self.token_max_vocab_size)
            data, dic = self.load_gold_data_WL(path, dic, train)

        else:
            if self.token_freq_threshold > 1 or self.token_max_vocab_size > 0:
                self.freq_tokens = self.get_frequent_tokens_SL(
                    path, self.token_freq_threshold, self.token_max_vocab_size)

            data, dic = self.load_gold_data_SL(path, dic, train)

        return data, dic


    def load_decode_data(self, path, data_format, dic=None):
        if data_format == constants.WL_FORMAT:
            data = self.load_decode_data_WL(path, dic)
        else:
            data = self.load_decode_data_SL(path, dic)
        return data


    def parse_commandline_input(self, line, dic):
        attr_delim = self.attr_delim if self.attr_delim else constants.SL_ATTR_DELIM
        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        if constants.ATTR_LABEL(1) in dic.tables:
            use_attr1 = True
            get_attr1_id = dic.tables[constants.ATTR_LABEL(1)].get_id
        else:
            use_attr1 = False
            get_attr1_id = None

        org_arr = line.split(' ')
        if use_attr1:
            org_attr1_seq = [
                self.preprocess_attribute(
                    elem.split(attr_delim)[1] if attr_delim in elem else constants.UNK_SYMBOL,
                    0, #self.attr_depths[0], 
                    None, #self.attr_target_labelsets[0]
                ) for elem in org_arr]
            org_attr1_seqs = [org_attr1_seq]
            attr1_seq = [get_attr1_id(attr) for attr in org_attr1_seq]
            attr1_seqs = [attr1_seq]
        else:
            org_attr1_seqs = []
            attr1_seqs = []

        org_token_seq = [elem.split(attr_delim)[0] for elem in org_arr]
        org_token_seqs = [org_token_seq]
        ptoken_seq = [self.preprocess_token(word) for word in org_token_seq]
        uni_seq = [get_unigram_id(word) for word in ptoken_seq]
        uni_seqs = [uni_seq]

        inputs = [uni_seqs, None, attr1_seqs] # TODO fix
        outputs = []
        orgdata = [org_token_seqs, org_attr1_seqs]
        return RestorableData(inputs, outputs, orgdata=orgdata)


    def load_gold_data_WL(self, path, dic, train=True):
        attr_delim = self.attr_delim if self.attr_delim else constants.WL_ATTR_DELIM
        num_attrs = len(self.attr_indexes)
        word_clm = self.token_index

        if not dic:
            dic = init_dictionary(num_attrs=num_attrs)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_ith_attr_id = []
        for i in range(num_attrs): 
            get_ith_attr_id.append(dic.tables[constants.ATTR_LABEL(i)].get_id)

        token_seqs = []
        attr_seqs_list = [[] for i in range(num_attrs)]

        ins_cnt = 0
        with open(path) as f:
            uni_seq = [] 
            attr_seq_list = [[] for i in range(num_attrs)]
     
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) == 0:
                    if len(uni_seq) > 0:
                        token_seqs.append(uni_seq)
                        uni_seq = []
                        for i, attr_seq in enumerate(attr_seq_list):
                            if self.attr_chunking_flags[i]:
                                attr_seq = [get_ith_attr_id[i](attr, update=train) for attr in 
                                            data_loader.get_labelseq_BIOES(attr_seq)]
                            attr_seqs_list[i].append(attr_seq)
                            attr_seq_list = [[] for i in range(num_attrs)]

                        ins_cnt += 1
                        if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                            print('Read', ins_cnt, 'sentences', file=sys.stderr)

                    continue

                elif line[0] == constants.COMMENT_SYM:
                    continue

                array = line.split(attr_delim)
                token = self.preprocess_token(array[word_clm])
                tlen = len(token)
                attrs = [None] * max(num_attrs, 1)
     
                for i in range(num_attrs):
                    attr = array[self.attr_indexes[i]] if len(array) > self.attr_indexes[i] else ''
                    attrs[i] = self.preprocess_attribute(
                        attr, self.attr_depths[i], self.attr_target_labelsets[i])

                    update_token = self.to_be_registered(token, train, self.freq_tokens, self.unigram_vocab)
                    uni_seq.append(get_unigram_id(token, update=update_token))
 
                    for i in range(num_attrs):
                        attr = attrs[i]
                        attr_tmp = attr if self.attr_chunking_flags[i] else get_ith_attr_id[i](attr, update=train)
                        attr_seq_list[i].append(attr_tmp)

            # register last sentenece
            if len(uni_seq) > 0:
                token_seqs.append(uni_seq)
                for i, attr_seq in enumerate(attr_seq_list):
                    if self.attr_chunking_flags[i]:
                        attr_seq = [get_ith_attr_id[i](attr, update=train) for attr in 
                                    data_loader.get_labelseq_BIOES(attr_seq)]
                    attr_seqs_list[i].append(attr_seq)

        inputs = [token_seqs]
        inputs.append(None) # bigram
        inputs.append(attr_seqs_list[1] if len(attr_seqs_list) > 1 else None)
     
        outputs = []
        if len(attr_seqs_list) > 0:
            outputs.append(attr_seqs_list[0])
            
        return Data(inputs, outputs), dic


    """
    Read data with SL (one Sentence in one Line) format.
    The following format is expected:
      word1_attr1 word2_attr2 ... wordn_attrn
    """
    def load_gold_data_SL(self, path, dic, train=True):
        attr_delim = self.attr_delim if self.attr_delim else constants.SL_ATTR_DELIM
        num_attrs = len(self.attr_indexes)
        word_clm = self.token_index

        if not dic:
            dic = init_dictionary(num_attrs=num_attrs)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_ith_attr_id = []
        for i in range(num_attrs): 
            get_ith_attr_id.append(dic.tables[constants.ATTR_LABEL(i)].get_id)

        token_seqs = []
        attr_seqs_list = [[] for i in range(num_attrs)]

        ins_cnt = 0
        with open(path) as f:
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) <= 1:
                    continue
     
                elif line[0] == constants.COMMENT_SYM:
                    continue
                
                entries = line.split(constants.SL_TOKEN_DELIM)
                uni_seq = []
                attr_seq_list = [[] for i in range(num_attrs)]
     
                for entry in entries:
                    array = entry.split(attr_delim)
                    token = self.preprocess_token(array[word_clm])
                    tlen = len(token)
                    attrs = [None] * max(num_attrs, 1)
     
                    for i in range(num_attrs):
                        attrs[i] = self.preprocess_attribute(
                            array[self.attr_indexes[i]], self.attr_depths[i], self.attr_target_labelsets[i])
     
                    update_token = self.to_be_registered(token, train, self.freq_tokens, self.unigram_vocab)
                    uni_seq.append(get_unigram_id(token, update=update_token))
 
                    for i in range(num_attrs):
                        attr = attrs[i]
                        attr_tmp = attr if self.attr_chunking_flags[i] else get_ith_attr_id[i](attr, update=train)
                        attr_seq_list[i].append(attr_tmp)
     
                token_seqs.append(uni_seq)
                for i, attr_seq in enumerate(attr_seq_list):
                    if self.attr_chunking_flags[i]:
                        attr_seq = [get_ith_attr_id[i](attr, update=train) for attr in 
                                    data_loader.get_labelseq_BIOES(attr_seq)]
                    attr_seqs_list[i].append(attr_seq)
     
                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
     
        inputs = [token_seqs]
        inputs.append(None) # bigram
        inputs.append(attr_seqs_list[1] if len(attr_seqs_list) > 1 else None)
     
        outputs = []
        if len(attr_seqs_list) > 0:
            outputs.append(attr_seqs_list[0])
     
        return Data(inputs, outputs), dic


    def load_decode_data_WL(self, path, dic):
        # to be implemented
        pass


    def load_decode_data_SL(self, path, dic):
        attr_delim = self.attr_delim if self.attr_delim else constants.SL_ATTR_DELIM
        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        if constants.ATTR_LABEL(1) in dic.tables:
            use_attr1 = True
            get_attr1_id = dic.tables[constants.ATTR_LABEL(1)].get_id
        else:
            use_attr1 = False
            get_attr1_id = None

        org_token_seqs = []
        org_attr1_seqs = []
        token_seqs = []
        attr1_seqs = []

        ins_cnt = 0
        with open(path) as f:
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) <= 1:
                    continue
     
                elif line[0] == constants.COMMENT_SYM:
                    continue
                
                org_arr = line.split(constants.SL_TOKEN_DELIM)
                if use_attr1:
                    org_attr1_seq = [
                        self.preprocess_attribute(
                            elem.split(attr_delim)[1],
                            self.attr_depths[0], 
                            self.attr_target_labelsets[0])
                        for elem in org_arr]
                    org_attr1_seqs.append(org_attr1_seq)
                    attr1_seq = [get_attr1_id(attr) for attr in org_attr1_seq]
                    attr1_seqs.append(attr1_seq)

                org_token_seq = [elem.split(attr_delim)[0] for elem in org_arr]
                org_token_seqs.append(org_token_seq)
                ptoken_seq = [self.preprocess_token(token) for token in org_token_seq]
                token_seq = [get_unigram_id(ptoken, update=ptoken in self.unigram_vocab) for ptoken in ptoken_seq]
                token_seqs.append(token_seq)
                    
                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
                    
        inputs = [token_seqs]
        inputs.append(None) # bigram
        inputs.append(attr1_seqs if attr1_seqs else None)
        outputs = []
        orgdata = [org_token_seqs, org_attr1_seqs]

        return RestorableData(inputs, outputs, orgdata=orgdata)


def init_dictionary(num_attrs=0): 
    dic = dictionary.Dictionary()

    # unigram
    dic.create_table(constants.UNIGRAM)
    dic.tables[constants.UNIGRAM].set_unk(constants.UNK_SYMBOL)

    # attributes
    for i in range(num_attrs):
        dic.create_table(constants.ATTR_LABEL(i))
        # dic.tables[constants.ATTR_LABEL(i)].set_unk(constants.UNK_SYMBOL)

    return dic

    
