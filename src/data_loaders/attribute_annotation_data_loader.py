import sys

import constants
import dictionary
from data_loaders import data_loader
from data_loaders.data_loader import DataLoader, Data, RestorableData


class AttributeAnnotationDataLoader(DataLoader):
    def __init__(self,
                 token_index=1,
                 label_index=2,
                 attr_indexes=[], 
                 attr_depths=[], 
                 attr_chunking_flags=[],
                 attr_target_labelsets=[],
                 attr_delim=None,
                 lowercasing=False, 
                 normalize_digits=True, 
    ):
        self.token_index = token_index
        self.label_index = label_index
        self.attr_indexes = attr_indexes
        self.attr_depths = attr_depths
        self.attr_chunking_flags = attr_chunking_flags
        self.attr_target_labelsets = attr_target_labelsets
        self.attr_delim = attr_delim
        self.lowercasing = lowercasing
        self.normalize_digits = normalize_digits


    def load_gold_data(self, path, data_format=None, dic=None, train=True):
        data, dic = self.load_gold_data_WL(path, dic, train)
        return data, dic


    def load_decode_data(self, path, data_format, dic=None):
        if data_format == constants.WL_FORMAT:
            data = self.load_decode_data_WL(path, dic)
        else:
            data = self.load_decode_data_SL(path, dic)
        return data


    def parse_commandline_input(self, line, dic):
        attr_delim = self.attr_delim if self.attr_delim else constants.SL_ATTR_DELIM
        num_attrs = len(self.attr_indexes)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        if constants.ATTR_LABEL(0) in dic.tables:
            use_attr0 = True
            get_attr0_id = dic.tables[constants.ATTR_LABEL(0)].get_id
        else:
            use_attr0 = False
            get_attr0_id = None

        org_arr = line.split(' ')
        if use_attr0:
            attr0_seq = [
                elem.split(attr_delim)[self.attr_indexes[0]] 
                if attr_delim in elem else ''
                for elem in org_arr]
            org_attr0_seq = [
                self.preprocess_attribute(attr, self.attr_depths[0], self.attr_target_labelsets[0]) 
                for attr in attr0_seq]
            org_attr0_seqs = [org_attr0_seq]
            attr0_seq = [get_attr0_id(attr) for attr in org_attr0_seq]
            attr0_seqs = [attr0_seq]
        else:
            org_attr0_seqs = []
            attr0_seqs = []

        org_token_seq = [elem.split(attr_delim)[0] for elem in org_arr]
        org_token_seqs = [org_token_seq]
        ptoken_seq = [self.preprocess_token(word) for word in org_token_seq]
        uni_seq = [get_unigram_id(word) for word in ptoken_seq]
        uni_seqs = [uni_seq]

        inputs = [uni_seqs]
        outputs = [attr0_seqs]
        orgdata = [org_token_seqs, org_attr0_seqs]

        return RestorableData(inputs, outputs, orgdata=orgdata)


    def load_gold_data_WL(self, path, dic, train=True):
        attr_delim = self.attr_delim if self.attr_delim else constants.WL_ATTR_DELIM
        num_attrs = len(self.attr_indexes)

        if not dic:
            dic = init_dictionary(num_attrs=num_attrs)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_label_id = dic.tables[constants.SEM_LABEL].get_id
        get_ith_attr_id = []
        for i in range(num_attrs): 
            get_ith_attr_id.append(dic.tables[constants.ATTR_LABEL(i)].get_id)

        token_seqs = []
        label_seqs = []          # list of semantic attribute sequences
        attr_seqs_list = [[] for i in range(num_attrs)]

        ins_cnt = 0
        word_clm = self.token_index
        label_clm = self.label_index

        with open(path) as f:
            uni_seq = [] 
            label_seq = []
            attr_seq_list = [[] for i in range(num_attrs)]
     
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) == 0:
                    if len(uni_seq) > 0:
                        token_seqs.append(uni_seq)
                        uni_seq = []
                        label_seqs.append(label_seq)
                        label_seq = []
                        
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
                    attrs[i] = self.preprocess_attribute(
                        array[self.attr_indexes[i]], self.attr_depths[i], self.attr_target_labelsets[i])
                    attr_tmp = attrs[i] if self.attr_chunking_flags[i] else get_ith_attr_id[i](
                        attrs[i], update=train)
                    attr_seq_list[i].append(attr_tmp)

                update_token = self.to_be_registered(token, train)
                uni_seq.append(get_unigram_id(token, update=update_token))

                label = array[label_clm] if len(array) > label_clm else constants.NONE_SYMBOL
                label_seq.append(get_label_id(label, update=train))

            # register last sentenece
            if len(uni_seq) > 0:
                token_seqs.append(uni_seq)
                label_seqs.append(label_seq)
                for i, attr_seq in enumerate(attr_seq_list):
                    if self.attr_chunking_flags[i]:
                        attr_seq = [get_ith_attr_id[i](attr, update=train) for attr in 
                                    data_loader.get_labelseq_BIOES(attr_seq)]
                    attr_seqs_list[i].append(attr_seq)

        inputs = [token_seqs]
        inputs.append(attr_seqs_list[0] if len(attr_seqs_list) > 0 else None)
        outputs = [label_seqs]

        return Data(inputs, outputs), dic


    def load_decode_data_WL(self, path, dic):
        attr_delim = self.attr_delim if self.attr_delim else constants.WL_ATTR_DELIM
        num_attrs = len(self.attr_indexes)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_attr_id = dic.tables[constants.ATTR_LABEL(0)].get_id if num_attrs > 0 else None

        org_token_seqs = []
        org_attr_seqs = []      # second or later attribute is ignored
        token_seqs = []
        attr_seqs = []

        ins_cnt = 0
        word_clm = self.token_index

        with open(path) as f:
            org_token_seq = []
            org_attr_seq = []
            token_seq = []
            attr_seq_list = []

            for line in f:
                line = self.normalize_input_line(line)
                if len(line) == 0:
                    if len(token_seq) > 0:
                        org_token_seqs.append(org_token_seq)
                        org_token_seq = []

                        token_seqs.append(token_seq)
                        token_seq = []

                        if num_attrs > 0:
                            if self.attr_chunking_flags[0]:
                                org_attr_seq = [attr for attr in data_loader.get_labelseq_BIOES(attr_seq)]
                            org_attr_seqs.append(org_attr_seq)

                            attr_seq = [get_attr_id(attr) for attr in org_attr_seq]
                            attr_seqs.append(attr_seq)
                            
                        org_attr_seq = []
                        attr_seq = []

                        ins_cnt += 1
                        if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                            print('Read', ins_cnt, 'sentences', file=sys.stderr)

                    continue

                elif line[0] == constants.COMMENT_SYM:
                    continue

                array = line.split(attr_delim)
                org_token = array[word_clm]
                org_token_seq.append(org_token)
                token_seq.append(get_unigram_id(self.preprocess_token(org_token)))

                attrs = [None] * max(num_attrs, 1)
                if num_attrs > 0:
                    attr = self.preprocess_attribute(
                        array[self.attr_indexes[0]], self.attr_depths[0], self.attr_target_labelsets[0])
                    org_attr_seq.append(attr)

            # register last sentenece
            if len(token_seq) > 0:
                org_token_seqs.append(org_token_seq)
                token_seqs.append(token_seq)

                if num_attrs > 0:
                    if self.attr_chunking_flags[0]:
                        org_attr_seq = [attr for attr in data_loader.get_labelseq_BIOES(attr_seq)]
                    org_attr_seqs.append(org_attr_seq)

                    attr_seq = [get_attr_id(attr) for attr in org_attr_seq]
                    attr_seqs.append(attr_seq)

        inputs = [token_seqs, None]
        outputs = []
        outputs.append(attr_seqs if num_attrs > 0 else None)
        orgdata = [org_token_seqs]
        orgdata.append(org_attr_seqs if num_attrs > 0 else None)

        return RestorableData(inputs, outputs, orgdata=orgdata)


    def load_decode_data_SL(self, path, dic):
        attr_delim = self.attr_delim if self.attr_delim else constants.SL_ATTR_DELIM
        num_attrs = len(self.attr_indexes)
        word_clm = self.token_index

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_attr_id = dic.tables[constants.ATTR_LABEL(0)].get_id if num_attrs > 0 else None

        org_token_seqs = []
        org_attr_seqs = []      # second or later attribute is ignored
        token_seqs = []
        attr_seqs = []

        ins_cnt = 0
        with open(path) as f:
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) <= 1:
                    continue
     
                elif line[0] == constants.COMMENT_SYM:
                    continue
                
                org_arr = line.split(constants.SL_TOKEN_DELIM)

                org_token_seq = [elem.split(attr_delim)[word_clm] for elem in org_arr]
                org_token_seqs.append(org_token_seq)
                token_seq = [get_unigram_id(self.preprocess_token(token)) for token in org_token_seq]
                token_seqs.append(token_seq)
 
                if num_attrs > 0:
                    org_token_seq = [elem.split(attr_delim)[0] for elem in org_arr]
                    org_attr_seq = [
                        self.preprocess_attribute(
                            elem.split(attr_delim)[self.attr_indexes[0]], 
                            self.attr_depths[0], self.attr_target_labelsets[0])
                        for elem in org_arr]
                    org_attr_seqs.append(org_attr_seq)
                    attr_seq = [get_attr_id(attr) for attr in org_attr_seq]
                    attr_seqs.append(attr_seq)

                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
     
        inputs = [token_seqs, None]
        outputs = []
        outputs.append(attr_seqs if num_attrs > 0 else None)
        orgdata = [org_token_seqs]
        orgdata.append(org_attr_seqs if num_attrs > 0 else None)

        return RestorableData(inputs, outputs, orgdata=orgdata)


def init_dictionary(num_attrs=0):
    dic = dictionary.Dictionary()

    # unigram
    dic.create_table(constants.UNIGRAM)
    dic.tables[constants.UNIGRAM].set_unk(constants.UNK_SYMBOL)

    # semantic label
    dic.create_table(constants.SEM_LABEL)

    # attributes
    for i in range(num_attrs):
        dic.create_table(constants.ATTR_LABEL(i))
        # dic.tables[constants.ATTR_LABEL(i)].set_unk(constants.UNK_SYMBOL)

    return dic
