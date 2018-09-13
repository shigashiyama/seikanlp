import constants
import dictionary
from data import data_loader
from data.data_loader import DataLoader, Data, RestorableData


class ParsingDataLoader(DataLoader):
    def __init__(self,
                 token_index=1,
                 head_index=2,
                 arc_index=3,
                 attr_indexes=[], 
                 attr_depths=[], 
                 attr_chunking_flags=[],
                 attr_target_labelsets=[],
                 attr_delim=None,
                 use_arc_label=False,
                 lowercasing=False, 
                 normalize_digits=True, 
                 token_max_vocab_size=-1,
                 token_freq_threshold=1, 
                 unigram_vocab=set(), 
    ):
        self.token_index = token_index
        self.head_index = head_index
        self.arc_index = arc_index
        self.attr_indexes = attr_indexes
        self.attr_depths = attr_depths
        self.attr_chunking_flags = attr_chunking_flags
        self.attr_target_labelsets = attr_target_labelsets
        self.attr_delim = attr_delim
        self.use_arc_label = use_arc_label
        self.lowercasing = lowercasing
        self.normalize_digits = normalize_digits
        self.token_max_vocab_size=token_max_vocab_size
        self.token_freq_threshold=token_freq_threshold
        self.unigram_vocab = unigram_vocab
        self.freq_tokens = set()


    def preprocess_token(self, token):
        if token == constants.ROOT_SYMBOL:
            return token
        else:
            return super().preprocess_token(token)


    def load_gold_data(self, path, data_format=None, dic=None, train=True):
        if self.token_freq_threshold > 1 or self.token_max_vocab_size > 0:
            self.freq_tokens = self.get_frequent_tokens_WL(
                path, self.token_freq_threshold, self.token_max_vocab_size)
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
        root_token = constants.ROOT_SYMBOL
        if constants.ATTR_LABEL(0) in dic.tables:
            use_attr0 = True
            get_attr0_id = dic.tables[constants.ATTR_LABEL(0)].get_id
        else:
            use_attr0 = False
            get_attr0_id = None

        org_arr = line.split(' ')
        if use_attr0:
            org_attr0_seq = [
                self.preprocess_attribute(
                    elem.split(attr_delim)[1] if attr_delim in elem else constants.UNK_SYMBOL,
                    0, #self.attr_depths[0], 
                    None, #self.attr_target_labelsets[0]
                ) for elem in org_arr]
            org_attr0_seq.insert(0, constants.ROOT_SYMBOL)
            org_attr0_seqs = [org_attr0_seq]
            attr0_seq = [get_attr0_id(attr) for attr in org_attr0_seq]
            attr0_seqs = [attr0_seq]
        else:
            org_attr0_seqs = []
            attr0_seqs = []

        org_token_seq = [elem.split(attr_delim)[0] for elem in org_arr]
        org_token_seq.insert(0, constants.ROOT_SYMBOL)
        org_token_seqs = [org_token_seq]
        ptoken_seq = [self.preprocess_token(word) for word in org_token_seq]
        uni_seq = [get_unigram_id(word) for word in ptoken_seq]
        uni_seqs = [uni_seq]

        inputs = [uni_seqs, None, None, None] # TODO fix
        outputs = [attr0_seqs]
        orgdata = [org_token_seqs, org_attr0_seqs]

        return RestorableData(inputs, outputs, orgdata=orgdata)


    def load_gold_data_WL(self, path, dic, train=True):
        attr_delim = self.attr_delim if self.attr_delim else constants.WL_ATTR_DELIM
        num_attrs = len(self.attr_indexes)

        if not dic:
            dic = init_dictionary(
                num_attrs=num_attrs,
                use_arc_label=self.use_arc_label)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_arc_id = dic.tables[constants.ARC_LABEL].get_id if self.use_arc_label else None
        get_ith_attr_id = []
        for i in range(num_attrs): 
            get_ith_attr_id.append(dic.tables[constants.ATTR_LABEL(i)].get_id)

        token_seqs = []
        head_seqs = []          # list of head id sequences
        arc_seqs = []           # list of arc label sequences
        attr_seqs_list = [[] for i in range(num_attrs)]

        ins_cnt = 0
        word_clm = self.token_index
        head_clm = self.head_index
        arc_clm = self.arc_index

        with open(path) as f:
            uni_seq = [get_unigram_id(constants.ROOT_SYMBOL)] 
            head_seq = [constants.NO_PARENTS_ID]
            arc_seq = [constants.NO_PARENTS_ID] if self.use_arc_label else None
            attr_seq_list = [[get_ith_attr_id[i](constants.ROOT_SYMBOL, update=train)] for i in range(num_attrs)]
     
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) == 0:
                    if len(uni_seq) > 0:
                        token_seqs.append(uni_seq)
                        uni_seq = [get_unigram_id(constants.ROOT_SYMBOL)]

                        head_seqs.append(head_seq)
                        head_seq = [constants.NO_PARENTS_ID]
                        
                        if self.use_arc_label:
                            arc_seqs.append(arc_seq)
                            arc_seq = [constants.NO_PARENTS_ID]

                        for i, attr_seq in enumerate(attr_seq_list):
                            if self.attr_chunking_flags[i]:
                                # TODO fix code for ROOT
                                attr_seq = [get_ith_attr_id[i](attr, update=train) for attr in 
                                            data_loader.get_labelseq_BIOES(attr_seq)]
                            attr_seqs_list[i].append(attr_seq)
                            attr_seq_list = [[get_ith_attr_id[i](constants.ROOT_SYMBOL)] for i in range(num_attrs)]

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

                update_token = self.to_be_registered(token, train, self.freq_tokens, self.unigram_vocab)
                uni_seq.append(get_unigram_id(token, update=update_token))

                head = int(array[head_clm])
                if head < 0:
                    head = 0
                head_seq.append(head)
     
                if self.use_arc_label:
                    arc = array[arc_clm]
                    arc_seq.append(get_arc_id(arc, update=train))

            # register last sentenece
            if len(uni_seq) > 1: # initialized sequence contains ROOT 
                # org_token_seqs.append(org_token_seq)
                token_seqs.append(uni_seq)
                head_seqs.append(head_seq)
                arc_seqs.append(arc_seq)
                for i, attr_seq in enumerate(attr_seq_list):
                    if self.attr_chunking_flags[i]:
                        attr_seq = [get_ith_attr_id[i](attr, update=train) for attr in 
                                    data_loader.get_labelseq_BIOES(attr_seq)]
                    attr_seqs_list[i].append(attr_seq)

        inputs = [token_seqs]
        # TODO fix
        inputs.append(None) # bigram
        inputs.append(None) # toktype
        inputs.append(attr_seqs_list[1] if len(attr_seqs_list) > 1 else None)
     
        outputs = []
        outputs.append(attr_seqs_list[0] if len(attr_seqs_list) > 0 else None)
        outputs.append(head_seqs)
        if self.use_arc_label:
            outputs.append(arc_seqs)

        return Data(inputs, outputs), dic


    def load_decode_data_WL(self, path, dic):
        # to be implemented
        pass


    def load_decode_data_SL(self, path, dic):
        attr_delim = self.attr_delim if self.attr_delim else constants.SL_ATTR_DELIM
        num_attrs = len(self.attr_indexes)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_attr_id = dic.tables[constants.ATTR_LABEL(0)].get_id if num_attrs > 0 else None
        root_token = constants.ROOT_SYMBOL
        if num_attrs > 0:
            root_attr_id = dic.tables[constants.ATTR_LABEL(0)].get_id(constants.ROOT_SYMBOL)

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

                org_token_seq = [elem.split(attr_delim)[0] for elem in org_arr]
                org_token_seq.insert(0, root_token)
                org_token_seqs.append(org_token_seq)
                ptoken_seq = [self.preprocess_token(token) for token in org_token_seq]
                token_seq = [get_unigram_id(ptoken, update=ptoken in self.unigram_vocab) for ptoken in ptoken_seq]
                token_seqs.append(token_seq)
 
                if num_attrs > 0:
                    org_token_seq = [elem.split(attr_delim)[0] for elem in org_arr]
                    org_attr_seq = [
                        self.preprocess_attribute(
                            elem.split(attr_delim)[1], self.attr_depths[0], self.attr_target_labelsets[0])
                        for elem in org_arr]
                    org_attr_seq.insert(0, root_token)
                    org_attr_seqs.append(org_attr_seq)
                    attr_seq = [get_attr_id(attr) for attr in org_attr_seq]
                    attr_seqs.append(attr_seq)

                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
     
        inputs = [token_seqs]
        # TODO fix
        inputs.append(None) # bigram
        inputs.append(None) # toktype
        inputs.append(attr_seqs if num_attrs > 1 else None)
        outputs = []
        outputs.append(attr_seqs if num_attrs > 0 else None)
        orgdata = [org_token_seqs]
        orgdata.append(org_attr_seqs if num_attrs > 0 else None)

        return RestorableData(inputs, outputs, orgdata=orgdata)


def init_dictionary(
        num_attrs=0,
        use_arc_label=False):

    dic = dictionary.Dictionary()

    # unigram
    dic.create_table(constants.UNIGRAM)
    dic.tables[constants.UNIGRAM].set_unk(constants.UNK_SYMBOL)
    dic.tables[constants.UNIGRAM].get_id(constants.ROOT_SYMBOL, update=True)

    # attributes
    for i in range(num_attrs):
        dic.create_table(constants.ATTR_LABEL(i))
        # dic.tables[constants.ATTR_LABEL(i)].set_unk(constants.UNK_SYMBOL)

    # arc label
    if use_arc_label:
        dic.create_table(constants.ARC_LABEL)

    return dic
