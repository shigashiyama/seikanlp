import constants
import dictionary
from data import data_loader
from data.data_loader import DataLoader, Data, RestorableData

import numpy as np


class SegmentationDataLoader(DataLoader):
    def __init__(self,
                 token_index=1,
                 attr_indexes=[],
                 attr_depths=[],
                 attr_target_labelsets=[],
                 attr_delim=None,
                 use_bigram=False,
                 use_tokentype=False,
                 use_chunk_trie=False,
                 bigram_max_vocab_size=-1,
                 bigram_freq_threshold=1,
                 chunk_max_vocab_size=-1,
                 chunk_freq_threshold=1,
                 max_chunk_len=4,
                 add_gold_chunk=True,
                 add_unknown_pretrained_chunk=True,
                 unigram_vocab=set(),
                 bigram_vocab=set(),
                 chunk_vocab=set(),
    ):
        self.token_index = token_index
        self.attr_indexes = attr_indexes
        self.attr_depths = attr_depths
        self.attr_target_labelsets = attr_target_labelsets
        self.attr_delim = attr_delim
        self.use_bigram = use_bigram
        self.use_tokentype = use_tokentype
        self.use_chunk_trie = use_chunk_trie
        self.bigram_max_vocab_size=bigram_max_vocab_size
        self.bigram_freq_threshold=bigram_freq_threshold
        self.chunk_max_vocab_size=chunk_max_vocab_size
        self.chunk_freq_threshold=chunk_freq_threshold
        self.max_chunk_len = max_chunk_len
        self.add_gold_chunk = add_gold_chunk
        self.add_unknown_pretrained_chunk = add_unknown_pretrained_chunk
        self.unigram_vocab=unigram_vocab
        self.bigram_vocab=bigram_vocab
        self.chunk_vocab=chunk_vocab
        self.freq_bigrams = set()
        self.freq_chunks = set()


    def register_chunks(self, sen, unigram_seq, get_chunk_id, label_seq=None, train=True):
        if train:
            spans_gold = get_segmentation_spans(label_seq)
        for n in range(1, self.max_chunk_len+1):
            span_ngrams = data_loader.create_all_char_ngram_indexes(unigram_seq, n)
            cid_ngrams = [unigram_seq[span[0]:span[1]] for span in span_ngrams]
            str_ngrams = [sen[span[0]:span[1]] for span in span_ngrams]
            for span, cn, sn in zip (span_ngrams, cid_ngrams, str_ngrams):
                is_pretrained_chunk = not self.chunk_vocab or sn in self.chunk_vocab
                if train:
                    is_gold_chunk = self.add_gold_chunk and span in spans_gold
                    pass_freq_filter = not self.freq_chunks or sn in self.freq_chunks
                    if pass_freq_filter and (is_gold_chunk or is_pretrained_chunk):
                        ci = get_chunk_id(cn, sn, True)
                else:
                    if self.add_unknown_pretrained_chunk and is_pretrained_chunk:
                        ci = get_chunk_id(cn, sn, True)


    def load_gold_data(self, path, data_format, dic=None, train=True):
        if data_format == constants.WL_FORMAT:
            if self.bigram_freq_threshold > 1 or self.bigram_max_vocab_size > 0:
                self.freq_bigrams = self.get_frequent_bigrams_WL(
                    path, self.bigram_freq_threshold, self.bigram_max_vocab_size)

            if self.chunk_freq_threshold > 1 or self.chunk_max_vocab_size > 0:
                self.freq_chunks = self.get_frequent_ngrams_WL(
                    path, self.chunk_freq_threshold, self.chunk_max_vocab_size, self.max_chunk_len)

            data, dic = self.load_gold_data_WL(path, dic, train)

        else:
            if self.bigram_freq_threshold > 1 or self.bigram_max_vocab_size > 0:
                self.freq_bigrams = self.get_frequent_bigrams_SL(
                    path, self.bigram_freq_threshold, self.bigram_max_vocab_size)

            if self.chunk_freq_threshold > 1 or self.chunk_max_vocab_size > 0:
                self.freq_chunks = self.get_frequent_ngrams_SL(
                    path, self.chunk_freq_threshold, self.chunk_max_vocab_size, self.max_chunk_len)

            data, dic = self.load_gold_data_SL(path, dic, train)

        return data, dic


    def load_decode_data(self, path, data_format, dic=None):
        return self.load_decode_data_SL(path, dic)


    def parse_commandline_input(self, line, dic, use_attention=False, use_concat=False):
        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_bigram_id = dic.tables[constants.BIGRAM].get_id if self.use_bigram else None
        get_toktype_id = dic.tables[constants.TOKEN_TYPE].get_id if self.use_tokentype else None
        get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id if self.use_chunk_trie else None

        org_token_seq = [char for char in line]
        org_token_seqs = [org_token_seq]

        uni_seq = [get_unigram_id(char) for char in line]
        uni_seqs = [uni_seq]

        bi_seqs = []
        if self.use_bigram:
            str_bigrams = data_loader.create_all_char_ngrams(line, 2)
            str_bigrams.append('{}{}'.format(line[-1], constants.EOS))
            bi_seq = [get_bigram_id(sb, update=False) for sb in str_bigrams]
            bi_seqs = [bi_seqs]

        if self.use_chunk_trie:
            self.register_chunks(line, uni_seq, get_chunk_id, train=False)

        type_seqs = []
        if self.use_tokentype:
            type_seq = [[get_toktype_id(get_char_type(char)) for char in line]]

        inputs = [uni_seqs, bi_seqs, type_seqs, None] # TODO fix
        outputs = []
        orgdata = [org_token_seqs]

        return RestorableData(inputs, outputs, orgdata=orgdata)


    def load_gold_data_WL(self, path, dic=None, train=True):
        attr_delim = self.attr_delim if self.attr_delim else constants.WL_ATTR_DELIM
        num_attrs = len(self.attr_indexes)

        if not dic:
            dic = init_dictionary(
                use_bigram=self.use_bigram,
                use_tokentype=self.use_tokentype,
                num_attrs=num_attrs,
                use_chunk_trie=self.use_chunk_trie)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_bigram_id = dic.tables[constants.BIGRAM].get_id if self.use_bigram else None
        get_toktype_id = dic.tables[constants.TOKEN_TYPE].get_id if self.use_tokentype else None
        get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id if self.use_chunk_trie else None
        get_seg_id = dic.tables[constants.SEG_LABEL].get_id
        
        token_seqs = []
        bigram_seqs = []
        toktype_seqs = []
        seg_seqs = []               # list of segmentation label sequences
        attr_seqs_list = [[] for i in range(num_attrs)]

        ins_cnt = 0
        word_clm = self.token_index
        with open(path) as f:
            uni_seq = []
            bi_seq = []
            type_seq = []
            seg_seq = []
            sen = ''

            for line in f:
                line = self.normalize_input_line(line)
                if len(line) == 0:
                    if len(uni_seq) > 0:
                        if self.use_bigram:
                            str_bigrams = data_loader.create_all_char_ngrams(sen, 2)
                            str_bigrams.append('{}{}'.format(sen[-1], constants.EOS))
                            bi_seq = [
                                get_bigram_id(sb, update=self.to_be_registered(
                                    sb, train, self.freq_bigrams, self.bigram_vocab))
                                for sb in str_bigrams]
     
                        if self.use_chunk_trie:
                            self.register_chunks(sen, uni_seq, get_chunk_id, seg_seq, train=train)
     
                        token_seqs.append(uni_seq)

                        uni_seq = []
                        if bi_seq:
                            bigram_seqs.append(bi_seq)
                            bi_seq = []
                        if type_seq:
                            toktype_seqs.append(type_seq)
                            type_seq = []
                        if seg_seq:
                            seg_seqs.append(seg_seq)
                            seg_seq = []
                        sen = ''
     
                        ins_cnt += 1
                        if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                            print('Read', ins_cnt, 'sentences', file=sys.stderr)

                    continue
                    
                elif line[0] == constants.COMMENT_SYM:
                    continue

                array = line.split(attr_delim)
                attr = ''
                token = array[word_clm]
                tlen = len(token)
                sen += token

                # only first attribute (usually POS) is used
                if num_attrs > 0:
                    attr = self.preprocess_attribute(
                        array[self.attr_indexes[0]], self.attr_depths[0], self.attr_target_labelsets[0])
                    if train:
                        for seg_lab in constants.SEG_LABELS:
                            get_seg_id('{}-{}'.format(seg_lab, attr), True)
     
                uni_seq.extend([get_unigram_id(token[i], True) for i in range(tlen)])

                seg_seq.extend(
                    [get_seg_id(
                        data_loader.get_label_BIES(i, tlen-1, cate=attr), update=train) for i in range(tlen)])
                if self.use_tokentype:
                    type_seq.extend(
                        [get_toktype_id(get_char_type(token[i]), update_tokens[i]) for i in range(tlen)])
                        
            # register last sentenece
            if uni_seq:
                if self.use_bigram:  # TODO check
                    str_bigrams = create_all_char_ngrams(pstr, 2)
                    str_bigrams.append('{}{}'.format(pstr[-1], constants.EOS))
                    bi_seq = [
                        get_bigram_id(sb, update=to_be_registered(sb, train, freq_bigrams, refer_bigrams))
                        for sb in str_bigrams]
     
                if self.use_chunk_trie:
                    self.register_chunks(sen, uni_seq, get_chunk_id, seg_seq, train=train)

                token_seqs.append(uni_seq)
                if bi_seq:
                    bigram_seqs.append(bi_seq)
                if type_seq:
                   toktype_seqs.append(type_seq)
                if seg_seq:
                    seg_seqs.append(seg_seq)
     
        inputs = [token_seqs]
        inputs.append(bigram_seqs if bigram_seqs else None)
        inputs.append(toktype_seqs if toktype_seqs else None)
        inputs.append(None) # TODO fix
        outputs = [seg_seqs]
     
        return Data(inputs, outputs), dic


    """
    Read data with SL (one Sentence in one Line) format.
    The following format is expected:
      word1_attr1 word2_attr2 ... wordn_attrn
    where attr can be ommited.
    """
    def load_gold_data_SL(self, path, dic=None, train=True):
        attr_delim = self.attr_delim if self.attr_delim else constants.SL_ATTR_DELIM
        num_attrs = len(self.attr_indexes)

        if not dic:
            dic = init_dictionary(
                use_bigram=self.use_bigram,
                use_tokentype=self.use_tokentype,
                num_attrs=num_attrs,
                use_chunk_trie=self.use_chunk_trie)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_bigram_id = dic.tables[constants.BIGRAM].get_id if self.use_bigram else None
        get_toktype_id = dic.tables[constants.TOKEN_TYPE].get_id if self.use_tokentype else None
        get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id if self.use_chunk_trie else None
        get_seg_id = dic.tables[constants.SEG_LABEL].get_id

        token_seqs = []
        bigram_seqs = []
        toktype_seqs = []
        seg_seqs = []           # list of segmentation label sequences

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
                bi_seq = []
                type_seq = []
                seg_seq = []
     
                for entry in entries:
                    array = entry.split(attr_delim)
                    attr = ''
                    token = array[0]
                    tlen = len(token)
     
                    # only first attribute (usually POS) is used
                    if num_attrs > 0:
                        attr = self.preprocess_attribute(
                            array[self.attr_indexes[0]], self.attr_depths[0], self.attr_target_labelsets[0])
                        if train:
                            for seg_lab in constants.SEG_LABELS:
                                get_seg_id('{}-{}'.format(seg_lab, attr), True)

                    uni_seq.extend([get_unigram_id(token[i], True) for i in range(tlen)])
                    seg_seq.extend([get_seg_id(data_loader.get_label_BIES(
                        i, tlen-1, cate=attr), update=train) for i in range(tlen)])
     
                    if self.use_tokentype:
                        type_seq.extend(
                            [get_toktype_id(get_char_type(ptoken[i]), update=train) for i in range(tlen)])

                if self.use_bigram:
                    str_bigrams = data_loader.create_all_char_ngrams(sen, 2)
                    str_bigrams.append('{}{}'.format(sen[-1], constants.EOS))
                    bi_seq = [
                        get_bigram_id(
                            sb, update=self.to_be_registered(sb, train, self.freq_bigrams, self.bigram_vocab))
                        for sb in str_bigrams]
  
                if self.use_chunk_trie:
                    self.register_chunks(sen, uni_seq, get_chunk_id, seg_seq, train=train)

                token_seqs.append(uni_seq)
                if bi_seq:
                    bigram_seqs.append(bi_seq)
                if type_seq:
                    toktype_seqs.append(type_seq)
                seg_seqs.append(seg_seq)
     
                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
     
        inputs = [token_seqs]
        inputs.append(bigram_seqs if bigram_seqs else None)
        inputs.append(toktype_seqs if toktype_seqs else None)
        inputs.append(None) # TODO fix
        outputs = [seg_seqs]
     
        return Data(inputs, outputs), dic


    def load_decode_data_SL(self, path, dic):
        num_attrs = len(self.attr_indexes)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_bigram_id = dic.tables[constants.BIGRAM].get_id if self.use_bigram else None
        get_toktype_id = dic.tables[constants.TOKEN_TYPE].get_id if self.use_tokentype else None
        get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id if self.use_chunk_trie else None
        get_seg_id = dic.tables[constants.SEG_LABEL].get_id
        get_ith_attr_id = []
        for i in range(num_attrs):
            get_ith_attr_id.append(dic.tables[constants.ATTR_LABEL(i)].get_id)

        org_token_seqs = []
        token_seqs = []
        bigram_seqs = []
        toktype_seqs = []

        ins_cnt = 0
        with open(path) as f:
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) <= 1:
                    continue

                elif line[0] == constants.COMMENT_SYM:
                    continue

                org_token_seqs.append([char for char in line])
                uni_seq = [get_unigram_id(char) for char in line]
                token_seqs.append(uni_seq)

                if self.use_tokentype:
                    toktype_seqs.append([get_toktype_id(get_char_type(char)) for char in line])

                if self.use_bigram:
                    str_bigrams = data_loader.create_all_char_ngrams(sen, 2)
                    str_bigrams.append('{}{}'.format(sen[-1], constants.EOS))
                    bi_seq = [
                        get_bigram_id(
                            sb, update=self.to_be_registered(sb, train, self.freq_bigrams, self.bigram_vocab))
                        for sb in str_bigrams]
  
                if self.use_chunk_trie:
                    self.register_chunks(sen, uni_seq, get_chunk_id, train=False)

                token_seqs.append(uni_seq)
                if self.use_bigram:
                    bigram_seqs.append(bi_seq)
                if self.use_tokentype:
                    toktype_seqs.append(type_seq)
     
                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
     
        inputs = [token_seqs]
        inputs.append(bigram_seqs if bigram_seqs else None)
        inputs.append(toktype_seqs if toktype_seqs else None)
        inputs.append(None) # TODO fix
        outputs = []
        orgdata = [org_token_seqs]
     
        return RestorableData(inputs, outputs, orgdata=orgdata)


def get_char_type(char):
    if len(char) != 1:
        return
 
    if char == '\u30fc':
        return constants.TYPE_LONG
    elif '\u3041' <= char <= '\u3093':
        return constants.TYPE_HIRA
    elif '\u30A1' <= char <= '\u30F4':
        return constants.TYPE_KATA
    elif '\u4e8c' <= char <= '\u9fa5':
        return constants.TYPE_KANJI
    elif '\uff10' <= char <= '\uff19' or '0' <= char <= '9':
        return constants.TYPE_DIGIT
    elif '\uff21' <= char <= '\uff5a' or 'A' <= char <= 'z':
        return constants.TYPE_ALPHA
    elif char == '\u3000' or char == ' ':
        return constants.TYPE_SPACE
    elif '!' <= char <= '~':
        return constants.TYPE_ASCII_OTHER
    else:
        return constants.TYPE_SYMBOL
    

def get_segmentation_spans(label_seq):
    spans = []
    first = -1
    for i, label in enumerate(label_seq):
        if label == 3: # 'S'
            spans.append((i, i+1))
        elif label == 0: # 'B'
            first = i
        elif label == 2 : # 'E'
            spans.append((first, i+1))
    return spans


def init_dictionary(
        use_bigram=False, 
        use_tokentype=False, 
        num_attrs=0, 
        use_chunk_trie=False): 

    dic = dictionary.Dictionary()

    # unigram
    dic.create_table(constants.UNIGRAM)
    dic.tables[constants.UNIGRAM].set_unk(constants.UNK_SYMBOL)

    # segmentation label
    dic.create_table(constants.SEG_LABEL)
    if num_attrs == 0:
        for lab in constants.SEG_LABELS:
            dic.tables[constants.SEG_LABEL].get_id(lab, update=True)

    # bigram
    if use_bigram:
        dic.create_table(constants.BIGRAM)
        dic.tables[constants.BIGRAM].set_unk(constants.UNK_SYMBOL)

    # token type
    if use_tokentype:
        dic.create_table(constants.TOKEN_TYPE)
        dic.tables[constants.TOKEN_TYPE].set_unk(constants.UNK_SYMBOL)

    # attributes
    # for i in range(num_attrs):
    #     dic.create_table(constants.ATTR_LABEL(i))
        # dic.tables[constants.ATTR_LABEL(i)].set_unk(constants.UNK_SYMBOL)

    # chunk
    if use_chunk_trie:
        dic.create_trie(constants.CHUNK)

    return dic


"""
  ave (average) / wave (weighed average):
    chunk_seq:  [w_0, ..., w_{m-1}]
    gchunk_seq: [gold_index(chunk_seq, c_0), ..., gold_index(chunk_seq, c_{n-1})]
    mask1:      [[exist(c_0, w_0), ..., exist(c_0, w_{m-1})],
                 ...
                 [exist(c_{n_1}, w_0), ..., exist(c_{n-1}, w_{m-1})]]

  con (concat):
    feat_seq:   [[word_id(c_0, 0), ..., word_id(c_{n-1}, 0)],
                 ...
                 [word_id(c_0, k-1), ..., word_id(c_{n-1}, k-1)]]

    gchunk_seq: [gold_index([0,...,k-1], c_0), ..., gold_index([0,...,k-1], c_{n-1})]
    mask0:      zero vectors for characters w/o no candidate words

  wcon (weghted concat):
    chunk_seq:  [w[0:0] ... w[n-1:n-1] w[0:1] ... w[n-2:n-1] ... w[0:k-1] ... w[k:n-1]]
    feat_seq:   the same as concat
    gchunk_seq: the same as concat
    mask0:      the same as concat
    mask1:      the same as ave/wave

"""
def add_chunk_sequences(
        data, dic, max_len=4, evaluate=True, use_attention=True, use_concat=True):
    feat_size = sum([h for h in range(max_len+1)])

    token_seqs = data.inputs[0]
    gold_label_seqs = data.outputs[0] if evaluate else None
    gold_chunk_seqs = []
    chunk_seqs = [] if not (use_concat and not use_attention) else None
    feat_seqs = [] if use_concat else None
    masks = []
    trie = dic.tries[constants.CHUNK]

    ins_cnt = 0
    for sen_id, tseq in enumerate(token_seqs):
        if ins_cnt > 0 and ins_cnt % 100000 == 0:
            print('Processed', ins_cnt, 'sentences', file=sys.stderr)
        ins_cnt += 1

        n = len(tseq)
        gchunk_seq = [-1] * n
        feats = [[0] * n for k in range(feat_size)] if use_concat else None
        mask0 = [] if use_concat else None
        mask1 = [] if use_attention or not use_concat else None

        if evaluate:
            lseq = gold_label_seqs[sen_id]
            spans_gold = get_segmentation_spans(lseq)
        else:
            lseq = spans_gold = None

        spans_found = []
        for i in range(n):
            res = trie.common_prefix_search(tseq, i, i + max_len)
            spans_found.extend(res)
        m = len(spans_found)

        if use_concat:
            if use_attention:
                chunk_seq = [0] * sum([max(0, n - h) for h in range(max_len)])
            else:
                chunk_seq = None
        else:
            chunk_seq = [None] * m

        for j, span in enumerate(spans_found):
            is_gold_span = evaluate and span in spans_gold
            if use_concat:
                j = get_chunk_index(span, n)
            cid = trie.get_chunk_id(tseq[span[0]:span[1]])
            if chunk_seq:
                chunk_seq[j] = cid

            for i in range(span[0], span[1]):
                if use_concat:
                    k = token_index2feat_index(i, span)
                    feats[k][i] = cid
                    mask0.append((k,i)) # (feat k, char i) has value; used for concatenation

                if use_attention or not use_concat:
                    mask1.append((i,j)) # (char i, word j) has value; used for attention

                if is_gold_span:
                    gchunk_seq[i] = k if use_concat else j

        # print('ID={}'.format(sen_id))
        # print('x', ' '.join([str(i)+':'+dic.tables[constants.UNIGRAM].id2str[t] for i,t in enumerate(tseq)]))
        # print('l', ' '.join([dic.tables[constants.SEG_LABEL].id2str[l] if l in dic.tables[constants.SEG_LABEL].id2str else '-1' for l in lseq]))
        # if chunk_seq:
        #     print('c', ' '.join([str(i)+':'+trie.id2chunk[c] for i,c in enumerate(chunk_seq)]))
        # print('f', [str(i)+':'+str([trie.id2chunk[c] if c >= 0 else '-1' for c in raw]) for i,raw in enumerate(feats)])
        # print('gc', gchunk_seq)
        # print('mask0', mask0)
        # print('mask1', mask1)
        # for i in range(n):
        #     print(i, token_index2feat_index(i, n, max_len))
        # print()

        # if chunk_seq:           # wave 精度に影響?
        if chunk_seq is not None:
            chunk_seqs.append(chunk_seq)
        if feats:
            feat_seqs.append(feats)
        masks.append((mask0, mask1))
        if evaluate:
            gold_chunk_seqs.append(gchunk_seq)

    data.inputs.append(chunk_seqs)
    data.inputs.append(feat_seqs)
    data.inputs.append(masks)
    if evaluate:
        data.outputs.append(gold_chunk_seqs)


def get_chunk_index(span, sen_len):
    chunk_len = span[1] - span[0]
    ci = sum([sen_len - i for i in range(chunk_len-1)]) + span[0]
    return ci


def token_index2feat_index(ti, span):
    chunk_len = span[1] - span[0]
    fi = sum([i for i in range(chunk_len)]) + (span[1] - ti - 1)
    return fi


FLOAT_MIN = -100000.0           # input for exp
def convert_mask_matrix(
        mask, n_tokens, n_chunks, feat_size, emb_dim, use_attention=True, xp=np):
    if use_attention:
        mask_val = FLOAT_MIN
        non_mask_val = 0.0
    else:
        mask_val = 0.0
        non_mask_val = 1.0

    pairs0 = mask[0]
    pairs1 = mask[1]

    # print(' ', n_tokens, n_chunks, feat_size, emb_dim, 'mask1='+str(len(mask[1])))
    # print(mask[1])

    if pairs0:                  # con or wcon
        mask0 = xp.zeros((feat_size, n_tokens, emb_dim), dtype='f')
        for k, i in pairs0:
            mask0[k][i] = xp.ones((emb_dim,), dtype='f')
        # mask0 = F.concat(mask0, axis=1)
    else:
        mask0 = None   

    if pairs1 is not None:      # ave, wave or wcon
        mask1 = xp.full((n_tokens, n_chunks), mask_val, 'f')
        for i, j in pairs1:
            # print(' -', i, j)
            mask1[i, j] = non_mask_val
    else:
        mask1 = None

    if use_attention:           # wave or wcon
        mask2 = xp.ones((n_tokens, n_chunks), 'f') # zero vectors for characters w/o no candidate words
        for i in range(n_tokens):
            tmp = [k1 == i for k1,k2 in pairs1]
            if len(tmp) == 0 or max(tmp) == False: # not (i,*) in index_pair
                mask2[i] = xp.zeros((n_chunks,), 'f')
    else:
        mask2 = None        

    return (mask0, mask1, mask2)
