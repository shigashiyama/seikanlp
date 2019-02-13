import pickle
import sys

import numpy as np

import constants
import dictionary
from data_loaders import data_loader
from data_loaders.data_loader import DataLoader, Data, RestorableData


class Vocabularies(object):     # use for FMeasureEvaluatorForEachVocab
    def __init__(self, *tries):
        self.tries = tries


    def __len__(self):
        return len(self.tries)


    def get_index(self, sen, span):
        cseq = sen[span[0]:span[1]]
        for i, trie in enumerate(self.tries):
            cid = trie.get_chunk_id(cseq)
            if cid > 0:         # not UNK
                return i

        return -1


class SegmentationDataLoader(DataLoader):
    def __init__(self,
                 token_index=1,
                 attr_indexes=[],
                 attr_depths=[],
                 attr_target_labelsets=[],
                 attr_delim=None,
                 use_bigram=False,
                 use_chunk_trie=False, # for a hybrid model
                 bigram_max_vocab_size=-1,
                 bigram_freq_threshold=1,
                 chunk_max_vocab_size=-1,
                 chunk_freq_threshold=1,
                 min_chunk_len=1,
                 max_chunk_len=4,
                 add_gold_chunk=True,
                 add_nongold_chunk=True,
                 add_unknown_pretrained_chunk=True,
                 unigram_vocab=set(),
                 bigram_vocab=set(),
                 chunk_vocab=set(),
                 generate_ngram_chunks=False,
                 trie_ext=None,
    ):
        self.token_index = token_index
        self.attr_indexes = attr_indexes
        self.attr_depths = attr_depths
        self.attr_target_labelsets = attr_target_labelsets
        self.attr_delim = attr_delim
        self.use_bigram = use_bigram
        self.use_chunk_trie = use_chunk_trie
        self.bigram_max_vocab_size=bigram_max_vocab_size
        self.bigram_freq_threshold=bigram_freq_threshold
        self.chunk_max_vocab_size=chunk_max_vocab_size
        self.chunk_freq_threshold=chunk_freq_threshold
        self.min_chunk_len = min_chunk_len
        self.max_chunk_len = max_chunk_len
        self.add_gold_chunk = add_gold_chunk
        self.add_nongold_chunk = add_nongold_chunk
        self.add_unknown_pretrained_chunk = add_unknown_pretrained_chunk
        self.unigram_vocab=unigram_vocab
        self.bigram_vocab=bigram_vocab
        self.chunk_vocab=chunk_vocab
        self.freq_bigrams = set()
        self.freq_chunks = set()
        self.generate_ngram_chunks = generate_ngram_chunks
        self.trie_ext = trie_ext


    def gen_vocabs(self, train_path, char_table, *pretrain_dics, data_format=None):
        trie0 = self.create_trie(train_path, char_table, data_format) # gold words in train data
        print('\nCreate word trie with size={} from {}'.format(len(trie0), train_path), file=sys.stderr)
        tries = [trie0]
        
        for i in range(len(pretrain_dics)):
            # i == 0: words in pretrain during training
            # i == 1: words in pretrain during testing

            trie = pretrain_dics[i].tries[constants.CHUNK]
            tries.append(trie)
            print('Load word trie with size={}'.format(len(trie)), file=sys.stderr)
        print('', file=sys.stderr)

        return Vocabularies(*tries)


    def create_trie(self, path, unigram_table, data_format=None):
        attr_delim = self.attr_delim if self.attr_delim else constants.SL_ATTR_DELIM

        trie = dictionary.MapTrie()
        with open(path) as f:
            if data_format == constants.WL_FORMAT:
                word_clm = self.token_index
                for line in f:
                    line = self.normalize_input_line(line)
                    if len(line) == 0:
                        continue
                    elif line[0] == constants.COMMENT_SYM:
                        continue
     
                    array = line.split(attr_delim)
                    token = array[word_clm]
                    tlen = len(token)
                    char_ids = [unigram_table.get_id(token[i]) for i in range(tlen)]
                    trie.get_chunk_id(char_ids, token, True)

            else:
                for line in f:
                    line = self.normalize_input_line(line)
                    if len(line) == 0:
                        continue
                    elif line[0] == constants.COMMENT_SYM:
                        continue
     
                    entries = line.split(constants.SL_TOKEN_DELIM)
                    for entry in entries:
                        array = entry.split(attr_delim)
                        token = array[0]
                        tlen = len(token)
                        char_ids = [unigram_table.get_id(token[i]) for i in range(tlen)]
                        trie.get_chunk_id(char_ids, token, True)

        return trie


    def register_chunks(self, sen, unigram_seq, get_chunk_id, label_seq=None, train=True):
        if train:
            spans_gold = get_segmentation_spans(label_seq)
        for n in range(self.min_chunk_len, self.max_chunk_len+1):
            span_ngrams = data_loader.create_all_char_ngram_indexes(unigram_seq, n)
            cid_ngrams = [unigram_seq[span[0]:span[1]] for span in span_ngrams]
            str_ngrams = [sen[span[0]:span[1]] for span in span_ngrams]
            for span, cn, sn in zip (span_ngrams, cid_ngrams, str_ngrams):
                is_pretrained_chunk = self.chunk_vocab and sn in self.chunk_vocab.wv.vocab
                is_generable_chunk = (self.chunk_vocab and 
                                      self.generate_ngram_chunks and
                                      (not self.trie_ext or self.trie_ext.get_chunk_id(sn) > 0) and
                                      sn in self.chunk_vocab.wv # for fasttext
                                      # and span in spans_gold # tmp cheat
                )

                if train:
                    is_gold_chunk = self.add_gold_chunk and span in spans_gold
                    is_pretrained_chunk = is_pretrained_chunk and self.add_nongold_chunk
                    pass_freq_filter = not self.freq_chunks or sn in self.freq_chunks
                    if pass_freq_filter and (is_gold_chunk or is_pretrained_chunk):
                        ci = get_chunk_id(cn, sn, True)
                else:
                    if (self.add_unknown_pretrained_chunk and
                        is_pretrained_chunk or is_generable_chunk
                    ):
                        ci = get_chunk_id(cn, sn, True)


    def load_gold_data(self, path, data_format, dic=None, train=True):
        if data_format == constants.WL_FORMAT:
            if self.bigram_freq_threshold > 1 or self.bigram_max_vocab_size > 0:
                self.freq_bigrams = self.get_frequent_bigrams_WL(
                    path, self.bigram_freq_threshold, self.bigram_max_vocab_size)

            if self.chunk_freq_threshold > 1 or self.chunk_max_vocab_size > 0:
                self.freq_chunks = self.get_frequent_ngrams_WL(
                    path, self.chunk_freq_threshold, 
                    self.chunk_max_vocab_size, self.min_chunk_len, self.max_chunk_len)

            data, dic = self.load_gold_data_WL(path, dic, train)

        else:
            if self.bigram_freq_threshold > 1 or self.bigram_max_vocab_size > 0:
                self.freq_bigrams = self.get_frequent_bigrams_SL(
                    path, self.bigram_freq_threshold, self.bigram_max_vocab_size)

            if self.chunk_freq_threshold > 1 or self.chunk_max_vocab_size > 0:
                self.freq_chunks = self.get_frequent_ngrams_SL(
                    path, self.chunk_freq_threshold, self.chunk_max_vocab_size, 
                    self.min_chunk_len, self.max_chunk_len)

            data, dic = self.load_gold_data_SL(path, dic, train)

        return data, dic


    def load_decode_data(self, path, data_format, dic=None):
        return self.load_decode_data_SL(path, dic)


    def parse_commandline_input(self, line, dic, use_attention=False, use_concat=False):
        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_bigram_id = dic.tables[constants.BIGRAM].get_id if self.use_bigram else None
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

        inputs = [uni_seqs, bi_seqs, None]
        outputs = []
        orgdata = [org_token_seqs]

        return RestorableData(inputs, outputs, orgdata=orgdata)


    def load_gold_data_WL(self, path, dic=None, train=True):
        attr_delim = self.attr_delim if self.attr_delim else constants.WL_ATTR_DELIM
        num_attrs = len(self.attr_indexes)
        word_clm = self.token_index

        if not dic:
            dic = init_dictionary(
                use_bigram=self.use_bigram,
                num_attrs=num_attrs,
                use_chunk_trie=self.use_chunk_trie)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_bigram_id = dic.tables[constants.BIGRAM].get_id if self.use_bigram else None
        get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id if self.use_chunk_trie else None
        get_seg_id = dic.tables[constants.SEG_LABEL].get_id
        
        token_seqs = []
        bigram_seqs = []
        seg_seqs = []               # list of segmentation label sequences
        attr_seqs_list = [[] for i in range(num_attrs)]

        ins_cnt = 0

        with open(path) as f:
            uni_seq = []
            bi_seq = []
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
                if seg_seq:
                    seg_seqs.append(seg_seq)
     
        inputs = [token_seqs]
        inputs.append(bigram_seqs if bigram_seqs else None)
        inputs.append(None)
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
        word_clm = self.token_index

        if not dic:
            dic = init_dictionary(
                use_bigram=self.use_bigram,
                num_attrs=num_attrs,
                use_chunk_trie=self.use_chunk_trie)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_bigram_id = dic.tables[constants.BIGRAM].get_id if self.use_bigram else None
        get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id if self.use_chunk_trie else None
        get_seg_id = dic.tables[constants.SEG_LABEL].get_id

        token_seqs = []
        bigram_seqs = []
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
                seg_seq = []
                raw_sen = ''

                for entry in entries:
                    array = entry.split(attr_delim)
                    attr = ''
                    token = array[word_clm]
                    tlen = len(token)
                    raw_sen += token
     
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
     
                if self.use_bigram:
                    str_bigrams = data_loader.create_all_char_ngrams(raw_sen, 2)
                    str_bigrams.append('{}{}'.format(raw_sen[-1], constants.EOS))
                    bi_seq = [
                        get_bigram_id(
                            sb, update=self.to_be_registered(sb, train, self.freq_bigrams, self.bigram_vocab))
                        for sb in str_bigrams]
  
                if self.use_chunk_trie:
                    self.register_chunks(raw_sen, uni_seq, get_chunk_id, seg_seq, train=train)

                token_seqs.append(uni_seq)
                if bi_seq:
                    bigram_seqs.append(bi_seq)
                seg_seqs.append(seg_seq)
     
                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
     
        inputs = [token_seqs]
        inputs.append(bigram_seqs if bigram_seqs else None)
        inputs.append(None)
        outputs = [seg_seqs]
     
        return Data(inputs, outputs), dic


    def load_decode_data_SL(self, path, dic):
        num_attrs = len(self.attr_indexes)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_bigram_id = dic.tables[constants.BIGRAM].get_id if self.use_bigram else None
        get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id if self.use_chunk_trie else None
        get_seg_id = dic.tables[constants.SEG_LABEL].get_id
        get_ith_attr_id = []
        for i in range(num_attrs):
            get_ith_attr_id.append(dic.tables[constants.ATTR_LABEL(i)].get_id)

        org_token_seqs = []
        token_seqs = []
        bigram_seqs = []

        ins_cnt = 0
        with open(path) as f:
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) == 0:
                    continue

                elif line[0] == constants.COMMENT_SYM:
                    continue

                org_token_seqs.append([char for char in line])
                uni_seq = [get_unigram_id(char) for char in line]

                if self.use_bigram:
                    str_bigrams = data_loader.create_all_char_ngrams(line, 2)
                    str_bigrams.append('{}{}'.format(line[-1], constants.EOS))
                    bi_seq = [
                        get_bigram_id(
                            sb, update=self.to_be_registered(
                                sb, False, self.freq_bigrams, self.bigram_vocab))
                        for sb in str_bigrams]
  
                if self.use_chunk_trie:
                    self.register_chunks(line, uni_seq, get_chunk_id, train=False)

                token_seqs.append(uni_seq)
                if self.use_bigram:
                    bigram_seqs.append(bi_seq)
                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
     
        inputs = [token_seqs]
        inputs.append(bigram_seqs if bigram_seqs else None)
        inputs.append(None)
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


def load_external_dictionary(path, dic=None, num_attrs=0):
    # if path.endswith('pickle'):
    #     with open(path, 'rb') as f:
    #         dic = pickle.load(f)
    #         dic.create_id2strs()
    #     return dic

    if not dic:
        dic = init_dictionary(num_attrs=num_attrs, use_chunk_trie=True)

    get_unigram_id = dic.tables[constants.UNIGRAM].get_id
    get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id
    attr_delim = constants.WL_ATTR_DELIM

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(constants.COMMENT_SYM):
                continue

            arr = line.split(attr_delim)
            if len(arr[0]) == 0:
                continue

            word = arr[0]
            char_ids = [get_unigram_id(char, update=True) for char in word]
            word_id = get_chunk_id(char_ids, word, update=True)
                
    dic.create_id2strs()
    return dic


def init_dictionary(
        use_bigram=False, 
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

    # chunk
    if use_chunk_trie:
        dic.init_trie(constants.CHUNK)

    return dic


"""
  avg / wavg
    chunk_seq:  [w_0, ..., w_{m-1}]
    gchunk_seq: [gold_index(chunk_seq, c_0), ..., gold_index(chunk_seq, c_{n-1})]
    mask_ij:      [[exist(c_0, w_0), ..., exist(c_0, w_{m-1})],
                 ...
                 [exist(c_{n_1}, w_0), ..., exist(c_{n-1}, w_{m-1})]]

  con / wcon:
    feat_seq:   [[word_id(c_0, 0), ..., word_id(c_{n-1}, 0)],
                 ...
                 [word_id(c_0, k-1), ..., word_id(c_{n-1}, k-1)]]

    gchunk_seq: [gold_index([0,...,k-1], c_0), ..., gold_index([0,...,k-1], c_{n-1})]
    mask_ik:      zero vectors for characters w/o no candidate words
"""
def add_chunk_sequences(
        data, dic, min_len=1, max_len=4, evaluate=True, model_type=constants.AVG):
    is_con = model_type == constants.CON
    is_wcon = model_type == constants.WCON
    is_att_based = model_type == constants.WAVG or model_type == constants.WCON
    is_con_based = model_type == constants.CON or model_type == constants.WCON

    trie = dic.tries[constants.CHUNK]
    token_seqs = data.inputs[0]
    gold_label_seqs = data.outputs[0] if evaluate else None
    gold_chunk_seqs = []
    chunk_seqs = [] if not is_con else None
    feat_seqs = [] if is_con_based else None
    feat_size = sum([h for h in range(min_len, max_len+1)]) if is_con_based else None
    masks = []

    ins_cnt = 0
    for sen_id, tseq in enumerate(token_seqs):
        if ins_cnt > 0 and ins_cnt % 100000 == 0:
            print('Processed', ins_cnt, 'sentences', file=sys.stderr)
        ins_cnt += 1

        n = len(tseq)
        gchunk_seq = [-1] * n
        feats = [[0] * n for k in range(feat_size)] if is_con_based else None
        mask_ij = [] if not is_con else None   # for biaffine
        mask_ik = [] if is_con_based else None # for concat
        table_ikj = [[-1 for k in range(feat_size)] for i in range(n)] if is_wcon else None # for wcon

        if evaluate:
            lseq = gold_label_seqs[sen_id]
            spans_gold = get_segmentation_spans(lseq)
        else:
            lseq = spans_gold = None

        spans_found = []
        for i in range(n):
            res = trie.common_prefix_search(tseq, i, i + max_len)
            for span in res:
                if min_len == 1 or span[1]-span[0] >= min_len:
                    spans_found.append(span)

        if not is_con:
            m = len(spans_found)
            chunk_seq = [None] * m

        for j, span in enumerate(spans_found):
            is_gold_span = evaluate and span in spans_gold
            cid = trie.get_chunk_id(tseq[span[0]:span[1]])
            if not is_con:
                chunk_seq[j] = cid

            for i in range(span[0], span[1]):
                if not is_con:
                    mask_ij.append((i,j)) # (char i, word j) has value; used for biaffine

                if is_con_based:
                    k = token_index2feat_index(i, span, min_chunk_len=min_len)
                    feats[k][i] = cid
                    mask_ik.append((i,k)) # (char i, feat k) has value; used for concatenation

                if is_wcon:
                    table_ikj[i][k] = j

                if is_gold_span:
                    gchunk_seq[i] = k if is_con_based else j

        # print('ID={}'.format(sen_id))
        # print('l', ' '.join([dic.tables[constants.SEG_LABEL].id2str[l] if l in dic.tables[constants.SEG_LABEL].id2str else '-1' for l in lseq]))
        # print('i:x:g', ' '.join([str(i)+':'+dic.tables[constants.UNIGRAM].id2str[t]+':'+str(k) for i,t,k in zip(range(n), tseq, gchunk_seq)]))
        # if chunk_seq:
        #     print('m', ' '.join([str(i)+':'+trie.id2chunk[c] for i,c in enumerate(chunk_seq)]))
        # print('f')
        # for i, raw in enumerate(feats):
        #     print(str(i)+':['+
        #           ' '.join([str(i)+':'+ (trie.id2chunk[c] if c > 0 else '-') for i,c in enumerate(raw)])+
        #           ']')

        # if mask_ij:
        #     mat = np.zeros((n, m), dtype='i')
        #     for i, j in mask_ij:
        #         mat[i][j] = 1
        #     print('mask_ij (atn)\n', mat)
        # if mask_ik:
        #     mat = np.zeros((n, feat_size), dtype='i')
        #     for i, k in mask_ik:
        #         mat[i][k] = 1
        #     print('mask_ik (con)\n', mat)
        # if table_ikj:
        #     print('table_ikj')
        #     for i in range(n):
        #         print(i, table_ikj[i])

        if not is_con:
            chunk_seqs.append(chunk_seq)
        if is_con_based:
            feat_seqs.append(feats)
        masks.append((mask_ij, mask_ik, table_ikj))
        if evaluate:
            gold_chunk_seqs.append(gchunk_seq)

    data.inputs.append(chunk_seqs)
    data.inputs.append(feat_seqs)
    data.inputs.append(masks)
    if evaluate:
        data.outputs.append(gold_chunk_seqs)


def token_index2feat_index(ti, span, min_chunk_len=1):
    chunk_len = span[1] - span[0]
    fi = sum([i for i in range(min_chunk_len, chunk_len)]) + (span[1] - ti - 1)
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


    pairs_ij = mask[0]
    pairs_ik = mask[1]
    table_ikj = mask[2]

    if pairs_ik:                  # con or wcon
        mask_ik = xp.zeros((n_tokens, feat_size), dtype='f')
        for i, k in pairs_ik:
            mask_ik[i][k] = np.float32(1)
    else:
        mask_ik = None   

    if pairs_ij is not None:      # ave, wave, or wcon
        mask_ij = xp.full((n_tokens, n_chunks), mask_val, 'f')
        for i, j in pairs_ij:
            mask_ij[i, j] = non_mask_val
    else:
        mask_ij = None

    mask_i = None
    if use_attention:           # for softmax
        mask_i = xp.ones((n_tokens, n_chunks), 'f') # zero vectors for characters w/o no candidate words
        for i in range(n_tokens):
            tmp = [k1 == i for k1,k2 in pairs_ij]
            if len(tmp) == 0 or max(tmp) == False: # not (i,*) in index_pair
                mask_i[i] = xp.zeros((n_chunks,), 'f')

    return (mask_ij, mask_i, mask_ik, table_ikj)
