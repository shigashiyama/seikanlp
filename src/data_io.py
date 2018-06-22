import sys
import re
import pickle
import argparse
import copy
from collections import Counter

import numpy as np

import chainer.functions as F

import util
import common
import constants
import dictionary


FLOAT_MIN = -100000.0           # input for exp

class Data(object):
    def __init__(self, inputs=None, outputs=None, featvecs=None):
        self.inputs = inputs      # list of input sequences e.g. chars, words
        self.outputs = outputs    # list of output sequences (label sequences) 
        self.featvecs = featvecs  # feature vectors other than input sequences


class RestorableData(Data):
    def __init__(self, inputs=None, outputs=None, featvecs=None, orgdata=None):
        super().__init__(inputs, outputs, featvecs)
        self.orgdata = orgdata


def load_decode_data(
        path, dic, task, data_format=constants.WL_FORMAT,
        use_pos=True, use_bigram=False, use_tokentype=False, use_subtoken=False, use_chunk_trie=False,
        subpos_depth=-1, lowercase=False, normalize_digits=False, max_chunk_len=0, 
        # freq_words=set(), freq_bigrams=set(),
        refer_unigrams=set(), refer_bigrams=set(), refer_chunks=set(),
        add_gold_chunk=False, add_unknown_pretrained_chunk=True):

    segmentation = common.is_segmentation_task(task)
    parsing = common.is_parsing_task(task)
    use_subtoken = use_subtoken and not segmentation

    if segmentation or data_format == constants.SL_FORMAT:
        data = load_decode_data_SL(
            path, dic, segmentation=segmentation, parsing=parsing, 
            use_pos=use_pos, use_bigram=use_bigram, use_tokentype=use_tokentype,
            use_subtoken=use_subtoken, use_chunk_trie=use_chunk_trie,
            subpos_depth=subpos_depth, lowercase=lowercase, normalize_digits=normalize_digits, 
            max_chunk_len=max_chunk_len, 
            refer_unigrams=refer_unigrams, refer_bigrams=refer_bigrams, refer_chunks=refer_chunks,
            add_gold_chunk=add_gold_chunk, add_unknown_pretrained_chunk=add_unknown_pretrained_chunk)

    elif data_format == constants.WL_FORMAT:
        data = load_decode_data_WL(
            path, dic, parsing=parsing,
            use_pos=use_pos, use_subtoken=use_subtoken, 
            subpos_depth=subpos_depth, lowercase=lowercase, normalize_digits=normalize_digits, 
            refer_unigrams=refer_unigrams, refer_bigrams=refer_bigrams, refer_chunks=refer_chunks,
            add_gold_chunk=add_gold_chunk, add_unknown_pretrained_chunk=add_unknown_pretrained_chunk)        
    else:
        print('Invalid data format: {}'.format(data_format))
        sys.exit()

    return data


def load_annotated_data(
        path, task, data_format, train=True, 
        use_bigram=False, use_tokentype=False, use_subtoken=False, use_chunk_trie=False, 
        attr_indexes=[], attr_depths=[], attr_target_labelsets=[],
        lowercase=False, normalize_digits=False, 
        bigram_max_vocab_size=-1, bigram_freq_threshold=1, 
        word_max_vocab_size=-1, word_freq_threshold=1, 
        max_chunk_len=0, dic=None, refer_unigrams=set(), refer_bigrams=set(), refer_chunks=set(),
        add_gold_chunk=False, add_unknown_pretrained_chunk=True):
    pos_seqs = []

    segmentation = common.is_segmentation_task(task)
    tagging = common.is_tagging_task(task)
    parsing = common.is_parsing_task(task)
    typed_parsing = common.is_typed_parsing_task(task)
    use_subtoken = use_subtoken and not segmentation

    freq_bigrams = set()
    freq_words = set()
    if segmentation:
        if train and use_bigram and bigram_freq_threshold > 1 or bigram_max_vocab_size > 0:
            if data_format == constants.WL_FORMAT:
                freq_bigrams = count_bigrams_WL(path, bigram_freq_threshold, bigram_max_vocab_size)
            else:
                freq_bigrams = count_bigrams_SL(path, bigram_freq_threshold, bigram_max_vocab_size)

        if train and use_chunk_trie and word_freq_threshold > 1 or word_max_vocab_size > 0:
            if data_format == constants.WL_FORMAT:
                freq_words = count_ngrams_WL(path, word_freq_threshold, word_max_vocab_size, max_chunk_len)
            else:
                freq_words = count_ngrams_SL(path, word_freq_threshold, word_max_vocab_size, max_chunk_len)
                
    else:
        if train and word_freq_threshold > 1 or word_max_vocab_size > 0:
            if data_format == constants.WL_FORMAT:
                freq_words = count_words_WL(
                    path, word_freq_threshold, word_max_vocab_size, 
                    lowercase=lowercase, normalize_digits=normalize_digits)
            else:
                freq_words = count_words_SL(
                    path, word_freq_threshold, word_max_vocab_size,
                    lowercase=lowercase, normalize_digits=normalize_digits)
                
    if data_format == constants.WL_FORMAT:
        data, dic = load_annotated_data_WL(
            path, train=train,
            segmentation=segmentation, tagging=tagging, parsing=parsing, typed_parsing=typed_parsing,
            use_bigram=use_bigram, use_tokentype=use_tokentype,
            use_subtoken=use_subtoken, use_chunk_trie=use_chunk_trie,
            attr_indexes=attr_indexes, attr_depths=attr_depths, attr_target_labelsets=attr_target_labelsets,
            lowercase=lowercase, normalize_digits=normalize_digits, max_chunk_len=max_chunk_len, 
            dic=dic, freq_words=freq_words, freq_bigrams=freq_bigrams,
            refer_unigrams=refer_unigrams, refer_bigrams=refer_bigrams, refer_chunks=refer_chunks,
            add_gold_chunk=add_gold_chunk, add_unknown_pretrained_chunk=add_unknown_pretrained_chunk)

    else:
        data, dic = load_annotated_data_SL(
            path, train=train, segmentation=segmentation, tagging=tagging, 
            use_bigram=use_bigram, use_tokentype=use_tokentype,
            use_subtoken=use_subtoken, use_chunk_trie=use_chunk_trie, 
            attr_indexes=attr_indexes, attr_depths=attr_depths, attr_target_labelsets=attr_target_labelsets,
            lowercase=lowercase, normalize_digits=normalize_digits, max_chunk_len=max_chunk_len, 
            dic=dic, freq_words=freq_words, freq_bigrams=freq_bigrams,
            refer_unigrams=refer_unigrams, refer_bigrams=refer_bigrams, refer_chunks=refer_chunks,
            add_gold_chunk=add_gold_chunk, add_unknown_pretrained_chunk=add_unknown_pretrained_chunk)

    return data, dic


def parse_commandline_input(
        line, dic, task, use_pos=False, subpos_depth=-1, 
        lowercase=False, normalize_digits=True, use_subtoken=False):
    segmentation = common.is_segmentation_task(task)
    parsing = common.is_parsing_task(task)

    attr_delim = constants.SL_ATTR_DELIM

    get_unigram_id = dic.tables[constants.UNIGRAM].get_id
    get_subtoken_id = dic.tables[constants.SUBTOKEN].get_id if use_subtoken else None
    get_pos_id = dic.tables[constants.POS_LABEL].get_id if use_pos else None
    if parsing:
        root_token = constants.ROOT_SYMBOL
        root_token_id = dic.tables[constants.UNIGRAM].get_id(constants.ROOT_SYMBOL)
        if use_subtoken:
            root_subtoken_id = dic.tables[constants.SUBTOKEN].get_id(constants.ROOT_SYMBOL)
        if use_pos:
            root_pos_id = dic.tables[constants.POS_LABEL].get_id(constants.ROOT_SYMBOL)

    if segmentation:
        token_seq = [get_unigram_id(preprocess_token(char, lowercase, normalize_digits)) for char in line]
        org_token_seq = [char for char in line]

    else:
        org_arr = line.split(constants.SL_TOKEN_DELIM)

        org_token_seq = [word for word in org_arr]
        if use_pos:
            org_token_seq = [elem.split(attr_delim)[0] for elem in org_arr]
        if parsing:
            org_token_seq.insert(0, root_token)

        pro_token_seq = [preprocess_token(word, lowercase, normalize_digits) for word in org_token_seq]
        token_seq = [get_unigram_id(pword) for pword in pro_token_seq]

        if use_pos:
            org_pos_seq = [get_subpos(
                elem.split(attr_delim)[1], subpos_depth) for elem in org_arr] if use_pos else None
            if parsing:
                org_pos_seq.insert(0, root_token)

            pos_seq = [get_pos_id(pos) for pos in org_pos_seq]

        if use_subtoken:
            # TODO fix a case that token contains <NUM> and does not equal to <NUM>
            subtoken_seq = ([
                [get_subtoken_id(pword)] if is_special_token(pword) 
                else [get_subtoken_id(pword[i]) for i in range(len(pword))]
                for pword in pro_token_seq])

    inputs = [[token_seq]]
    if use_subtoken:
        inputs.append([subtoken_seq])

    outputs = []
    if use_pos:
        outputs.append([pos_seq])

    orgdata = [[org_token_seq]]
    if use_pos:
        orgdata.append([org_pos_seq])

    return RestorableData(inputs, outputs, orgdata=orgdata)
                

def load_decode_data_SL(
        path, dic, segmentation=False, parsing=False, 
        use_pos=False, use_bigram=False, use_tokentype=False, use_subtoken=False, use_chunk_trie=False,
        subpos_depth=-1, lowercase=False, normalize_digits=True, max_chunk_len=0,
        refer_unigrams=set(), refer_bigrams=set(), refer_chunks=set(),
        add_gold_chunk=False, add_unknown_pretrained_chunk=True):

    attr_delim = constants.SL_ATTR_DELIM

    ins_cnt = 0

    token_seqs = []
    bigram_seqs = []            # only available for segmentation
    toktype_seqs = []           # only available for segmentation
    subtoken_seqs = []
    pos_seqs = []
    org_token_seqs = []
    org_pos_seqs = []

    get_unigram_id = dic.tables[constants.UNIGRAM].get_id
    get_bigram_id = dic.tables[constants.BIGRAM].get_id if use_bigram else None
    get_toktype_id = dic.tables[constants.TOKEN_TYPE].get_id if use_tokentype else None
    get_subtoken_id = dic.tables[constants.SUBTOKEN].get_id if use_subtoken else None
    get_seg_id = dic.tables[constants.SEG_LABEL].get_id if segmentation else None
    get_pos_id = dic.tables[constants.POS_LABEL].get_id if use_pos else None
    get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id if use_chunk_trie else None

    if parsing:
        root_token = constants.ROOT_SYMBOL
        root_token_id = dic.tables[constants.UNIGRAM].get_id(constants.ROOT_SYMBOL)
        if use_subtoken:
            root_subtoken_id = dic.tables[constants.SUBTOKEN].get_id(constants.ROOT_SYMBOL)
        if use_pos:
            root_pos_id = dic.tables[constants.POS_LABEL].get_id(constants.ROOT_SYMBOL)
        
    with open(path) as f:
        for line in f:
            # line = re.sub(' +', ' ', line).strip('\t\n')
            line = re.sub(' +', ' ', line).strip(' \t\n')
            if len(line) < 1:
                continue

            elif line.startswith(constants.ATTR_DELIM_TXT):
                attr_delim = line.split(constants.KEY_VALUE_DELIM)[1]
                print('Read attribute delimiter:', attr_delim, file=sys.stderr)
                continue

            elif line.startswith(constants.POS_CLM_TXT):
                pos_clm = int(line.split('=')[1]) - 1
                print('Read 1st label column id:', pos_clm+1, file=sys.stderr)
                if pos_clm < 0:
                    use_pos = False
                    pos_seqs = pos_seq = []
                continue

            if segmentation:    # raw text: seg or seg_tag
                uni_seq = [get_unigram_id(char) for char in line]
                token_seqs.append(uni_seq)
                org_token_seqs.append([char for char in line])

                if use_tokentype:
                    toktype_seqs.append([get_toktype_id(get_char_type(char)) for char in line])

                if use_bigram:
                    str_bigrams = create_all_char_ngrams(line, 2)
                    str_bigrams.append('{}{}'.format(line[-1], constants.EOS))
                    bi_seq = [get_bigram_id(sb, update=sb in refer_bigrams) for sb in str_bigrams]
                    bigram_seqs.append(bi_seq)

                # if use_chunk_trie:
                #     cids = token_seqs[-1]
                #     for n in range(1, max_chunk_len+1):
                #         cid_ngrams = create_all_char_ngrams(cids, n)
                #         str_ngrams = create_all_char_ngrams(line, n)
                #         for cn, sn in zip (cid_ngrams, str_ngrams):
                #             if sn in refer_chunks:
                #                 get_chunk_id(cn, sn, True)


            else:               # tokeniized text w/ or w/o pos: tag or (t)dep
                org_arr = line.split(constants.SL_TOKEN_DELIM)

                org_token_seq = [word for word in org_arr]
                if parsing:
                    org_token_seq = [elem.split(attr_delim)[0] for elem in org_arr]
                    org_token_seq.insert(0, root_token)
                org_token_seqs.append(org_token_seq)

                pro_token_seq = [preprocess_token(
                    word, lowercase, normalize_digits, refer_unigrams) for word in org_token_seq]
                token_seq = [get_unigram_id(pword, update=pword in refer_unigrams) for pword in pro_token_seq]
                token_seqs.append(token_seq)

                if use_pos:
                    org_pos_seq = [get_subpos(
                        elem.split(attr_delim)[1], subpos_depth) for elem in org_arr] if use_pos else None
                    if parsing:
                        org_pos_seq.insert(0, root_token)
                    org_pos_seqs.append(org_pos_seq)

                    pos_seq = [get_pos_id(pos) for pos in org_pos_seq]
                    pos_seqs.append(pos_seq)

                if use_subtoken:
                    # TODO fix a case that token contains <NUM> and does not equal to <NUM>
                    subtoken_seq = ([
                        [get_subtoken_id(pword)] if is_special_token(pword) 
                        else [get_subtoken_id(pword[i]) for i in range(len(pword))]
                        for pword in pro_token_seq])
                    subtoken_seqs.append(subtoken_seq)

            if segmentation:
                if use_bigram:
                    str_bigrams = create_all_char_ngrams(line, 2)
                    str_bigrams.append('{}{}'.format(line[-1], constants.EOS))
                    bi_seq = [
                        get_bigram_id(sb, update=to_be_registered(sb, train, set(), refer_bigrams))
                        for sb in str_bigrams]

                if use_chunk_trie:
                    for n in range(1, max_chunk_len+1):
                        span_ngrams = create_all_char_ngram_indexes(uni_seq, n)
                        cid_ngrams = [uni_seq[span[0]:span[1]] for span in span_ngrams]
                        str_ngrams = [line[span[0]:span[1]] for span in span_ngrams]
                        for span, cn, sn in zip (span_ngrams, cid_ngrams, str_ngrams):
                            is_pretrained_chunk = not refer_chunks or sn in refer_chunks
                            if add_unknown_pretrained_chunk and is_pretrained_chunk:
                                ci = get_chunk_id(cn, sn, True)


            ins_cnt += 1
            if ins_cnt % 100000 == 0:
                print('read', ins_cnt, 'sentences', file=sys.stderr)

    inputs = [token_seqs]
    inputs.append(bigram_seqs if bigram_seqs else None)
    inputs.append(toktype_seqs if toktype_seqs else None)
    inputs.append(subtoken_seqs if subtoken_seqs else None)

    outputs = []
    if use_pos:
        outputs.append(pos_seqs)
        
    orgdata = [org_token_seqs]
    if use_pos:
        orgdata.append(org_pos_seqs)

    return RestorableData(inputs, outputs, orgdata=orgdata)


# TODO fix
def load_decode_data_WL(
        path, dic, parsing=False, 
        use_pos=True, use_subtoken=False, subpos_depth=-1, lowercase=False, normalize_digits=True, 
        refer_unigrams=set(), refer_bigrams=set(), refer_chunks=set(),
        add_gold_chunk=False, add_unknown_pretrained_chunk=True):

    ins_cnt = 0

    token_seqs = []
    subtoken_seqs = []
    pos_seqs = []
    org_token_seqs = []
    org_pos_seqs = []

    get_unigram_id = dic.tables[constants.UNIGRAM].get_id
    get_subtoken_id = dic.tables[constants.SUBTOKEN].get_id if use_subtoken else None
    get_pos_id = dic.tables[constants.POS_LABEL].get_id if use_pos else None

    if parsing:
        root_token_id = dic.tables[constants.UNIGRAM].get_id(constants.ROOT_SYMBOL)
        if use_pos:
            root_pos_id = dic.tables[constants.POS_LABEL].get_id(constants.ROOT_SYMBOL)

    ouni_seq_ = [constants.ROOT_SYMBOL] if parsing else []
    opos_seq_ = [constants.ROOT_SYMBOL] if parsing and use_pos else []
    uni_seq_ = [get_unigram_id(constants.ROOT_SYMBOL)] if parsing else []
    sub_seq_ = [[get_subtoken_id(constants.ROOT_SYMBOL)]] if parsing and use_subtoken else []
    pos_seq_ = [get_pos_id(constants.ROOT_SYMBOL)] if parsing and use_pos else []

    attr_delim = constants.WL_ATTR_DELIM
    word_clm = 0
    pos_clm = 1

    with open(path) as f:
        ouni_seq = ouni_seq_.copy()
        opos_seq = opos_seq_.copy()
        uni_seq = uni_seq_.copy()
        sub_seq = sub_seq_.copy()
        pos_seq = pos_seq_.copy()

        for line in f:
            line = re.sub(' +', ' ', line)
            line = line.strip(' \t\n')
            if len(line) <= 1:
                if len(uni_seq) - (1 if parsing else 0) > 0:
                    token_seqs.append(uni_seq)
                    uni_seq = uni_seq_.copy()
                    org_token_seqs.append(ouni_seq)
                    ouni_seq = ouni_seq_.copy()
                    if sub_seq:
                        subtoken_seqs.append(sub_seq)
                        sub_seq = sub_seq_.copy()
                    if pos_seq:
                        pos_seqs.append(pos_seq)
                        pos_seq = pos_seq_.copy()
                    if opos_seq:
                        org_pos_seqs.append(opos_seq)
                        opos_seq = opos_seq_.copy()

                    ins_cnt += 1
                    if ins_cnt % 100000 == 0:
                        print('Read', ins_cnt, 'sentences', file=sys.stderr)

                continue

            elif line[0] == constants.COMMENT_SYM:
                if line.startswith(constants.ATTR_DELIM_TXT):
                    attr_delim = line.split(constants.KEY_VALUE_DELIM)[1]
                    print('Read attribute delimiter: \'{}\''.format(attr_delim), file=sys.stderr)
                    continue

                elif line.startswith(constants.WORD_CLM_TXT):
                    word_clm = int(line.split('=')[1]) - 1
                    print('Read word column id:', word_clm+1, file=sys.stderr)
                    continue

                elif line.startswith(constants.POS_CLM_TXT):
                    pos_clm = int(line.split('=')[1]) - 1
                    print('Read 1st label column id:', pos_clm+1, file=sys.stderr)
                    if pos_clm < 0:
                        use_pos = False
                        pos_seqs = pos_seq = pos_seq_ = []
                    continue

                else:
                    continue    

            array = line.split(attr_delim)
            word = array[word_clm]
            ouni_seq.append(word)
            pword = preprocess_token(word, lowercase, normalize_digits, refer_unigrams)
            uni_seq.append(get_unigram_id(pword, update=pword in refer_unigrams))

            if use_pos:
                pos = get_subpos(array[pos_clm], subpos_depth)
                opos_seq.append(pos)
                pos_seq.append(get_pos_id(pos))

            if use_subtoken:
                # TODO fix a case that token contains <NUM> and does not equal to <NUM>
                if is_special_token(pword):
                    sub_seq.append([get_subtoken_id(pword)])
                else:
                    sub_seq.append([get_subtoken_id(pword[i]) for i in range(len(pword))])

        n_non_words = 1 if parsing else 0
        if len(uni_seq) - n_non_words > 0:
            org_token_seqs.append(ouni_seq)
            token_seqs.append(uni_seq)
            if opos_seq:
                org_pos_seqs.append(opos_seq)
            if subtoken_seqs:
                subtoken_seqs.append(sub_seq)
            if pos_seq:
                pos_seqs.append(pos_seq)

    # TODO fix
    inputs = [token_seqs]
    if subtoken_seqs:
        inputs.append(subtoken_seqs)
    outputs = []
    if pos_seqs:
        outputs.append(pos_seqs)
    orgdata = [org_token_seqs]
    if org_pos_seqs:
        orgdata.append(org_pos_seqs)

    return RestorableData(inputs, outputs, orgdata=orgdata)


"""
Read data with SL (one sentence in one line) format.
If tagging == True, the following format is expected for joint segmentation and POS tagging:
  word1_pos1 word2_pos2 ... wordn_posn

otherwise, the following format is expected for segmentation:
  word1 word2 ... wordn
"""
def load_annotated_data_SL(
        path, train=True, segmentation=True, tagging=False,
        use_bigram=False, use_tokentype=False, use_subtoken=False, use_chunk_trie=False, 
        attr_indexes=[], attr_depths=[], attr_target_labelsets=[],
        lowercase=False, normalize_digits=True, max_chunk_len=0, 
        dic=None, freq_words=set(), freq_bigrams=set(),
        refer_unigrams=set(), refer_bigrams=set(), refer_chunks=set(),
        add_gold_chunk=False, add_unknown_pretrained_chunk=True):

    num_attrs = len(attr_indexes)
    if not dic:
        dic = dictionary.init_dictionary(
            use_unigram=True, use_bigram=use_bigram, use_subtoken=use_subtoken, 
            use_tokentype=use_tokentype, num_attrs=num_attrs,
            use_seg_label=segmentation, use_chunk_trie=use_chunk_trie, use_root=False)

    attr_delim = constants.SL_ATTR_DELIM
    ins_cnt = 0
    attr_dics = []

    get_unigram_id = dic.tables[constants.UNIGRAM].get_id
    get_bigram_id = dic.tables[constants.BIGRAM].get_id if use_bigram else None
    get_toktype_id = dic.tables[constants.TOKEN_TYPE].get_id if use_tokentype else None
    get_subtoken_id = dic.tables[constants.SUBTOKEN].get_id if use_subtoken else None
    get_seg_id = dic.tables[constants.SEG_LABEL].get_id if segmentation else None
    get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id if use_chunk_trie else None
    get_ith_attr_id = []
    for i in range(num_attrs): 
        get_ith_attr_id.append(dic.tables[constants.ATTR_LABEL(i)].get_id)

    token_seqs = []
    bigram_seqs = []            # only available for segmentation
    toktype_seqs = []           # only available for segmentation
    subtoken_seqs = []
    seg_seqs = []               # list of segmentation label sequences
    attr_seqs_list = [[] for i in range(num_attrs)]

    with open(path) as f:
        # ccounter = Counter()    # tmp
        # n_chunks = 0            # tmp
        for line in f:
            oline = line
            line = re.sub(' +', ' ', line).strip(' \t\n')
            if len(line) <= 1:
                continue

            elif line[0] == constants.COMMENT_SYM:
                if line.startswith(constants.ATTR_DELIM_TXT):
                    attr_delim = line.split(constants.KEY_VALUE_DELIM)[1]
                    print('Read attribute delimiter:', attr_delim, file=sys.stderr)
                    continue

                else:
                    continue
            
            entries = line.split(constants.SL_TOKEN_DELIM)
            uni_seq = []
            bi_seq = []
            type_seq = []
            sub_seq = []
            seg_seq = []
            attr_seq_list = [[] for i in range(num_attrs)]
            pstr = ''

            for entry in entries:
                array = entry.split(attr_delim)
                pword = preprocess_token(array[0], lowercase, normalize_digits, refer_unigrams)
                pstr += pword

                if tagging:
                    if len(array) < 2:
                        continue

                    for i in range(num_attrs):
                        attrs[i] = array[attr_dics[i][constants.ATTR_INDEX]-1]
                        if attr_target_labelsets[i]:
                            if not attrs[i] in attr_target_labelsets[i]:
                                attrs[i] = '' # ignore

                        if attr_depths[i] > 0:
                            attrs[i] = get_subpos(attrs[i], attr_depths[i])

                        if segmentation and train and i == 0: # tmp
                            for seg_lab in constants.SEG_LABELS:
                                get_seg_id('{}-{}'.format(seg_lab, attrs[i]), True)

                else:
                    attrs = [None] * max(num_attrs, 1)
                       
                wlen = len(pword)

                if segmentation:
                    uni_seq.extend([get_unigram_id(pword[i], True) for i in range(wlen)])

                    # tentative: attrs[0] indicates POS
                    seg_seq.extend(
                        [get_seg_id(
                            get_label_BIES(i, wlen-1, cate=attrs[0]), update=train) for i in range(wlen)])

                    if use_tokentype:
                        type_seq.extend(
                            [get_toktype_id(get_char_type(pword[i]), update=train) for i in range(wlen)])

                else:
                    update_token = to_be_registered(pword, train, freq_words, refer_unigrams)
                    uni_seq.append(get_unigram_id(pword, update=update_token))

                    if use_subtoken:
                        # TODO fix a case that token contains <NUM> and does not equal to <NUM>
                        if is_special_token(pword):
                            sub_seq.append([get_subtoken_id(pword, update_token)])
                        else:
                            sub_seq.append(
                                [get_subtoken_id(pword[i], update_token) for i in range(wlen)])

                    for i in range(num_attrs):
                        attr = attrs[i]
                        if attr_dics[i][constants.ATTR_CHUNKING] == True:
                            tmp = attr
                        else:
                            tmp = get_ith_attr_id[i](attr, update=train)
                        attr_seq_list[i].append(tmp)

            if segmentation:
                if use_bigram:
                    str_bigrams = create_all_char_ngrams(pstr, 2)
                    str_bigrams.append('{}{}'.format(pstr[-1], constants.EOS))
                    bi_seq = [
                        get_bigram_id(sb, update=to_be_registered(sb, train, freq_bigrams, refer_bigrams))
                        for sb in str_bigrams]

                if use_chunk_trie:
                    spans_gold = get_segmentation_spans(seg_seq)
                    for n in range(1, max_chunk_len+1):
                        span_ngrams = create_all_char_ngram_indexes(uni_seq, n)
                        cid_ngrams = [uni_seq[span[0]:span[1]] for span in span_ngrams]
                        str_ngrams = [pstr[span[0]:span[1]] for span in span_ngrams]
                        for span, cn, sn in zip (span_ngrams, cid_ngrams, str_ngrams):
                            is_pretrained_chunk = not refer_chunks or sn in refer_chunks
                            if train:
                                is_gold_chunk = add_gold_chunk and span in spans_gold
                                pass_freq_filter = not freq_words or sn in freq_words
                                if pass_freq_filter and (is_gold_chunk or is_pretrained_chunk):
                                    ci = get_chunk_id(cn, sn, True)
                            else:
                                if add_unknown_pretrained_chunk and is_pretrained_chunk:
                                    ci = get_chunk_id(cn, sn, True)

            ins_cnt += 1
            
            if uni_seq:
                token_seqs.append(uni_seq)
            if bi_seq:
                bigram_seqs.append(bi_seq)
            if type_seq:
                toktype_seqs.append(type_seq)
            if sub_seq:    # TODO bug
                subtoken_seqs.append(sub_seq)
            if seg_seq:
                seg_seqs.append(seg_seq)
            for i, attr_seq in enumerate(attr_seq_list):
                if attr_dics[i][constants.ATTR_CHUNKING]:
                    attr_seq = [get_ith_attr_id[i](attr, update=train) for attr in 
                                get_labelseq_BIOES(attr_seq)]
                attr_seqs_list[i].append(attr_seq)

            if ins_cnt % 100000 == 0:
                print('Read', ins_cnt, 'sentences', file=sys.stderr)

    inputs = [token_seqs]
    inputs.append(bigram_seqs if bigram_seqs else None)
    inputs.append(toktype_seqs if toktype_seqs else None)
    inputs.append(subtoken_seqs if subtoken_seqs else None)
        
    outputs = []
    if seg_seqs:
        outputs.append(seg_seqs)
    for attr_seqs in attr_seqs_list:
        outputs.append(attr_seqs)

    # tmp
    # cave = 0
    # for leng in ccounter.keys():
    #     ccount = ccounter[leng]
    #     crate = ccount / n_chunks
    #     cave += leng * crate
    #     print('len {}\t{}\t{}'.format(leng, ccount, crate*100))
    # print('ave {}'.format(cave))

    return Data(inputs, outputs), dic


def load_annotated_data_WL(
        path, train=True, segmentation=True, tagging=False, parsing=False, typed_parsing=False, 
        use_bigram=False, use_tokentype=False, use_subtoken=False, use_chunk_trie=False, 
        attr_indexes=[], attr_depths=[], attr_target_labelsets=[],
        lowercase=False, normalize_digits=True, max_chunk_len=0, 
        dic=None, freq_words=set(), freq_bigrams=set(),
        refer_unigrams=set(), refer_bigrams=set(), refer_chunks=set(),
        add_gold_chunk=False, add_unknown_pretrained_chunk=True):

    num_attrs = len(attr_indexes)
    if not dic:
        dic = dictionary.init_dictionary(
            use_unigram=True, use_bigram=use_bigram, use_subtoken=use_subtoken, 
            use_tokentype=use_tokentype, num_attrs=num_attrs,
            use_seg_label=segmentation, use_arc_label=typed_parsing, 
            use_chunk_trie=use_chunk_trie, use_root=parsing)

    attr_delim = constants.WL_ATTR_DELIM
    ins_cnt = 0
    word_clm = 0
    head_clm = 2
    arc_clm = 3
    attr_dics = []

    token_seqs = []
    bigram_seqs = []            # only available for segmentation
    toktype_seqs = []           # only available for segmentation
    subtoken_seqs = []
    seg_seqs = []               # list of segmentation label sequences
    head_seqs = []              # list of head id sequences
    arc_seqs = []               # list of arc label sequences
    attr_seqs_list = [[] for i in range(num_attrs)]
    # attr_seqs_list = [None] * num_attrs
    # for i in range(num_attrs):
    #     attr_seqs_list[i] = []
    
    get_unigram_id = dic.tables[constants.UNIGRAM].get_id
    get_bigram_id = dic.tables[constants.BIGRAM].get_id if use_bigram else None
    get_toktype_id = dic.tables[constants.TOKEN_TYPE].get_id if use_tokentype else None
    get_subtoken_id = dic.tables[constants.SUBTOKEN].get_id if use_subtoken else None
    get_seg_id = dic.tables[constants.SEG_LABEL].get_id if segmentation else None
    get_arc_id = dic.tables[constants.ARC_LABEL].get_id if typed_parsing else None
    get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id if use_chunk_trie else None
    get_ith_attr_id = []
    for i in range(num_attrs): 
        get_ith_attr_id.append(dic.tables[constants.ATTR_LABEL(i)].get_id)

    uni_seq_ = [get_unigram_id(constants.ROOT_SYMBOL)] if parsing else []
    bi_seq_ = []                # not used for parsing
    type_seq_ = []              # not used for parsing
    sub_seq_ = [[get_subtoken_id(constants.ROOT_SYMBOL)]] if parsing and use_subtoken else []
    seg_seq_ = []
    head_seq_ = [constants.NO_PARENTS_ID] if parsing else []
    arc_seq_ = [constants.NO_PARENTS_ID] if typed_parsing else []
    def init_attr_seq_list(num_attrs, get_ith_attr_id, parsing):
        if parsing:
            return [[get_ith_attr_id[i](constants.ROOT_SYMBOL)] for i in range(num_attrs)]
        else:
            return [[] for i in range(num_attrs)]

    with open(path) as f:
        uni_seq = uni_seq_.copy()
        bi_seq = bi_seq_.copy()
        type_seq = type_seq_.copy()
        sub_seq = sub_seq_.copy()
        seg_seq = seg_seq_.copy()
        head_seq = head_seq_.copy() 
        arc_seq = arc_seq_.copy() 
        attr_seq_list = init_attr_seq_list(num_attrs, get_ith_attr_id, parsing)
        pstr = ''

        for line in f:
            line = re.sub(' +', ' ', line).strip('\n')
            tmp = line.strip('\t')
            if not tmp:
                line = tmp
            
            if len(line) > 0 and line[0] == constants.COMMENT_SYM:
                if line.startswith(constants.ATTR_DELIM_TXT):
                    attr_delim = line.split(constants.KEY_VALUE_DELIM)[1]
                    print('Read attribute delimiter: \'{}\''.format(attr_delim), file=sys.stderr)

                elif line.startswith(constants.WORD_CLM_TXT):
                    word_clm = int(line.split('=')[1]) - 1
                    print('Read word column id:', word_clm+1, file=sys.stderr)

                elif line.startswith(constants.HEAD_CLM_TXT):
                    head_clm = int(line.split('=')[1]) - 1
                    print('Read head label column id:', head_clm+1, file=sys.stderr)

                elif line.startswith(constants.ARC_CLM_TXT) and typed_parsing:
                    arc_clm = int(line.split('=')[1]) - 1
                    print('Read arc label column id:', arc_clm+1, file=sys.stderr)

                elif line.startswith(constants.ATTR_CLM_TXT):
                    # e.g., '#ATTR_COLUMN:index=X,chunking=X'
                    array = line.split(constants.KEY_VALUE_DELIM)[1].split(constants.ATTR_INFO_DELIM)
                    info = {}
                    ignore_flag = True

                    for kv in array:
                        pair = kv.split(constants.ATTR_INFO_DELIM2)
                        key = pair[0]
                        val = pair[1]

                        if key == constants.ATTR_INDEX:
                            if int(val) in attr_indexes:
                                info[key] = int(val)
                                ignore_flag = False
                            else:
                                break

                        elif key == constants.ATTR_CHUNKING:
                            info[key] = val.lower() == 't'
                            if info[key]:
                                get_ith_attr_id[len(attr_dics)](constants.O, update=train)

                    if not ignore_flag:
                        if not constants.ATTR_CHUNKING in info:
                            info[constants.ATTR_CHUNKING] = False
                        index = attr_indexes.index(info[constants.ATTR_INDEX])
                        attr_dics.insert(index, info)
                        print('Read attribute label info:', info, file=sys.stderr)
                    
                continue

            elif len(line) < 1:
                if len(uni_seq) - (1 if parsing else 0) > 0:
                    if segmentation:
                        if use_bigram:
                            str_bigrams = create_all_char_ngrams(pstr, 2)
                            str_bigrams.append('{}{}'.format(pstr[-1], constants.EOS))
                            bi_seq = [
                                get_bigram_id(
                                    sb, update=to_be_registered(sb, train, freq_bigrams, refer_bigrams))
                                for sb in str_bigrams]

                        if use_chunk_trie:
                            spans_gold = get_segmentation_spans(seg_seq)
                            for n in range(1, max_chunk_len+1):
                                span_ngrams = create_all_char_ngram_indexes(uni_seq, n)
                                cid_ngrams = [uni_seq[span[0]:span[1]] for span in span_ngrams]
                                str_ngrams = [pstr[span[0]:span[1]] for span in span_ngrams]
                                for span, cn, sn in zip (span_ngrams, cid_ngrams, str_ngrams):
                                    is_pretrained_chunk = not refer_chunks or sn in refer_chunks
                                    if train:
                                        is_gold_chunk = add_gold_chunk and span in spans_gold
                                        pass_freq_filter = not freq_words or sn in freq_words
                                        if pass_freq_filter and (is_gold_chunk or is_pretrained_chunk):
                                            ci = get_chunk_id(cn, sn, True)
                                    else:
                                        if add_unknown_pretrained_chunk and is_pretrained_chunk:
                                            ci = get_chunk_id(cn, sn, True)

                    if uni_seq:
                        token_seqs.append(uni_seq)
                        uni_seq = uni_seq_.copy()
                    if bi_seq:
                        bigram_seqs.append(bi_seq)
                        bi_seq = bi_seq_.copy()
                    if type_seq:
                        toktype_seqs.append(type_seq)
                        type_seq = type_seq_.copy()
                    if sub_seq:
                        subtoken_seqs.append(sub_seq)
                        sub_seq = sub_seq_.copy()
                    if seg_seq:
                        seg_seqs.append(seg_seq)
                        seg_seq = seg_seq_.copy()
                    if head_seq:
                        head_seqs.append(head_seq)
                        head_seq = head_seq_.copy()
                    if arc_seq:
                        arc_seqs.append(arc_seq)
                        arc_seq = arc_seq_.copy()
                    for i, attr_seq in enumerate(attr_seq_list):
                        if attr_dics[i][constants.ATTR_CHUNKING]:
                            if train:
                                for attr in attr_seq:
                                    if attr:
                                        [get_ith_attr_id[i]('{}-{}'.format(seg_lab, attr), True)
                                         for seg_lab in constants.SEG_LABELS]

                            attr_seq = [get_ith_attr_id[i](attr, update=train) for attr in 
                                        get_labelseq_BIOES(attr_seq)]
                        attr_seqs_list[i].append(attr_seq)
                        attr_seq_list = init_attr_seq_list(num_attrs, get_ith_attr_id, parsing)
                    pstr = ''

                    ins_cnt += 1
                    if ins_cnt % 100000 == 0:
                        print('Read', ins_cnt, 'sentences', file=sys.stderr)

                continue

            array = line.split(attr_delim)
            pword = preprocess_token(array[word_clm], lowercase, normalize_digits, refer_unigrams)
            pstr += pword
            wlen = len(pword)

            attrs = [None] * max(num_attrs, 1)
            # attrs = [None] * num_attrs

            for i in range(num_attrs):
                attrs[i] = array[attr_dics[i][constants.ATTR_INDEX]-1]
                if attr_target_labelsets[i]:
                    if not attrs[i] in attr_target_labelsets[i]:
                        attrs[i] = '' # ignore

                if attr_depths[i] > 0:
                    attrs[i] = get_subpos(attrs[i], attr_depths[i])

                if segmentation and train and i == 0: # tmp
                    for seg_lab in constants.SEG_LABELS:
                        get_seg_id('{}-{}'.format(seg_lab, attrs[i]), True)

            if segmentation:
                uni_seq.extend([get_unigram_id(pword[i], True) for i in range(wlen)])

                # tentative: attrs[0] indicates POS
                seg_seq.extend(
                    [get_seg_id(
                        get_label_BIES(i, wlen-1, cate=attrs[0]), update=train) for i in range(wlen)])

                if use_tokentype:
                    type_seq.extend(
                        [get_toktype_id(get_char_type(pword[i]), update_tokens[i]) for i in range(wlen)])
                    
            else:
                update_token = to_be_registered(pword, train, freq_words, refer_unigrams)
                uni_seq.append(get_unigram_id(pword, update=update_token))

                if use_subtoken:
                    # TODO fix a case that token contains <NUM> and does not equal to <NUM>
                    if is_special_token(pword):
                        sub_seq.append([get_subtoken_id(pword, update_token)])
                    else:
                        sub_seq.append([get_subtoken_id(pword[i], update_token) for i in range(len(pword))])

                for i in range(num_attrs):
                    attr = attrs[i]
                    if attr_dics[i][constants.ATTR_CHUNKING] == True:
                        tmp = attr
                    else:
                        tmp = get_ith_attr_id[i](attr, update=train)
                    attr_seq_list[i].append(tmp)

                if parsing:
                    head = int(array[head_clm])
                    if head < 0:
                        head = 0
                    head_seq.append(head)

                if typed_parsing:
                    arc = array[arc_clm]
                    arc_seq.append(get_arc_id(arc, update=train))

        n_non_words = 1 if parsing else 0
        if len(uni_seq) - n_non_words > 0:
            if segmentation:
                if use_bigram:  # TODO confirm
                    str_bigrams = create_all_char_ngrams(pstr, 2)
                    str_bigrams.append('{}{}'.format(pstr[-1], constants.EOS))
                    bi_seq = [
                        get_bigram_id(sb, update=to_be_registered(sb, train, freq_bigrams, refer_bigrams))
                        for sb in str_bigrams]

                if use_chunk_trie:
                    if use_chunk_trie:
                        spans_gold = get_segmentation_spans(seg_seq)
                        for n in range(1, max_chunk_len+1):
                            span_ngrams = create_all_char_ngram_indexes(uni_seq, n)
                            cid_ngrams = [uni_seq[span[0]:span[1]] for span in span_ngrams]
                            str_ngrams = [pstr[span[0]:span[1]] for span in span_ngrams]
                            for span, cn, sn in zip (span_ngrams, cid_ngrams, str_ngrams):
                                is_pretrained_chunk = not refer_chunks or sn in refer_chunks
                                if train:
                                    is_gold_chunk = add_gold_chunk and span in spans_gold
                                    pass_freq_filter = not freq_words or sn in freq_words
                                    if pass_freq_filter and (is_gold_chunk or is_pretrained_chunk):
                                        ci = get_chunk_id(cn, sn, True)
                                else:
                                    if add_unknown_pretrained_chunk and is_pretrained_chunk:
                                        ci = get_chunk_id(cn, sn, True)

            token_seqs.append(uni_seq)
            if bi_seq:
                bigram_seqs.append(bi_seq)
            if type_seq:
               toktype_seqs.append(type_seq)
            if sub_seq:
                subtoken_seqs.append(sub_seq)
            if seg_seq:
                seg_seqs.append(seg_seq)
            for i, attr_seq in enumerate(attr_seq_list):
                if attr_dics[i][constants.ATTR_CHUNKING]:
                    attr_seq = [get_ith_attr_id[i](attr, update=train) for attr in 
                                get_labelseq_BIOES(attr_seq)]
                attr_seqs_list[i].append(attr_seq)

    inputs = [token_seqs]
    inputs.append(bigram_seqs if bigram_seqs else None)
    inputs.append(toktype_seqs if toktype_seqs else None)
    inputs.append(subtoken_seqs if subtoken_seqs else None)

    outputs = []
    if seg_seqs:
        outputs.append(seg_seqs)
    for attr_seqs in attr_seqs_list:
        outputs.append(attr_seqs)
    if not attr_seqs_list and parsing:
        outputs.append(None)
    if head_seqs:
        outputs.append(head_seqs)
    if arc_seqs:
        outputs.append(arc_seqs)

    return Data(inputs, outputs), dic


# for segmentation task
def count_bigrams_SL(path, freq_threshold, max_vocab_size=-1):
    counter = {}
    attr_delim = constants.WL_ATTR_DELIM
    ins_cnt = 0

    with open(path) as f:
        for line in f:
            line = re.sub(' +', ' ', line).strip(' \t\n')
            if len(line) < 1:
                continue

            elif line.startswith(constants.ATTR_DELIM_TXT):
                attr_delim = line.split(constants.KEY_VALUE_DELIM)[1]
                print('Read attribute delimiter:', attr_delim, file=sys.stderr)
                continue
            
            pstr = ''
            entries = line.split()
            for entry in entries:
                word = entry.split(attr_delim)[0]
                pstr += word

            str_bigrams = create_all_char_ngrams(pstr, 2)
            str_bigrams.append('{}{}'.format(pstr[-1], constants.EOS))
            for sb in str_bigrams:
                if sb in counter:
                    counter[sb] += 1
                else:
                    counter[sb] = 0

            ins_cnt += 1
            if ins_cnt % 100000 == 0:
                print('Read', ins_cnt, 'sentences', file=sys.stderr)

    if max_vocab_size > 0:
        counter2 = Counter(counter)
        for pair in counter2.most_common(max_vocab_size):
            del counter[pair[0]]
        
        print('keep {} bigrams from {} bigrams (max vocab size={})'.format(
            len(counter), len(counter2), max_vocab_size), file=sys.stderr)

    freq_bigrams = set()
    for word in counter:
        if counter[word] >= freq_threshold:
            freq_bigrams.add(word)

    print('keep {} bigrams from {} bigrams (frequency threshold={})'.format(
        len(freq_bigrams), len(counter), freq_threshold), file=sys.stderr)
    
    return freq_bigrams


# for segmentation task
def count_bigrams_WL(path, freq_threshold, max_vocab_size=-1):
    counter = {}
    attr_delim = constants.WL_ATTR_DELIM
    word_clm = 0                          
    ins_cnt = 0

    with open(path) as f:
        pstr = ''
        for line in f:
            line = re.sub(' +', ' ', line).strip(' \t\n')
            if len(line) < 1:
                str_bigrams = create_all_char_ngrams(pstr, 2)
                str_bigrams.append('{}{}'.format(pstr[-1], constants.EOS))
                for sb in str_bigrams:
                    if sb in counter:
                        counter[sb] += 1
                    else:
                        counter[sb] = 0

                pstr = ''
                ins_cnt += 1
                if ins_cnt % 100000 == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)

                continue

            elif line[0] == constants.COMMENT_SYM:
                if line.startswith(constants.ATTR_DELIM_TXT):
                    attr_delim = line.split(constants.KEY_VALUE_DELIM)[1]
                    print('Read attribute delimiter: \'{}\''.format(attr_delim), file=sys.stderr)

                elif line.startswith(constants.WORD_CLM_TXT):
                    word_clm = int(line.split('=')[1]) - 1
                    print('Read word column id:', word_clm+1, file=sys.stderr)

                continue
            
            array = line.split(attr_delim)
            word = array[word_clm] #preprocess_token(array[word_clm], lowercase, normalize_digits) #tmp
            pstr += word

        if pstr:
            str_bigrams = create_all_char_ngrams(pstr, 2)
            str_bigrams.append('{}{}'.format(pstr[-1], constants.EOS))
            for sb in str_bigrams:
                if sb in counter:
                    counter[sb] += 1
                else:
                    counter[sb] = 0

    if max_vocab_size > 0:
        counter2 = Counter(counter)
        for pair in counter2.most_common(max_vocab_size):
            del counter[pair[0]]
        
        print('keep {} bigrams from {} bigrams (max vocab size={})'.format(
            len(counter), len(counter2), max_vocab_size), file=sys.stderr)

    freq_bigrams = set()
    for word in counter:
        if counter[word] >= freq_threshold:
            freq_bigrams.add(word)

    print('keep {} bigrams from {} bigrams (frequency threshold={})'.format(
        len(freq_bigrams), len(counter), freq_threshold), file=sys.stderr)
    
    return freq_bigrams


# for segmentation task
def count_ngrams_SL(path, freq_threshold, max_vocab_size=-1, max_word_length=4):
    counter = {}
    attr_delim = constants.WL_ATTR_DELIM
    ins_cnt = 0

    with open(path) as f:
        for line in f:
            line = re.sub(' +', ' ', line).strip(' \t\n')
            if len(line) < 1:
                continue

            elif line.startswith(constants.ATTR_DELIM_TXT):
                attr_delim = line.split(constants.KEY_VALUE_DELIM)[1]
                print('Read attribute delimiter:', attr_delim, file=sys.stderr)
                continue
            
            pstr = ''
            entries = line.split()
            for entry in entries:
                word = entry.split(attr_delim)[0]
                pstr += word

            str_ngrams = create_all_char_ngrams(pstr, 2)
            for n in range(1, max_word_length+1):
                str_ngrams = create_all_char_ngrams(pstr, n)
                for sb in str_ngrams:
                    if sb in counter:
                        counter[sb] += 1
                    else:
                        counter[sb] = 0

            ins_cnt += 1
            if ins_cnt % 100000 == 0:
                print('Read', ins_cnt, 'sentences', file=sys.stderr)

    if max_vocab_size > 0:
        counter2 = Counter(counter)
        for pair in counter2.most_common(max_vocab_size):
            del counter[pair[0]]
        
        print('keep {} ngrams from {} ngrams (max vocab size={})'.format(
            len(counter), len(counter2), max_vocab_size), file=sys.stderr)

    freq_ngrams = set()
    for word in counter:
        if counter[word] >= freq_threshold:
            freq_ngrams.add(word)

    print('keep {} ngrams from {} ngrams (frequency threshold={})'.format(
        len(freq_ngrams), len(counter), freq_threshold), file=sys.stderr)
    
    return freq_ngrams


# for segmentation task
def count_ngrams_WL(path, freq_threshold, max_vocab_size=-1, max_word_length=4):
    counter = {}
    attr_delim = constants.WL_ATTR_DELIM
    word_clm = 0                          
    ins_cnt = 0

    with open(path) as f:
        pstr = ''
        for line in f:
            line = re.sub(' +', ' ', line).strip(' \t\n')
            if len(line) < 1:
                for n in range(1, max_word_length+1):
                    str_ngrams = create_all_char_ngrams(pstr, n)
                    for sb in str_ngrams:
                        if sb in counter:
                            counter[sb] += 1
                        else:
                            counter[sb] = 0

                pstr = ''
                ins_cnt += 1
                if ins_cnt % 100000 == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)

                continue

            elif line[0] == constants.COMMENT_SYM:
                if line.startswith(constants.ATTR_DELIM_TXT):
                    attr_delim = line.split(constants.KEY_VALUE_DELIM)[1]
                    print('Read attribute delimiter: \'{}\''.format(attr_delim), file=sys.stderr)

                elif line.startswith(constants.WORD_CLM_TXT):
                    word_clm = int(line.split('=')[1]) - 1
                    print('Read word column id:', word_clm+1, file=sys.stderr)

                continue
            
            array = line.split(attr_delim)
            word = array[word_clm]
            pstr += word

        if pstr:
            for n in range(1, max_word_length+1):
                str_ngrams = create_all_char_ngrams(pstr, n)
                for sb in str_ngrams:
                    if sb in counter:
                        counter[sb] += 1
                    else:
                        counter[sb] = 0

    if max_vocab_size > 0:
        counter2 = Counter(counter)
        for pair in counter2.most_common(max_vocab_size):
            del counter[pair[0]]
        
        print('keep {} ngrams from {} ngrams (max vocab size={})'.format(
            len(counter), len(counter2), max_vocab_size), file=sys.stderr)

    freq_ngrams = set()
    for word in counter:
        if counter[word] >= freq_threshold:
            freq_ngrams.add(word)

    print('keep {} ngrams from {} ngrams (frequency threshold={})'.format(
        len(freq_ngrams), len(counter), freq_threshold), file=sys.stderr)
    
    return freq_ngrams


# for non-segmentation tasks
def count_words_SL(path, freq_threshold, max_vocab_size=-1, lowercase=False, normalize_digits=True):
    counter = {}
    attr_delim = constants.WL_ATTR_DELIM
    ins_cnt = 0

    with open(path) as f:
        for line in f:
            line = re.sub(' +', ' ', line).strip(' \t\n')
            if len(line) < 1:
                continue

            elif line.startswith(constants.ATTR_DELIM_TXT):
                attr_delim = line.split(constants.KEY_VALUE_DELIM)[1]
                print('Read attribute delimiter:', attr_delim, file=sys.stderr)
                continue
            
            entries = line.split()
            for entry in entries:
                attrs = entry.split(attr_delim)
                word = preprocess_token(attrs[0], lowercase, normalize_digits)
                wlen = len(word)

                if word in counter:
                    counter[word] += 1
                else:
                    counter[word] = 0

            ins_cnt += 1
            if ins_cnt % 100000 == 0:
                print('Read', ins_cnt, 'sentences', file=sys.stderr)

    if max_vocab_size > 0:
        counter2 = Counter(counter)
        for pair in counter2.most_common(max_vocab_size):
            del counter[pair[0]]
        
        print('keep {} words from {} words (max vocab size={})'.format(
            len(counter), len(counter2), max_vocab_size), file=sys.stderr)

    freq_words = set()
    for word in counter:
        if counter[word] >= freq_threshold:
            freq_words.add(word)

    print('keep {} words from {} words (frequency threshold={})'.format(
        len(freq_words), len(counter), freq_threshold), file=sys.stderr)
    
    return freq_words


# for non-segmentation tasks
def count_words_WL(path, freq_threshold, max_vocab_size=-1, lowercase=False, normalize_digits=True):
    counter = {}
    attr_delim = constants.WL_ATTR_DELIM
    word_clm = 0
    ins_cnt = 0

    with open(path) as f:
        for line in f:
            line = re.sub(' +', ' ', line).strip(' \n\t')

            if len(line) < 1:
                if ins_cnt > 0 and ins_cnt % 100000 == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
                ins_cnt += 1
                continue

            elif line[0] == constants.COMMENT_SYM:
                if line.startswith(constants.ATTR_DELIM_TXT):
                    attr_delim = line.split(constants.KEY_VALUE_DELIM)[1]
                    print('Read attribute delimiter: \'{}\''.format(attr_delim), file=sys.stderr)

                elif line.startswith(constants.WORD_CLM_TXT):
                    word_clm = int(line.split('=')[1]) - 1
                    print('Read word column id:', word_clm+1, file=sys.stderr)

                continue

            array = line.split(attr_delim)
            word = preprocess_token(array[word_clm], lowercase, normalize_digits)
            wlen = len(word)

            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 0

    if max_vocab_size > 0:
        org_counter = counter
        mc = Counter(org_counter).most_common(max_vocab_size)
        print('keep {} words from {} words (max vocab size={})'.format(
            len(mc), len(org_counter), max_vocab_size), file=sys.stderr)
        counter = {pair[0]:pair[1] for pair in mc}

    freq_tokens = set()
    for word in counter:
        if counter[word] >= freq_threshold:
            freq_tokens.add(word)

    print('keep {} words from {} tokens (frequency threshold={})'.format(
        len(freq_tokens), len(counter), freq_threshold), file=sys.stderr)
    
    return freq_tokens


# for segmentation
def load_external_dictionary(path, use_pos=False):
    if path.endswith('pickle'):
        with open(path, 'rb') as f:
            dic = pickle.load(f)
            dic.create_id2strs()
        return dic

    dic = dictionary.init_dictionary(
        use_unigram=True, use_seg_label=True, use_pos_label=use_pos, use_chunk_trie=True)
    get_unigram_id = dic.tables[constants.UNIGRAM].get_id
    get_pos_id = dic.tables[constants.POS_LABEL].get_id if use_pos else None
    get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id

    use_pos = False            # pos information in dictionary is not used yet
    n_elems = 2 if use_pos else 1
    attr_delim = constants.WL_ATTR_DELIM

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(constants.ATTR_DELIM_TXT):
                attr_delim = line.split(constants.KEY_VALUE_DELIM)[1]
                print('Read attribute delimiter: \'{}\''.format(attr_delim), file=sys.stderr)
                continue

            elif line.startswith(constants.POS_CLM_TXT):
                pos_clm = int(line.split('=')[1]) - 1
                print('Read 1st label column id:', pos_clm+1, file=sys.stderr)
                if pos_clm < 0:
                    use_pos = False
                    n_elems = 1
                continue

            arr = line.split(attr_delim)
            if len(arr) < n_elems or len(arr[0]) == 0:
                continue

            word = arr[0]
            char_ids = [get_unigram_id(char, update=True) for char in word]
            word_id = get_chunk_id(char_ids, word, update=True)
            if use_pos:
                pos = arr[1]
                get_pos_id(pos, update=True)
                
    dic.create_id2strs()
    return dic


def load_pickled_data(filename_wo_ext, load_dic=True):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'rb') as f:
        obj = pickle.load(f)
        inputs = obj[0]
        outputs = obj[1]
        featvecs = obj[2]

    return Data(inputs, outputs, featvecs)


def dump_pickled_data(filename_wo_ext, data):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'wb') as f:
        obj = (data.inputs, data.outputs, data.featvecs)
        pickle.dump(obj, f)


def is_special_token(token):
    if token == constants.NUM_SYMBOL:
        return True
    elif token == constants.ROOT_SYMBOL:
        return True
    else:
        return False

def preprocess_token(token, lowercase=False, normalize_digits=False, refer_tokens=set()):
    if not is_special_token(token):
        if lowercase:
            token = token.lower()
        if normalize_digits and (not refer_tokens or not token in refer_tokens):
            token = re.sub(r'[0-9]+', constants.NUM_SYMBOL, token)
    return token


def to_be_registered(token, train, freq_tokens=set(), refer_vocab=set()):
    if train:
        if not freq_tokens or token in freq_tokens:
            return True            
    else:
        if token in refer_vocab:
            return True

    return False


# def add_padding_to_subtokens(subtokenseq, padding_id):
#     max_len = max([len(subs) for subs in subtokenseq])
#     for subs in subtokenseq:
#         diff = max_len - len(subs)
#         if diff > 0:
#             subs.extend([padding_id for i in range(diff)])


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
    

def create_all_char_ngrams(chars, n):
    seq_len = len(chars)
    if n > seq_len or n <= 0 or seq_len == 0:
        return []

    ngrams = []
    for i in range(seq_len-n+1):
        ngrams.append(chars[i:i+n])
    return ngrams
    

def create_all_char_ngram_indexes(chars, n):
    seq_len = len(chars)
    if n > seq_len or n <= 0 or seq_len == 0:
        return []

    index_pairs = []
    for i in range(seq_len-n+1):
        index_pairs.append((i,i+n))
    return index_pairs


def get_label_BI(index, cate=None):
    prefix = 'B' if index == 0 else 'I'
    suffix = '' if cate is None else '-' + cate
    return prefix + suffix


def get_label_BIES(index, last, cate=None):
    if last == 0:
        prefix = 'S'
    else:
        if index == 0:
            prefix = 'B'
        elif index < last:
            prefix = 'I'
        else:
            prefix = 'E'
    suffix = '' if cate is None else '-' + cate
    return '{}{}'.format(prefix, suffix)


def get_labelseq_BIOES(seq):
    N = len(seq)
    ret = [constants.O] * N
    for i, elem in enumerate(seq):
        if not elem:
            continue

        prev = seq[i-1] if (i > 0 and seq[i-1]) else None
        next = seq[i+1] if (i < N-1 and seq[i+1]) else None

        if prev != elem:
            if elem == next:
                ret[i] = '{}-{}'.format(constants.B, elem)
            else:
                ret[i] = '{}-{}'.format(constants.S, elem)
        else:
            if elem == next:
                ret[i] = '{}-{}'.format(constants.I, elem)
            else:
                ret[i] = '{}-{}'.format(constants.E, elem)

    return ret


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


def get_subpos(pos, depth):
    if depth == 1:
        subpos = pos.split(constants.POS_SEPARATOR)[0]
    elif depth > 1:
        subpos = constants.POS_SEPARATOR.join(pos.split(constants.POS_SEPARATOR)[0:depth])
    else:
        subpos = pos
    return subpos


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

        # print('x', ' '.join([str(i)+':'+dic.tables[constants.UNIGRAM].id2str[t] for i,t in enumerate(tseq)]))
        # print('l', ' '.join([dic.tables[constants.SEG_LABEL].id2str[l] for l in lseq]))
        # if chunk_seq:
        #     print('c', ' '.join([str(i)+':'+trie.id2chunk[c] for i,c in enumerate(chunk_seq)]))
        # print('f', [str(i)+':'+str([trie.id2chunk[c] if c >= 0 else '-1' for c in raw]) for i,raw in enumerate(feats)])
        # print('gc', gchunk_seq)
        # print('mask0', mask0)
        # print('mask1', mask1)
        # for i in range(n):
        #     print(i, token_index2feat_index(i, n, max_len))
        # print()

        if chunk_seq:
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
