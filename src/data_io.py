import sys
import re
import pickle
import argparse
import copy
from collections import Counter

import common
import constants
import dictionary


class Data(object):
    def __init__(self, inputs=None, outputs=None, featvecs=None):
        self.inputs = inputs      # list of input sequences e.g. chars, words
        self.outputs = outputs    # list of output sequences (label sequences) 
        self.featvecs = featvecs  # feature vectors other than input sequences


class RestorableData(Data):
    def __init__(self, inputs=None, outputs=None, featvecs=None, orgdata=None):
        super().__init__(inputs, outputs, featvecs)
        self.orgdata = orgdata


# TODO add refer_chunks
def load_decode_data(
        path, dic, task, data_format=constants.WL_FORMAT,
        use_pos=True, use_subtoken=False, 
        subpos_depth=-1, lowercase=False, normalize_digits=False, refer_tokens=set()):

    segmentation = common.is_segmentation_task(task)
    parsing = common.is_parsing_task(task)
    attribute_annotation = common.is_attribute_annotation_task(task)
    use_subtoken = use_subtoken and not segmentation

    if segmentation or data_format == constants.SL_FORMAT:
        data = load_decode_data_SL(
            path, dic, segmentation=segmentation, parsing=parsing, attribute_annotation=attribute_annotation,
            use_pos=use_pos, subpos_depth=subpos_depth, lowercase=lowercase, 
            normalize_digits=normalize_digits, use_subtoken=use_subtoken, refer_tokens=refer_tokens)

    elif data_format == constants.WL_FORMAT:
        data = load_decode_data_WL(
            path, dic, parsing=parsing, use_pos=use_pos, subpos_depth=subpos_depth, lowercase=lowercase,
            normalize_digits=normalize_digits, use_subtoken=use_subtoken, refer_tokens=refer_tokens)
        
    else:
        print('Invalid data format: {}'.format(data_format))
        sys.exit()

    return data


def load_annotated_data(
        path, task, data_format, train=True, 
        use_pos=True, use_subtoken=False, use_chunk_trie=False, 
        subpos_depth=-1, lowercase=False, normalize_digits=False, 
        max_vocab_size=-1, freq_threshold=1, dic=None, refer_tokens=set(), refer_chunks=set()):
    pos_seqs = []

    segmentation = common.is_segmentation_task(task)
    tagging = common.is_tagging_task(task)
    parsing = common.is_parsing_task(task)
    typed_parsing = common.is_typed_parsing_task(task)
    attr_annotation = common.is_attribute_annotation_task(task)
    use_subtoken = use_subtoken and not segmentation

    if data_format == constants.WL_FORMAT:
        if not train:
            freq_tokens = set()

        elif freq_threshold > 1 or max_vocab_size > 0:
            freq_tokens = count_tokens_WL(
                path, freq_threshold, max_vocab_size=max_vocab_size, 
                segmentation=segmentation, lowercase=lowercase, normalize_digits=normalize_digits)
        else:
            freq_tokens = set()

        data, dic = load_annotated_data_WL(
            path, train=train,
            segmentation=segmentation, tagging=tagging, parsing=parsing, typed_parsing=typed_parsing,
            attribute_annotation=attr_annotation,
            use_pos=use_pos, subpos_depth=subpos_depth, 
            use_subtoken=use_subtoken, use_chunk_trie=use_chunk_trie,
            lowercase=lowercase, normalize_digits=normalize_digits,
            dic=dic, freq_tokens=freq_tokens, refer_tokens=refer_tokens, refer_chunks=refer_chunks)

    elif data_format == constants.SL_FORMAT and not parsing:
        if not train:
            freq_tokens = set()

        elif freq_threshold > 1 or max_vocab_size > 0:
            freq_tokens = count_tokens_SL(
                path, freq_threshold, max_vocab_size=max_vocab_size, 
                segmentation=segmentation, lowercase=lowercase, normalize_digits=normalize_digits)
        else:
            freq_tokens = set()

        data, dic = load_annotated_data_SL(
            path, train=train, segmentation=segmentation, tagging=tagging, 
            use_pos=use_pos, subpos_depth=subpos_depth, use_subtoken=use_subtoken, 
            use_chunk_trie=use_chunk_trie, 
            lowercase=lowercase, normalize_digits=normalize_digits,
            dic=dic, freq_tokens=freq_tokens, refer_tokens=refer_tokens)
        
    else:
        print('Invalid data format: {}'.format(data_format))
        sys.exit()

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
        path, dic, segmentation=False, parsing=False, attribute_annotation=False, 
        use_pos=False, subpos_depth=-1, lowercase=False, normalize_digits=True, use_subtoken=False,
        refer_tokens=set()):
    attr_delim = constants.SL_ATTR_DELIM

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
        root_token = constants.ROOT_SYMBOL
        root_token_id = dic.tables[constants.UNIGRAM].get_id(constants.ROOT_SYMBOL)
        if use_subtoken:
            root_subtoken_id = dic.tables[constants.SUBTOKEN].get_id(constants.ROOT_SYMBOL)
        if use_pos:
            root_pos_id = dic.tables[constants.POS_LABEL].get_id(constants.ROOT_SYMBOL)
        
    with open(path) as f:
        for line in f:
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
                token_seqs.append(
                    [get_unigram_id(preprocess_token(
                        char, lowercase, normalize_digits, refer_tokens)) for char in line])
                org_token_seqs.append([char for char in line])

            else:               # tokeniized text w/ or w/o pos: tag or (t)dep
                org_arr = line.split(constants.SL_TOKEN_DELIM)

                org_token_seq = [word for word in org_arr]
                if parsing or attribute_annotation:
                    org_token_seq = [elem.split(attr_delim)[0] for elem in org_arr]
                if parsing:
                    org_token_seq.insert(0, root_token)
                org_token_seqs.append(org_token_seq)

                pro_token_seq = [preprocess_token(
                    word, lowercase, normalize_digits, refer_tokens) for word in org_token_seq]
                token_seq = [get_unigram_id(pword, update=pword in refer_tokens) for pword in pro_token_seq]
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

            ins_cnt += 1
            if ins_cnt % 100000 == 0:
                print('read', ins_cnt, 'sentences', file=sys.stderr)

    inputs = [token_seqs]
    if subtoken_seqs:
        inputs.append(subtoken_seqs)

    outputs = []
    if use_pos:
        outputs.append(pos_seqs)
        
    orgdata = [org_token_seqs]
    if use_pos:
        orgdata.append(org_pos_seqs)

    return RestorableData(inputs, outputs, orgdata=orgdata)


def load_decode_data_WL(
        path, dic, parsing=False, use_pos=True, subpos_depth=-1,
        lowercase=False, normalize_digits=True, use_subtoken=False, refer_tokens=set()):
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

    ot_seq_ = [constants.ROOT_SYMBOL] if parsing else []
    opos_seq_ = [constants.ROOT_SYMBOL] if parsing and use_pos else []
    t_seq_ = [get_unigram_id(constants.ROOT_SYMBOL)] if parsing else []
    st_seq_ = [[get_subtoken_id(constants.ROOT_SYMBOL)]] if parsing and use_subtoken else []
    pos_seq_ = [get_pos_id(constants.ROOT_SYMBOL)] if parsing and use_pos else []

    attr_delim = constants.WL_ATTR_DELIM
    word_clm = 0
    pos_clm = 1

    with open(path) as f:
        ot_seq = ot_seq_.copy()
        opos_seq = opos_seq_.copy()
        t_seq = t_seq_.copy()
        st_seq = st_seq_.copy()
        pos_seq = pos_seq_.copy() 

        for line in f:
            line = re.sub(' +', ' ', line)
            line = line.strip(' \t\n')
            if len(line) < 1:
                if len(t_seq) - (1 if parsing else 0) > 0:
                    token_seqs.append(t_seq)
                    t_seq = t_seq_.copy()
                    org_token_seqs.append(ot_seq)
                    ot_seq = ot_seq_.copy()
                    if st_seq:
                        subtoken_seqs.append(st_seq)
                        st_seq = st_seq_.copy()
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

            array = line.split(attr_delim)
            word = array[word_clm]
            ot_seq.append(word)
            pword = preprocess_token(word, lowercase, normalize_digits, refer_tokens)
            t_seq.append(get_unigram_id(pword, update=pword in refer_tokens))

            if use_pos:
                pos = get_subpos(array[pos_clm], subpos_depth)
                opos_seq.append(pos)
                pos_seq.append(get_pos_id(pos))

            if use_subtoken:
                # TODO fix a case that token contains <NUM> and does not equal to <NUM>
                if is_special_token(pword):
                    st_seq.append([get_subtoken_id(pword)])
                else:
                    st_seq.append([get_subtoken_id(pword[i]) for i in range(len(pword))])

        n_non_words = 1 if parsing else 0
        if len(t_seq) - n_non_words > 0:
            org_token_seqs.append(ot_seq)
            token_seqs.append(t_seq)
            if opos_seq:
                org_pos_seqs.append(opos_seq)
            if subtoken_seqs:
                subtoken_seqs.append(st_seq)
            if pos_seq:
                pos_seqs.append(pos_seq)

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
        use_pos=True, subpos_depth=-1, use_subtoken=False, use_chunk_trie=False, 
        lowercase=False, normalize_digits=True,
        dic=None, freq_tokens=set(), refer_tokens=set(), refer_chunks=set()):
    use_pos = tagging 
    if not dic:
        dic = dictionary.init_dictionary(
            use_unigram=True, use_bigram=False, use_subtoken=use_subtoken, use_token_type=False,
            use_seg_label=segmentation, use_pos_label=use_pos, 
            use_chunk_trie=use_chunk_trie, use_root=False)

    attr_delim = constants.SL_ATTR_DELIM

    ins_cnt = 0

    token_seqs = []
    subtoken_seqs = []
    seg_seqs = []               # list of segmentation label sequences
    pos_seqs = []               # list of POS label sequences

    get_unigram_id = dic.tables[constants.UNIGRAM].get_id
    get_subtoken_id = dic.tables[constants.SUBTOKEN].get_id if use_subtoken else None
    get_seg_id = dic.tables[constants.SEG_LABEL].get_id if segmentation else None
    get_pos_id = dic.tables[constants.POS_LABEL].get_id if use_pos else None
    get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id if use_chunk_trie else None

    with open(path) as f:
        for line in f:
            line = re.sub(' +', ' ', line).strip(' \t\n')
            if len(line) < 1:
                continue

            elif line.startswith(constants.ATTR_DELIM_TXT):
                attr_delim = line.split(constants.KEY_VALUE_DELIM)[1]
                print('Read attribute delimiter:', attr_delim, file=sys.stderr)
                continue
            
            entries = line.split(constants.SL_TOKEN_DELIM)
            t_seq = []
            st_seq = []
            seg_seq = []
            pos_seq = []

            for entry in entries:
                attrs = entry.split(attr_delim)
                word = preprocess_token(attrs[0], lowercase, normalize_digits, refer_tokens)
                
                if tagging:
                    if len(attrs) < 2:
                        continue # todo
                    pos = get_subpos(attrs[1], subpos_depth)

                    if segmentation and train:
                        for seg_lab in constants.SEG_LABELS:
                            get_seg_id('{}-{}'.format(seg_lab, pos) , True)
                else:
                    pos = None    
                        
                wlen = len(word)

                if segmentation:
                    update_chunk = use_chunk_trie and to_be_registered(pword, train, set(), refer_chunks)
                    update_tokens = [
                        update_chunk or to_be_registered(word[i], train, freq_tokens, refer_tokens) 
                        for i in range(wlen)]

                    t_seq.extend([get_unigram_id(word[i], update_tokens[i]) for i in range(wlen)])
                    seg_seq.extend(
                        [get_seg_id(
                            get_label_BIES(i, wlen-1, cate=pos), update=train) for i in range(wlen)])

                    if use_chunk_trie and (update_chunk or update_token == [True for i in range(wlen)]):
                        ids = chars[-wlen:]
                        get_chunk_id(ids, word, True)

                else:
                    update_token = to_be_registered(word, train, freq_tokens, refer_tokens)
                    t_seq.append(get_unigram_id(word, update=update_token))
                    pos_seq.append(get_pos_id(pos, update=train))

                    if use_subtoken:
                        # TODO fix a case that token contains <NUM> and does not equal to <NUM>
                        if is_special_token(pword):
                            st_seq.append([get_subtoken_id(word, update_token)])
                        else:
                            st_seq.append(
                                [get_subtoken_id(word[i], update_token) for i in range(wlen)])

            ins_cnt += 1

            token_seqs.append(t_seq)
            if subtoken_seqs:
                subtoken_seqs.append(st_seq)
            if seg_seq:
                seg_seqs.append(seg_seq)
            if pos_seq:
                pos_seqs.append(pos_seq)

            if ins_cnt % 100000 == 0:
                print('Read', ins_cnt, 'sentences', file=sys.stderr)

    inputs = [token_seqs]
    if subtoken_seqs:
        inputs.append(subtoken_seqs)

    outputs = []
    if seg_seqs:
        outputs.append(seg_seqs)
    if pos_seqs:
        outputs.append(pos_seqs)

    return Data(inputs, outputs), dic


"""
Read data with WL (one word in a line) format.
If tagging == True, the following format is expected for joint segmentation and POS tagging:
  word1 \t pos1 \t otther attributes ...
  word2 \t pos2 \t otther attributes ...
  ...

otherwise, the following format is expected for segmentation:
  word1 \t otther attributes ...
  word2 \t otther attributes ...
  ...
"""
def load_annotated_data_WL(
        path, train=True, 
        segmentation=True, tagging=False, parsing=False, typed_parsing=False, attribute_annotation=False, 
        use_pos=True, subpos_depth=-1, use_subtoken=False, use_chunk_trie=False, 
        lowercase=False, normalize_digits=True,
        dic=None, freq_tokens=set(), refer_tokens=set(), refer_chunks=set()):
    if not dic:
        dic = dictionary.init_dictionary(
            use_unigram=True, use_bigram=False, use_subtoken=use_subtoken, use_token_type=False,
            use_seg_label=segmentation, use_pos_label=use_pos, use_arc_label=typed_parsing, 
            use_attr_label=attribute_annotation, use_chunk_trie=use_chunk_trie, use_root=parsing)

    attr_delim = constants.WL_ATTR_DELIM
    word_clm = 0
    pos_clm = 1
    head_clm = 2
    arc_clm = 3
    attr_clm = 3

    ins_cnt = 0

    token_seqs = []
    subtoken_seqs = []
    seg_seqs = []        # list of segmentation label sequences
    pos_seqs = []        # list of POS label sequences
    head_seqs = []       # list of head id sequences
    arc_seqs = []        # list of arc label sequences
    attr_seqs = []       # list of attribute label sequences
    
    get_unigram_id = dic.tables[constants.UNIGRAM].get_id
    get_subtoken_id = dic.tables[constants.SUBTOKEN].get_id if use_subtoken else None
    get_seg_id = dic.tables[constants.SEG_LABEL].get_id if segmentation else None
    get_pos_id = dic.tables[constants.POS_LABEL].get_id if use_pos else None
    get_arc_id = dic.tables[constants.ARC_LABEL].get_id if typed_parsing else None
    get_attr_id = dic.tables[constants.ATTR_LABEL].get_id if attribute_annotation else None
    get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id if use_chunk_trie else None

    t_seq_ = [get_unigram_id(constants.ROOT_SYMBOL)] if parsing else []
    st_seq_ = [[get_subtoken_id(constants.ROOT_SYMBOL)]] if parsing and use_subtoken else []
    seg_seq_ = []
    pos_seq_ = [get_pos_id(constants.ROOT_SYMBOL)] if parsing and use_pos else []
    head_seq_ = [constants.NO_PARENTS_ID] if parsing else []
    arc_seq_ = [constants.NO_PARENTS_ID] if typed_parsing else []
    attr_seq_ = []

    with open(path) as f:
        t_seq = t_seq_.copy()
        st_seq = st_seq_.copy()
        seg_seq = seg_seq_.copy()
        pos_seq = pos_seq_.copy() 
        head_seq = head_seq_.copy() 
        arc_seq = arc_seq_.copy() 
        attr_seq = attr_seq_.copy()

        for line in f:
            line = re.sub(' +', ' ', line).strip(' \t\n')

            if len(line) < 1:
                if len(t_seq) - (1 if parsing else 0) > 0:
                    token_seqs.append(t_seq)
                    t_seq = t_seq_.copy()
                    if st_seq:
                        subtoken_seqs.append(st_seq)
                        st_seq = st_seq_.copy()
                    if seg_seq:
                        seg_seqs.append(seg_seq)
                        seg_seq = seg_seq_.copy()
                    if pos_seq:
                        pos_seqs.append(pos_seq)
                        pos_seq = pos_seq_.copy()
                    if head_seq:
                        head_seqs.append(head_seq)
                        head_seq = head_seq_.copy()
                    if arc_seq:
                        arc_seqs.append(arc_seq)
                        arc_seq = arc_seq_.copy()
                    if attr_seq:
                        attr_seqs.append(attr_seq)
                        attr_seq = attr_seq_.copy()

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

                elif line.startswith(constants.POS_CLM_TXT):
                    pos_clm = int(line.split('=')[1]) - 1
                    print('Read 1st label column id:', pos_clm+1, file=sys.stderr)

                    if pos_clm < 0:
                        use_pos = False
                        pos_seqs = pos_seq = pos_seq_ = []
                        if dic.has_table(constants.POS_LABEL):
                            del dic.tables[constants.POS_LABEL]
                        if tagging: 
                            print('POS label is mandatory for POS tagging', file=sys.stderr)
                            sys.exit()

                elif line.startswith(constants.HEAD_CLM_TXT):
                    head_clm = int(line.split('=')[1]) - 1
                    print('Read 2nd label column id:', head_clm+1, file=sys.stderr)

                elif line.startswith(constants.ARC_CLM_TXT):
                    arc_clm = int(line.split('=')[1]) - 1
                    print('Read 3rd label column id:', arc_clm+1, file=sys.stderr)

                elif line.startswith(constants.ATTR_CLM_TXT):
                    attr_clm = int(line.split('=')[1]) - 1
                    print('Read 4th label column id:', attr_clm+1, file=sys.stderr)

                continue

            array = line.split(attr_delim)
            pword = preprocess_token(array[word_clm], lowercase, normalize_digits, refer_tokens)

            if use_pos:
                pos = get_subpos(array[pos_clm], subpos_depth)
                if segmentation and train:
                    for seg_lab in constants.SEG_LABELS:
                        get_seg_id('{}-{}'.format(seg_lab, pos) , True)
            else:
                pos = None

            wlen = len(pword)
            if segmentation:
                update_chunk = use_chunk_trie and to_be_registered(pword, train, set(), refer_chunks)
                update_tokens = [
                    update_chunk or to_be_registered(pword[i], train, freq_tokens, refer_tokens) 
                    for i in range(wlen)]
                
                t_seq.extend([get_unigram_id(pword[i], update_tokens[i]) for i in range(wlen)])
                seg_seq.extend(
                    [get_seg_id(
                        get_label_BIES(i, wlen-1, cate=pos), update=train) for i in range(wlen)])

                if use_chunk_trie and (update_chunk or update_tokens == [True for i in range(wlen)]):
                    token_ids = t_seq[-wlen:]
                    get_chunk_id(token_ids, pword, True)

            else:
                update_token = to_be_registered(pword, train, freq_tokens, refer_tokens)
                t_seq.append(get_unigram_id(pword, update=update_token))

                if use_subtoken:
                    # TODO fix a case that token contains <NUM> and does not equal to <NUM>
                    if is_special_token(pword):
                        st_seq.append([get_subtoken_id(pword, update_token)])
                    else:
                        st_seq.append([get_subtoken_id(pword[i], update_token) for i in range(len(pword))])

            if use_pos:
                pos_seq.append(get_pos_id(pos, update=train))

            if parsing:
                head = int(array[head_clm])
                if head < 0:
                    head = 0
                head_seq.append(head)

            if typed_parsing:
                arc = array[arc_clm]
                arc_seq.append(get_arc_id(arc, update=train))
                
            if attribute_annotation:
                if len(array) > attr_clm:
                    attr = array[attr_clm]
                else:
                    attr = constants.NONE_SYMBOL
                attr_seq.append(get_attr_id(attr, update=train))

        n_non_words = 1 if parsing else 0
        if len(t_seq) - n_non_words > 0:
            token_seqs.append(t_seq)
            if subtoken_seqs:
                subtoken_seqs.append(st_seq)
            if seg_seq:
                seg_seqs.append(seg_seq)
            if pos_seq:
                pos_seqs.append(pos_seq)
            if head_seq:
                head_seqs.append(head_seq)
            if arc_seq:
                arc_seqs.append(arc_seq)
            if attr_seq:
                attr_seqs.append(attr_seq)

    inputs = [token_seqs]
    if subtoken_seqs:
        inputs.append(subtoken_seqs)

    outputs = []
    if seg_seqs:
        outputs.append(seg_seqs)
    if pos_seqs:
        outputs.append(pos_seqs)
    elif parsing or attribute_annotation:
        outputs.append(None)
    if head_seqs:
        outputs.append(head_seqs)
    if arc_seqs:
        outputs.append(arc_seqs)
    if attr_seqs:
        outputs.append(attr_seqs)

    return Data(inputs, outputs), dic


def count_tokens_SL(path, freq_threshold, max_vocab_size=-1,
                    segmentation=True, lowercase=False, normalize_digits=True,
                    refer_tokens=set()):
    counter = {}
    attr_delim = constants.WL_ATTR_DELIM
    word_clm = 0
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
            
            entries = split()
            for entry in entries:
                attrs = entry.split(attr_delim)
                word = preprocess_token(attrs[0], lowercase, normalize_digits)
                wlen = len(word)

                if segmentation:
                    for i in range(wlen):
                        if word in counter:
                            counter[word[i]] += 1
                        else:
                            counter[word[i]] = 0

                else:
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
        
        print('keep {} tokens from {} tokens (max vocab size={})'.format(
            len(counter), len(counter2), max_vocab_size), file=sys.stderr)

    freq_tokens = set()
    for word in counter:
        if counter[word] >= freq_threshold:
            freq_tokens.add(word)

    print('keep {} tokens from {} tokens (frequency threshold={})'.format(
        len(freq_tokens), len(counter), freq_threshold), file=sys.stderr)
    
    return freq_tokens


def count_tokens_WL(path, freq_threshold, max_vocab_size=-1,
                    segmentation=True, lowercase=False, normalize_digits=True,
                    refer_tokens=set()):
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

            if segmentation:
                for i in range(wlen):
                    if word in counter:
                        counter[word[i]] += 1
                    else:
                        counter[word[i]] = 0

            else:
                if word in counter:
                    counter[word] += 1
                else:
                    counter[word] = 0

    if max_vocab_size > 0:
        org_counter = counter
        mc = Counter(org_counter).most_common(max_vocab_size)
        print('keep {} tokens from {} tokens (max vocab size={})'.format(
            len(mc), len(org_counter), max_vocab_size), file=sys.stderr)
        counter = {pair[0]:pair[1] for pair in mc}

    freq_tokens = set()
    for word in counter:
        if counter[word] >= freq_threshold:
            freq_tokens.add(word)

    print('keep {} tokens from {} tokens (frequency threshold={})'.format(
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
    if token in refer_vocab:
        return True
    elif train and (not freq_tokens or token in freq_tokens):
        return True
    else:
        return False


# def add_padding_to_subtokens(subtokenseq, padding_id):
#     max_len = max([len(subs) for subs in subtokenseq])
#     for subs in subtokenseq:
#         diff = max_len - len(subs)
#         if diff > 0:
#             subs.extend([padding_id for i in range(diff)])


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


def get_subpos(pos, depth):
    if depth == 1:
        subpos = pos.split(constants.POS_SEPARATOR)[0]
    elif depth > 1:
        subpos = constants.POS_SEPARATOR.join(pos.split(constants.POS_SEPARATOR)[0:depth])
    else:
        subpos = pos
    return subpos
