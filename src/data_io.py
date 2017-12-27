import sys
import re
import pickle
import argparse
import copy
from collections import Counter

import constants
import dictionary


class Data(object):
    def __init__(self, inputs=None, outputs=None, featvecs=None):
        self.inputs = inputs      # list of input sequences e.g. chars, words
        self.outputs = outputs    # list of output sequences (label sequences) 
        self.featvecs = featvecs  # feature vectors other than input sequences


def load_raw_text_for_segmentation(path, dic):
    tokenseqs = []
    ins_cnt = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue

            ins = [dic.get_token_id(char) for char in line]
            tokenseqs.append(ins)

            ins_cnt += 1
            if ins_cnt % 100000 == 0:
                print('read', ins_cnt, 'sentences', file=sys.stderr)

    return tokenseqs


def load_raw_text_for_tagging(path, dic):
    tokenseqs = []
    ins_cnt = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue

            array = line.split(' ')
            ins = [dic.get_token_id(word) for word in array]
            tokenseqs.append(ins)

            ins_cnt += 1
            if ins_cnt % 100000 == 0:
                print('read', ins_cnt, 'sentences', file=sys.stderr)

    return tokenseqs


"""
Read data with SL (one sentence in one line) format.
If tagging == True, the following format is expected for joint segmentation and POS tagging:
  word1_pos1 word2_pos2 ... wordn_posn

otherwise, the following format is expected for segmentation:
  word1 word2 ... wordn
"""
def load_data_SL(path, segmentation=True, tagging=False,
                 update_tokens=True, update_labels=True, 
                 lowercase=False, normalize_digits=True,
                 subpos_depth=-1, use_subtoken=False, create_chunk_trie=False, 
                 freq_tokens=set(), dic=None, refer_vocab=set()):
    if not dic:
        dic = dictionary.init_dictionary(
            use_unigram=True, use_bigram=False, use_subtoken=use_subtoken, use_token_type=False,
            use_seg_label=segmentation, use_pos_label=read_pos, 
            use_chunk_trie=create_chunk_trie, use_root=parsing)

    delim = constants.DELIM1_SYMBOL

    ins_cnt = 0

    tokenseqs = []
    subtokenseqs = []
    seg_seqs = []               # list of segmentation label sequences
    pos_seqs = []               # list of POS label sequences

    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue

            elif line.startswith(constants.DELIM_TXT):
                delim = line.split(constants.KEY_VALUE_SEPARATOR)[1]
                print('Read delimiter:', delim, file=sys.stderr)
                continue
            
            entries = re.sub(' +', ' ', line).split()
            t_seq = []
            st_seq = []
            seg_seq = []
            pos_seq = []

            for entry in entries:
                attrs = entry.split(delim)
                word = attrs[0]
                if lowercase:
                    word = word.lower()
                if normalize_digits and not word in refer_vocab:
                    word = re.sub(r'[0-9]+', constants.NUM_SYMBOL, word)
                
                if tagging:
                    pos = attrs[1]
                    if subpos_depth == 1:
                        pos = pos.split(constants.POS_SEPARATOR)[0]
                    elif subpos_depth > 1:
                        pos = constants.POS_SEPARATOR.join(
                            pos.split(constants.POS_SEPARATOR)[0:subpos_depth])

                    if segmentation and update_labels:
                        for seg_lab in constants.SEG_LABELS:
                            dic.seg_label_indices.get_id('{}-{}'.format(seg_lab, pos) , True)
                else:
                    pos = None    
                        
                wlen = len(word)

                if segmentation:
                    update_token = [
                        update_each_token(word[i], update_tokens, freq_tokens, refer_vocab) for i in range(wlen)]
                    t_seq.extend([dic.tables['unigram'].get_id(word[i], update_token[i]) for i in range(wlen)])
                    seg_seq.extend(
                        [dic.tables['seg_label'].get_id(
                            get_label_BIES(i, wlen-1, cate=pos), update=update_labels) for i in range(wlen)])

                    if create_chunk_trie and update_token == [True for i in range(wlen)]: # update word trie
                        ids = chars[-wlen:]
                        dic.chunk_trie.get_chunk_id(ids, True)

                else:
                    update_token = update_each_token(word, update_tokens, freq_tokens, refer_vocab)
                    t_seq.append(dic.tables['unigram'].get_id(word, update=update_token))
                    pos_seq.append(dic.tables['pos_label'].get_id(pos, update=update_labels))

                    if use_subtoken:
                        update_token = [
                            update_each_token(word[i], update_tokens, refer_vocab) for i in range(wlen)]
                        st_seq.append(
                            [dic.tables['subtoken'].get_id(word[i], update_token[i]) for i in range(wlen)])

            ins_cnt += 1

            tokenseqs.append(t_seq)
            if subtokenseqs:
                subtokenseqs.append(st_seq)
            if seg_seq:
                seg_seqs.append(seg_seq)
            if pos_seq:
                pos_seqs.append(pos_seq)

            if ins_cnt % 100000 == 0:
                print('Read', ins_cnt, 'sentences', file=sys.stderr)

    inputs = [tokenseqs]
    if subtokenseqs:
        inputs.append(subtokenseqs)

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
def load_data_WL(path, segmentation=True, tagging=False, parsing=False, typed_parsing=False,
                 update_tokens=True, update_labels=True, 
                 lowercase=False, normalize_digits=True,
                 subpos_depth=-1, predict_parent_existence=False,
                 use_subtoken=False, create_chunk_trie=False, 
                 freq_tokens=set(), dic=None, refer_vocab=set()):
    read_pos = tagging or parsing
    pred_parex = predict_parent_existence
    if not dic:
        dic = dictionary.init_dictionary(
            use_unigram=True, use_bigram=False, use_subtoken=use_subtoken, use_token_type=False,
            use_seg_label=segmentation, use_pos_label=read_pos, use_arc_label=typed_parsing, 
            use_chunk_trie=create_chunk_trie, use_root=parsing)
    # predict_parent_existence is unused

    delim = constants.DELIM2_SYMBOL
    word_clm = 0
    pos_clm = 1
    head_clm = 2
    arc_clm = 3

    ins_cnt = 0

    tokenseqs = []
    subtokenseqs = []
    seg_seqs = []        # list of segmentation label sequences
    pos_seqs = []        # list of POS label sequences
    head_seqs = []       # list of head id sequences
    arc_seqs = []        # list of arc label sequences

    t_seq_ = [dic.tables['unigram'].get_id(constants.ROOT_SYMBOL)] if parsing else []
    st_seq_ = [[dic.tables['subtoken'].get_id(constants.ROOT_SYMBOL)]] if parsing and use_subtoken else []
    seg_seq_ = []
    pos_seq_ = [dic.tables['pos_label'].get_id(constants.ROOT_SYMBOL)] if parsing and read_pos else []
    head_seq_ = [constants.NO_PARENTS_ID] if parsing else []
    arc_seq_ = [constants.NO_PARENTS_ID] if typed_parsing else []

    with open(path) as f:
        t_seq = t_seq_.copy()
        st_seq = st_seq_.copy()
        seg_seq = seg_seq_.copy()
        pos_seq = pos_seq_.copy() 
        head_seq = head_seq_.copy() 
        arc_seq = arc_seq_.copy() 

        for line in f:
            line = re.sub(' +', ' ', line).strip('\n\t')

            if len(line) < 1:
                if len(t_seq) - (1 if parsing else 0) > 0:
                    tokenseqs.append(t_seq)
                    t_seq = t_seq_.copy()
                    if st_seq:
                        # add_padding_to_subtokens(st_seq, dic.pad_id)
                        subtokenseqs.append(st_seq)
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

                    ins_cnt += 1
                    if ins_cnt % 100000 == 0:
                        print('Read', ins_cnt, 'sentences', file=sys.stderr)
                continue

            elif line[0] == constants.COMMENT_SYM:

                if line.startswith(constants.DELIM_TXT):
                    delim = line.split(constants.KEY_VALUE_SEPARATOR)[1]
                    print('Read delimiter: \'{}\''.format(delim), file=sys.stderr)

                elif line.startswith(constants.WORD_CLM_TXT):
                    word_clm = int(line.split('=')[1]) - 1
                    print('Read word column id:', word_clm+1, file=sys.stderr)

                elif line.startswith(constants.POS_CLM_TXT):
                    pos_clm = int(line.split('=')[1]) - 1
                    print('Read 1st label column id:', pos_clm+1, file=sys.stderr)

                    if pos_clm < 0:
                        read_pos = False
                        pos_seq = []
                        if tagging: 
                            print('POS label is mandatory for POS tagging', file=sys.stderr)
                            sys.exit()

                elif line.startswith(constants.HEAD_CLM_TXT):
                    head_clm = int(line.split('=')[1]) - 1
                    print('Read 2nd label column id:', head_clm+1, file=sys.stderr)

                elif line.startswith(constants.ARC_CLM_TXT):
                    arc_clm = int(line.split('=')[1]) - 1
                    print('Read 3rd label column id:', arc_clm+1, file=sys.stderr)

                continue

            array = line.split(delim)
            word = array[word_clm]
            if lowercase:
                word = word.lower()
            if normalize_digits and not word in refer_vocab:
                word = re.sub(r'[0-9]+', constants.NUM_SYMBOL, word)

            if read_pos:
                pos = array[pos_clm]
                if subpos_depth == 1:
                    pos = pos.split(constants.POS_SEPARATOR)[0]
                elif subpos_depth > 1:
                    pos = constants.POS_SEPARATOR.join(
                        pos.split(constants.POS_SEPARATOR)[0:subpos_depth])

                if segmentation and update_labels:
                    for seg_lab in constants.SEG_LABELS:
                        dic.seg_label_indices.get_id('{}-{}'.format(seg_lab, pos) , True)
            else:
                pos = None

            if parsing:
                head = array[head_clm]

            if typed_parsing:
                arc = array[arc_clm]

            wlen = len(word)

            if segmentation:
                update_token = [
                    update_each_token(word[i], update_tokens, freq_tokens, refer_vocab) for i in range(wlen)]
                t_seq.extend([dic.tables['unigram'].get_id(word[i], update_token[i]) for i in range(wlen)])
                seg_seq.extend(
                    [dic.tables['seg_label'].get_id(
                        get_label_BIES(i, wlen-1, cate=pos), update=update_labels) for i in range(wlen)])

                if create_chunk_trie and update_token == [True for i in range(wlen)]: # update chunk trie
                    token_ids = t_seq[-wlen:]
                    dic.tries['chunk'].get_chunk_id(token_ids, True)

            else:
                update_token = update_each_token(word, update_tokens, freq_tokens, refer_vocab)
                t_seq.append(dic.tables['unigram'].get_id(word, update=update_token))

                if use_subtoken:
                    update_token = [update_each_token(word[i], update_tokens, refer_vocab) for i in range(wlen)]
                    st_seq.append(
                        [dic.tables['subtoken'].get_id(word[i], update_token[i]) for i in range(wlen)])

            if read_pos:
                pos_seq.append(dic.tables['pos_label'].get_id(pos, update=update_labels))

            if parsing:
                head = int(head)
                if head < 0:
                    head = 0
                head_seq.append(head)

            if typed_parsing:
                arc_seq.append(dic.tables['arc_label'].get_id(arc, update=update_labels))
                
        n_non_words = 1 if parsing else 0
        if len(t_seq) - n_non_words > 0:
            tokenseqs.append(t_seq)
            if subtokenseqs:
                subtokenseqs.append(st_seq)
            if seg_seq:
                seg_seqs.append(seg_seq)
            if pos_seq:
                pos_seqs.append(pos_seq)
            if head_seq:
                head_seqs.append(head_seq)
            if arc_seq:
                arc_seqs.append(arc_seq)

    inputs = [tokenseqs]
    if subtokenseqs:
        inputs.append(subtokenseqs)

    outputs = []
    if seg_seqs:
        outputs.append(seg_seqs)
    if pos_seqs:
        outputs.append(pos_seqs)
    elif parsing:
        outputs.append(None)
    if head_seqs:
        outputs.append(head_seqs)
    if arc_seqs:
        outputs.append(arc_seqs)

    return Data(inputs, outputs), dic


# def init_list(cond=False, elem=None, org_list=None):
#     if org_list is not None:
#         ret = org_list.copy()
#     else:
#         ret = []
#     if cond:
#         ret.append(elem)
#     return ret


# def add_padding_to_subtokens(subtokenseq, padding_id):
#     max_len = max([len(subs) for subs in subtokenseq])
#     for subs in subtokenseq:
#         diff = max_len - len(subs)
#         if diff > 0:
#             subs.extend([padding_id for i in range(diff)])


def update_each_token(token, update_tokens, freq_tokens=set(), refer_vocab=set()):
    if token in refer_vocab:
        return True
    elif update_tokens and (not freq_tokens or token in freq_tokens):
        return True
    else:
        return False


def count_tokens_SL(path, freq_threshold, max_vocab_size=-1,
                    segmentation=True, lowercase=False, normalize_digits=True,
                    refer_vocab=set()):
    counter = {}
    delim = constants.DELIM2_SYMBOL
    word_clm = 0
    ins_cnt = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue

            elif line.startswith(constants.DELIM_TXT):
                delim = line.split(constants.KEY_VALUE_SEPARATOR)[1]
                print('Read delimiter:', delim, file=sys.stderr)
                continue
            
            entries = re.sub(' +', ' ', line).split()
            for entry in entries:
                attrs = entry.split(delim)
                word = attrs[0]
                if lowercase:
                    word = word.lower()
                if normalize_digits and not word in refer_vocab:
                    word = re.sub(r'[0-9]+', constants.NUM_SYMBOL, word)
                        
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
                    refer_vocab=set()):
    counter = {}
    delim = constants.DELIM2_SYMBOL
    word_clm = 0
    ins_cnt = 0

    with open(path) as f:
        for line in f:
            line = re.sub(' +', ' ', line).strip('\n\t')

            if len(line) < 1:
                if ins_cnt > 0 and ins_cnt % 100000 == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
                ins_cnt += 1
                continue

            elif line[0] == constants.COMMENT_SYM:
                if line.startswith(constants.DELIM_TXT):
                    delim = line.split(constants.KEY_VALUE_SEPARATOR)[1]
                    print('Read delimiter: \'{}\''.format(delim), file=sys.stderr)

                elif line.startswith(constants.WORD_CLM_TXT):
                    word_clm = int(line.split('=')[1]) - 1
                    print('Read word column id:', word_clm+1, file=sys.stderr)

                continue

            array = line.split(delim)
            word = array[word_clm]
            if lowercase:
                word = word.lower()
            if normalize_digits and not word in refer_vocab:
                word = re.sub(r'[0-9]+', constants.NUM_SYMBOL, word)

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


def load_data(data_format, path, read_pos=True, update_tokens=True, update_labels=True, 
              use_subtoken=False, create_chunk_trie=False, subpos_depth=-1, predict_parent_existence=False,
              lowercase=False, normalize_digits=False, 
              #no_freq_update=False,
              max_vocab_size=-1, freq_threshold=1, dic=None, refer_vocab=set(), limit=-1):
    pos_seqs = []

    segmentation = True if 'seg' in data_format else False
    tagging = True if 'tag' in data_format else False
    parsing = True if 'dep' in data_format else False
    typed_parsing = True if 'tdep' in data_format else False
    use_subtoken = use_subtoken and not segmentation

    if (data_format == 'wl_seg' or data_format == 'wl_seg_tag' or data_format == 'wl_tag' or
        data_format == 'wl_dual_seg' or data_format == 'wl_dual_seg_tag' or data_format == 'wl_dual_tag' or
        data_format == 'wl_dep' or data_format == 'wl_tdep' or
        data_format == 'wl_tag_dep' or data_format == 'wl_tag_tdep'
    ):
        if not update_tokens:
            freq_tokens = set()

        # elif no_freq_update and dic:
        #     freq_tokens = set(dic.id2word) # skip count tokens

        elif freq_threshold > 1 or max_vocab_size > 0:
            freq_tokens = count_tokens_WL(
                path, freq_threshold, max_vocab_size=max_vocab_size, 
                segmentation=segmentation, lowercase=lowercase, normalize_digits=normalize_digits)
        else:
            freq_tokens = set()

        data, dic = load_data_WL(
            path, segmentation=segmentation, tagging=tagging, parsing=parsing, typed_parsing=typed_parsing,
            update_tokens=update_tokens, update_labels=update_labels, 
            lowercase=lowercase, normalize_digits=normalize_digits,
            subpos_depth=subpos_depth, predict_parent_existence=predict_parent_existence,
            use_subtoken=use_subtoken, create_chunk_trie=create_chunk_trie,
            freq_tokens=freq_tokens, dic=dic, refer_vocab=refer_vocab)

    elif data_format == 'sl_seg' or data_format == 'sl_seg_tag' or data_format == 'sl_tag':
        if not update_tokens:
            freq_tokens = set()

        # elif no_freq_update and dic:
        #     freq_tokens = set(dic.id2word) # skip count tokens

        elif freq_threshold > 1 or max_vocab_size > 0:
            freq_tokens = count_tokens_SL(
                path, freq_threshold, max_vocab_size=max_vocab_size, 
                segmentation=segmentation, lowercase=lowercase, normalize_digits=normalize_digits)
        else:
            freq_tokens = set()

        data, dic = load_data_SL(
            path, segmentation=segmentation, tagging=tagging, 
            update_tokens=update_tokens, update_labels=update_labels,
            lowercase=lowercase, normalize_digits=normalize_digits,
            subpos_depth=subpos_depth, use_subtoken=use_subtoken, create_chunk_trie=create_chunk_trie, 
            freq_tokens=freq_tokens, dic=dic, refer_vocab=refer_vocab)
        
    else:
        data = None
        print('Invalid data format: {}'.format(data_format))
        sys.exit()

    return data, dic


def load_pickled_data(filename_wo_ext, load_dic=True):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'rb') as f:
        obj = pickle.load(f)
        tokenseqs = obj[0]
        labels = obj[1]

    return Data(tokenseqs, labels)


def dump_pickled_data(filename_wo_ext, data):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'wb') as f:
        obj = (data.tokenseqs, data.labels)
        pickle.dump(obj, f)

