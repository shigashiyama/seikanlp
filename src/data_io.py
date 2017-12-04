import sys
import re
import pickle
import argparse
import copy

import constants
import dictionary


class Data(object):
    def __init__(self, instances, labels):
        self.instances = instances
        self.labels = labels    # list of label sequences (e.g. seg, pos, dep and arc)
        self.features = None

        # self.seg_labels = seg_labels
        # self.pos_labels = pos_labels
        # self.dep_labels = dep_labels
        # self.arc_labels = arc_labesl

    def set_features(self, features):
        self.features = features


def load_raw_text_for_segmentation(path, indices):
    instances = []
    ins_cnt = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue

            ins = [indices.get_token_id(char) for char in line]
            instances.append(ins)

            ins_cnt += 1
            if ins_cnt % 100000 == 0:
                print('read', ins_cnt, 'instances', file=sys.stderr)

    return instances


def load_raw_text_for_tagging(path, indices):
    instances = []
    ins_cnt = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue

            array = line.split(' ')
            ins = [indices.get_token_id(word) for word in array]
            instances.append(ins)

            ins_cnt += 1
            if ins_cnt % 100000 == 0:
                print('read', ins_cnt, 'instances', file=sys.stderr)

    return instances


"""
Read data with SL (one sentence in one line) format.
If tagging == True, the following format is expected for joint segmentation and POS tagging:
  word1_pos1 word2_pos2 ... wordn_posn

otherwise, the following format is expected for segmentation:
  word1 word2 ... wordn
"""
def load_data_SL(path, segmentation=True, tagging=False,
                 update_token=True, update_label=True, 
                 lowercase=False, normalize_digits=True,
                 subpos_depth=-1, create_word_trie=False, indices=None, refer_vocab=set()):
    if not indices:
        indices = dictionary.Dictionary(
            use_seg_label=segmentation, use_pos_label=tagging, use_word_trie=create_word_trie)

    delim = constants.DELIM1_SYMBOL

    ins_cnt = 0
    word_cnt = 0
    token_cnt = 0

    instances = []
    seg_seqs = []        # list of segmentation label sequences
    pos_seqs = []         # list of POS label sequences

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
            ins = []
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

                    if segmentation and update_label:
                        for seg_lab in constants.SEG_LABELS:
                            indices.seg_label_indices.get_id('{}-{}'.format(seg_lab, pos) , True)
                else:
                    pos = None    
                        
                wlen = len(word)
                update_this_token = update_token or word in refer_vocab

                if segmentation:
                    ins.extend([indices.token_indices.get_id(word[i], update_this_token) for i in range(wlen)])
                    seg_seq.extend(
                        [indices.seg_label_indices.get_id(
                            get_label_BIES(i, wlen-1, cate=pos), update=update_label) for i in range(wlen)])

                    if update_this_token and create_word_trie: # update word trie
                        ids = chars[-wlen:]
                        indices.word_trie.get_word_id(ids, True)

                else:
                    ins.append(indices.token_indices.get_id(word, update=update_this_token))
                    pos_seq.append(indices.pos_label_indices.get_id(pos, update=update_label))

            ins_cnt += 1

            instances.append(ins)
            if seg_seq:
                seg_seqs.append(seg_seq)
            if pos_seq:
                pos_seqs.append(pos_seq)

            if ins_cnt % 100000 == 0:
                print('Read', ins_cnt, 'instances', file=sys.stderr)

    label_seqs_list = []
    if seg_seqs:
        label_seqs_list.append(seg_seqs)
    if pos_seqs:
        label_seqs_list.append(pos_seqs)

    return Data(instances, label_seqs_list), indices


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
                 update_token=True, update_label=True, 
                 lowercase=False, normalize_digits=True,
                 subpos_depth=-1, create_word_trie=False, indices=None, refer_vocab=set()):
    read_pos = tagging or parsing
    if not indices:
        indices = dictionary.Dictionary(
            use_seg_label=segmentation, use_pos_label=read_pos, 
            use_arc_label=typed_parsing, use_word_trie=create_word_trie, use_root=parsing)

    delim = constants.DELIM2_SYMBOL

    word_clm = 0
    pos_clm = 1
    dep_clm = 2
    arc_clm = 3

    ins_cnt = 0
    word_cnt = 0
    token_cnt = 0

    instances = []
    seg_seqs = []        # list of segmentation label sequences
    pos_seqs = []        # list of POS label sequences
    dep_seqs = []        # list of dependency label sequences
    arc_seqs = []        # list of arc label sequences

    with open(path) as f:
        ins = [indices.root_id] if parsing else []
        seg_seq = []
        pos_seq = [indices.root_id] if (parsing and read_pos) else []
        dep_seq = [-1] if parsing else []
        arc_seq = [-1] if parsing else []

        for line in f:
            line = re.sub(' +', ' ', line).strip('\n\t')

            if len(line) < 1:
                if len(ins) - (1 if parsing else 0) > 0:
                    instances.append(ins)
                    ins = [indices.root_id] if parsing else []
                    if seg_seq:
                        seg_seqs.append(seg_seq)
                        seg_seq = []
                    if pos_seq:
                        pos_seqs.append(pos_seq)
                        pos_seq = [indices.root_id] if (parsing and read_pos) else []
                    if dep_seq:
                        dep_seqs.append(dep_seq)
                        dep_seq = [-1] if parsing else []
                    if arc_seq:
                        arc_seqs.append(arc_seq)
                        arc_seq = [-1] if parsing else []

                    ins_cnt += 1
                    if ins_cnt % 100000 == 0:
                        print('Read', ins_cnt, 'instances', file=sys.stderr)
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

                elif line.startswith(constants.DEP_CLM_TXT):
                    dep_clm = int(line.split('=')[1]) - 1
                    print('Read 2nd label column id:', dep_clm+1, file=sys.stderr)

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

                if segmentation and update_label:
                    for seg_lab in constants.SEG_LABELS:
                        indices.seg_label_indices.get_id('{}-{}'.format(seg_lab, pos) , True)
            else:
                pos = None

            if parsing:
                dep = array[dep_clm]

            if typed_parsing:
                arc = array[arc_clm]

            wlen = len(word)
            update_this_token = update_token or word in refer_vocab
            if segmentation:
                ins.extend([indices.token_indices.get_id(word[i], update_this_token) for i in range(wlen)])
                seg_seq.extend(
                    [indices.seg_label_indices.get_id(
                        get_label_BIES(i, wlen-1, cate=pos), update=update_label) for i in range(wlen)])

                if update_this_token and create_word_trie: # update word trie
                    ids = ins[-wlen:]
                    indices.word_trie.get_word_id(ids, True)

            else:
                ins.append(indices.token_indices.get_id(word, update=update_this_token))

            if read_pos:
                pos_seq.append(indices.pos_label_indices.get_id(pos, update=update_label))

            if parsing:
                dep_seq.append(int(dep.replace('-1', '0')))

            if typed_parsing:
                arc_seq.append(indices.arc_label_indices.get_id(arc, update=update_label))
                

            word_cnt += 1
            token_cnt += len(word)

        if len(ins) - (1 if parsing else 0) > 0:
            instances.append(ins)
            if seg_seq:
                seg_seqs.append(seg_seq)
            if pos_seq:
                pos_seqs.append(pos_seq)
            if dep_seq:
                dep_seqs.append(dep_seq)
            if arc_seq:
                arc_seqs.append(arc_seq)

    label_seqs_list = []
    if seg_seqs:
        label_seqs_list.append(seg_seqs)
    if pos_seqs:
        label_seqs_list.append(pos_seqs)
    elif parsing:
        label_seqs_list.append(None)
    if dep_seqs:
        label_seqs_list.append(dep_seqs)
    if arc_seqs:
        label_seqs_list.append(arc_seqs)

    return Data(instances, label_seqs_list), indices


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


def load_data(data_format, path, read_pos=True, update_token=True, update_label=True, create_word_trie=False,
              subpos_depth=-1, lowercase=False, normalize_digits=False, 
              indices=None, refer_vocab=set(), limit=-1):
    pos_seqs = []

    segmentation = True if 'seg' in data_format else False
    tagging = True if 'tag' in data_format else False
    parsing = True if 'dep' in data_format else False
    typed_parsing = True if 'tdep' in data_format else False

    if (data_format == 'wl_seg' or data_format == 'wl_seg_tag' or data_format == 'wl_tag' or
        data_format == 'wl_dep' or data_format == 'wl_tdep' or
        data_format == 'wl_tag_dep' or data_format == 'wl_tag_tdep'
    ):
        data, indices = load_data_WL(
            path, segmentation=segmentation, tagging=tagging, parsing=parsing, typed_parsing=typed_parsing,
            update_token=update_token, update_label=update_label, 
            lowercase=lowercase, normalize_digits=normalize_digits,
            subpos_depth=subpos_depth, create_word_trie=create_word_trie,
            indices=indices, refer_vocab=refer_vocab)

    elif data_format == 'sl_seg' or data_format == 'sl_seg_tag' or data_format == 'sl_tag':
        data, indices = load_data_SL(
            path, segmentation=segmentation, tagging=tagging, 
            update_token=update_token, update_label=update_label,
            lowercase=lowercase, normalize_digits=normalize_digits,
            subpos_depth=subpos_depth, create_word_trie=create_word_trie, 
            indices=indices, refer_vocab=refer_vocab)
        
    else:
        data = None

    return data, indices


def load_pickled_data(filename_wo_ext, load_indices=True):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'rb') as f:
        obj = pickle.load(f)
        instances = obj[0]
        labels = obj[1]

    return Data(instances, labels)


def dump_pickled_data(filename_wo_ext, data, pos_labels=None):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'wb') as f:
        obj = (data.instances, data.labels)
        pickle.dump(obj, f)

