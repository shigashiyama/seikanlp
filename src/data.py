import sys
import re
import pickle
import argparse
import copy

import lattice
import models


NUM_SYMBOL = '<NUM>'

DELIM_TXT = '#DELIMITER'
WORD_CLM_TXT = '#WORD_COLUMN'
LAB1_CLM_TXT = '#POS_COLUMN'


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
                 subpos_depth=-1, ws_dict_feat=False, indices=None, refer_vocab=set()):
    if not indices:
        if ws_dict_feat:
            indices = lattice.MorphologyDictionary()
        else:
            indices = lattice.IndicesPair()
            if not tagging:
                indices.init_label_indices('BIES')

    delim = '_'

    ins_cnt = 0
    word_cnt = 0
    token_cnt = 0

    instance_list = []
    slab_seq_list = []        # list of segmentation label sequences
    lab1_seq_list = []        # list of 1st label sequences (typically POS label)
    lab2_seq_list = []        # list of 1st label sequences (typically modifier index)
    lab3_seq_list = []        # list of 1st label sequences (typically arc label)

    with open(path) as f:

        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue

            elif line.startswith(DELIM_TXT):
                delim = line.split('=')[1]
                print('Read delimiter:', delim, file=sys.stderr)
                continue
            
            entries = re.sub(' +', ' ', line).split()
            ins = []
            slab_seq = []
            lab1_seq = []

            for entry in entries:
                attrs = entry.split(delim)
                word = attrs[0]
                
                if tagging:
                    lab1 = attrs[1]
                    if subpos_depth == 1:
                        lab1 = lab1.split('-')[0]
                    elif subpos_depth > 1:
                        lab1 = '_'.join(lab1.split('-')[0:subpos_depth])

                    if update_label and segmentation:
                        for seg_lab in 'BIES':
                            indices.label_indices.get_id('{}-{}'.format(seg_lab, lab1) , True)
                else:
                    lab1 = None    
                        
                wlen = len(word)
                update_this_token = update_token or word in refer_vocab

                if segmentation:
                    ins.extend([indices.token_indices.get_id(word[i], update_this_token) for i in range(wlen)])
                    slab_seq.extend(
                        [indices.label_indices.get_id(
                            get_label_BIES(i, wlen-1, cate=lab1), update=update_label) for i in range(wlen)])

                    if update_this_token and ws_dict_feat: # update indices
                        ids = chars[-wlen:]
                        indices.chunk_trie.get_word_id(ids, True)

                else:
                    ins.append(indices.token_indices.get_id(word, update=update_this_token))
                    lab1_seq.append(indices.label_indices.get_id(lab1, update=update_label))

            ins_cnt += 1

            instance_list.append(ins)
            if slab_seq:
                slab_seq_list.append(slab_seq)
            if lab1_seq:
                lab1_seq_list.append(lab1_seq)

            if ins_cnt % 100000 == 0:
                print('Read', ins_cnt, 'instances', file=sys.stderr)

    if segmentation:
        return instance_list, slab_seq_list, indices
    else:
        return instance_list, lab1_seq_list, indices


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
def load_data_WL(path, segmentation=True, tagging=False, parsing=False,
                  update_token=True, update_label=True, subpos_depth=-1, ws_dict_feat=False,
                  indices=None, refer_vocab=set()):
    if not indices:
        if ws_dict_feat:
            indices = lattice.MorphologyDictionary()
        else:
            indices = lattice.IndicesPair()

    delim = '\t'
    word_clm = 0
    lab1_clm = 1
    lab2_clm = 2
    lab3_clm = 3

    ins_cnt = 0
    word_cnt = 0
    token_cnt = 0

    instance_list = []
    slab_seq_list = []        # list of segmentation label sequences
    lab1_seq_list = []        # list of 1st label sequences (typically POS label)
    lab2_seq_list = []        # list of 1st label sequences (typically modifier index)
    lab3_seq_list = []        # list of 1st label sequences (typically arc label)

    with open(path) as f:
        ins = []
        slab_seq = []
        lab1_seq = []
        lab2_seq = []
        lab3_seq = []

        for line in f:
            line = line.strip()

            if len(line) < 1:
                if ins:
                    instance_list.append(ins)
                    ins = []
                    if slab_seq:
                        slab_seq_list.append(slab_seq)
                        slab_seq = []
                    if lab1_seq:
                        lab1_seq_list.append(lab1_seq)
                        lab1_seq = []
                    if lab2_seq:
                        lab2_seq_list.append(lab2_seq)
                        lab2_seq = []
                    if lab3_seq:
                        lab3_seq_list.append(lab3_seq)
                        lab3_seq = []

                    ins_cnt += 1
                    if ins_cnt % 100000 == 0:
                        print('Read', ins_cnt, 'instances', file=sys.stderr)
                continue

            elif line[0] == '#':
                if line.startswith(DELIM_TXT):
                    delim = line.split('=')[1]
                    print('Read delimiter:', delim, file=sys.stderr)

                elif line.startswith(WORD_CLM_TXT):
                    word_clm = int(line.split('=')[1]) - 1
                    print('Read word column id:', word_clm+1, file=sys.stderr)

                elif line.startswith(LAB1_CLM_TXT):
                    lab1_clm = int(line.split('=')[1]) - 1
                    print('Read 1st label column id:', lab1_clm+1, file=sys.stderr)

                continue

            array = line.split(delim)
            word = array[word_clm]
            if parsing:
                pass

            if tagging or parsing:
                lab1 = array[lab1_clm]
                if subpos_depth == 1:
                    lab1 = lab1.split('-')[0]
                elif subpos_depth > 1:
                    lab1 = '-'.join(lab1.split('-')[0:subpos_depth])

                if update_label and segmentation:
                    for seg_lab in 'BIES':
                        indices.label_indices.get_id('{}-{}'.format(seg_lab, lab1) , True)
            else:
                lab1 = None

            wlen = len(word)
            update_this_token = update_token or word in refer_vocab
            if segmentation:
                ins.extend([indices.token_indices.get_id(word[i], update_this_token) for i in range(wlen)])
                slab_seq.extend(
                    [indices.label_indices.get_id(
                        get_label_BIES(i, wlen-1, cate=lab1), update=update_label) for i in range(wlen)])

                if update_this_token and ws_dict_feat: # update indices
                    ids = ins[-wlen:]
                    indices.chunk_trie.get_word_id(ids, True)

            elif tagging:
                ins.append(indices.token_indices.get_id(word, update=update_this_token))
                lab1_seq.append(indices.label_indices.get_id(lab1, update=update_label))

            else:
                pass

            word_cnt += 1
            token_cnt += len(word)

        if ins:
            instance_list.append(ins)
            if slab_seq:
                slab_seq_list.append(slab_seq)
            if lab1_seq:
                lab1_seq_list.append(lab1_seq)
            if lab2_seq:
                lab2_seq_list.append(lab2_seq)
            if lab3_seq:
                lab3_seq_list.append(lab3_seq)

    if segmentation:
        return instance_list, slab_seq_list, indices
    elif tagging:
        return instance_list, lab1_seq_list, indices
    else:
        return None


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


def load_data(data_format, path, read_pos=True, update_token=True, update_label=True, ws_dict_feat=False,
              subpos_depth=-1, lowercase=False, normalize_digits=False, 
              indices=None, refer_vocab=set(), limit=-1):
    pos_seqs = []

    segmentation = True if 'seg' in data_format else False
    tagging = True if 'tag' in data_format else False

    if data_format == 'wl_seg' or data_format == 'wl_seg_tag' or data_format == 'wl_tag':
        instances, label_seqs, indices = load_data_WL(
            path, segmentation=segmentation, tagging=tagging,
            update_token=update_token, update_label=update_label, 
            subpos_depth=subpos_depth, ws_dict_feat=ws_dict_feat,
            indices=indices, refer_vocab=refer_vocab)

    elif data_format == 'sl_seg' or data_format == 'sl_seg_tag' or data_format == 'sl_tag':
        instances, label_seqs, indices = load_data_SL(
            path, segmentation=segmentation, tagging=tagging, 
            update_token=update_token, update_label=update_label,
            subpos_depth=subpos_depth, ws_dict_feat=ws_dict_feat, 
            indices=indices, refer_vocab=refer_vocab)
        
    # if data_format == 'bccwj_seg_lattice':
    #     instances, label_seqs, pos_seqs, indices = load_bccwj_data_for_lattice_ma(
    #         path, read_pos, update_token=update_token, subpos_depth=subpos_depth, 
    #         indices=indices, limit=limit)

    # elif data_format == 'wsj':
    #     instances, label_seqs, indices = load_wsj_data(
    #         path, update_token=update_token, update_label=update_label, 
    #         lowercase=lowercase, normalize_digits=normalize_digits, 
    #         indices=indices, refer_vocab=refer_vocab, limit=limit)

    # elif data_format == 'conll2003':
    #     instances, label_seqs, indices = load_conll2003_data(
    #         path, update_token=update_token, update_label=update_label, 
    #         lowercase=lowercase, normalize_digits=normalize_digits, 
    #         indices=indices, refer_vocab=refer_vocab, limit=limit)
    else:
        print('Error: invalid data format: {}'.format(data_format), file=sys.stderr)
        sys.exit()

    return instances, label_seqs, pos_seqs, indices


def load_pickled_data(filename_wo_ext, load_indices=True):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'rb') as f:
        obj = pickle.load(f)
        instances = obj[0]
        labels = obj[1]
        pos_labels = obj[2]

    return instances, labels, pos_labels


def dump_pickled_data(filename_wo_ext, instances, labels, pos_labels=None):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'wb') as f:
        obj = (instances, labels, pos_labels)
        pickle.dump(obj, f)

