import sys
import re
import pickle
import argparse
import copy
import enum
import numpy as np

import lattice.lattice as lattice
import models
import read_embedding as emb

import chainer


NUM_SYMBOL = '<NUM>'
Schema = enum.Enum("Schema", "BI BIES")


"""for BCCWJ word segmentation or POS tagging

example of input data:
  B       最後    名詞-普通名詞-一般
  I       の      助詞-格助詞$
  I       会話    名詞-普通名詞-サ変可能
"""
def load_bccwj_data_for_wordseg(path, update_token=True, update_label=True, subpos_depth=-1, 
                                dic=None, refer_vocab=set(), do_segmentation=True, limit=-1):
    if not dic:
        dic = lattice.IndicesPair()
        if subpos_depth == 0:
            dic.init_label_indices(schema)

    token_ind = dic.token_indices
    label_ind = dic.label_indices
        
    instances = []
    lab_seqs = []

    ins_cnt = 0
    word_cnt = 0
    token_cnt = 0

    print("Read file:", path)
    with open(path) as f:
        bof = True
        ins = []
        labs = []
        for line in f:
            if len(line) < 2:
                continue

            line = line.rstrip('\n').split('\t')
            bos = line[2]
            word = line[5]
            pos = line[3]
            if do_segmentation and subpos_depth == 0:
                pos = None
            elif subpos_depth == 1:
                pos = pos.split('-')[0]
            elif subpos_depth > 1:
                pos = '_'.join(pos.split('-')[0:subpos_depth])

            if not bof and bos == 'B':
                instances.append(ins)
                lab_seqs.append(labs)
                ins = []
                labs = []
                ins_cnt += 1
                if ins_cnt % 100000 == 0:
                    print('read', ins_cnt, 'instances')

            if limit > 0 and ins_cnt >= limit:
                break


            if do_segmentation:
                wlen = len(word)
                ins.extend([token_ind.get_id(word[i], update_token) for i in range(wlen)])

                labs.extend(
                    [label_ind.get_id(
                        get_label_BIES(i, wlen-1, cate=pos), update=update_label) for i in range(wlen)])
            else:
                update_this_token = update_token or word in refer_vocab
                ins.append(token_ind.get_id(word, update=update_this_token))
                labs.append(label_ind.get_id(pos, update=update_label))

            bof = False
            word_cnt += 1
            token_cnt += len(word)

        if limit <= 0:
            instances.append(ins)
            lab_seqs.append(labs)
            ins_cnt += 1

    return instances, lab_seqs, dic



def load_bccwj_data_for_lattice_ma(
        path, read_pos=True, update_token=True, subpos_depth=-1, dic=None, limit=-1):
    if not dic:
        dic = lattice.MorphologicalDictitionary()
        
    instances = []
    slab_seqs = []              # list of segmentation labels (B, I, E, S)
    plab_seqs = []              # list of pos labels

    ins_cnt = 0

    print("Read file:", path)
    with open(path) as f:
        bof = True
        ins = []
        slabs = []
        plabs = []
        for line in f:
            if len(line) < 2:
                continue

            line = line.rstrip('\n').split('\t')
            bos = line[2]
            word = line[5]
            if read_pos:
                pos = line[3]
                if subpos_depth == 1:
                    pos = pos.split('-')[0]
                elif subpos_depth > 1:
                    pos = '_'.join(pos.split('-')[0:subpos_depth])
            else:
                pos = lattice.DUMMY_POS

            if not bof and bos == 'B':
                instances.append(ins)
                slab_seqs.append(slabs)
                if subpos_depth != 0:
                    plab_seqs.append(plabs)
                ins = []
                slabs = []
                plabs = []
                ins_cnt += 1

            if limit > 0 and ins_cnt >= limit:
                break

            wlen = len(word)
            first_char_index = len(ins)
            char_ids, word_id, pos_id = dic.get_entries(word, pos, update=update_token)
            ins.extend(char_ids)
            slabs.extend([dic.get_label_id(get_label_BIES(i, wlen-1)) for i in range(wlen)])
            #if not read_pos:
            plabs.append((first_char_index, first_char_index + wlen, pos_id))

            bof = False

        if limit <= 0:
            instances.append(ins)
            slab_seqs.append(slabs)
            #if not read_pos:
            plab_seqs.append(plabs)
            ins_cnt += 1

    if update_token:
        dic.create_id2token()
        dic.create_id2pos()

    return instances, slab_seqs, plab_seqs, dic


""" for Chinese word segmentation

example of input data:
  偶尔  有  老乡  拥  上来  想  看 ...
  说  ，  这  “  不是  一种  风险  ，  而是  一种  保证  。
...
"""
def load_cws_data(path, update_token=True, update_label=True, dic=None, refer_vocab=set(), limit=-1):

    if not dic:
        dic = lattice.IndicesPair()
        dic.init_label_indices('BIES')

    token_ind = dic.token_indices
    label_ind = dic.label_indices
        
    instances = []
    lab_seqs = []

    ins_cnt = 0

    print("Read file:", path)
    with open(path) as f:
        ins = []
        labs = []
        for line in f:
            if limit > 0 and ins_cnt >= limit:
                break

            line = line.rstrip('\n').strip()
            if len(line) < 1:
                continue

            words = line.replace('  ', ' ').split(' ')
            # print('sen:', words)
            for word in words:
                wlen = len(word)
                update_this_token = update_token or word in refer_vocab
                ins.extend([token_ind.get_id(word[i], update_this_token) for i in range(wlen)])
                labs.extend(
                    [label_ind.get_id(get_label_BIES(i, wlen-1), update_label) for i in range(wlen)])

            instances.append(ins)
            lab_seqs.append(labs)

            # print(ins)
            # print([id2token[id] for id in ins])
            # print([(id2label[id]+' ') for id in lab])
            # print()

            ins = []
            labs = []
            ins_cnt += 1

    return instances, lab_seqs, dic


""" for PTB WSJ (CoNLL-2005)

example of input data:
  0001  1  0 B-NP    NNP   Pierre          NOFUNC          Vinken            1 B-S/B-NP/B-NP
  0001  1  1 I-NP    NNP   Vinken          NP-SBJ          join              8 I-S/I-NP/I-NP
  0001  1  2 O       COMMA COMMA           NOFUNC          Vinken            1 I-S/I-NP
  ...
  0001  1 17 O       .     .               NOFUNC          join              8 I-S
"""
def load_wsj_data(path, update_token=True, update_label=True, lowercase=True, normalize_digits=True,
                  dic=None, refer_vocab=set(), limit=-1):

    if not dic:
        label_ind = lattice.TokenIndices(unk_symbol=lattice.DUMMY_POS) # '#' が val で出てくる
        dic = lattice.IndicesPair(label_indices=label_ind)

    token_ind = dic.token_indices
    label_ind = dic.label_indices

    instances = []
    label_seqs = []

    ins_cnt = 0

    print("Read file:", path)
    with open(path) as f:
        ins = []
        labs = []

        for line in f:
            if line[0] == '#':
                continue

            if len(line) < 2:
                if len(ins) > 0:
                    # print([id2token[wi] for wi in ins])
                    # print([id2label[li] for li in lab])
                    # print()
                    
                    instances.append(ins)
                    label_seqs.append(labs)
                    ins = []
                    labs = []
                    ins_cnt += 1
                continue

            if limit > 0 and ins_cnt >= limit:
                break

            line = re.split(' +', line.rstrip('\n'))
            word = line[6].rstrip()
            pos = line[5].rstrip()
            chunk = line[4].rstrip()

            if lowercase:
                word = word.lower()

            if normalize_digits and not word in refer_vocab:
                word = re.sub(r'[0-9]+', NUM_SYMBOL, word)

            update_this_token = update_token or word in refer_vocab
            ins.append(token_ind.get_id(word, update_this_token))
            labs.append(label_ind.get_id(pos, update_label))

        if limit <= 0:
            instances.append(ins)
            label_seqs.append(labs)
            ins_cnt += 1

    return instances, label_seqs, dic


"""for CoNLL-2003 NER
単語単位で BIES フォーマットに変換

example of input data:
  EU NNP I-NP I-ORG
  rejects VBZ I-VP O
  German JJ I-NP I-MISC
  call NN I-NP O
"""
def load_conll2003_data(path, update_token=True, update_label=True, lowercase=True, normalize_digits=True,
                        dic=None, refer_vocab=set(), limit=-1):

    if not dic:
        dic = lattice.IndicesPair()
    token_ind = dic.token_indices
    label_ind = dic.label_indices

    instances = []
    label_seqs = []
    
    ins_cnt = 0

    print("Read file:", path)
    with open(path) as f:
        ins = []
        org_labs = []

        for line in f:
            if len(line) < 2:
                if len(ins) > 0:
                    # print([id2token[wi] for wi in ins])
                    # print([id2label[li] for li in lab])
                    # print()
                    
                    instances.append(ins)
                    label_seqs.append(convert_label_sequence(org_labs, label_ind, update_label))
                    # print(ins)
                    # print(label_seqs[-1])
                    # print()

                    ins = []
                    org_labs = []
                    ins_cnt += 1
                continue

            if limit > 0 and ins_cnt >= limit:
                break

            line = line.rstrip('\n').split(' ')
            word = line[0]
            pos = line[1]
            chunk = line[2]
            label = line[3]

            if word == '-DOCSTART-':
                continue

            if lowercase:
                word = word.lower()

            if normalize_digits:
            # if normalize_digits and not word in refer_vocab:
                word = re.sub(r'[0-9]+', NUM_SYMBOL, word)

            update_this_token = update_token or word in refer_vocab
            ins.append(token_ind.get_id(word, update_this_token))
            org_labs.append(label)

        if limit <= 0:
            instances.append(ins)
            label_seqs.append(convert_label_sequence(org_labs, label_ind, update_label))
            ins_cnt += 1

    return instances, label_seqs, dic

    
def convert_label_sequence(org_labs, label_ind, update_label):
    org_labs = org_labs.copy()
    org_labs.append('<END>') # dummy
    O_id = label_ind.get_id('O', update_label)

    new_labs = []
    new_labs_debug = []
    stack = []

    for i in range(len(org_labs)):
        if len(stack) == 0:
            if org_labs[i] == '<END>':
                break
            elif org_labs[i] == 'O':
                new_labs.append(O_id)
                # new_labs_debug.append('O')
            else:
                stack.append(org_labs[i])

        else:
            if org_labs[i] == stack[-1]:
                stack.append(org_labs[i])
            else:
                cate = stack[-1].split('-')[1]
                chunk_len = len(stack)
                
                new_labs.extend(
                    [label_ind.get_id(
                        get_label_BIES(j, chunk_len-1, cate=cate), update_label) for j in range(chunk_len)])
                # new_labs_debug.extend(
                #     [get_label_BIES(j, chunk_len-1, cate=cate) for j in range(chunk_len)])
                    
                stack = []
                if org_labs[i] == '<END>':
                    pass
                elif org_labs[i] == 'O':
                    new_labs.append(O_id)
                    # new_labs_debug.append('O')
                else:
                    stack.append(org_labs[i])
                            
    # print(ins)
    # print(org_labs)
    # print(new_labs_debug)
    # print(new_labs)
    # print()

    return new_labs


# example of input data:
# B       最後    名詞-普通名詞-一般$
# I       の      助詞-格助詞$
# I       会話    名詞-普通名詞-サ変可能$
#
def process_bccwj_data_for_kytea(input, output, subpos_depth=1):
    wf = open(output, 'w')

    instances = []
    labels = []

    bof = True

    print("Read file:", input)
    with open(input) as f:
        ins = []

        for line in f:
            if len(line) < 2:
                continue

            line = line.rstrip('\n').split('\t')
            bos = line[2]
            word = line[5]
            pos = line[3]
            if subpos_depth == 1:
                pos = pos.split('-')[0]
            elif subpos_depth > 1:
                pos = '_'.join(pos.split('-')[0:subpos_depth])

            if not bof and bos == 'B':
              write_for_kytea(wf, ins)
              ins = []
            ins.append('%s/%s' % (word, pos))
                
            bof = False

    wf.close()
    print('Output file:', output)

    return


def write_for_kytea(f, ins):
    if len(ins) == 0:
        return
    
    f.write('%s' % ins[0])
    for i in range(1, len(ins)):
        #print(' %s' % ins[i])
        f.write(' %s' % ins[i])
    f.write('\n')


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


def load_data(data_format, path, read_pos=True, update_token=True, update_label=True, subpos_depth=-1,
               lowercase=False, normalize_digits=False, dic=None, refer_vocab=set(), limit=-1):
    pos_seqs = []

    if data_format == 'bccwj_ws_lattice':
        instances, label_seqs, pos_seqs, dic = load_bccwj_data_for_lattice_ma(
            path, read_pos, update_token=update_token, subpos_depth=subpos_depth, 
            dic=dic, limit=limit)

    elif data_format == 'bccwj_ws' or data_format == 'bccwj_pos':
        do_segmentation = True if data_format == 'bccwj_ws' else False

        instances, label_seqs, dic = load_bccwj_data_for_wordseg(
            path, update_token=update_token, update_label=update_label, subpos_depth=subpos_depth, 
            dic=dic, refer_vocab=refer_vocab, do_segmentation=do_segmentation, limit=limit)

    elif data_format == 'cws':
        instances, label_seqs, dic = load_cws_data(
            path, update_token=update_token, update_label=update_label,
            dic=dic, refer_vocab=refer_vocab, limit=limit)

    elif data_format == 'wsj':
        instances, label_seqs, dic = load_wsj_data(
            path, update_token=update_token, update_label=update_label, 
            lowercase=lowercase, normalize_digits=normalize_digits, 
            dic=dic, refer_vocab=refer_vocab, limit=limit)

    elif data_format == 'conll2003':
        instances, label_seqs, dic = load_conll2003_data(
            path, update_token=update_token, update_label=update_label, 
            lowercase=lowercase, normalize_digits=normalize_digits, 
            dic=dic, refer_vocab=refer_vocab, limit=limit)
    else:
        print("invalid data format")
        sys.exit()

    return instances, label_seqs, pos_seqs, dic


def load_pickled_data(filename_wo_ext, load_token2id=True, load_label2id=True, load_params=True):
    ins_dump_path = filename_wo_ext + '.pickle'
    lab_dump_path = filename_wo_ext + '.pickle'        
    token2id_dump_path = filename_wo_ext + '.t2i.pickle'
    label2id_dump_path = filename_wo_ext + '.l2i.pickle'
    params_dump_path = filename_wo_ext + '.param.pickle'

    with open(ins_dump_path, 'rb') as f:
        instances = pickle.load(f)
        print('load pickled data:', ins_dump_path)

    with open(lab_dump_path, 'rb') as f:
        labels = pickle.load(f)

    if load_token2id:
        with open(token2id_dump_path, 'rb') as f:
            token2id = pickle.load(f)
    else:
        token2id = None

    if load_label2id:
        with open(label2id_dump_path, 'rb') as f:
            label2id = pickle.load(f)
    else:
        label2id = None

    if load_params:
        with open(params_dump_path, 'rb') as f:
            params = pickle.load(f)
    else:
        params = None

    return instances, labels, token2id, label2id, params


def dump_pickled_data(filename_wo_ext, instances, labels, dic=None, params=None):
    ins_dump_path = filename_wo_ext + '.pickle'
    lab_dump_path = filename_wo_ext + '.pickle'        
    dic_dump_path = filename_wo_ext + '.dic.pickle'
    params_dump_path = filename_wo_ext + '.param.pickle'

    with open(ins_dump_path, 'wb') as f:
        pickle.dump(instances, f)
        print('dump pickled data:', ins_dump_path)

    with open(lab_dump_path, 'wb') as f:
        pickle.dump(labels, f)

    if dic:
        with open(dic_dump_path, 'wb') as f:
            pickle.dump(dic, f)

    if params:
        with open(params_dump_path, 'wb') as f:
            pickle.dump(params, f)


def read_map(path):
    if path.endswith('bin'):
        with open(path, 'rb') as f:
            dic = pickle.load(f)
    else:
        dic = {}
        with open(path, 'r') as f:
            for line in f:
                arr = line.strip().split('\t')
                if len(arr) < 2:
                    continue

                dic.update({arr[0]:arr[1]})

    return dic


def write_map(dic, path):
    if path.endswith('bin'):
        with open(path, 'wb') as f:
            pickle.dump(dic, f)
    else:
        with open(path, 'w') as f:
            for token, index in dic.items():
                f.write('%s\t%d\n' % (token, index))


def read_param_file(path):
    params = {}

    with open(path, 'r') as f:
        for line in f:
            arr = line.strip().split(' ')
            if len(arr) < 2:
                continue
            
            params.update({arr[0]:arr[1]})

    return params


def load_model_from_params(params, model_path='', dic=None, token_indices_updated=None, 
                           embed_model=None, gpu=-1):
    if not 'joint' in params:
        params['joint'] = False
    else:
        params['joint'] = str(params['joint']).lower() == 'true'

    if not 'embed_dim' in params:
        params['embed_dim'] = 300
    else:
        params['embed_dim'] = int(params['embed_dim'])

    if not 'rnn_layer' in params:
        params['rnn_layer'] = 1
    else:
        params['rnn_layer'] = int(params['rnn_layer'])

    if not 'rnn_hidden_unit' in params:
        params['rnn_hidden_unit'] = 500
    else:
        params['rnn_hidden_unit'] = int(params['rnn_hidden_unit'])

    if not 'rnn_unit_type' in params:
        params['rnn_unit_type'] = 'lstm'

    if not 'rnn_bidirection' in params:
        params['rnn_bidirection'] = False
    else:
        params['rnn_bidirection'] = str(params['rnn_bidirection']).lower() == 'true'

    if not 'lattice' in params:
        params['lattice'] = False
    else:
        params['lattice'] = str(params['lattice']).lower() == 'true'

    if not 'crf' in params:
        params['crf'] = False
    else:
        params['crf'] = str(params['crf']).lower() == 'true'

    if not 'linear_activation' in params:
        params['linear_activation'] = 'identity'

    if not 'left_contexts' in params:
        params['left_contexts'] = 0
    else:
        params['left_contexts'] = int(params['left_contexts'])

    if not 'right_contexts' in params:
        params['right_contexts'] = 0
    else:
        params['right_contexts'] = int(params['right_contexts'])

    if not 'dropout' in params:
        params['dropout'] = 0
    else:
        params['dropout'] = int(params['dropout'])
    
    if not embed_model and 'embed_path' in params:
        #TODO token2id_update != None の場合の pre-trained embedding の拡張は未対応
        dic.create_id2token()
        embed_model = emb.read_model(params['embed_path'])
        embed = emb.construct_lookup_table(dic.id2token, embed_model, gpu=gpu)
        embed_dim = embed.W.shape[1]
    else:
        embed = None

    if not dic:
        token2id = read_map(token2id_path)
        label2id = read_map(label2id_path)
        dic = lattice.IndicesPair(
            lattice.TokenIndices(token2id), 
            lattice.TokenIndices(label2id))
    dic.create_id2label()

    if params['lattice']:
        rnn = models.RNN_LatticeCRF(
            params['rnn_layer'], len(dic.token_indices), params['embed_dim'], params['rnn_hidden_unit'], 
            len(dic.label_indices), dic, dropout=params['dropout'], rnn_unit_type=params['rnn_unit_type'], 
            rnn_bidirection=params['rnn_bidirection'], linear_activation=params['linear_activation'], 
            n_left_contexts=params['left_contexts'], n_right_contexts=params['right_contexts'], 
            init_embed=embed, gpu=gpu)

    elif params['crf']:
        rnn = models.RNN_CRF(
            params['rnn_layer'], len(dic.token_indices), params['embed_dim'], params['rnn_hidden_unit'], 
            len(dic.label_indices), dropout=params['dropout'], rnn_unit_type=params['rnn_unit_type'], 
            rnn_bidirection=params['rnn_bidirection'], linear_activation=params['linear_activation'], 
            n_left_contexts=params['left_contexts'], n_right_contexts=params['right_contexts'], 
            init_embed=embed, gpu=gpu)

    else:
        rnn = models.RNN(
            params['rnn_layer'], len(token_indices), params['embed_dim'], params['rnn_hidden_unit'], 
            len(label_indices), dropout=params['dropout'], rnn_unit_type=params['rnn_unit_type'], 
            rnn_bidirection=params['rnn_bidirection'], linear_activation=params['linear_activation'], 
            # n_left_contexts=params['left_contexts'], n_right_contexts=params['right_contexts'], 
            init_embed=embed, gpu=gpu)

    if params['lattice'] or params['joint']:
        model = models.JointMorphologicalAnalyzer(rnn, dic.id2label)
        #TODO Joint_RNN の実装
    else:
        model = models.SequenceTagger(rnn, dic.id2label)

    if model_path:
        chainer.serializers.load_npz(model_path, model)

    if token_indices_updated:
        model.grow_lookup_table(token_indices_updated, gpu=gpu)

    if gpu >= 0:
        model.to_gpu()

    return model

        
def write_param_file(params, path):
    with open(path, 'w') as f:
        f.write('# %s\n\n' % path)
        f.write('# paths\n')
        f.write('token2id_path %s\n' % params['token2id_path'])
        f.write('label2id_path %s\n' % params['label2id_path'])
        if 'embed_path' in params:
            f.write('embed_path %s\n' % params['embed_path'])

        f.write('# model parameters\n')
        f.write('embed_dim %d\n' % params['embed_dim'])
        f.write('rnn_layer %d\n' % params['rnn_layer'])
        f.write('rnn_hiddlen_unit %d\n' %  params['rnn_hidden_unit'])
        f.write('rnn_unit_type %s'  % params['rnn_unit_type'])
        f.write('rnn_bidirection %s\n' % str(params['rnn_bidirection']))
        f.write('crf %s\n' % str(params['crf']))
        f.write('linear_activation %s\n' % params['linear_activation'])
        f.write('left_contexts %d\n' % params['left_contexts'])
        f.write('right_contexts %d\n' % params['right_contexts'])
        f.write('dropout %d\n' % params['dropout'])

        f.write('# others\n')
        f.write('model_date %s\n' % params['model_date'])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', default='')
    parser.add_argument('--input_format', '-f', default='')
    parser.add_argument('--subpos_depth', '-s', type=int, default=-1)
    parser.add_argument('--tag_schema', '-t', default='BIES')
    parser.add_argument('--limit', '-l', type=int, default=-1)
    parser.add_argument('--output_filename', '-o', default='')
    parser.add_argument('--output_format', '-x', default='bin')
    args = parser.parse_args()

    
    dic_path = 'unidic/lex4kytea_zen.txt'
    dic = lattice.load_dictionary(dic_path, read_pos=args.joint)
    dic_pic_path = 'unidic/lex4kytea_zen_li.pickle'
    with open(dic_pic_path, 'wb') as f:
        pickle.dump(dic, f)
    # dic_pic_path = 'unidic/lex4kytea_zen_li.pickle'
    # with open(dic_pic_path, 'rb') as f:
    #     dic = pickle.load(f)

    instances, slab_seqs, plab_seqs, _ = read_bccwj_data_for_joint_ma(args.input_path, subpos_depth=args.subpos_depth, dic=dic, limit=-1)


    t2i_tmp = list(dic.token_indices.token2id.items())
    id2slabel = {v:k for k,v in dic.label_indices.token2id.items()}
    id2pos = {v:k for k,v in dic.pos_indices.token2id.items()}

    print('vocab =', len(dic.token_indices))
    print('data length:', len(instances))
    print()
    print('instances:', instances[:3], '...', instances[len(instances)-3:])
    print('seg labels:', slab_seqs[:3], '...', slab_seqs[len(slab_seqs)-3:])
    print('pos labels:', plab_seqs[:3], '...', plab_seqs[len(plab_seqs)-3:])
    print()
    print('token2id:', t2i_tmp[:10], '...', t2i_tmp[len(t2i_tmp)-10:])
    print('labels:', id2slabel)
    print('poss:', id2pos)

    # instances, labels, token2id, label2id = read_data(
    #     args.input_format, args.input_path, subpos_depth=args.subpos_depth, schema=args.tag_schema, limit=args.limit)
    #
    # if args.output_filename:
    #     write_map(token2id, args.output_filename + '.t2i.' + args.output_format)
    #     write_map(label2id, args.output_filename + '.l2i.' + args.output_format)
