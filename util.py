import re
import pickle
import argparse
import copy
import enum
import numpy as np

import models
import read_embedding as emb

UNK_SYMBOL = '<UNK>'
NUM_SYMBOL = '<NUM>'
Schema = enum.Enum("Schema", "BI BIES")

# for PTB WSJ (CoNLL-2005)
#
# example of input data:
# 0001  1  0 B-NP    NNP   Pierre          NOFUNC          Vinken            1 B-S/B-NP/B-NP
# 0001  1  1 I-NP    NNP   Vinken          NP-SBJ          join              8 I-S/I-NP/I-NP
# 0001  1  2 O       COMMA COMMA           NOFUNC          Vinken            1 I-S/I-NP
# ...
# 0001  1 17 O       .     .               NOFUNC          join              8 I-S
#
def read_wsj_data(path, token2id={}, label2id={}, update_token=True, 
                  update_label=True, refer_vocab=set(), limit=-1,
                  lowercase=True, replace_digits=True):
    # id2token = {}
    # id2label = {}

    instances = []
    labels = []
    if len(token2id) == 0:
        token2id = {UNK_SYMBOL: np.int32(0)}
    if len(label2id) == 0:
        label2id = {UNK_SYMBOL: np.int32(0)} # '#' が val で出てくる

    print("Read file:", path)
    ins_cnt = 0
    word_cnt = 0

    with open(path) as f:
        ins = []
        lab = []

        for line in f:
            if line[0] == '#':
                continue

            if len(line) < 2:
                if len(ins) > 0:
                    # print([id2token[wi] for wi in ins])
                    # print([id2label[li] for li in lab])
                    # print()
                    
                    instances.append(ins)
                    labels.append(lab)
                    ins = []
                    lab = []
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

            if replace_digits:
                word = re.sub(r'[0-9]+', NUM_SYMBOL, word)

            update_this_token = update_token or word in refer_vocab
            wi = get_id(word, token2id, update_this_token)
            li = get_id(pos, label2id, update_label)
            ins.append(wi)
            lab.append(li)
            # print(wi, word)
            # print(li, pos)
            # id2token[wi] = word
            # id2label[li] = pos

            word_cnt += 1

        if limit <= 0:
            instances.append(ins)
            labels.append(lab)
            ins_cnt += 1

    return instances, labels, token2id, label2id


# for CoNLL-2003 NER
# 単語単位で BI or BIES フォーマットに変換
#
# example of input data:
# EU NNP I-NP I-ORG
# rejects VBZ I-VP O
# German JJ I-NP I-MISC
# call NN I-NP O
#
def read_conll2003_data(path, token2id={}, label2id={}, update_token=True, 
                        update_label=True, refer_vocab=set(), schema='BI', limit=-1,
                        lowercase=True, replace_digits=True):

    if schema == 'BI':
        sch = Schema.BI
    else:
        sch = Schema.BIES

    id2token = {}
    id2label = {}

    if len(token2id) == 0:
        token2id = {UNK_SYMBOL: np.int32(0)}
    if len(label2id) == 0:
        label2id = {}

    instances = []
    labels = []
    ins_cnt = 0
    word_cnt = 0

    print("Read file:", path)
    with open(path) as f:
        ins = []
        org_lab = []

        for line in f:
            if len(line) < 2:
                if len(ins) > 0:
                    # print([id2token[wi] for wi in ins])
                    # print([id2label[li] for li in lab])
                    # print()
                    
                    instances.append(ins)
                    labels.append(convert_label_sequence(org_lab, sch, label2id, update_label))
                    # print(ins)
                    # print(labels[-1])
                    # print()

                    ins = []
                    org_lab = []
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

            if replace_digits:
                word = re.sub(r'[0-9]+', NUM_SYMBOL, word)

            update_this_token = update_token or word in refer_vocab
            ins.append(get_id(word, token2id, update_this_token))
            org_lab.append(label)

            word_cnt += 1

        if limit <= 0:
            instances.append(ins)
            labels.append(convert_label_sequence(org_lab, sch, label2id, update_label))
            ins_cnt += 1

    return instances, labels, token2id, label2id

    
def convert_label_sequence(org_lab, sch, label2id, update_label):
    org_lab = org_lab.copy()
    org_lab.append('<END>') # dummy
    O_id = get_id('O', label2id, update_label)

    new_lab = []
    new_lab_debug = []
    stack = []

    for i in range(len(org_lab)):
        if len(stack) == 0:
            if org_lab[i] == '<END>':
                break
            elif org_lab[i] == 'O':
                new_lab.append(O_id)
                # new_lab_debug.append('O')
            else:
                stack.append(org_lab[i])

        else:
            if org_lab[i] == stack[-1]:
                stack.append(org_lab[i])
            else:
                cate = stack[-1].split('-')[1]
                chunk_len = len(stack)
                
                if sch == Schema.BI:
                    new_lab.extend(
                        [get_id(get_label_BI(j, cate=cate), 
                                label2id, update_label) for j in range(chunk_len)]
                    )
                    # new_lab_debug.extend(
                    #     [get_label_BI(j, cate=cate) for j in range(chunk_len)]
                    # )
                else:
                    new_lab.extend(
                        [get_id(get_label_BIES(j, chunk_len-1, cate=cate), 
                                label2id, update_label) for j in range(chunk_len)]
                    )
                    # new_lab_debug.extend(
                    #     [get_label_BIES(j, chunk_len-1, cate=cate) for j in range(chunk_len)]
                    # )
                    
                    stack = []
                    if org_lab[i] == '<END>':
                        pass
                    elif org_lab[i] == 'O':
                        new_lab.append(O_id)
                        # new_lab_debug.append('O')
                    else:
                        stack.append(org_lab[i])
                            
    # print(ins)
    # print(org_lab)
    # print(new_lab_debug)
    # print(new_lab)
    # print()

    return new_lab

# for BCCWJ pos tagging
#
# example of input data:
# B       最後    名詞-普通名詞-一般$
# I       の      助詞-格助詞$
# I       会話    名詞-普通名詞-サ変可能$
#
def read_bccwj_data_for_postag(path, token2id={}, label2id={}, cate_row=-1, subpos_depth=-1,
                               update_token=True, update_label=True, refer_vocab=set(), limit=-1):
    instances = []
    labels = []
    if len(token2id) == 0:
        token2id = {UNK_SYMBOL: np.int32(0)}
    if len(label2id) == 0:
        label2id = {}

    print("Read file:", path)
    ins_cnt = 0
    word_cnt = 0

    with open(path) as f:
        bof = True
        ins = []
        lab = []

        for line in f:
            if len(line) < 2:
                continue

            line = line.rstrip('\n').split('\t')
            bos = line[2]
            word = line[5]
            label = line[3]     # pos
            
            if subpos_depth == 1:
                label = label.split('-')[0]
            elif subpos_depth > 1:
                label = '_'.join(label.split('-')[0:subpos_depth])

            if not bof and bos == 'B':
                # if len(ins) > 1:

                instances.append(ins)
                labels.append(lab)
                ins = []
                lab = []
                ins_cnt += 1

                # else:
                #     print(ins)

            if limit > 0 and ins_cnt >= limit:
                break

            update_this_token = update_token or word in refer_vocab
            wi = get_id(word, token2id, update_this_token)
            li = get_id(label, label2id, update_label)
            ins.append(wi)
            lab.append(li)
            # print(wi, word)
            # print(li, label)

            bof = False
            word_cnt += 1

        if limit <= 0:
            instances.append(ins)
            labels.append(lab)
            ins_cnt += 1

    print(ins_cnt, word_cnt)
    return instances, labels, token2id, label2id


# for BCCWJ word segmentation
# 文字単位で BI or BIES フォーマットに変換
#
# example of input data:
# B       最後    名詞-普通名詞-一般$
# I       の      助詞-格助詞$
# I       会話    名詞-普通名詞-サ変可能$
#
def read_bccwj_data_for_wordseg(path, token2id={}, label2id={}, cate_row=-1, 
                                update_token=True, update_label=True, schema='BI', limit=-1):
    if schema == 'BI':
        sch = Schema.BI
    else:
        sch = Schema.BIES

    instances = []
    labels = []
    if len(token2id) == 0:
        token2id = {UNK_SYMBOL: np.int32(0)}
    if len(label2id) == 0:
        label2id = {}

    print("Read file:", path)
    ins_cnt = 0
    word_cnt = 0
    token_cnt = 0

    with open(path) as f:
        bof = True
        ins = []
        lab = []
        for line in f:
            if len(line) < 2:
                continue

            line = line.rstrip('\n').split('\t')
            bos = line[2]
            word = line[5]
            cate = None if cate_row < 1 else pair[cate_row]

            if not bof and bos == 'B':
                instances.append(ins)
                labels.append(lab)
                ins = []
                lab = []
                ins_cnt += 1

            if limit > 0 and ins_cnt >= limit:
                break

            wlen = len(word)
            ins.extend(
                [get_id(word[i], token2id, update_token) for i in range(wlen)]
                )

            if sch == Schema.BI:
                lab.extend(
                    [get_id(get_label_BI(i, cate=cate), label2id, update_label) for i in range(wlen)]
                )
            else:
                lab.extend(
                    [get_id(get_label_BIES(i, wlen-1, cate=cate), label2id, update_label) for i in range(wlen)]
                )

            bof = False
            word_cnt += 1
            token_cnt += len(word)

        if limit <= 0:
            instances.append(ins)
            labels.append(lab)
            ins_cnt += 1

    #print(ins_cnt, word_cnt, token_cnt)
    return instances, labels, token2id, label2id


# 文字単位で BI or BIES フォーマットに変換
#
# example of input data:
# 偶尔  有  老乡  拥  上来  想  看 ...
# 说  ，  这  “  不是  一种  风险  ，  而是  一种  保证  。
# ...
def read_cws_data(path, token2id={}, label2id={}, cate_row=-1, 
                  update_token=True, update_label=True, schema='BI', limit=-1):
    if schema == 'BI':
        sch = Schema.BI
    else:
        sch = Schema.BIES

    instances = []
    labels = []
    if len(token2id) == 0:
        token2id = {UNK_SYMBOL: np.int32(0)}
    if len(label2id) == 0:
        label2id = {}

    print("Read file:", path)
    ins_cnt = 0

    id2token = {}
    id2label = {}

    with open(path) as f:
        ins = []
        lab = []
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
                ins.extend(
                    [get_id(word[i], token2id, update_token) for i in range(wlen)]
                )
                # id2token.update(
                #     {get_id(word[i], token2id, update_token): word[i] for i in range(wlen)}
                # )

                if sch == Schema.BI:
                    lab.extend(
                        [get_id(get_label_BI(i), label2id, update_label) for i in range(wlen)]
                    )
                else:
                    lab.extend(
                        [get_id(get_label_BIES(i, wlen-1), 
                                label2id, update_label) for i in range(wlen)]
                    )
                    # id2label.update(
                    #     {get_id(get_label_BIES(i, wlen-1), label2id, update_label):get_label_BIES(i, wlen-1) for i in range(wlen)}
                    # )

            instances.append(ins)
            labels.append(lab)
            # print(ins)
            # print([id2token[id] for id in ins])
            # print([(id2label[id]+' ') for id in lab])
            # print()
            ins = []
            lab = []
            ins_cnt += 1

    return instances, labels, token2id, label2id


# example of input data:
# B       最後    名詞-普通名詞-一般$
# I       の      助詞-格助詞$
# I       会話    名詞-普通名詞-サ変可能$
#
def process_bccwj_data_for_kytea(input, output, subpos_depth=1):
    wf = open(output, 'w')

    instances = []
    labels = []

    print("Read file:", input)

    bof = True
    with open(input) as f:
        ins = []

        for line in f:
            if len(line) < 2:
                continue

            line = line.rstrip('\n').split('\t')
            bos = line[2]
            word = line[5]
            pos = line[3].rstrip('$')
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
    return prefix + suffix


def get_id(string, string2id, update=True):
    if string in string2id:
        return string2id[string]
    elif update:
        id = np.int32(len(string2id))
        string2id[string] = id
        return id
    else:
        if not UNK_SYMBOL in string2id:
            print("WARN: "+string+" is unknown and <UNK> is not registered.")
        return string2id[UNK_SYMBOL]


def read_data(data_format, path, token2id={}, label2id={}, subpos_depth=-1, update_token=True, 
              update_label=True, refer_vocab=set(), schema='BIES', limit=-1, lowercase=True, replace_digits=True):

    if data_format == 'bccwj_ws' or data_format == 'cws':
        read_data = read_bccwj_data_for_wordseg if data_format == 'bccwj_ws' else read_cws_data

        instances, labels, token2id, label2id = read_data(
            path, token2id=token2id, label2id=label2id, update_token=update_token, 
            update_label=update_label, schema=schema, limit=limit)

    elif data_format == 'bccwj_pos':
        read_data = read_bccwj_data_for_postag
        
        instances, labels, token2id, label2id = read_data(
            path, token2id=token2id, label2id=label2id, update_token=update_token, 
            update_label=update_label, subpos_depth=subpos_depth, refer_vocab=refer_vocab, limit=limit)

    elif data_format == 'wsj':
        read_data = read_wsj_data

        instances, labels, token2id, label2id = read_data(
            path, token2id=token2id, label2id=label2id, update_token=update_token, update_label=update_label, 
            refer_vocab=refer_vocab, limit=limit,
            lowercase=lowercase, replace_digits=replace_digits)

    elif data_format == 'conll2003':
        read_data = read_conll2003_data

        instances, labels, token2id, label2id = read_data(
            path, token2id=token2id, label2id=label2id, update_token=update_token, update_label=update_label, 
            refer_vocab=refer_vocab, schema=schema, limit=limit, 
            lowercase=lowercase, replace_digits=replace_digits)
    else:
        return

    return instances, labels, token2id, label2id


def read_dictionary(path):
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


def write_dictionary(dic, path):
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


def load_model_from_params(params, model_path='', token2id=None, label2id=None, embed_model=None, gpu=-1):
    if not 'embed_dim' in params:
        params['embed_dim'] = 300
    else:
        params['embed_dim'] = int(params['embed_dim'])

    if not 'rnn_layer' in params:
        params['rnn_layer'] = 1
    else:
        params['rnn_layer'] = int(params['rnn_layer'])

    if not 'rnn_hiddlen_unit' in params:
        params['rnn_hidden_unit'] = 500
    else:
        params['rnn_hidden_unit'] = int(params['rnn_hidden_unit'])

    if not 'rnn_unit_type' in params:
        params['rnn_unit_type'] = 'lstm'

    if not 'rnn_bidirection' in params:
        params['rnn_bidirection'] = False
    else:
        params['rnn_bidirection'] = str(params['rnn_bidirection']).lower() == 'true'

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
        
    if not 'tag_schema' in params:
        params['tag_schema'] = 'BIES'
    
    if not embed_model and not 'embed_path' in params:
        id2token = {v:k for k,v in token2id.items()}
        embed_model = emb.read_model(params['embed_path'])
        embed = emb.construct_lookup_table(id2token, embed_model, gpu=gpu)
        embed_dim = embed.W.shape[1]
    else:
        embed = None

    if not token2id:
        token2id = read_dictionary(token2id_path)
    if not label2id:
        label2id = read_dictionary(label2id_path)
    id2label = {v:k for k,v in label2id.items()}

    if params['crf']:
        rnn = models.RNN_CRF(
            params['rnn_layer'], len(token2id), params['embed_dim'], params['rnn_hidden_unit'], 
            len(label2id), dropout=params['dropout'], rnn_unit_type=params['rnn_unit_type'], 
            rnn_bidirection=params['rnn_bidirection'], linear_activation=params['linear_activation'], 
            n_left_contexts=params['left_contexts'], n_right_contexts=params['right_contexts'], 
            init_embed=embed, gpu=gpu)
    else:
        rnn = models.RNN(
            params['rnn_layer'], len(token2id), params['embed_dim'], params['rnn_hidden_unit'], 
            len(label2id), dropout=params['dropout'], rnn_unit_type=params['rnn_unit_type'], 
            rnn_bidirection=params['rnn_bidirection'], linear_activation=params['linear_activation'], 
            # n_left_contexts=params['left_contexts'], n_right_contexts=params['right_contexts'], 
            init_embed=embed, gpu=gpu)

    model = models.SequenceTagger(rnn, id2label)
    if model_path:
        chainer.serializers.load_npz(model_path, model)
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
        f.write('tag_schema %s\n' % params['tag_schema'])
    

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

    instances, labels, token2id, label2id = read_data(
        args.input_format, args.input_path, subpos_depth=args.subpos_depth, schema=args.tag_schema, limit=args.limit)

    t2i_tmp = list(token2id.items())
    id2token = {v:k for k,v in token2id.items()}
    id2label = {v:k for k,v in label2id.items()}

    print('vocab =', len(token2id))
    print('data length:', len(instances))
    print()
    print('instances:', instances[:3], '...', instances[len(instances)-3:])
    print('labels:', labels[:3], '...', labels[len(labels)-3:])
    print()
    print('token2id:', t2i_tmp[:3], '...', t2i_tmp[len(t2i_tmp)-3:])
    print('label2id:', label2id)

    if args.output_filename:
        write_dictionary(token2id, args.output_filename + '_t2i.' + args.output_format)
        write_dictionary(label2id, args.output_filename + '_l2i.' + args.output_format)
