import re
import copy
import enum
import numpy as np


UNK_SYMBOL = '<UNK>'
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
                  update_label=True, refer_vocab=set(), limit=-1):

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
                        update_label=True, refer_vocab=set(), schema='BI', limit=-1):

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
              update_label=True, refer_vocab=set(), schema='BIES', limit=-1):
    if data_format == 'bccwj_ws' or data_format == 'cws':
        read_data = read_bccwj_data_for_wordseg if data_format == 'bccwj_ws' else read_cws_data

        instances, labels, token2id, label2id = read_data(
            path, token2id=token2id, label2id=label2id, update_token=update_token, 
            update_label=update_label, schema=schema, limit=limit)

    elif data_format == 'bccwj_pos' or data_format == 'wsj':
        read_data = read_bccwj_data_for_postag if data_format == 'bccwj_pos' else read_wsj_data

        instances, labels, token2id, label2id = read_data(
            path, token2id=token2id, label2id=label2id, update_token=update_token, 
            update_label=update_label, subpos_depth=subpos_depth, refer_vocab=refer_vocab, limit=limit)

    elif data_format == 'conll2003':
        read_data = read_conll2003_data

        instances, labels, token2id, label2id = read_data(
            path, token2id=token2id, label2id=label2id, update_token=update_token, update_label=update_label, 
            subpos_depth=subpos_depth, refer_vocab=refer_vocab, schema=schema, limit=limit)
    else:
        return

    return instances, labels, token2id, label2id


if __name__ == '__main__':
    pass
