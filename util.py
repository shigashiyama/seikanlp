import numpy as np


UNK_SYMBOL = '<UNK>'

# for CoNLL-2003 NER
# 単語単位で BIO2 フォーマットに変換 (未実施：先頭の I -> B へ置き換え)
#
# example of input data:
# EU NNP I-NP I-ORG
# rejects VBZ I-VP O
# German JJ I-NP I-MISC
# call NN I-NP O
#
def create_data(path, token2id={}, label2id={}, 
                token_update=True, label_update=True, limit=-1):

    # id2token = {}
    # id2label = {}

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
        ins = []
        lab = []

        for line in f:
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

            line = line.rstrip('\n').split(' ')
            word = line[0]
            pos = line[1]
            chunk = line[2]
            label = line[3]
            if word == '-DOCSTART-':
                continue

            wi = get_id(word, token2id, token_update)
            li = get_id(label, label2id, label_update)
            ins.append(wi)
            lab.append(li)
            # print(wi, word)
            # print(li, label)
            # id2token[wi] = word
            # id2label[li] = label

            word_cnt += 1

        if limit <= 0:
            instances.append(ins)
            labels.append(lab)
            ins_cnt += 1

    return instances, labels, token2id, label2id

    
# for BCCWJ word segmentation
# 文字単位で BIO2 フォーマットに変換
#
# example of input data:
# B       最後    名詞-普通名詞-一般$
# I       の      助詞-格助詞$
# I       会話    名詞-普通名詞-サ変可能$
#
def create_data_per_char(path, token2id={}, label2id={}, cate_row=-1, 
                         token_update=True, label_update=True, limit=-1):
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

            pair = line.rstrip('\n').split('\t')
            bos = pair[0]
            word = pair[1]
            cate = None if cate_row < 1 else pair[cate_row]

            if not bof and bos == 'B':
                instances.append(ins)
                labels.append(lab)
                ins = []
                lab = []
                ins_cnt += 1

            if limit > 0 and ins_cnt >= limit:
                break

            ins.extend(
                [get_id(word[i], token2id, token_update) for i in range(len(word))]
                )
            lab.extend(
                [get_id(get_label(i, cate), label2id, label_update) for i in range(len(word))]
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


def get_label(index, cate):
    prefix = 'B' if index == 0 else 'I'
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
