import numpy as np

import constants

from data.segmentation_data_loader import get_segmentation_spans


FLOAT_MIN = -100000.0           # input for exp

def add_chunk_sequences(
        data, dic, max_len=4, evaluate=True, use_attention=False, use_concat=False, xp=np):
    if use_concat:
        if use_attention:
            add_chunk_sequences_wconcat(data, dic, max_len=max_len, evaluate=evaluate,
                                        use_attention=True, use_concat=True)
        else:
            add_chunk_sequences_concat(data, dic, max_len=max_len, evaluate=evaluate)

    else:
        add_chunk_sequences_average(
            data, dic, max_len=max_len, min_value_mask=use_attention,
            evaluate=evaluate, restrict_memory=False, xp=xp)


# deprecated
def add_chunk_sequences_average(
        data, dic, max_len=4, oracle=False, min_value_mask=True, evaluate=True, restrict_memory=False, xp=np):
    print('add_chunk_sequences_average, attention=', min_value_mask)
    if min_value_mask:          # for attention
        mask_val = FLOAT_MIN
        non_mask_val = 0.0
    else:
        mask_val = 0.0
        non_mask_val = 1.0

    token_seqs = data.inputs[0]
    gold_label_seqs = data.outputs[0] if evaluate else None
    gold_chunk_seqs = []
    chunk_seqs = []
    masks = []
    trie = dic.tries[constants.CHUNK]

    ins_cnt = 0
    for sen_id, tseq in enumerate(token_seqs):
        if ins_cnt > 0 and ins_cnt % 100000 == 0:
            print('Processed', ins_cnt, 'sentences', file=sys.stderr)
        ins_cnt += 1
        lseq = gold_label_seqs[sen_id] if evaluate else None

        n = len(tseq)
        gchunk_seq = [-1] * n
        spans_gold = get_segmentation_spans(lseq) if evaluate else None

        spans_found = []
        for i in range(n):
            res = trie.common_prefix_search(tseq, i, i + max_len)
            spans_found.extend(res)

        m = len(spans_found)
        chunk_seq = [None] * m

        mask1 = [] if restrict_memory else xp.full((n, m), mask_val, dtype='f')
        i_found = []
        for j, span in enumerate(spans_found):
            chunk_seq[j] = trie.get_chunk_id(tseq[span[0]:span[1]])
            is_gold_span = evaluate and span in spans_gold
            for i in range(span[0], span[1]):
                if is_gold_span:
                    gchunk_seq[i] = j
                if not oracle or is_gold_span:
                    if restrict_memory:
                        mask1.append((i, j))
                    else:
                        mask1[i, j] = non_mask_val
                        i_found.append(i)

        if not restrict_memory and min_value_mask:
            mask2 = xp.ones((n,m), dtype='f')
            for i in range(n):
                if not i in i_found:
                    mask2[i] = xp.zeros((m,), dtype='f')
        else:
            mask2 = None
        mask = (None, mask1, mask2)

        #     mask = (mask1, mask2)
        # else:
        #     mask = mask1
                
        # print('x', ' '.join([str(i)+':'+dic.tables[constants.UNIGRAM].id2str[t] for i,t in enumerate(tseq)]))
        # print('l', ' '.join([dic.tables[constants.SEG_LABEL].id2str[l] for l in lseq]))
        # print('c', ' '.join([str(i)+':'+trie.id2chunk[c] for i,c in enumerate(chunk_seq)]))
        # print('gc', gchunk_seq)
        # print('gc', ' '.join([str(i)+':'+trie.id2chunk[chunk_seq[c]] if c else str(i)+':-1' for i,c in enumerate(gchunk_seq)]))
        # print(mask1)
        # print(mask2)

        chunk_seqs.append(chunk_seq)
        masks.append(mask)
        if evaluate:
            gold_chunk_seqs.append(gchunk_seq)

    data.inputs.append(chunk_seqs)
    data.inputs.append([])      # feat_seqs for concat
    data.inputs.append(masks)
    if evaluate:
        data.outputs.append(gold_chunk_seqs)


# deprecated
def add_chunk_sequences_concat(
        data, dic, max_len=4, evaluate=True):
    print('add_chunk_sequences_concat')
    feat_size = sum([h for h in range(max_len+1)])

    token_seqs = data.inputs[0]
    gold_label_seqs = data.outputs[0] if evaluate else None
    gold_chunk_seqs = []
    feat_seqs = []
    masks = []
    trie = dic.tries[constants.CHUNK]

    ins_cnt = 0
    for sen_id, tseq in enumerate(token_seqs):
        if ins_cnt > 0 and ins_cnt % 100000 == 0:
            print('Processed', ins_cnt, 'sentences', file=sys.stderr)
        ins_cnt += 1

        n = len(tseq)
        gchunk_seq = [-1] * n
        feats = [[0] * n for k in range(feat_size)]
        mask = []

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

        for span in (spans_found):
            is_gold_span = evaluate and span in spans_gold
            cid = trie.get_chunk_id(tseq[span[0]:span[1]])
            clen = span[1] - span[0]

            for i in range(span[0], span[1]):
                k = sum([h for h in range(clen)]) + (i - span[0])
                feats[k][i] = cid
                mask.append((k,i))
                if is_gold_span:
                    gchunk_seq[i] = k

        # print('x', ' '.join([str(i)+':'+dic.tables[constants.UNIGRAM].id2str[t] for i,t in enumerate(tseq)]))
        # print('l', ' '.join([dic.tables[constants.SEG_LABEL].id2str[l] for l in lseq]))
        # print('c', [str(i)+':'+str([trie.id2chunk[c] if c >= 0 else '-1' for c in raw]) for i,raw in enumerate(feats)])
        # print('gc', gchunk_seq)
        # print('mask', mask)

        feat_seqs.append(feats)
        masks.append(mask)
        if evaluate:
            gold_chunk_seqs.append(gchunk_seq)

    data.inputs.append([])      # chunk_seqs for average
    data.inputs.append(feat_seqs)
    data.inputs.append(masks)
    if evaluate:
        data.outputs.append(gold_chunk_seqs)


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
def add_chunk_sequences_wconcat(
        data, dic, max_len=4, evaluate=True, use_attention=True, use_concat=True):
    print('add_chunk_sequences_wconcat')
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
        # print('c', ' '.join([str(i)+':'+trie.id2chunk[c] for i,c in enumerate(chunk_seq)]))
        # print('f', [str(i)+':'+str([trie.id2chunk[c] if c >= 0 else '-1' for c in raw]) for i,raw in enumerate(feats)])
        # print('gc', gchunk_seq)
        # print('mask0', mask0)
        # print('mask1', mask1)
        # for i in range(n):
        #     print(i, char_index2feat_indexs(i, n, max_len))
        # print()

        #RM if chunk_seq is not None:
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
        mask, n_tokens, n_chunks, feat_size, emb_dim, use_attention=False, use_concat=False, xp=np):
    if use_concat:
        if use_attention:
            ret = convert_mask_matrix_wconcat(
                mask, n_tokens, n_chunks, feat_size, emb_dim, use_attention=True, xp=xp)
        else:
            ret = convert_mask_matrix_concat(mask, n_tokens, feat_size, emb_dim, xp=xp)
    else:
        ret = convert_mask_matrix_average(mask, n_tokens, n_chunks, min_value_mask=use_attention, xp=xp)

    return ret


# deprecated
def convert_mask_matrix_average(index_pairs, n_tokens, n_chunks, min_value_mask=True, xp=np):
    if min_value_mask:          # for attention
        mask_val = FLOAT_MIN
        non_mask_val = 0.0
    else:
        mask_val = 0.0
        non_mask_val = 1.0

    shape = (n_tokens, n_chunks)
    mask1 = xp.full(shape, mask_val, dtype='f')
    for i, j in index_pairs:
        mask1[i, j] = non_mask_val

    if min_value_mask:
        mask2 = xp.ones(shape, dtype='f') # zero vectors for characters w/o no candidate words
        for i in range(n_tokens):
            tmp = [k1 == i for k1,k2 in index_pairs]
            if len(tmp) == 0 or max(tmp) == False: # not (i,*) in index_pair
                mask2[i] = xp.zeros((n_chunks,), dtype='f')
    else:
        mask2 = None

    mask = (None, mask1, mask2)

    return mask


# deprecated
def convert_mask_matrix_concat(index_pairs, sen_len, feat_size, emb_dim, xp=np):
    shape = (feat_size, sen_len, emb_dim)
    ret = xp.zeros(shape, dtype='f')
    
    for k, i in index_pairs:
        ret[k][i] = xp.ones((emb_dim,), dtype='f')
        
    # ret = F.concat(ret, axis=1)

    return (ret, None, None)


def convert_mask_matrix_wconcat(
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
