"""
- (Zaremba 2015) の著者らによる torch 実装
  - (Zaremba 2015) RECURRENT NEURAL NETWORK REGULARIZATION, ICLR
  - https://github.com/tomsercu/lstm
-> 同等のプログラムの PFN による chainer 実装 -> 改造

"""

import copy
import enum
from collections import Counter
from datetime import datetime

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.functions.loss import softmax_cross_entropy
from chainer.links.connection.n_step_rnn import argsort_list_descent, permutate_list
from chainer.functions.math import minmax, exponential, logsumexp
from chainer import initializers
from chainer import Variable


from eval.conlleval import conlleval
import lattice
import features
from util import Timer



"""
Base model that consists of embedding layer, recurrent network (RNN) layers and linear layer.

Args:
    n_rnn_layers:
        the number of (vertical) layers of recurrent network
    n_vocab:
        size of vocabulary
    n_embed_dim:
        dimention of word embedding
    n_rnn_units:
        the number of units of RNN
    n_labels:
        the number of labels that input instances will be classified into
    dropout:
        dropout ratio of RNN
    rnn_unit_type:
        unit type of RNN: lstm, gru or plain
    rnn_bidirection:
        use bi-directional RNN or not
    linear_activation:
        activation function of linear layer: identity, relu, tanh, sigmoid
    init_embed:
        pre-trained embedding matrix
    feat_extractor:
        FeatureExtractor object to extract additional features 
    gpu:
        gpu device id
"""
class RNNBase(chainer.Chain):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            init_embed=None, feat_extractor=None, gpu=-1):
        super(RNNBase, self).__init__()

        with self.init_scope():
            self.act = get_activation(linear_activation)
            if not self.act:
                print('unsupported activation function.')
                sys.exit()

            if init_embed:
                n_vocab = init_embed.W.shape[0]
                embed_dim = init_embed.W.shape[1]

            # init fields
            self.embed_dim = embed_dim
            if feat_extractor:
                self.feat_extractor = feat_extractor
                self.input_vec_size = self.embed_dim + self.feat_extractor.dim
            else:
                self.feat_extractor = None
                self.input_vec_size = self.embed_dim

            # init layers
            self.embed = L.EmbedID(n_vocab, self.embed_dim) if init_embed == None else init_embed

            self.rnn_unit_type = rnn_unit_type
            if rnn_unit_type == 'lstm':
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiLSTM(n_rnn_layers, self.input_vec_size, n_rnn_units, dropout)
                else:
                    self.rnn_unit = L.NStepLSTM(n_rnn_layers, embed_dim, n_rnn_units, dropout)
            elif rnn_unit_type == 'gru':
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiGRU(n_rnn_layers, self.input_vec_size, n_rnn_units, dropout) 
                else:
                    self.rnn_unit = L.NStepGRU(n_rnn_layers, embed_dim, n_rnn_units, dropout)
            else:
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiRNNTanh(n_rnn_layers, self.input_vec_size, n_rnn_units, dropout) 
                else:
                    self.rnn_unit = L.NStepRNNTanh(n_rnn_layers, embed_dim, n_rnn_units, dropout)

            self.linear = L.Linear(n_rnn_units * (2 if rnn_bidirection else 1), n_labels)

            print('## parameters')
            print('# embed:', self.embed.W.shape)
            print('# rnn unit:', self.rnn_unit)
            if self.rnn_unit_type == 'lstm':
                i = 0
                for c in self.rnn_unit._children:
                    print('#   param', i)
                    print('#      0 -', c.w0.shape, '+', c.b0.shape)
                    print('#      1 -', c.w1.shape, '+', c.b1.shape)
                    print('#      2 -', c.w2.shape, '+', c.b2.shape)
                    print('#      3 -', c.w3.shape, '+', c.b3.shape)
                    print('#      4 -', c.w4.shape, '+', c.b4.shape)
                    print('#      5 -', c.w5.shape, '+', c.b5.shape)
                    print('#      6 -', c.w6.shape, '+', c.b6.shape)
                    print('#      7 -', c.w7.shape, '+', c.b7.shape)
                    i += 1
            print('# linear:', self.linear.W.shape, '+', self.linear.b.shape)
            print('# linear_activation:', self.act)


    # create input vector
    def create_features(self, xs):
        exs = []
        for x in xs:
            if self.feat_extractor:
                emb = self.embed(x)
                feat = self.feat_extractor.extract_features(x)
                ex = F.concat((emb, feat), 1)
            else:
                ex = self.embed(x)
                
            exs.append(ex)
        xs = exs
        return xs


    ## unused
    def get_id_array(self, start, width, gpu=-1):
        ids = np.array([], dtype=np.int32)
        for i in range(start, start + width):
            ids = np.append(ids, np.int32(i))
        return cuda.to_gpu(ids) if gpu >= 0 else ids


class RNN(RNNBase):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            init_embed=None, feat_extractor=None, gpu=-1):
        super(RNN, self).__init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            init_embed=None, feat_extractor=None, gpu=-1)

        self.loss_fun = softmax_cross_entropy.softmax_cross_entropy


    def __call__(self, xs, ts, train=True):
        xs = self.create_features(xs)

        with chainer.using_config('train', train):
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, xs)
            else:
                hy, hs = self.rnn_unit(None, xs)
            ys = [self.act(self.linear(h)) for h in hs]

        loss = None
        ps = []
        for y, t in zip(ys, ts):
            if loss is not None:
                loss += self.loss_fun(y, t)
            else:
                loss = self.loss_fun(y, t)
                ps.append([np.argmax(yi.data) for yi in y])

        return loss, ps


class RNN_CRF(RNNBase):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            init_embed=None, feat_extractor=None, gpu=-1):
        super(RNN_CRF, self).__init__(
            n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout, rnn_unit_type, 
            rnn_bidirection, linear_activation, init_embed, feat_extractor, gpu)

        with self.init_scope():
            self.crf = L.CRF1d(n_labels)

            print('# crf cost:', self.crf.cost.shape)
            print()


    def __call__(self, xs, ts, train=True):
        xs = self.create_features(xs)

        with chainer.using_config('train', train):
            # rnn layers
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, xs)
            else:
                hy, hs = self.rnn_unit(None, xs)

            # linear layer
            if not self.act or self.act == 'identity':
                hs = [self.linear(h) for h in hs]                
            else:
                hs = [self.act(self.linear(h)) for h in hs]

            # crf layer
            indices = argsort_list_descent(hs)
            trans_hs = F.transpose_sequence(permutate_list(hs, indices, inv=False))
            trans_ts = F.transpose_sequence(permutate_list(ts, indices, inv=False))
            loss = self.crf(trans_hs, trans_ts)
            score, trans_ys = self.crf.argmax(trans_hs)
            ys = permutate_list(F.transpose_sequence(trans_ys), indices, inv=True)
            ys = [y.data for y in ys]

            ################
            # loss = chainer.Variable(cuda.cupy.array(0, dtype=np.float32))
            # trans_ys = []
            # for i in range(len(hs)):
            #     hs_tmp = hs[i:i+1]
            #     ts_tmp = ts[i:i+1]
            #     t0 = datetime.now()
            #     indices = argsort_list_descent(hs_tmp)
            #     trans_hs = F.transpose_sequence(permutate_list(hs_tmp, indices, inv=False))
            #     trans_ts = F.transpose_sequence(permutate_list(ts_tmp, indices, inv=False))
            #     t1 = datetime.now()
            #     loss += self.crf(trans_hs, trans_ts)
            #     t2 = datetime.now()
            #     score, trans_y = self.crf.argmax(trans_hs)
            #     t3 = datetime.now()
            #     #trans_ys.append(trans_y)
            #     # print('  transpose     : {}'.format((t1-t0).seconds+(t1-t0).microseconds/10**6))
            #     # print('  crf forward   : {}'.format((t2-t1).seconds+(t2-t1).microseconds/10**6))
            #     # print('  crf argmax    : {}'.format((t3-t2).seconds+(t3-t2).microseconds/10**6))
            # ys = [None]
            ################

        return loss, ys
        

# not finished to implement
class DualRNN(chainer.Chain):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            init_embed=None, feat_extractor=None, gpu=-1):
        super(DualRNN, self).__init__()

        with self.init_scope():
            self.rnn_crf1 = RNN_CRF(
                n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout, rnn_unit_type, 
                rnn_bidirection, linear_activation, init_embed, gpu)
            self.rnn_ff2 = RNN(
                n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout, rnn_unit_type, 
                rnn_bidirection, linear_activation, init_embed, gpu)


    def __call__(self, cxs, cts, wts, train=True):
        char_loss, cys = self.rnn_crf1(cxs, cts)

        
        #convert_to_word_seqs(cxs, cys, morph_dic)
        word_loss, wys = self.rnn_ff2(wxs, wts)
        loss = char_loss + word_loss

        return


    def call_rnn_ff2(self, xs, ts, train=True):
        xs = self.create_features(xs)

        with chainer.using_config('train', train):
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, xs)
            else:
                hy, hs = self.rnn_unit(None, xs)
            ys = [self.act(self.linear(h)) for h in hs]

        loss = None
        ps = []
        for y, t in zip(ys, ts):
            if loss is not None:
                loss += self.loss_fun(y, t)
            else:
                loss = self.loss_fun(y, t)
                ps.append([np.argmax(yi.data) for yi in y])

        return loss, ps


class RNN_LatticeCRF(RNNBase):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, morph_dic, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            init_embed=None, feat_extractor=None, gpu=-1):
        super(RNN_LatticeCRF, self).__init__(
            n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout, rnn_unit_type, 
            rnn_bidirection, linear_activation, init_embed, feat_extractor, gpu)

        with self.init_scope():
            self.morph_dic = morph_dic
            self.lattice_crf = LatticeCRF(self.morph_dic, gpu)

    def __call__(self, xs, ts_seg, ts_pos, train=True):
        # ts_seg: unused
        embed_xs = self.create_features(xs)

        with chainer.using_config('train', train):
            # rnn layers
            if self.rnn_unit_type == 'lstm':
                hy, cy, hs = self.rnn_unit(None, None, embed_xs)
            else:
                hy, hs = self.rnn_unit(None, embed_xs)

            # linear layer
            if not self.act or self.act == 'identity':
                hs = [self.linear(h) for h in hs]
            else:
                hs = [self.act(self.linear(h)) for h in hs]

            # lattice crf layer
            loss, ys_seg, ys_pos = self.lattice_crf(xs, hs, ts_pos)
            #print(len(xs), xs[0].shape, xs[0])
            #print(len(embed_xs), xs[0].shape)
            #print(len(hs), hs[0].shape)

        return loss, ys_seg, ys_pos


class LatticeCRF(chainer.link.Link):
    def __init__(self, morph_dic, gpu=-1):
        super(LatticeCRF, self).__init__()
        self.gpu = gpu
        self.morph_dic = morph_dic

        n_label = len(self.morph_dic.label_indices)
        n_pos = len(self.morph_dic.pos_indices)
        self.predict_pos = n_pos > 1

        with self.init_scope():
            # prev_tag -> next_tag
            self.cost_seg = chainer.variable.Parameter(0, (n_label, n_label))
            if self.predict_pos:
                # prev_pos -> next_pos
                self.cost_pos = chainer.variable.Parameter(0, (n_pos, n_pos))
                # score b/w chars of current pos to prevent length bias
                self.cost_inter_pos = chainer.variable.Parameter(0, (n_pos,))
            else:
                self.cost_pos = None
                self.cost_inter_pos = None

        print('# crf cost_seg:', self.cost_seg.shape)
        print('# crf cost_pos:', self.cost_pos.shape if self.predict_pos else None)
        print('# crf cost_inter_pos:', self.cost_inter_pos.shape if self.predict_pos else None)
        print()

        self.debug = False


    def __call__(self, xs, hs, ts_pos):
        xp = cuda.cupy if self.gpu >= 0 else np

        ys_seg = []
        ys_pos = []
        loss = Variable(xp.array(0, dtype=np.float32))

        for x, h, t_pos in zip(xs, hs, ts_pos):
            t0 = datetime.now()
            lat = lattice.Lattice(x, self.morph_dic, self.debug)
            t1 = datetime.now()
            lat.prepare_forward(xp)
            t2 = datetime.now()
            loss += self.forward(lat, h, t_pos)
            t3 = datetime.now()
            y_seg, y_pos, words = self.argmax(lat)
            t4 = datetime.now()
            # print('  create lattice : {}'.format((t1-t0).seconds+(t1-t0).microseconds/10**6))
            # print('  prepare forward: {}'.format((t2-t1).seconds+(t2-t1).microseconds/10**6))
            # print('  forward       : {}'.format((t3-t2).seconds+(t3-t2).microseconds/10**6))
            # print('  argmax        : {}'.format((t4-t3).seconds+(t4-t3).microseconds/10**6))

            # print('y:', decode(x, y_pos, self.morph_dic))
            # print('t:', decode(x, t_pos, self.morph_dic))
            # print('t:', t_pos)
            # print([self.morph_dic.get_token(ti) for ti in x)
            # print(y_seg)
            # print(y_pos)
            # print(words)

            ys_seg.append(y_seg)
            ys_pos.append(y_pos)

        return loss, ys_seg, ys_pos


    def forward(self, lat, h, t_pos):
        xp = cuda.cupy if self.gpu >= 0 else np

        gold_score = Variable(xp.array(0, dtype=np.float32))
        gold_nodes = t_pos
        gold_next_ind = 0
        gold_next = gold_nodes[0]
        gold_prev = None

        if self.debug:
            print('\nnew instance:', lat.sen)

        T = len(lat.sen)
        for t in range(1, T + 1):
            if self.debug:
                print('\nt={}'.format(t))

            begins = lat.end2begins.get(t)
            if begins:
                num_nodes = sum([len(entry[2]) for entry in begins])
                deltas_t = [Variable(xp.array(0, dtype='f')) for m in range(num_nodes)]
                scores_t = [Variable(xp.array(0, dtype='f')) for m in range(num_nodes)]

            for i, entry in enumerate(begins):
                s = entry[0]    # begin index
                word_id = entry[1]
                pos_ids = entry[2]
                word_len = t - s
                if self.debug:
                    print('  i={}:\n    span={}, wi={}, poss={}'.format(i, (s, t), word_id, pos_ids))

                cur_nodes_begin = sum([len(entry[2]) for entry in begins[:i]])

                # node score
                node_score = self.calc_node_score_for_characters(s, t, h)
                if self.debug:
                    print('    * char score={}'.format(node_score.data))
                # node_score += node_score_word

                # edge score b/w current and previous nodes
                prev_begins = lat.end2begins.get(s)
                if not prev_begins:
                    for k, pos_id in enumerate(pos_ids):
                        cur_ind = cur_nodes_begin + k
                        deltas_t[cur_ind] += node_score
                        scores_t[cur_ind] += node_score
                        if self.predict_pos:
                            cont_pos_score = self.cost_inter_pos[pos_id] * (word_len - 1)
                            deltas_t[cur_ind] += cont_pos_score
                            scores_t[cur_ind] += cont_pos_score
                        if self.debug:
                            print('    node score={}, contpos score={} (len_={})'.format(
                                node_score.data, cont_pos_score.data if self.predict_pos else 0, word_len-1))

                    if self.debug:
                        if self.debug:
                            print("    delta[{}] {}".format(t, deltas_t))
                            print("    score[{}] {}".format(t, scores_t))

                    if gold_next and gold_next[0] == s and gold_next[1] == t and gold_next[2] in pos_ids:
                        tmp = float(gold_score.data)
                        gold_score += node_score
                        if self.predict_pos:
                            gold_score += self.cost_inter_pos[gold_next[2]] * (word_len - 1)
                        gold_prev = gold_next
                        gold_next_ind += 1
                        gold_next = gold_nodes[gold_next_ind] if gold_next_ind < len(gold_nodes) else None
                        if self.debug:
                            print("\n    @ gold {} node score {} -> {}".format(
                                gold_prev, tmp, gold_score.data))

                else:
                    num_prev_nodes = sum([len(entry[2]) for entry in prev_begins])
                    edge_scores_chartran = [Variable(xp.array(0, dtype='f')) for m in range(num_prev_nodes)]

                    for j, prev_entry in enumerate(prev_begins):
                        r = prev_entry[0]
                        prev_pos_ids = prev_entry[2]
                        prev_nodes_begin = sum([len(entry[2]) for entry in prev_begins[:j]])
                        prev_nodes_end = prev_nodes_begin + len(prev_pos_ids)
                        edge_score_chartran = self.calc_edge_score_for_characters(r, s, t)
                        if self.debug:
                            print('    j={}-{}:\n      span={}, poss={} - chartran score={}'.format(
                                prev_nodes_begin, prev_nodes_end-1, (r, s), prev_pos_ids, 
                                edge_score_chartran.data))

                        for k in range(prev_nodes_begin, prev_nodes_end):
                            edge_scores_chartran[k] += edge_score_chartran


                    for k, pos_id in enumerate(pos_ids):
                        if self.debug and self.predict_pos:
                            print('    current pos={}'.format(pos_id))
                        cur_ind = cur_nodes_begin + k

                        if self.predict_pos:
                            cont_pos_score = self.cost_inter_pos[pos_id] * (word_len - 1)
                            if self.debug:
                                print('    contpos score={} (len_={})'.format(cont_pos_score, word_len-1))
                            edge_scores_postran = [
                                Variable(xp.array(0, dtype='f')) for m in range(num_prev_nodes)]

                        for j, prev_entry in enumerate(prev_begins):
                            r = prev_entry[0]
                            prev_pos_ids = prev_entry[2]
                            prev_nodes_begin = sum([len(entry[2]) for entry in prev_begins[:j]])
                            
                            for l, prev_pos_id in enumerate(prev_pos_ids):
                                prev_ind = prev_nodes_begin + l
                                if self.predict_pos:
                                    edge_score_postran = self.cost_pos[prev_pos_id][pos_id]
                                    edge_scores_postran[prev_ind] += edge_score_postran + cont_pos_score
                                    if self.debug:
                                        print('      prev pos={} ({}): postran score={} + contpos score={}'.format(prev_pos_id, prev_ind, edge_score_postran.data, cont_pos_score.data))

                                if gold_next and gold_next[0] == s and gold_next[1] == t and gold_next[2] == pos_id and gold_prev[0] == r and gold_prev[1] == s and gold_prev[2] == prev_pos_id:
                                    tmp = float(gold_score.data)
                                    gold_score += node_score + edge_scores_chartran[prev_ind]
                                    if self.predict_pos:
                                        gold_score += edge_scores_postran[prev_ind]

                                    if self.debug:
                                        print("\n@ gold {} edge score {} -> {} (prev={})".format(
                                            gold_next, tmp, gold_score.data, gold_prev))

                                    gold_prev = gold_next
                                    gold_next_ind += 1
                                    gold_next = gold_nodes[gold_next_ind] if gold_next_ind < len(gold_nodes) else None

                        if self.predict_pos:
                            edge_scores = F.concat(
                                [F.expand_dims(cts + pts, axis=0) for cts, pts in zip(
                                    edge_scores_chartran, edge_scores_postran)], axis=0)
                        else:
                            edge_scores = F.concat(
                                [F.expand_dims(var, axis=0) for var in edge_scores_chartran], axis=0)

                        tmp_val_for_deltas = logsumexp.logsumexp(lat.deltas[s] + edge_scores)
                        tmp_vals_for_scores = lat.scores[s] + edge_scores
                        if self.debug:
                            print("\n    escores {} + delta[{}] {} | lse={} | node score={}".format(
                                edge_scores.data, s, lat.deltas[s].data, tmp_val_for_deltas, node_score.data))
                        deltas_t[cur_ind] += tmp_val_for_deltas + node_score
                        scores_t[cur_ind] += minmax.max(tmp_vals_for_scores) + node_score
                        lat.prev_pointers[t][cur_ind] = minmax.argmax(tmp_vals_for_scores).data
                        if self.debug:
                            print("    delta[{}] {}".format(t, deltas_t))
                            print("    score[{}] {}".format(t, scores_t))
                            print("    prev best {}\n".format(lat.prev_pointers[t]))

                lat.deltas[t] = F.concat([F.expand_dims(val, axis=0) for val in deltas_t], axis=0)
                lat.scores[t] = F.concat([F.expand_dims(val, axis=0) for val in scores_t], axis=0)

        logz = logsumexp.logsumexp(lat.deltas[T])
        loss = logz - gold_score
        if self.debug:
            print(lat.deltas[T].data, 'lse->', logz.data)
            print(loss.data, '=', logz.data, '-', gold_score.data)

        return loss


    def argmax(self, lat):
        xp = cuda.cupy if self.gpu >= 0 else np
        if self.debug:
            print('\nargmax')

        y_seg = []
        y_pos = []
        wis = []

        t = len(lat.sen)
        prev_best = int(xp.argmax(lat.scores[t].data))
        while t > 0:
            begins = lat.end2begins.get(t)
            if self.debug:
                print(t, begins)

            i = j = prev_best
            for begin in begins:
                if j >= len(begin[2]):
                    j -= len(begin[2])
                    continue
                else:
                    if self.debug:
                        print(' ', i, j, begin)
                    s = begin[0]
                    wi = begin[1]
                    pos = begin[2][j]
                    node = (s, t, pos)
                    break
            prev_best = lat.prev_pointers[t][i]

            # update lists
            y_pos.append(node)
            wis.append(wi)
            if t - s == 1:
                tag = 3 # S
                y_seg.append(tag)
            else:
                for j in range(t-1, s-1, -1):
                    if j == s:
                        tag = 0 # B
                    elif j == t - 1:
                        tag = 2 # E
                    else:
                        tag = 1 # I
                    y_seg.append(tag)
            t = s

        y_pos.reverse()
        y_seg.reverse()

        return y_seg, y_pos, wis


    def calc_node_score_for_characters(self, node_begin, node_end, hidden_vecs):
        # BIES スキーマのみ対応
        if node_end - node_begin == 1:
            tag = 3 #self.morph_dic.get_label_id('S')
            score = hidden_vecs[node_begin][tag]
            # if self.debug:
            #     print('* N1 score for {}-th elem of {}, {}: {} -> {}'.format(
            #         0, (node_begin, node_end), tag, hidden_vecs[node_begin][tag].data, score.data))

        else:
            score = None
            prev_tag = None
            for i in range(node_begin, node_end):
                if i == node_begin:
                    tag = 0 # B
                elif i == node_end - 1:
                    tag = 2 # E
                else:
                    tag = 1 # I

                score = (hidden_vecs[i][tag] + score) if score is not None else hidden_vecs[i][tag]
                # if self.debug:
                #     print('* N2 score for {}-th elem of {}, {}: {} -> {}'.format(
                #         i, (node_begin, node_end), tag, hidden_vecs[i][tag].data, score.data))
                
                if i > node_begin:
                    cost = self.cost_seg[prev_tag][tag]
                    score += cost
                    # if self.debug:
                    #     print('* NE score ({}) to ({}): {} -> {}'.format(
                    #         prev_tag, tag, cost.data, score.data))

                prev_tag = tag

        return score


    def calc_edge_score_for_characters(self, prev_begin, prev_end, next_end):
        # BIES スキーマのみ対応
        if prev_end - prev_begin == 1:
            prev_last_tag = 3 # S
        else:
            prev_last_tag = 2 # #

        next_begin = prev_end
        if next_end - next_begin == 1:
            next_first_tag = 3 # S
        else:
            next_first_tag = 0 # B

        score = self.cost_seg[prev_last_tag][next_first_tag]

        # if self.debug:
        #     print('* E score for {} ({}) and {} ({}): {}'.format(
        #         (prev_begin, prev_end), prev_last_tag, (next_begin, next_end), next_first_tag, score.data))
        return score


class SequenceTagger(chainer.link.Chain):
    compute_fscore = True

    def __init__(self, predictor, id2label):
        super(SequenceTagger, self).__init__(predictor=predictor)
        self.id2label = id2label
        
    def __call__(self, *args, **kwargs):
        assert len(args) >= 2
        xs = args[0]
        ts = args[1]
        loss, ys = self.predictor(*args, **kwargs)

        eval_counts = None
        if self.compute_fscore:
            for x, t, y in zip(xs, ts, ys):
                generator = self.generate_lines(x, t, y)
                eval_counts = conlleval.merge_counts(eval_counts, conlleval.evaluate(generator))

        return loss, eval_counts


    # def decode(self, *args, **kwargs):
    #     _, ps = self.pridictor(*args, **kwargs)
    #     return ps


    def generate_lines(self, x, t, y, is_str=False):
        i = 0
        while True:
            if i == len(x):
                raise StopIteration
            x_str = str(x[i])
            t_str = t[i] if is_str else self.id2label[int(t[i])]
            y_str = y[i] if is_str else self.id2label[int(y[i])]

            yield [x_str, t_str, y_str]

            i += 1


    def grow_lookup_table(self, token2id_new, gpu=-1):
        weight1 = self.predictor.embed.W
        diff = len(token2id_new) - len(weight1)
        weight2 = chainer.variable.Parameter(initializers.normal.Normal(1.0), (diff, weight1.shape[1]))
        weight = F.concat((weight1, weight2), 0)
        embed = L.EmbedID(0, 0)
        embed.W = chainer.Parameter(initializer=weight.data)
        self.predictor.embed = embed

        print('# grow vocab size: %d -> %d' % (weight1.shape[0], weight.shape[0]))
        print('# embed:', self.predictor.embed.W.shape)


class JointMorphologicalAnalyzer(SequenceTagger):
    # def __init(self, predictor):
    #     super(JointMorphologicalAnalyzer, self).__init__(predictor=predictor)


    def __init__(self, predictor, id2label):
        super(JointMorphologicalAnalyzer, self).__init__(predictor, id2label)

        
    def __call__(self, *args, **kwargs):
        assert len(args) >= 2
        xs = args[0]
        ts_seg = args[1]
        ts_pos = args[2]
        loss, ys_seg, ys_pos = self.predictor(*args, **kwargs)

        if self.compute_fscore:
            ecounts_seg = None
            ecounts_pos = None
            
            for x, t_seg, t_pos, y_seg, y_pos in zip(xs, ts_seg, ts_pos, ys_seg, ys_pos):
                generator_seg = self.generate_lines(x, t_seg, y_seg)
                ecounts_seg = conlleval.merge_counts(ecounts_seg, conlleval.evaluate(generator_seg))

                t_seg_pos = self.convert_to_joint_labels(t_seg, t_pos)
                y_seg_pos = self.convert_to_joint_labels(y_seg, y_pos)
                generator_pos = self.generate_lines(x, t_seg_pos, y_seg_pos, is_str=True)
                ecounts_pos = conlleval.merge_counts(ecounts_seg, conlleval.evaluate(generator_pos))
                
                # print([self.predictor.morph_dic.get_label(int(y)) for y in y_seg])
                # print([self.predictor.morph_dic.get_label(int(t)) for t in t_seg])
                # print(y_seg_pos)
                # print(t_seg_pos)
                # print()

        return loss, ecounts_seg, ecounts_pos


    def convert_to_joint_labels(self, seg_labels, node_seq):
        res = [None] * node_seq[-1][1]
        i = 0
        for node in node_seq:
            pos = self.predictor.morph_dic.get_pos(node[2])
            for j in range(node[0], node[1]):
                seg_label = self.predictor.morph_dic.get_label(int(seg_labels[j]))
                res[i] = '{}-{}'.format(seg_label, pos)
                i += 1

        return res


    def decode(self, sen, node_seq):
        return decode(sen, node_seq, self.morph_dic)
        

def convert_to_word_seqs(cxs, cys, morph_dic):
    batch_size = len(cxs) 
    wxs = [None] * batch_size
    wys = [None] * batch_size
    for k, ins in enumerate(zip(cxs, cys)):
        cx = ins[0]
        cy = ins[1]
        sen_len = len(ins[0])
        wx = []

        prev_i = 0
        next_i = 0
        while next_i < sen_len:
            if cy[next_i] == 0 or cy[next_i] == 3:
                w = ''.join([morph_dic.get_token(cxj) for cxj in cx[prev_i:next_i+1]])
                wx.append(morph_dic.get_word_id(w))

        wxs[k] = wx

    return wxs
    


def decode(sen, pos_seq, morph_dic):
    seq = []

    for node in pos_seq:
        if node[1] - node[0] == 1:
            ti = int(sen[node[0]])
            word = morph_dic.get_token(ti)
        else:
            word = ''
            for i in range(node[0], node[1]):
                ti = int(sen[i])
                word += morph_dic.get_token(ti)
        pos = morph_dic.get_pos(node[2])
        seq.append('{}/{}'.format(word, pos))

    return seq
        

def get_activation(activation):
    if not activation  or activation == 'identity':
        return F.identity
    elif activation == 'relu':
        return F.relu
    elif activation == 'tanh':
        return F.tanh
    elif activation == 'sigmoid':
        return F.sigmoid
    else:
        return


def add(var_list1, var_list2):
    len_list = len(var_list1)
    ret = [None] * len_list
    for i, var1, var2 in zip(range(len_list), var_list1, var_list2):
        ret[i] = var1 + var2
    return ret


# unused
def argmax_max_for_list(var_list):
    i = -1
    ret = None
    for j, var in enumerate(var_list):
        if ret is not None and var.data > ret.data:
            ret = var
            i = j
        else:
            ret = var
            i = j
    return i, ret


# unused
def logsumexp_for_list(var_list, xp=np):
    _, m = argmax_max(var_list)
    ret = chainer.Variable(xp.array(0, dtype='f'))
    for var in var_list:
        ret += exponential.exp(var - m)
    return exponential.log(ret) + m


# unused
def expand_variable(var, ind_from, ind_to, size, xp=np):
    var_ext = [F.expand_dims(var if (ind_from <= i and i < ind_to) else chainer.Variable(xp.array(0, dtype=np.float32)), axis=0) for i in range(size)]
    return var_ext
