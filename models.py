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
import lattice.lattice as lattice
from util import Timer


class RNNBase(chainer.Chain):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            n_left_contexts=0, n_right_contexts=0, init_embed=None, gpu=-1):
        super(RNNBase, self).__init__()

        with self.init_scope():
            self.act = get_activation(linear_activation)
            if not self.act:
                print('unsupported activation function.')
                sys.exit()

            if init_embed != None:
                n_vocab = init_embed.W.shape[0]
                embed_dim = init_embed.W.shape[1]

            # padding indices for context window
            self.left_padding_ids = self.get_id_array(n_vocab, n_left_contexts, gpu)
            self.right_padding_ids = self.get_id_array(n_vocab + n_left_contexts, n_right_contexts, gpu)
            self.empty_array = cuda.cupy.array(
                [], dtype=np.float32) if gpu >= 0 else np.array([], dtype=np.float32)
            # init fields
            self.embed_dim = embed_dim
            self.context_size = 1 + n_left_contexts + n_right_contexts
            self.input_vec_size = self.embed_dim * self.context_size
            vocab_size = n_vocab + n_left_contexts + n_right_contexts
            rnn_in = embed_dim * self.context_size

            # init layers
            self.lookup = L.EmbedID(vocab_size, self.embed_dim) if init_embed == None else init_embed

            self.rnn_unit_type = rnn_unit_type
            if rnn_unit_type == 'lstm':
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiLSTM(n_rnn_layers, rnn_in, n_rnn_units, dropout)
                else:
                    self.rnn_unit = L.NStepLSTM(n_rnn_layers, embed_dim, n_rnn_units, dropout)
            elif rnn_unit_type == 'gru':
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiGRU(n_rnn_layers, rnn_in, n_rnn_units, dropout) 
                else:
                    self.rnn_unit = L.NStepGRU(n_rnn_layers, embed_dim, n_rnn_units, dropout)
            else:
                if rnn_bidirection:
                    self.rnn_unit = L.NStepBiRNNTanh(n_rnn_layers, rnn_in, n_rnn_units, dropout) 
                else:
                    self.rnn_unit = L.NStepRNNTanh(n_rnn_layers, embed_dim, n_rnn_units, dropout)

            self.linear = L.Linear(n_rnn_units * (2 if rnn_bidirection else 1), n_labels)

            print('## parameters')
            print('# lookup:', self.lookup.W.shape)
            print('# rnn unit:', self.rnn_unit)
            # i = 0
            # for c in self.rnn_unit._children:
            #     print('#   param', i)
            #     print('#      0 -', c.w0.shape, '+', c.b0.shape)
            #     print('#      1 -', c.w1.shape, '+', c.b1.shape)
            #     print('#      2 -', c.w2.shape, '+', c.b2.shape)
            #     print('#      3 -', c.w3.shape, '+', c.b3.shape)
            #     print('#      4 -', c.w4.shape, '+', c.b4.shape)
            #     print('#      5 -', c.w5.shape, '+', c.b5.shape)
            #     print('#      6 -', c.w6.shape, '+', c.b6.shape)
            #     print('#      7 -', c.w7.shape, '+', c.b7.shape)
            #     i += 1
            print('# linear:', self.linear.W.shape, '+', self.linear.b.shape)
            print('# linear_activation:', self.act)

            # tmp
            self.vocab_size = vocab_size
            self.n_rnn_layers = n_rnn_layers
            self.rnn_unit_in = rnn_in
            self.n_rnn_units = n_rnn_units
            self.dropout = dropout
            self.rnn_bidirection = rnn_bidirection
            self.n_labels = n_labels


    # create input embeded vector considering context window
    def embed(self, xs):
        exs = []
        for x in xs:
            if self.context_size > 1:
                embeddings = F.concat((self.lookup(self.left_padding_ids),
                                       self.lookup(x),
                                       self.lookup(self.right_padding_ids)), 0)
                embeddings = F.reshape(embeddings, (len(x) + self.context_size - 1, self.embed_dim))

                ex = self.empty_array.copy()
                for i in range(len(x)):
                    for j in range(i, i + self.context_size):
                        ex = F.concat((ex, embeddings[j]), 0)
                ex = F.reshape(ex, (len(x), self.input_vec_size))
            else:
                ex = self.lookup(x)
                
            exs.append(ex)
        xs = exs
        return xs


    def get_id_array(self, start, width, gpu=-1):
        ids = np.array([], dtype=np.int32)
        for i in range(start, start + width):
            ids = np.append(ids, np.int32(i))
        return cuda.to_gpu(ids) if gpu >= 0 else ids


class RNN(RNNBase):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            n_left_contexts=0, n_right_contexts=0, init_embed=None, gpu=-1):
        super(RNN, self).__init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            n_left_contexts=0, n_right_contexts=0, init_embed=None, gpu=-1)

        self.loss_fun = softmax_cross_entropy.softmax_cross_entropy

        print()
        

    def __call__(self, xs, ts, train=True):
        #xs = [self.embed(x) for x in xs]
        xs = self.embed(xs)

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
            n_left_contexts=0, n_right_contexts=0, init_embed=None, gpu=-1):
        super(RNN_CRF, self).__init__(
            n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout, rnn_unit_type, 
            rnn_bidirection, linear_activation, n_left_contexts, n_right_contexts, init_embed, gpu)

        with self.init_scope():
            self.crf = L.CRF1d(n_labels)

            print('# crf cost:', self.crf.cost.shape)
            print()


    def __call__(self, xs, ts, train=True):
        xs = self.embed(xs)

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


    # temporaly code for memory checking
    # def reset(self):
    #     print('reset')
    #     del self.lookup, self.rnn_unit, self.linear, self.crf
    #     gc.collect()

    #     self.lookup = L.EmbedID(self.vocab_size, self.embed_dim)
    #     self.rnn_unit = L.NStepBiLSTM(self.n_rnn_layers, self.rnn_unit_in, self.n_rnn_units, self.dropout) if self.rnn_bidirection else L.NStepLSTM(self.n_rnn_layers, self.embed_dim, self.n_rnn_units, self.dropout)
    #     self.linear = L.Linear(self.n_rnn_units * (2 if self.rnn_bidirection else 1), self.n_labels)
    #     self.crf = L.CRF1d(self.n_labels)

    #     self.lookup = self.lookup.to_gpu()
    #     self.rnn_unit = self.rnn_unit.to_gpu()
    #     self.linear = self.linear.to_gpu()
    #     self.crf = self.crf.to_gpu()
        

class RNN_LatticeCRF(RNNBase):
    def __init__(
            self, n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, morph_dic, dropout=0, 
            rnn_unit_type='lstm', rnn_bidirection=True, linear_activation='identity', 
            n_left_contexts=0, n_right_contexts=0, init_embed=None, gpu=-1):
        super(RNN_LatticeCRF, self).__init__(
            n_rnn_layers, n_vocab, embed_dim, n_rnn_units, n_labels, dropout, rnn_unit_type, 
            rnn_bidirection, linear_activation, n_left_contexts, n_right_contexts, init_embed, gpu)

        with self.init_scope():
            self.lattice_crf = LatticeCRF(morph_dic, gpu)

    def __call__(self, xs, ts_seg, ts_pos, train=True):
        # ts_seg: unused
        embed_xs = self.embed(xs)

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
        print()

        self.debug = False


    def __call__(self, xs, hs, ts_pos):
        xp = cuda.cupy if self.gpu >= 0 else np

        ys_seg = []
        ys_pos = []
        loss = Variable(xp.array(0, dtype=np.float32))

        for x, h, t_pos in zip(xs, hs, ts_pos):
            t0 = datetime.now()
            #lat = lattice.Lattice(x, self.morph_dic, self.debug)
            #lat = lattice.Lattice2(x, self.morph_dic, self.debug)
            #lat = lattice.Lattice3(x, self.morph_dic, self.debug)
            lat = lattice.Lattice2(x, self.morph_dic, self.debug)
            t1 = datetime.now()
            lat.prepare_forward(xp)
            t2 = datetime.now()
            #loss += self.forward(lat, h, t_pos)
            #loss += self.forward2(lat, h, t_pos)
            #loss += self.forward3(lat, h, t_pos)
            loss += self.forward4(lat, h, t_pos)
            t3 = datetime.now()
            #y_seg, y_pos, words = self.argmax(lat)
            #y_seg, y_pos, words = self.argmax2(lat)
            #y_seg, y_pos, words = self.argmax3(lat)
            y_seg, y_pos, words = self.argmax2(lat)
            t4 = datetime.now()
            # print('  create lattice : {}'.format((t1-t0).seconds+(t1-t0).microseconds/10**6))
            # print('  prepare forward: {}'.format((t2-t1).seconds+(t2-t1).microseconds/10**6))
            # print('  forward       : {}'.format((t3-t2).seconds+(t3-t2).microseconds/10**6))
            # print('  argmax        : {}'.format((t4-t3).seconds+(t4-t3).microseconds/10**6))

            # print('y:', decode(x, y_pos, self.morph_dic))
            # print('t:', decode(x, t_pos, self.morph_dic))
            # print([self.morph_dic.get_token(ti) for ti in x)
            # print(y_seg)
            # print(y_pos)
            # print(words)

            ys_seg.append(y_seg)
            ys_pos.append(y_pos)

        return loss, ys_seg, ys_pos


    def forward4(self, lat, h, t_pos):
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
                #cur_nodes_end = cur_nodes_begin + len(pos_ids)

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

        # print('calc',   tca.elapsed)
        # print('init  ', ti.elapsed)
        # print('expand', te.elapsed)
        # print('total', tsum.elapsed)
        # print()
        
        #return gold_score
        return loss


    # 動作未検証
    def forward3(self, lat, h, t_pos):
        t_refer = Timer()
        ta = Timer()
        te = Timer()
        tc = Timer()
        t_calc = Timer()
        t_total = Timer()

        t_total.start()

        xp = cuda.cupy if self.gpu >= 0 else np

        ta.start()
        gold_score = chainer.Variable(xp.array(0, dtype=np.float32))
        ta.stop()
        gold_nodes = t_pos
        gold_next_ind = 0
        gold_next = gold_nodes[0]
        gold_prev = None

        if self.debug:
            print('\nnew instance:', lat.sen)
            print('lattice:', lat.eid2nodes)

        T = len(lat.sen)
        for t in range(1, T + 1):
            begins = lat.end2begins.get(t)
            num_nodes = sum([len(entry[2]) for entry in begins])

            smallest_s = begins[-1][0]
            subst_node_char = [False] * (t - smallest_s) # E(0), I(-1), ..., I(-N+1) の代入済みフラグ
            subst_node_chartran_BE = False
            subst_node_chartran_BI_IE = False
            # print('t:', t, begins)
            # print('smallest_s', smallest_s, subst_node_char)

            deltas_t = lat.deltas[t]
            scores_t = lat.scores[t]
            pp_t = lat.prev_pointers[t]

            for i, entry in enumerate(begins):
                s = entry[0]    # begin index
                word_id = entry[1]
                pos_ids = entry[2]
                word_len = t - s

                begins_prev_to_i = begins[:i]

                # node score for char
                for cur_ind in range(s, t):
                    if cur_ind == s:
                        to_subst = True
                        if word_len == 1:                   # back=0
                            char_tag = 3                    # S
                            target = (0, len(pos_ids))      # (from, to)
                        else:                               # back>=1
                            char_tag = 0                    # B
                            target = (sum([len(entry[2]) for entry in begins_prev_to_i]), len(pos_ids))
                    else:                                   # tag=I or E
                        cur_posi = t - cur_ind - 1
                        # print('posi', cur_posi, '|', s, cur_ind, t)
                        if not subst_node_char[cur_posi]:
                            to_subst = True
                            char_tag = 2 if cur_ind == t - 1 else 1 # E / I
                            target = (sum([len(entry[2]) for entry in begins_prev_to_i]), num_nodes)
                            subst_node_char[cur_posi] = True
                        else:
                            to_subst = False

                    if to_subst:
                        t_refer.start()
                        char_score = h[cur_ind][char_tag]
                        t_refer.stop()
                        for k in range(target[0], target[1]):
                            t_calc.start()
                            deltas_t[k] += char_score
                            scores_t[k] += char_score
                            t_calc.stop()

                # node score for char-transition
                if word_len >= 2:
                    if not subst_node_chartran_BE:
                        t_refer.start()
                        char_trans_score = self.cost_seg[0][2] # B-E
                        t_refer.stop()
                        target = (sum([len(entry[2]) for entry in begins_prev_to_i]), len(pos_ids))
                        subst_node_chartran_BE = True

                        for k in range(target[0], target[1]):
                            t_calc.start()
                            deltas_t[k] += char_trans_score
                            scores_t[k] += char_trans_score
                            t_calc.stop()

                    if not subst_node_chartran_BI_IE:
                        t_refer.start()
                        tmp1 = self.cost_seg[0][1]
                        tmp2 = self.cost_seg[1][2]
                        t_refer.stop()
                        t_calc.start()
                        char_trans_score = tmp1 + tmp2
                        t_calc.stop()
                        #char_trans_score = self.cost_seg[0][1] + self.cost_seg[1][2] # B-I and I-E
                        target = (sum([len(entry[2]) for entry in begins_prev_to_i]), num_nodes)
                        subst_node_chartran_BI_IE = True

                        for k in range(target[0], target[1]):
                            t_calc.start()
                            deltas_t[k] += char_trans_score
                            scores_t[k] += char_trans_score
                            t_calc.stop()

                if word_len >= 3:
                    t_refer.start()
                    char_trans_score = self.cost_seg[1][1] # I-I
                    t_refer.stop()
                    target = (sum([len(entry[2]) for entry in begins_prev_to_i]), num_nodes)

                    for k in range(target[0], target[1]):
                        t_calc.start()
                        deltas_t[k] += char_trans_score
                        scores_t[k] += char_trans_score
                        t_calc.stop()

                # edge score for char-transition b/w current and previous nodes
                prev_begins = lat.end2begins.get(s)
                if prev_begins:
                    # print('  prev:', prev_begins)
                    num_prev_nodes = sum([len(entry[2]) for entry in prev_begins])
                    ta.start()
                    edge_scores = [chainer.Variable(xp.array(0, dtype='f')) for k in range(num_prev_nodes)]
                    ta.stop()

                    first_pb = prev_begins[0]
                    if first_pb[1] - first_pb[0] == 1:
                        target_SX = (0, len(first_pb[2]))
                        target_EX = (target_SX[1], num_prev_nodes)
                    else:
                        target_SX = (0, num_prev_nodes)
                        target_EX = None

                    cur_first_tag = 3 if word_len == 1 else 0               # X = S or B
                    t_refer.start()
                    char_trans_score1 = self.cost_seg[3][cur_first_tag]     # S-X
                    t_refer.stop()

                    for k in range(target_SX[0], target_SX[1]):
                        t_calc.start()
                        edge_scores[k] += char_trans_score1
                        t_calc.stop()

                    if target_EX:
                        t_refer.start()
                        char_trans_score2 = self.cost_seg[2][cur_first_tag] # E-X
                        t_refer.stop()
                        for k in range(target_EX[0], target_EX[1]):
                            t_calc.start()
                            edge_scores[k] += char_trans_score2
                            t_calc.stop()

                    t_calc.start()
                    lse_prev_score = logsumexp(add(lat.deltas[s], edge_scores), xp=xp) # TODO
                    t_calc.stop()
                    t_calc.start()
                    deltas_t[i] += lse_prev_score
                    t_calc.stop()

                    #TODO 品詞ごとの個別処理

                    scores_s = lat.scores[s]
                    t_calc.start()
                    prev_node2part_score = add(scores_s, edge_scores)
                    t_calc.stop()                    

                    t_calc.start()
                    _, best_part_score = argmax_max(prev_node2part_score)
                    scores_t[i] += best_part_score
                    t_calc.stop()

                    t_calc.start()
                    prev_best, _ = argmax_max(prev_node2part_score)
                    t_calc.stop()
                    from_idx = sum([len(entry[2]) for entry in begins_prev_to_i])
                    to_idx = from_idx + len(pos_ids)
                    for j in range(from_idx, to_idx):
                        pp_t[j] = prev_best
                        # print(' ({},{})->{} | {}'.format(t, j, lat.prev_pointers[t][i], prev_begins))

                # gold score
                if gold_next and gold_next[0] == s and gold_next[1] == t and gold_next[2] in pos_ids:
                    gi = sum([len(entry[2]) for entry in begins_prev_to_i]) + pos_ids.index(gold_next[2])
                    t_calc.start()
                    gold_score += scores_t[gi]
                    t_calc.stop()

                    if gold_prev:
                        prev_last_tag = 3 if gold_prev[1] - gold_prev[0] == 1 else 0 # S or B
                        t_refer.start()
                        tmp = self.cost_seg[prev_last_tag][cur_first_tag]
                        t_refer.stop()
                        t_calc.start()
                        gold_score += tmp
                        t_calc.stop()
                    gold_prev = gold_next
                    gold_next_ind += 1
                    gold_next = gold_nodes[gold_next_ind] if gold_next_ind < len(gold_nodes) else None

                    if self.debug:
                        print("\n@ gold node {} score {} -> {}".format(
                            gold_next, lat.scores[t][gi], gold_score.data))

        t_calc.start()
        logz = logsumexp(lat.deltas[T], xp=xp)
        loss = logz - gold_score
        t_calc.stop()
        if self.debug:
            print(lat.deltas[T].data, 'lse->', logz.data)
            print(loss.data, '=', logz.data, '-', gold_score.data)
            
        t_total.stop()
        # print('refer:', t_refer.elapsed)
        # print('calc',   t_calc.elapsed)
        # print('  init  ', ta.elapsed)
        # print('  expand', te.elapsed)
        # print('  concat', tc.elapsed)
        # print('total', t_total.elapsed)
        # print()
        
        return loss


    # 動作未検証
    def forward2(self, lat, h, t_pos):
        t_refer = Timer()
        t_alloc = Timer()
        ta = Timer()
        te = Timer()
        tc = Timer()
        t_calc = Timer()
        t_total = Timer()

        t_total.start()

        xp = cuda.cupy if self.gpu >= 0 else np

        t_alloc.start()
        ta.start()
        gold_score = chainer.Variable(xp.array(0, dtype=np.float32))
        t_alloc.stop()
        ta.stop()
        gold_nodes = t_pos
        gold_next_ind = 0
        gold_next = gold_nodes[0]
        gold_prev = None

        if self.debug:
            print('\nnew instance:', lat.sen)
            print('lattice:', lat.eid2nodes)

        T = len(lat.sen)
        for t in range(1, T + 1):
            begins = lat.end2begins.get(t)
            num_nodes = sum([len(entry[2]) for entry in begins])

            smallest_s = begins[-1][0]
            subst_node_char = [False] * (t - smallest_s) # E(0), I(-1), ..., I(-N+1) の代入済みフラグ
            subst_node_chartran_BE = False
            subst_node_chartran_BI_IE = False
            # print('t:', t, begins)
            # print('smallest_s', smallest_s, subst_node_char)

            for i, entry in enumerate(begins):
                # print('  i:', i, entry)

                s = entry[0]    # begin index
                word_id = entry[1]
                pos_ids = entry[2]
                word_len = t - s

                # node score for char
                for cur_ind in range(s, t):
                    if cur_ind == s:
                        to_subst = True
                        if word_len == 1:                   # back=0
                            char_tag = 3                    # S
                            target = (0, len(pos_ids))      # (from, to)
                        else:                               # back>=1
                            char_tag = 0                    # B
                            target = (sum([len(entry[2]) for entry in begins[:i]]), len(pos_ids))
                    else:                                   # tag=I or E
                        cur_posi = t - cur_ind - 1
                        # print('posi', cur_posi, '|', s, cur_ind, t)
                        if not subst_node_char[cur_posi]:
                            to_subst = True
                            char_tag = 2 if cur_ind == t - 1 else 1 # E / I
                            target = (sum([len(entry[2]) for entry in begins[:i]]), num_nodes)
                            subst_node_char[cur_posi] = True
                        else:
                            to_subst = False

                    if to_subst:
                        t_refer.start()
                        char_score = h[cur_ind][char_tag]
                        t_refer.stop()
                        t_alloc.start()
                        char_scores = expand_variable(char_score, target[0], target[1], num_nodes, xp=xp, ta=ta, te=te, tc=tc)
                        t_alloc.stop()
                        t_calc.start()
                        lat.deltas[t] += char_scores
                        lat.scores[t] += char_scores
                        t_calc.stop()

                # node score for char-transition
                if word_len >= 2:
                    if not subst_node_chartran_BE:
                        t_refer.start()
                        char_trans_score = self.cost_seg[0][2] # B-E
                        t_refer.stop()
                        target = (sum([len(entry[2]) for entry in begins[:i]]), len(pos_ids))
                        subst_node_chartran_BE = True

                        t_alloc.start()
                        char_trans_scores = expand_variable(
                            char_trans_score, target[0], target[1], num_nodes, xp=xp, ta=ta, te=te, tc=tc)
                        t_alloc.stop()
                        t_calc.start()
                        lat.deltas[t] += char_trans_scores
                        lat.scores[t] += char_trans_scores
                        t_calc.stop()

                    if not subst_node_chartran_BI_IE:
                        t_refer.start()
                        tmp1 = self.cost_seg[0][1]
                        tmp2 = self.cost_seg[1][2]
                        t_refer.stop()
                        t_calc.start()
                        char_trans_score = tmp1 + tmp2
                        t_calc.stop()
                        #char_trans_score = self.cost_seg[0][1] + self.cost_seg[1][2] # B-I and I-E
                        target = (sum([len(entry[2]) for entry in begins[:i]]), num_nodes)
                        subst_node_chartran_BI_IE = True

                        t_alloc.start()
                        char_trans_scores = expand_variable(
                            char_trans_score, target[0], target[1], num_nodes, xp=xp, ta=ta, te=te, tc=tc)
                        t_alloc.stop()
                        t_calc.start()
                        lat.deltas[t] += char_trans_scores
                        lat.scores[t] += char_trans_scores
                        t_calc.stop()

                if word_len >= 3:
                    t_refer.start()
                    char_trans_score = self.cost_seg[1][1] # I-I
                    t_refer.stop()
                    target = (sum([len(entry[2]) for entry in begins[:i]]), num_nodes)
                    t_alloc.start()
                    char_trans_scores = expand_variable(
                        char_trans_score, target[0], target[1], num_nodes, xp=xp, ta=ta, te=te, tc=tc)
                    t_alloc.stop()
                    t_calc.start()
                    lat.deltas[t] += char_trans_scores
                    lat.scores[t] += char_trans_scores
                    t_calc.stop()

                # edge score for char-transition b/w current and previous nodes
                prev_begins = lat.end2begins.get(s)
                if prev_begins:
                    # print('  prev:', prev_begins)
                    num_prev_nodes = sum([len(entry[2]) for entry in prev_begins])

                    first_pb = prev_begins[0]
                    if first_pb[1] - first_pb[0] == 1:
                        target_SX = (0, len(first_pb[2]))
                        target_EX = (target_SX[1], num_prev_nodes)
                    else:
                        target_SX = (0, num_prev_nodes)
                        target_EX = None

                    cur_first_tag = 3 if word_len == 1 else 0               # X = S or B
                    t_refer.start()
                    char_trans_score1 = self.cost_seg[3][cur_first_tag]     # S-X
                    t_refer.stop()
                    t_alloc.start()
                    char_trans_scores1 = expand_variable(
                        char_trans_score1, target_SX[0], target_SX[1], num_prev_nodes, xp=xp, ta=ta, te=te, tc=tc)
                    t_alloc.stop()
                    if target_EX:
                        t_refer.start()
                        char_trans_score2 = self.cost_seg[2][cur_first_tag] # E-X
                        t_refer.stop()
                        t_alloc.start()
                        char_trans_scores2 = expand_variable(
                            char_trans_score2, target_EX[0], target_EX[1], num_prev_nodes, xp=xp, ta=ta, te=te, tc=tc)
                        t_alloc.stop()
                    else:
                        t_alloc.start()
                        char_trans_scores2 = chainer.Variable(xp.zeros(num_prev_nodes, dtype='f'))
                        t_alloc.stop()

                    t_calc.start()
                    edge_scores = char_trans_scores1 + char_trans_scores2
                    lse_prev_score = logsumexp(lat.deltas[s] + edge_scores)
                    t_calc.stop()
                    t_alloc.start()
                    lse_prev_scores_1 = expand_variable(lse_prev_score, i, i, num_nodes, xp=xp, ta=ta, te=te, tc=tc)
                    t_alloc.stop()
                    t_calc.start()
                    lat.deltas[t] += lse_prev_scores_1
                    t_calc.stop()

                    #TODO 品詞ごとの個別処理

                    t_calc.start()
                    prev_node2part_score = lat.scores[s] + edge_scores
                    best_part_score = minmax.max(prev_node2part_score)
                    t_calc.stop()
                    t_alloc.start()
                    best_part_scores_1 = expand_variable(best_part_score, i, i, num_nodes, xp=xp, ta=ta, te=te, tc=tc)
                    t_alloc.stop()
                    t_calc.start()
                    lat.scores[t] += best_part_scores_1
                    t_calc.stop()

                    t_refer.start()
                    prev_best = minmax.argmax(prev_node2part_score).data
                    t_refer.stop()
                    from_idx = sum([len(entry[2]) for entry in begins[:i]])
                    to_idx = from_idx + len(pos_ids)
                    for j in range(from_idx, to_idx):
                        lat.prev_pointers[t][j] = prev_best
                        # print(' ({},{})->{} | {}'.format(t, j, lat.prev_pointers[t][i], prev_begins))

                # gold score
                if gold_next and gold_next[0] == s and gold_next[1] == t and gold_next[2] in pos_ids:
                    gi = sum([len(entry[2]) for entry in begins[:i]]) + pos_ids.index(gold_next[2])
                    t_refer.start()
                    tmp = lat.scores[t][gi]
                    t_refer.stop()
                    t_calc.start()
                    gold_score += tmp
                    t_calc.stop()

                    if gold_prev:
                        prev_last_tag = 3 if gold_prev[1] - gold_prev[0] == 1 else 0 # S or B
                        t_refer.start()
                        tmp = self.cost_seg[prev_last_tag][cur_first_tag]
                        t_refer.stop()
                        t_calc.start()
                        gold_score += tmp
                        t_calc.stop()
                    gold_prev = gold_next
                    gold_next_ind += 1
                    gold_next = gold_nodes[gold_next_ind] if gold_next_ind < len(gold_nodes) else None

                    if self.debug:
                        print("\n@ gold node {} score {} -> {}".format(
                            gold_next, lat.scores[t][gi], gold_score.data))

        t_calc.start()
        logz = logsumexp(lat.deltas[T])
        loss = logz - gold_score
        t_calc.stop()
        if self.debug:
            print(lat.deltas[T].data, 'lse->', logz.data)
            print(loss.data, '=', logz.data, '-', gold_score.data)

        t_total.stop()
        print('refer:', t_refer.elapsed)
        print('calc',   t_calc.elapsed)
        print('alloc', t_alloc.elapsed)
        print('  init  ', ta.elapsed)
        print('  expand', te.elapsed)
        print('  concat', tc.elapsed)
        print('total', t_total.elapsed)
        print()
        
        return loss
    

    def forward(self, lat, h, t_pos):
        t_refer = Timer()
        t_alloc = Timer()
        ta = Timer()
        te = Timer()
        tc = Timer()
        t_calc = Timer()
        t_total = Timer()

        t_total.start()

        xp = cuda.cupy if self.gpu >= 0 else np

        t_alloc.start()
        ta.start()
        gold_score = Variable(xp.array(0, dtype=np.float32))
        t_alloc.stop()
        ta.stop()
        gold_nodes = t_pos

        if self.debug:
            print('\nnew instance:', lat.sen)
            print('lattice:', lat.eid2nodes)

        T = len(lat.sen)
        for t in range(1, T + 1):
            nodes = lat.eid2nodes.get(t)
            len_nodes = len(nodes)
            for i, node in enumerate(nodes):
                #print('in:', t, i, node)

                s = node[0]
                prev_nodes = lat.eid2nodes.get(s)
                node_score = self.calc_node_score(node, h, tr=t_refer, ts=t_calc)
                t_alloc.start()
                node_scores_1 = expand_variable(node_score, i, i+1, len_nodes, xp=xp, ta=ta, te=te, tc=tc)
                t_alloc.stop()
                is_gold_node = node in gold_nodes
                if is_gold_node:
                    t_calc.start()
                    gold_score += node_score
                    t_calc.stop()
                    if self.debug:
                        print("\n@ gold node {} score {} -> {}".format(
                            node, node_score.data, gold_score.data))

                t_calc.start()
                lat.deltas[t] += node_scores_1
                lat.scores[t] += node_scores_1
                t_calc.stop()
                if self.debug:
                    print("\n[{},{}] {} <- {}".format(t, i, node, prev_nodes))
                    print("  nscores", node_scores_1.data)
                    print("  delta[{}]".format(t), lat.deltas[t].data)

                if prev_nodes:
                    edge_scores = [None] * len(prev_nodes)
                    for j, prev_node in enumerate(prev_nodes):
                        edge_score = self.calc_edge_score(prev_node, node, tr=t_refer, ts=t_calc)
                        te.start()
                        edge_scores[j] = F.expand_dims(edge_score, axis=0)
                        te.stop()
                    tc.start()
                    edge_scores = F.concat(edge_scores, axis=0)
                    tc.stop()

                    # F.concat([F.expand_dims(self.calc_edge_score(prev_node, node), axis=0)
                    #   for prev_node in prev_nodes], axis=0)
                    
                    if is_gold_node:
                        for k in range(len(prev_nodes)):
                            if prev_nodes[k] in gold_nodes:
                                t_refer.start()
                                tmp = edge_scores[k]
                                t_refer.stop()
                                t_calc.start()
                                gold_score += tmp
                                t_calc.stop()
                                if self.debug:
                                    print("  @ gold prev node {} score {} -> {}".format(
                                        prev_nodes[k], edge_scores[k].data, gold_score.data))
                                break

                    t_calc.start()
                    lse_prev_score = logsumexp.logsumexp(lat.deltas[s] + edge_scores)
                    t_calc.stop()
                    if self.debug:
                        print("  escores {} | delta[{}] {} | lse=> {}".format(
                            edge_scores.data, s, lat.deltas[s].data, lse_prev_score.data))

                    lse_prev_scores_1 = expand_variable(lse_prev_score, i, i+1, len_nodes, xp=xp, ta=ta, te=te, tc=tc)
                    t_calc.start()
                    lat.deltas[t] += lse_prev_scores_1
                    t_calc.stop()
                    if self.debug:
                        print("  delta[{}]".format(t), lat.deltas[t].data)

                    t_calc.start()
                    prev_node2part_score = lat.scores[s] + edge_scores
                    t_calc.stop()
                    best_part_scores_1 = expand_variable(
                        minmax.max(prev_node2part_score), i, i+1, len_nodes, xp=xp, ta=ta, te=te, tc=tc)
                    t_calc.start()
                    #print('t={},i={},node={}'.format(t, i, node))
                    lat.prev_pointers[t][i] = minmax.argmax(prev_node2part_score).data
                    lat.scores[t] += best_part_scores_1
                    t_calc.stop()
                if self.debug:
                    print("  scores[{}]".format(t), lat.scores[t].data)

        t_calc.start()
        logz = logsumexp.logsumexp(lat.deltas[T])
        loss = logz - gold_score
        t_calc.stop()
        if self.debug:
            print(lat.deltas[T].data, 'lse->', logz.data)
            print(loss.data, '=', logz.data, '-', gold_score.data)

        t_total.stop()
        # print('refer:', t_refer.elapsed)
        # print('calc',   t_calc.elapsed)
        # print('alloc', t_alloc.elapsed)
        # print('  init  ', ta.elapsed)
        # print('  expand', te.elapsed)
        # print('  concat', tc.elapsed)
        # print('total', t_total.elapsed)
        # print()

        return loss


    def argmax3(self, lat):
        xp = cuda.cupy if self.gpu >= 0 else np

        y_seg = []
        y_pos = []
        wis = []

        t = len(lat.sen)
        prev_best, _ = argmax_max(lat.scores[t])
        # for k in range(1, t+1):
        #     print('prev ptr', k, ':', lat.prev_pointers[k])

        while t > 0:
            begins = lat.end2begins.get(t)

            i = j = prev_best
            for begin in begins:
                if j >= len(begin[2]):
                    j -= len(begin[2])
                    continue
                else:
                    s = begin[0]
                    wi = begin[1]
                    pos = begin[2][j]
                    node = (s, t, pos)
                    break

            # print('{} {} ({}): {} | {}'.format(t, i, prev_best, node, begins))
            prev_best = lat.prev_pointers[t][i]
            # print('t', t, begins)
            # print('i', i, node_info)

            # s = node_info[0]
            # wi = node_info[1]
            # j = sum([len(entry[2]) for entry in begins[:i]])
            # print(j, begins[:i])
            # pos = node_info[2][i-j]

            # update lists
            y_pos.append(node)
            wis.append(wi)
            if t - s == 1:
                tag = 3 # S
                #y_seg.insert(0, tag)
                y_seg.append(tag)
            else:
                insert_to = 0
                for j in range(t-1, s-1, -1):
                    if j == s:
                        tag = 0 # B
                    elif j == t - 1:
                        tag = 2 # E
                    else:
                        tag = 1 # I
                    y_seg.append(tag)

            t = s

        return y_seg, y_pos, wis


    # 動作未検証
    def argmax2(self, lat):
        xp = cuda.cupy if self.gpu >= 0 else np

        y_seg = []
        y_pos = []
        wis = []

        t = len(lat.sen)
        
        prev_best = int(xp.argmax(lat.scores[t].data))
        # for k in range(1, t+1):
        #     print('prev ptr', k, ':', lat.prev_pointers[k])

        while t > 0:
            begins = lat.end2begins.get(t)

            i = j = prev_best
            for begin in begins:
                if j >= len(begin[2]):
                    j -= len(begin[2])
                    continue
                else:
                    s = begin[0]
                    wi = begin[1]
                    pos = begin[2][j]
                    node = (s, t, pos)
                    break
            # print('{} {} ({}): {} | {}'.format(t, i, prev_best, node, begins))
            prev_best = lat.prev_pointers[t][i]
            # print('t', t, begins)
            # print('i', i, node_info)

            # s = node_info[0]
            # wi = node_info[1]
            # j = sum([len(entry[2]) for entry in begins[:i]])
            # print(j, begins[:i])
            # pos = node_info[2][i-j]

            # update lists
            y_pos.append(node)
            wis.append(wi)
            if t - s == 1:
                tag = 3 # S
                #y_seg.insert(0, tag)
                y_seg.append(tag)
            else:
                insert_to = 0
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


    def argmax(self, lat):
        xp = cuda.cupy if self.gpu >= 0 else np

        y_seg = []
        y_pos = []
        words = []

        t = len(lat.sen)
        prev_best = int(xp.argmax(lat.scores[t].data))
        if self.debug:
            node = lat.eid2nodes.get(t)[prev_best]
            print('+ {},{}: {} ... {}'.format(t, prev_best, node, lat.scores[t].data))
        while t > 0:
            i = prev_best
            prev_best = lat.prev_pointers[t][i]
            node = lat.eid2nodes.get(t)[i]
            if self.debug:
                print('+ {},{}: {} prev-> {},{}'.format(t, i, node, node[0], prev_best))

            # update lists
            y_pos.insert(0, node)
            
            if lattice.node_len(node) == 1:
                tag = 3 #self.morph_dic.get_label_id('S')
                y_seg.insert(0, tag)
                ti = int(lat.sen[node[0]])
                word = self.morph_dic.get_token(ti)
            else:
                word = ''
                insert_to = 0
                for i in range(node[0], node[1]):
                    if i == node[0]:
                        tag = 0 #self.morph_dic.get_label_id('B')
                    elif i == node[1] - 1:
                        tag = 2 #self.morph_dic.get_label_id('E')
                    else:
                        tag = 1 #self.morph_dic.get_label_id('I')

                    y_seg.insert(insert_to, tag)
                    insert_to += 1

                    ti = int(lat.sen[i])
                    word += self.morph_dic.get_token(ti)

            words.insert(0, word)

            t = node[0]

        return y_seg, y_pos, words


    def calc_node_score(self, node, hidden_vecs, tr=None, ts=None):
        # BIES スキーマのみ対応
        if lattice.node_len(node) == 1:
            tag = 3 #self.morph_dic.get_label_id('S')
            if tr:
                tr.start()
            score = hidden_vecs[node[0]][tag]
            if tr:
                tr.stop()
            if self.debug:
                print('* N1 score for {}-th elem of {}, {}: {} -> {}'.format(
                    0, node, tag, hidden_vecs[node[0]][tag].data, score.data))

        else:
            score = None
            prev_tag = None
            for i in range(node[0], node[1]):
                if i == node[0]:
                    tag = 0 #self.morph_dic.get_label_id('B')
                elif i == node[1] - 1:
                    tag = 2 #self.morph_dic.get_label_id('E')
                else:
                    tag = 1 #self.morph_dic.get_label_id('I')

                if tr:
                    tr.start()
                score = (hidden_vecs[i][tag] + score) if score is not None else hidden_vecs[i][tag]
                if tr:
                    tr.stop()
                if self.debug:
                    print('* N2 score for {}-th elem of {}, {}: {} -> {}'.format(
                        i, node, tag, hidden_vecs[i][tag].data, score.data))
                
                if i > node[0]:
                    if tr:
                        tr.start()
                    cost = self.cost_seg[prev_tag][tag]
                    if tr:
                        tr.stop()
                    if ts:
                        ts.start()
                    score += cost
                    if ts:
                        ts.stop()
                    if self.debug:
                        print('* NE score ({}) to ({}): {} -> {}'.format(
                            prev_tag, tag, cost.data, score.data))

                prev_tag = tag

        return score


    def calc_edge_score(self, prev_node, next_node, tr=None, ts=None):
        # BIES スキーマのみ対応
        if lattice.node_len(prev_node) == 1:
            prev_last_tag = 3 #self.morph_dic.get_label_id('S')
        else:
            prev_last_tag = 2 #self.morph_dic.get_label_id('E')

        if lattice.node_len(next_node) == 1:
            next_first_tag = 3 #self.morph_dic.get_label_id('S')
        else:
            next_first_tag = 0 #self.morph_dic.get_label_id('B')

        if tr:
            tr.start()
        score = self.cost_seg[prev_last_tag][next_first_tag]
        if tr:
            tr.stop()

        if self.predict_pos:
            if tr:
                tr.start()
            tmp = self.cost_pos[prev_node[2]][next_node[2]]
            if tr:
                tr.stop()
            if ts:
                ts.start()
            score += tmp
            if ts:
                ts.stop()

        if self.debug:
            print('* E score for {} ({}) and {} ({}): {}'.format(
                prev_node, prev_last_tag, next_node, next_first_tag, score.data))
            if self.predict_pos:
                print('* E score for {} and {}: {}'.format(prev_node, next_node, score.data))
        return score


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


    # def get_char_score(self, cur_ind, begin_ind, end_ind, hidden_vecs):
    #     len_word = end_ind - begin_ind
        
    #     if len_word == 1:
    #         tag = 3 #self.morph_dic.get_label_id('S')
    #     else:
    #         if cur_ind == begin_ind:
    #             tag = 0 #self.morph_dic.get_label_id('B')
    #         elif cur_ind == end_ind - 1:
    #             tag = 2 #self.morph_dic.get_label_id('E')
    #         else:
    #             tag = 1 #self.morph_dic.get_label_id('I')

    #     score = hidden_vecs[cur_ind][tag]
    #     if self.debug:
    #         print('* N1 score for {} of {}, {}: {} -> {}'.format(
    #             cur_ind, (begin_index, end_index), tag, hidden_vecs[cur_ind][tag].data, score.data))

    #     return score


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


    def generate_lines(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                raise StopIteration
            x_str = str(x[i])
            t_str = self.id2label[int(t[i])]
            y_str = self.id2label[int(y[i])]

            yield [x_str, t_str, y_str]

            i += 1


    def grow_lookup_table(self, token2id_new, gpu=-1):
        n_left_contexts = len(self.predictor.left_padding_ids)
        n_right_contexts = len(self.predictor.right_padding_ids)
        padding_size = n_left_contexts + n_right_contexts


        weight1 = self.predictor.lookup.W if padding_size > 0 else self.predictor.lookup.W[-padding_size:]
        diff = len(token2id_new) - len(weight1)
        weight2 = chainer.variable.Parameter(initializers.normal.Normal(1.0), (diff, weight1.shape[1]))
        weight = F.concat((weight1, weight2), 0)
        if padding_size > 0:
            n_vocab = len(weight)
            self.predictor.left_padding_ids = predictor.get_id_array(n_vocab, n_left_contexts, gpu)
            self.predictor.right_padding_ids = predictor.get_id_array(n_vocab + n_left_contexts, 
                                                                      n_right_contexts, gpu)
            weight3 = predictor.lookup.W[:-padding_size]
            weight = F.concat((weight, weight3), 0)

        embed = L.EmbedID(0, 0)
        embed.W = chainer.Parameter(initializer=weight.data)
        self.predictor.lookup = embed

        print('# grow vocab size: %d -> %d' % (weight1.shape[0], weight.shape[0]))
        print('# lookup:', self.predictor.lookup.W.shape)


class JointMorphologicalAnalyzer(SequenceTagger):
    def __init(self, predictor):
        super(JointMorphologicalAnalyzer, self).__init__(predictor=predictor)


    def __init__(self, predictor, id2label):
        super(SequenceTagger, self).__init__(predictor=predictor)
        self.id2label = id2label

        
    def __call__(self, *args, **kwargs):
        assert len(args) >= 2
        xs = args[0]
        ts_seg = args[1]
        ts_pos = args[2]
        loss, ys_seg, ys_pos = self.predictor(*args, **kwargs)

        if self.compute_fscore:
            ecounts_seg = None
            ecounts_pos = Counter()
            for x, t_seg, t_pos, y_seg, y_pos in zip(xs, ts_seg, ts_pos, ys_seg, ys_pos):
                generator = self.generate_lines(x, t_seg, y_seg)
                ecounts_seg = conlleval.merge_counts(ecounts_seg, conlleval.evaluate(generator))
                ecounts_pos['all'] += len(t_pos)
                ecounts_pos['correct'] += len(t_pos) - len(set(t_pos) - set(y_pos))

        return loss, ecounts_seg, ecounts_pos


    def decode(self, sen, pos_seq):
        return decode(sen, pos_seq, self.morph_dic)
        

def decode(sen, pos_seq, morph_dic):
    seq = []

    for node in pos_seq:
        if lattice.node_len(node) == 1:
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

    
# def expand_variable(var, index, size, xp=np):
#     var_ext = [F.expand_dims(var if i == index else chainer.Variable(xp.array(0, dtype=np.float32)), axis=0) for i in range(size)]
#     var_ext = F.concat(var_ext, axis=0)
#     return var_ext


#def expand_variable(var, ind_from, ind_to, size, xp=np):
def expand_variable(var, ind_from, ind_to, size, xp=np, ta=None, te=None, tc=None):
    var_ext = [None] * size
    for i in range(0, ind_from):
        if ta:
            ta.start()
        tmp = chainer.Variable(xp.array(0, dtype=np.float32))
        if ta:
            ta.stop()
        if te:
            te.start()
        var_ext[i] = F.expand_dims(tmp, axis=0)
        if te:
            te.stop()
    for i in range(ind_from, ind_to):
        if te:
            te.start()
        var_ext[i] = F.expand_dims(var, axis=0)
        if te:
            te.stop()
    for i in range(ind_to, size):
        if ta:
            ta.start()
        tmp = chainer.Variable(xp.array(0, dtype=np.float32))
        if ta:
            ta.stop()
        if te:
            te.start()
        var_ext[i] = F.expand_dims(tmp, axis=0)
        if te:
            te.stop()

    #var_ext = [F.expand_dims(var if (ind_from <= i and i < ind_to) else chainer.Variable(xp.array(0, dtype=np.float32)), axis=0) for i in range(size)]

    if tc:
        tc.start()
    var_ext = F.concat(var_ext, axis=0)
    if tc:
        tc.stop()
    return var_ext
