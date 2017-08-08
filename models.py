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

from chainer.functions.math.logsumexp import logsumexp
from chainer.functions.math import minmax

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.functions.loss import softmax_cross_entropy
from chainer.links.connection.n_step_rnn import argsort_list_descent, permutate_list
from chainer import initializers

from eval.conlleval import conlleval
import lattice.lattice as lattice


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
            self.empty_array = cuda.cupy.array([], dtype=np.float32) if gpu >= 0 else np.array([], dtype=np.float32)
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
            else:
                self.cost_pos = None

        print('# crf cost_seg:', self.cost_seg.shape)
        print('# crf cost_pos:', self.cost_pos.shape if self.predict_pos else None)
        print()

        self.debug=False


    def __call__(self, xs, hs, ts_pos):
        xp = cuda.cupy if self.gpu >= 0 else np

        ys_seg = []
        ys_pos = []
        loss = chainer.Variable(xp.array(0, dtype=np.float32))

        i = 0

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
            # print('  create lattice: {}'.format((t1-t0).seconds+(t1-t0).microseconds/10**6))
            # print('  prepare fw    : {}'.format((t2-t1).seconds+(t2-t1).microseconds/10**6))
            # print('  forward       : {}'.format((t3-t2).seconds+(t3-t2).microseconds/10**6))
            # print('  argmax        : {}'.format((t4-t3).seconds+(t4-t3).microseconds/10**6))

            # print("\ninstance", i)
            # i += 1
            # print('y:', decode(x, y_pos, self.morph_dic))
            # print('t:', decode(x, t_pos, self.morph_dic))
            # print([self.morph_dic.get_token(ti) for ti in x)
            # print(y_seg)
            # print(y_pos)
            # print(words)

            ys_seg.append(y_seg)
            ys_pos.append(y_pos)

        return loss, ys_seg, ys_pos


    def forward(self, lat, h, t_pos):

        xp = cuda.cupy if self.gpu >= 0 else np

        gold_score = chainer.Variable(xp.array(0, dtype=np.float32))
        gold_nodes = t_pos

        if self.debug:
            print('\nnew instance:', lat.sen)
            print('lattice:', lat.eid2nodes)

        T = len(lat.sen)
        for t in range(1, T + 1):
            nodes = lat.eid2nodes.get(t)
            for i in range(len(nodes)):
                len_nodes = len(nodes)
                node = nodes[i]
                s = node[0]
                prev_nodes = lat.eid2nodes.get(s)
                node_score = self.calc_node_score(node, h)
                #node_scores_b = F.broadcast_to(node_score, (len(nodes),))
                node_scores_1 = expand_variable(node_score, i, len_nodes, xp=xp)
                is_gold_node = node in gold_nodes
                if is_gold_node:
                    gold_score += node_score
                    if self.debug:
                        print("\n@ gold node {} score {} -> {}".format(
                            node, node_score.data, gold_score.data))

                #lat.deltas[t] += node_scores_b # incorrect!
                lat.deltas[t] += node_scores_1
                lat.scores[t] += node_scores_1
                if self.debug:
                    print("\n[{},{}] {} <- {}".format(t, i, node, prev_nodes))
                    print("  nscores", node_scores_1.data)
                    print("  delta[{}]".format(t), lat.deltas[t].data)

                if prev_nodes:
                    edge_scores = F.concat(
                        [F.expand_dims(self.calc_edge_score(prev_node, node), axis=0) 
                         for prev_node in prev_nodes], axis=0)
                    
                    if is_gold_node:
                        for j in range(len(prev_nodes)):
                            if prev_nodes[j] in gold_nodes:
                                gold_score += edge_scores[j]
                                if self.debug:
                                    print("  @ gold prev node {} score {} -> {}".format(
                                        prev_nodes[j], edge_scores[j].data, gold_score.data))
                                break

                    lse_prev_score = logsumexp(lat.deltas[s] + edge_scores)
                    if self.debug:
                        print("  escores {} | delta[{}] {} | lse=> {}".format(
                            edge_scores.data, s, lat.deltas[s].data, lse_prev_score.data))

                    lse_prev_scores_1 = expand_variable(lse_prev_score, i, len_nodes, xp=xp)
                    lat.deltas[t] += lse_prev_scores_1
                    if self.debug:
                        print("  delta[{}]".format(t), lat.deltas[t].data)

                    prev_node2part_score = lat.scores[s] + edge_scores
                    best_part_scores_1 = expand_variable(minmax.max(prev_node2part_score), i, len_nodes, xp=xp)
                    lat.prev_pointers[t][i] = minmax.argmax(prev_node2part_score).data
                    lat.scores[t] += best_part_scores_1
                if self.debug:
                    print("  scores[{}]".format(t), lat.scores[t].data)

        logz = logsumexp(lat.deltas[T])
        loss = logz - gold_score
        if self.debug:
            print(lat.deltas[T].data, 'lse->', logz.data)
            print(loss.data, '=', logz.data, '-', gold_score.data)

        return loss


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


    def calc_node_score(self, node, hidden_vecs):
        # BIES スキーマのみ対応
        if lattice.node_len(node) == 1:
            tag = 3 #self.morph_dic.get_label_id('S')
            score = hidden_vecs[node[0]][tag]
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

                score = (hidden_vecs[i][tag] + score) if score is not None else hidden_vecs[i][tag]
                if self.debug:
                    print('* N2 score for {}-th elem of {}, {}: {} -> {}'.format(
                        i, node, tag, hidden_vecs[i][tag].data, score.data))
                
                if i > node[0]:
                    cost = self.cost_seg[prev_tag][tag]
                    score += cost
                    if self.debug:
                        print('* NE score ({}) to ({}): {} -> {}'.format(
                            prev_tag, tag, cost.data, score.data))

                prev_tag = tag

        return score


    def calc_edge_score(self, prev_node, next_node):
        # BIES スキーマのみ対応
        if lattice.node_len(prev_node) == 1:
            prev_last_tag = 3 #self.morph_dic.get_label_id('S')
        else:
            prev_last_tag = 2 #self.morph_dic.get_label_id('E')

        if lattice.node_len(next_node) == 1:
            next_first_tag = 3 #self.morph_dic.get_label_id('S')
        else:
            next_first_tag = 0 #self.morph_dic.get_label_id('B')
        score = self.cost_seg[prev_last_tag][next_first_tag]

        if self.predict_pos:
            score += self.cost_pos[prev_node[2]][next_node[2]]

        if self.debug:
            print('* E score for {} ({}) and {} ({}): {}'.format(
                prev_node, prev_last_tag, next_node, next_first_tag, score.data))
            if self.predict_pos:
                print('* E score for {} and {}: {}'.format(prev_node, next_node, score.data))
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

        if self.compute_fscore:
            eval_counts = None
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


def expand_variable(var, index, size, xp=np):
    var_ext = [F.expand_dims(var if i == index else chainer.Variable(xp.array(0, dtype=np.float32)), axis=0) for i in range(size)]
    var_ext = F.concat(var_ext, axis=0)
    return var_ext
