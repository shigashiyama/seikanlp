import sys

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import initializers

import numpy as np


class MLP(chainer.Chain):
    def __init__(self, n_input, n_units, n_hidden_units=0, n_layers=1, output_activation=F.relu, dropout=0):
        super().__init__()
        with self.init_scope():
            layers = [None] * n_layers
            self.acts = [None] * n_layers
            n_hidden_units = n_hidden_units if n_hidden_units > 0 else n_units

            for i in range(n_layers):
                if i == 0:
                    n_left = n_input
                    n_right = n_units if n_layers == 1 else n_hidden_units
                    act = output_activation if n_layers == 1 else F.relu

                elif i == n_layers - 1:
                    n_left = n_hidden_units
                    n_right = n_units
                    act = output_activation

                else:
                    n_left = n_right = n_hidden_units
                    act = F.relu

                layers[i] = L.Linear(n_left, n_right)
                self.acts[i] = act
            
            self.layers = chainer.ChainList(*layers)
            self.dropout = dropout

        for i in range(n_layers):
            print('#   Affine {}-th layer:                 W={}, b={}, dropout={}, act={}'.format(
                i, self.layers[i].W.shape, self.layers[i].b.shape, self.dropout, 
                self.acts[i].__name__), file=sys.stderr)


    def __call__(self, xs, start_index=0, per_element=True):
        hs_prev = xs
        hs = None

        if per_element:
            for i in range(len(self.layers)):
                hs = [self.acts[i](
                    self.layers[i](
                        F.dropout(h_prev, self.dropout))) for h_prev in hs_prev]
                hs_prev = hs

        else:
            for i in range(len(self.layers)):
                hs = self.acts[i](
                    self.layers[i](
                        F.dropout(hs_prev, self.dropout)))
                hs_prev = hs

        return hs


class BiaffineCombination(chainer.Chain):
    def __init__(self, left_size, right_size, use_U=False, use_V=False, use_b=False):
        super().__init__()
        
        with self.init_scope():
            initialW = None
            w_shape = (left_size, right_size)
            self.W = chainer.variable.Parameter(initializers._get_initializer(initialW), w_shape)

            if use_U:
                initialU = None
                u_shape = (left_size, 1)
                self.U = chainer.variable.Parameter(initializers._get_initializer(initialU), u_shape)
            else:
                self.U = None

            if use_V:
                initialV = None
                v_shape = (1, right_size)
                self.V = chainer.variable.Parameter(initializers._get_initializer(initialV), v_shape)
            else:
                self.V = None

            if use_b:
                initialb = 0
                b_shape = 1
                self.b = chainer.variable.Parameter(initialb, b_shape)
            else:
                self.b = None

    def __call__(self, x1, x2):
        # inputs: x1 = [x1_1 ... x1_i ... x1_n1]; dim(x1_i)=d1=left_size
        #         x2 = [x2_1 ... x2_j ... x2_n2]; dim(x2_j)=d2=right_size
        # output: o_ij = x1_i * W * x2_j + x2_j * U + b

        n1 = x1.shape[0]
        n2 = x2.shape[0]
        x2T = F.transpose(x2)
        x1_W = F.matmul(x1, self.W)                       # (n1, d1) * (d1, d2) => (n1, d2)
        res = F.matmul(x1_W, x2T)                         # (n1, d2) * (d2, n2) => (n1, n2)

        if self.U is not None:
            x1_U = F.broadcast_to(F.matmul(x1, self.U), (n1, n2)) # (n1, d1) * (d1, 1)  => (n1, 1) -> (n1, n2)
            # print('x1*U', x1_U.shape)
            res = res + x1_U

        if self.V is not None: # TODO fix
            V_x2 = F.broadcast_to(F.matmul(self.V, x2T), (n1, n2)) # (1, d2) * (d2, n2) => (1, n2) -> (n1, n2)
            res = res + V_x2

        if self.b is not None:
            b = F.broadcast_to(self.b, (n1, n2))
            res = res + b

        return res


# implementation of Variational RNN in https://arxiv.org/abs/1512.05287
class VariationalLSTM(chainer.Chain):
    def __init__(self,
                 bidirection,
                 n_layers,
                 n_inputs,
                 n_units,
                 dropout=0,
    ):
        super().__init__()
        with self.init_scope():
            self.bidirection = bidirection
            self.n_layers = n_layers
            self.dropout = dropout
            fw_layers = [None] * n_layers
            bw_layers = [None] * n_layers if bidirection else None

            # 1st layer
            for i in range(n_layers):
                if i == 0:
                    n_in = n_inputs
                elif not bidirection:
                    n_in = n_units
                else:
                    n_in = n_units * 2
                n_out = n_units

                # print(n_in, n_out)
                fw_layers[i] = L.StatelessLSTM(n_in, n_out)
                if self.bidirection:
                    bw_layers[i] = L.StatelessLSTM(n_in, n_out)

            self.fw_layers = chainer.ChainList(*fw_layers)
            self.bw_layers = chainer.ChainList(*bw_layers) if self.bidirection else None

            for i in range(n_layers):
                print('#   {}-th forward : lateral.W={} upward.W={}'.format(
                    i,
                    self.fw_layers[i].lateral.W.shape,
                    self.fw_layers[i].upward.W.shape),
                    file=sys.stderr)

                if self.bidirection:
                    print('#   {}-th backward: lateral.W={} upward.W={}'.format(
                        i,
                        self.bw_layers[i].lateral.W.shape,
                        self.bw_layers[i].upward.W.shape),
                          file=sys.stderr)


    def __call__(self, xs, padding_mask=None): 
        vs = xs        # xs: (n, b, d) ... max_sen_len, batch, emb_dim
        for i in range(self.n_layers):
            hs_fw = self.rnn_steps(True, i, vs, padding_mask)

            if self.bidirection:
                hs_bw = self.rnn_steps(False, i, vs, padding_mask)
                vs = [F.concat((hs_fw_t, hs_bw_t), axis=1) for hs_fw_t, hs_bw_t in zip(hs_fw, hs_bw)]
            else:
                vs = hs_fw

        return vs


    def rnn_steps(self, is_forward, layer_index, vs, padding_mask=None):
        xp = cuda.get_array_module(vs[0])

        hs = []
        cs_t_prev = hs_t_prev = None
        dmask_v = dmask_h = None # dropout mask
        lstm = self.fw_layers[layer_index] if is_forward else self.bw_layers[layer_index]
        directed_range = range(0, len(vs)) if is_forward else range(len(vs)-1, -1, -1)

        for t in directed_range:
            vs_t = vs[t]
            if 0.0 < self.dropout < 1.0:
                if dmask_v is None:
                    dmask_v = self.gen_dropout_mask(vs_t.shape, xp=xp)
                if dmask_h is None and hs_t_prev is not None:
                    dmask_h = self.gen_dropout_mask(hs_t_prev.shape, xp=xp)
                cs_t, hs_t = lstm(
                    cs_t_prev, 
                    dmask_h * hs_t_prev if (hs_t_prev is not None) else None, 
                    dmask_v * vs_t)
            else:
                cs_t, hs_t = lstm(cs_t_prev, hs_t_prev, vs_t)

            pmask_t = padding_mask[t]
            cs_t *= pmask_t
            hs_t *= pmask_t
            cs_t_prev = cs_t 
            hs_t_prev = hs_t 
            hs.append(hs_t)

        if not is_forward:
            hs.reverse()

        return hs


    def gen_dropout_mask(self, shape, xp=np):
        scale = np.float32(1. / (1 - self.dropout))
        flag = xp.random.rand(*shape) >= self.dropout
        return scale * flag
