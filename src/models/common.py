import sys

import chainer
import chainer.functions as F
import chainer.links as L


class MLP(chainer.Chain):
    def __init__(self, n_input, n_units, n_hidden_units=0, n_layers=1, output_activation=F.relu, 
                 dropout=0, file=sys.stderr):
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
                self.acts[i].__name__), file=file)


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
