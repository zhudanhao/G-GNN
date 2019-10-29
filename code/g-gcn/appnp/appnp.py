"""
APPNP implementation in DGL.
References
----------
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""
import torch
import torch.nn as nn
import dgl.function as fn


class GraphPropagation(nn.Module):
    def __init__(self,
                 g,
                 edge_drop,
                 alpha,
                 k):
        super(GraphPropagation, self).__init__()
        self.g = g
        self.alpha = alpha
        self.k = k
        if edge_drop:
            self.edge_drop = nn.Dropout(edge_drop)
        else:
            self.edge_drop = 0.

    def forward(self, h):
        self.cached_h = h
        for _ in range(self.k):
            # normalization by square root of src degree
            h = h * self.g.ndata['norm']
            self.g.ndata['h'] = h
            if self.edge_drop:
                # performing edge dropout
                ed = self.edge_drop(torch.ones((self.g.number_of_edges(), 1), device=h.device))
                self.g.edata['d'] = ed
                self.g.update_all(fn.src_mul_edge(src='h', edge='d', out='m'),
                                  fn.sum(msg='m', out='h'))
            else:
                self.g.update_all(fn.copy_src(src='h', out='m'),
                                  fn.sum(msg='m', out='h'))
            h = self.g.ndata.pop('h')
            # normalization by square root of dst degree
            h = h * self.g.ndata['norm']
            # update h using teleport probability alpha
            h = h * (1 - self.alpha) + self.cached_h * self.alpha
        return h


class APPNP(nn.Module):
    def __init__(self,
                 g,
                 in_feats1,
                 in_feats2,
                 in_feats3,
                 hiddens,
                 n_classes,
                 activation,
                 feat_drop,
                 edge_drop,
                 alpha,
                 k,
                 alpha1,
                 alpha2,
                 alpha3):
        super(APPNP, self).__init__()
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        # input layer
        self.layers1.append(nn.Linear(in_feats1, hiddens[0]))
        self.layers2.append(nn.Linear(in_feats2, hiddens[0]))
        self.layers3.append(nn.Linear(in_feats3, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers1.append(nn.Linear(hiddens[i - 1], hiddens[i]))
            self.layers2.append(nn.Linear(hiddens[i - 1], hiddens[i]))
            self.layers3.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers1.append(nn.Linear(hiddens[-1], n_classes))
        self.layers2.append(nn.Linear(hiddens[-1], n_classes))
        self.layers3.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate1 = GraphPropagation(g, edge_drop, alpha, k)
        self.propagate2 = GraphPropagation(g, edge_drop, alpha, k)
        self.propagate3 = GraphPropagation(g, edge_drop, alpha, k)
        self.reset_parameters()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

    def reset_parameters(self):
        for layer in self.layers1:
            layer.reset_parameters()
        for layer in self.layers2:
            layer.reset_parameters()
        for layer in self.layers3:
            layer.reset_parameters()

    def forward(self, features1,features2,features3):
        alpha1 =  self.alpha1
        alpha2 =  self.alpha2
        alpha3 =  self.alpha3
        # prediction step
        h1 = features1
        h2 = features2
        h3 = features3
        
        h1,h2,h3 = self.feat_drop(h1),self.feat_drop(h2),self.feat_drop(h3)
        h1,h2,h3 = self.activation(self.layers1[0](h1)),self.activation(self.layers2[0](h2)),self.activation(self.layers3[0](h3))
        for layer in self.layers1[1:-1]:
            h1 = self.activation(layer(h1))
        for layer in self.layers2[1:-1]:
            h2 = self.activation(layer(h2))
        for layer in self.layers3[1:-1]:
            h3 = self.activation(layer(h3))
        h1 = self.layers1[-1](self.feat_drop(h1))
        h2 = self.layers2[-1](self.feat_drop(h2))
        h3 = self.layers3[-1](self.feat_drop(h3))
        # propagation step
        h1,h2,h3 = self.propagate1(h1),self.propagate2(h2),self.propagate3(h3)
        return alpha1*h1+alpha2*h2+alpha3*h3
