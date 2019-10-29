"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import GraphConv

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats1,in_feats2,in_feats3,out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear1 = nn.Linear(in_feats1, out_feats)
        self.activation = activation
        self.linear2 = nn.Linear(in_feats2, out_feats)
        self.linear3 = nn.Linear(in_feats3, out_feats)
        
        

    def forward(self, node):
        h1 = self.linear1(node.data['h1'])  
        h2 = self.linear2(node.data['h2'])
        h3 = self.linear3(node.data['h3'])
        if self.activation is not None:
            h1 = self.activation(h1)
            h2 = self.activation(h2)
            h3 = self.activation(h3)
        return {'h1' : h1,'h2':h2,'h3':h3}
    
class GCN_layer(nn.Module):
    def __init__(self, in_feats1,in_feats2,in_feats3, out_feats, activation=None):
        super(GCN_layer, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats1,in_feats2,in_feats3, out_feats, activation)
        self.gcn_msg1 = fn.copy_src(src='h1', out='m1')
        self.gcn_reduce1 = fn.sum(msg='m1', out='h1')
        self.gcn_msg2 = fn.copy_src(src='h2', out='m2')
        self.gcn_reduce2 = fn.sum(msg='m2', out='h2')
        self.gcn_msg3 = fn.copy_src(src='h3', out='m3')
        self.gcn_reduce3 = fn.sum(msg='m3', out='h3')

    def forward(self, feature1,feature2,feature3,g):
        
        ##norm
        norm1 = torch.pow(g.in_degrees().float(), -0.5)
        shp1 = norm1.shape + (1,) * (feature1.dim() - 1)
        norm1 = torch.reshape(norm1, shp1).to(feature1.device)
        
        norm2 = torch.pow(g.in_degrees().float(), -0.5)
        shp2 = norm2.shape + (1,) * (feature2.dim() - 1)
        norm2 = torch.reshape(norm2, shp2).to(feature2.device)
        
        norm3 = torch.pow(g.in_degrees().float(), -0.5)
        shp3 = norm3.shape + (1,) * (feature3.dim() - 1)
        norm3 = torch.reshape(norm3, shp3).to(feature3.device)
        
        feature1 = feature1 * norm1
        feature2 = feature2 * norm2
        feature3 = feature3 * norm3
        ##
        
        g.ndata['h1'] = feature1
        g.ndata['h2'] = feature2
        g.ndata['h3'] = feature3
        g.update_all(self.gcn_msg1, self.gcn_reduce1)
        g.update_all(self.gcn_msg2, self.gcn_reduce2)
        g.update_all(self.gcn_msg3, self.gcn_reduce3)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h1')*norm1, g.ndata.pop('h2')*norm2,g.ndata.pop('h3')*norm3
    
    
class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats1,
                 in_feats2,
                 in_feats3,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 alpha1,
                 alpha2,
                 alpha3):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCN_layer(in_feats1, in_feats2,in_feats3, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCN_layer(n_hidden,n_hidden,n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GCN_layer(n_hidden,n_hidden,n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

    def forward(self, features1,features2,features3):
        alpha1 =  self.alpha1
        alpha2 =  self.alpha2
        alpha3 =  self.alpha3
        #print('alpha1:{:.4f},alpha2:{:.4f},alpha3:{:.4f}'.format(alpha1,alpha2,alpha3))
        h1,h2,h3 = features1,features2,features3
        for i, layer in enumerate(self.layers):
            if i != 0:
                h1,h2,h3 = self.dropout(h1),self.dropout(h2),self.dropout(h3)
            h1,h2,h3 = layer(h1,h2,h3, self.g)
        return alpha1*h1+alpha2*h2+alpha3*h3
