"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import time
import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import dgl.function as fn
from sklearn import preprocessing

class Aggregator(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation=None, bias=True):
        super(Aggregator, self).__init__()
        self.g = g
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # (F, EF) or (2F, EF)
        self.activation = activation
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, node):
        nei = node.mailbox['m']  # (B, N, F)
        h = node.data['h']  # (B, F)
        h = self.concat(h, nei, node)  # (B, F) or (B, 2F)
        h = self.linear(h)   # (B, EF)
        if self.activation:
            h = self.activation(h)
        norm = torch.pow(h, 2)
        norm = torch.sum(norm, 1, keepdim=True)
        norm = torch.pow(norm, -0.5)
        norm[torch.isinf(norm)] = 0
        # h = h * norm
        return {'h': h}

    @abc.abstractmethod
    def concat(self, h, nei, nodes):
        raise NotImplementedError


class MeanAggregator(Aggregator):
    def __init__(self, g, in_feats, out_feats, activation, bias):
        super(MeanAggregator, self).__init__(g, in_feats, out_feats, activation, bias)

    def concat(self, h, nei, nodes):
        degs = self.g.in_degrees(nodes.nodes()).float()
        if h.is_cuda:
            degs = degs.cuda(h.device)
        concatenate = torch.cat((nei, h.unsqueeze(1)), 1)
        concatenate = torch.sum(concatenate, 1) / degs.unsqueeze(1)
        return concatenate  # (B, F)


class PoolingAggregator(Aggregator):
    def __init__(self, g, in_feats, out_feats, activation, bias):  # (2F, F)
        super(PoolingAggregator, self).__init__(g, in_feats*2, out_feats, activation, bias)
        self.mlp = PoolingAggregator.MLP(in_feats, in_feats, F.relu, False, True)

    def concat(self, h, nei, nodes):
        nei = self.mlp(nei)  # (B, F)
        concatenate = torch.cat((nei, h), 1)  # (B, 2F)
        return concatenate

    class MLP(nn.Module):
        def __init__(self, in_feats, out_feats, activation, dropout, bias):  # (F, F)
            super(PoolingAggregator.MLP, self).__init__()
            self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # (F, F)
            self.dropout = nn.Dropout(p=dropout)
            self.activation = activation
            nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

        def forward(self, nei):
            nei = self.dropout(nei)  # (B, N, F)
            nei = self.linear(nei)
            if self.activation:
                nei = self.activation(nei)
            max_value = torch.max(nei, dim=1)[0]  # (B, F)
            return max_value


class GraphSAGELayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 aggregator_type,
                 bias=True,
                 ):
        super(GraphSAGELayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(p=dropout)
        if aggregator_type == "pooling":
            self.aggregator = PoolingAggregator(g, in_feats, out_feats, activation, bias)
        else:
            self.aggregator = MeanAggregator(g, in_feats, out_feats, activation, bias)

    def forward(self, h):
        h = self.dropout(h)
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'), self.aggregator)
        h = self.g.ndata.pop('h')
        return h


class GraphSAGE(nn.Module):
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
                 aggregator_type,
                 alpha1,
                 alpha2,
                 alpha3):
        super(GraphSAGE, self).__init__()
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        

        # input layer
        self.layers1.append(GraphSAGELayer(g, in_feats1, n_hidden, activation, dropout, aggregator_type))
        self.layers2.append(GraphSAGELayer(g, in_feats2, n_hidden, activation, dropout, aggregator_type))
        self.layers3.append(GraphSAGELayer(g, in_feats3, n_hidden, activation, dropout, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers1.append(GraphSAGELayer(g, n_hidden, n_hidden, activation, dropout, aggregator_type))
            self.layers2.append(GraphSAGELayer(g, n_hidden, n_hidden, activation, dropout, aggregator_type))
            self.layers3.append(GraphSAGELayer(g, n_hidden, n_hidden, activation, dropout, aggregator_type))
        # output layer
        self.layers1.append(GraphSAGELayer(g, n_hidden, n_classes, None, dropout, aggregator_type))
        self.layers2.append(GraphSAGELayer(g, n_hidden, n_classes, None, dropout, aggregator_type))
        self.layers3.append(GraphSAGELayer(g, n_hidden, n_classes, None, dropout, aggregator_type))
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        
        
    def forward(self, features1,features2,features3):
        h1,h2,h3 = features1,features2,features3
        for layer in self.layers1:
            h1 = layer(h1)
        for layer in self.layers2:
            h2 = layer(h2)
        for layer in self.layers3:
            h3 = layer(h3)
        alpha1 =  self.alpha1
        alpha2 =  self.alpha2
        alpha3 =  self.alpha3
        return alpha1*h1+alpha2*h2+alpha3*h3
    
    

def evaluate(model,  features1,features2,features3, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model( features1,features2,features3)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    
    
    data = load_data(args)
    ##
    ##
    structure_features = np.load('../../pretrained/'+args.dataset+'_structure_8d.npy')
    attr_features = np.load('../../pretrained/'+args.dataset+'_attr_8d.npy')
    
    
    
    
    structure_features = preprocessing.scale(structure_features, axis=1, with_mean=True,with_std=True,copy=True)
    #structure_features = preprocessing.scale(structure_features, axis=0, with_mean=True,with_std=True,copy=True)
    structure_features = torch.FloatTensor(structure_features).cuda()
    
    attr_features = preprocessing.scale(attr_features, axis=1, with_mean=True,with_std=True,copy=True)
    #attr_features = preprocessing.scale(attr_features, axis=0, with_mean=True,with_std=True,copy=True)
    attr_features = torch.FloatTensor(attr_features).cuda()
    
    
    in_feats2 = structure_features.shape[1]
    in_feats3 = attr_features.shape[1]
    print(structure_features.shape,attr_features.shape)
    ##
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)
    in_feats1 = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    num_feats = features.shape[1]
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d
      #number of features %d""" %
          (n_edges, n_classes,
           train_mask.sum().item(),
           val_mask.sum().item(),
           test_mask.sum().item(),
           num_feats))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    # graph preprocess and calculate normalization factor
    g = DGLGraph(data.graph)
    n_edges = g.number_of_edges()

    
    
    alpha2_set = [0,0.001,0.002,0.004,0.006,0.008,0.01,0.02,0.03,0.04,0.05]
    alpha3_set = [0,0.001,0.002,0.004,0.006,0.008,0.01,0.02,0.03,0.04,0.05]
    alpha1 = 1
    for alpha2 in alpha2_set:
        for alpha3 in alpha3_set:
            result = []
            for iter in range(10):
                # create GraphSAGE model
                model = GraphSAGE(g,
                                  in_feats1,
                                  in_feats2,
                                  in_feats3,
                                  args.n_hidden,
                                  n_classes,
                                  args.n_layers,
                                  F.relu,
                                  args.dropout,
                                  args.aggregator_type,
                                  alpha1,
                                  alpha2,
                                  alpha3)
                                  

                if cuda:
                    model.cuda()
                loss_fcn = torch.nn.CrossEntropyLoss()

                # use optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

                # initialize graph
                dur = []
                best_val_acc = 0
                best_test_acc = 0
                for epoch in range(args.n_epochs):
                    model.train()
                    if epoch >= 3:
                        t0 = time.time()
                    # forward
                    logits = model(features,structure_features,attr_features)
                    loss = loss_fcn(logits[train_mask], labels[train_mask])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if epoch >= 3:
                        dur.append(time.time() - t0)

                    val_acc = evaluate(model, features,structure_features,attr_features, labels, val_mask)
                    if val_acc>=best_val_acc:
                        best_val_acc = val_acc
                        best_test_acc = evaluate(model, features, structure_features,attr_features, labels, test_mask)
                result.append(best_test_acc)

            print(alpha2,alpha3,np.average(result),result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Weight for L2 loss")
    
    args = parser.parse_args()
    print(args)

    main(args)
