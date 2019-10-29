import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import dgl
from appnp import APPNP
from sklearn import preprocessing

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
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.sum().item(),
           val_mask.sum().item(),
           test_mask.sum().item()))

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

    # graph preprocess and calculate normalization factor
    g = DGLGraph(data.graph)
    n_edges = g.number_of_edges()
    # add self loop
    g.add_edges(g.nodes(), g.nodes())
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create APPNP model
    #alpha2_set = [0,0.001,0.002,0.004,0.006,0.008,0.01,0.02,0.03,0.04,0.05]
    #alpha3_set = [0,0.001,0.002,0.004,0.006,0.008,0.01,0.02,0.03,0.04,0.05]
    alpha2_set = [0.002]
    alpha3_set = [0]
    alpha1 = 1
    for alpha2 in alpha2_set:
        for alpha3 in alpha3_set:
            result = []
            for iter in range(30):
                model = APPNP(g,
                              in_feats1,
                              in_feats2,
                              in_feats3,
                              args.hidden_sizes,
                              n_classes,
                              F.relu,
                              args.in_drop,
                              args.edge_drop,
                              args.alpha,
                              args.k,
                              alpha1,
                              alpha2,
                              alpha3)


                if cuda:
                    model.cuda()
                loss_fcn = torch.nn.CrossEntropyLoss()

                # use optimizer
                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=args.lr,
                                             weight_decay=args.weight_decay)

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
    parser = argparse.ArgumentParser(description='APPNP')
    register_data_args(parser)
    parser.add_argument("--in-drop", type=float, default=0.5,
                        help="input feature dropout")
    parser.add_argument("--edge-drop", type=float, default=0.5,
                        help="edge propagation dropout")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[64],
                        help="hidden unit sizes for appnp")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of propagation steps")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Teleport Probability")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)
