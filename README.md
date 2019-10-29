# G-GNN
Dependencies
------------
- PyTorch 0.4.1+
- dgl 0.3
- python3
- requests
- Tensorflow 1.12.0 (optional)
gpu version
------------

This is the code for paper "Pre-train and Learn: Preserve Global Information for Graph Neural Networks". The method conducts semi-supervised learning on the graph data.

The method consists of two stages. First, we pretrain two vectors of the structure vector and the attribute vector for each node in a unsupervised manner. Second, we propose a parallel GNN based model to learn from the two pretrained vectors and the orignial attributes.

The code of the first stage is in 'code/pretrain'. It depends on Tensorflow 1.12.0 and dgl 0.3 . We also upload some pretrained vectors in 'pretrain/', so you can also skip the first stage without suffering from install Tensorflow.

The code of the second stage is in 'code/g-gcn'. It depends dgl 0.3 and the corresponding pytorch. You can get the results in the paper with the following commond:

For G-APPNP:
Run with following. It will take some time since the result is the average value of 10 experiments.
```bash
python3 train.py --dataset cora --gpu 0
python3 train.py --dataset citeseer --gpu 0 --galpha=0.002   --gbeta=0
python3 train.py --dataset pubmed --gpu 0   --galpha=0   --gbeta=0.03
```
On cora, citeseer and pubmed, the precison will be around 84.31, 72 and 81.95.

