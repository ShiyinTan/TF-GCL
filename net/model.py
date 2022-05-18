import dgl
import dgl.function as fn
from dgl.data import DGLDataset
from dgl.nn import SAGEConv
import itertools
import pickle as pkl
import time
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

import gc


# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


## user defined model
class MyFeatureExtractModel(nn.Module):
    def __init__(self, num_nodes, emb_size, h_feats_list=[128], concat=False):
        super().__init__()
        self.sage_layers = nn.ModuleList()
        self.concat = concat
        h_feats_pre = emb_size
        # print(h_feats_list)
        for h_feats in h_feats_list:
            self.sage_layers.append(GraphSAGE(h_feats_pre, h_feats))
            h_feats_pre = h_feats
        # self.Sage_layer_1 = GraphSAGE(emb_size, h_feats_1)
        # self.Sage_layer_2 = GraphSAGE(h_feats_1, h_feats_2)

    def forward(self, train_graph, nodes_features):
        outputs = [nodes_features]
        hids_features = nodes_features
        for i, l in enumerate(self.sage_layers):
            if self.concat:
                hids_features = l(train_graph, hids_features)
                outputs.append(hids_features)
            else:
                hids_features = l(train_graph, hids_features)
        if self.concat:
            outputs = torch.cat(outputs, -1)
        else:
            outputs = hids_features
        # hids_features = self.Sage_layer_1(train_graph, nodes_features)
        # hids_features = self.Sage_layer_2(dataset.train_graph, hids_features)
        return outputs


"""
define predictor
"""
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


