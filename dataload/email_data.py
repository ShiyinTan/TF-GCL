import numpy as np
import pandas as pd
import networkx as nx
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.data import DGLDataset

from .dataload_utils import split_train_test_data, from_edges_data_to_tensor

"""
usage: 
# dataset = MathoverflowDataset()
# graph = dataset[0]
"""
class EmailDataset(DGLDataset):
    def __init__(self, name='my_dataset', slice_interval=300, time_span=60*60*24, verbose=True, tf_decay_rate=0.03,
                robust_test=False, robust_ratio=0.0):
        self.slice_interval = slice_interval
        self.time_span = time_span
        self.verb = verbose
        self.decay_rate = tf_decay_rate
        self.robust_test = robust_test
        self.robust_ratio = robust_ratio
        super().__init__(name=name)
        
    def process(self):
        # 使用split_timestamp来划分训练集和测试集
        # split_timestamp回溯slice_interval天为测试集
        # nodes_data = pd.read_csv('./members.csv')
        edges_data = pd.read_csv('./data/email-Eu-core/email-Eu-core-temporal_reindex.txt', 
                                 header=None, sep=' ', names=['Src', 'Dst', 'Timestamp'])
        if self.verb:
            print("edges timestamp from: ")
            print(time.localtime(edges_data.Timestamp.min()))
            print("edges timestamp to: ")
            print(time.localtime(edges_data.Timestamp.max()))
        
        self.num_nodes = max(edges_data['Src'].max(),edges_data['Dst'].max())+1
        
        self.split_timestamp = edges_data.Timestamp.max()-self.slice_interval*self.time_span
#         train_edges_data = edges_data[edges_data.Timestamp < self.split_timestamp]
#         test_edges_data = edges_data[edges_data.Timestamp >= self.split_timestamp]
        train_edges_data, test_edges_data = split_train_test_data(edges_data, self.split_timestamp)

        # robustness test with noise edges
        if self.robust_test:
            num_noise_edge = int(len(train_edges_data) * self.robust_ratio)
            noise_u = np.random.randint(0, self.num_nodes, (num_noise_edge))
            noise_v = np.random.randint(0, self.num_nodes, (num_noise_edge))
            noise_ts = np.random.randint(train_edges_data['Timestamp'].min(), train_edges_data['Timestamp'].max(), (num_noise_edge))
            noise_edges = pd.DataFrame({'Src':noise_u,'Dst':noise_v,'Timestamp':noise_ts})
            train_edges_data = pd.concat([train_edges_data, noise_edges], ignore_index=True)
            if self.verb:
                print('noise_edge_num', num_noise_edge)
        
        edges_src = torch.from_numpy(pd.concat([edges_data['Src'], edges_data['Dst']]).to_numpy(dtype=np.int32))
        edges_dst = torch.from_numpy(pd.concat([edges_data['Dst'], edges_data['Src']]).to_numpy(dtype=np.int32))
        edge_ts = torch.from_numpy(pd.concat([edges_data['Timestamp'], edges_data['Timestamp']]).to_numpy(dtype=np.int32))

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=self.num_nodes)
        self.graph.edata['timestamp'] = edge_ts
        # node_features = torch.from_numpy(nodes_data['Age'].to_numpy())
        # self.graph.ndata['feat'] = node_features
        
        ## train dataset
        train_edge_src, train_edge_dst, train_edge_ts = from_edges_data_to_tensor(train_edges_data)
#         train_edge_src = torch.from_numpy(pd.concat([train_edges_data['Src'], train_edges_data['Dst']]).to_numpy(dtype=np.int32))
#         train_edge_dst = torch.from_numpy(pd.concat([train_edges_data['Dst'], train_edges_data['Src']]).to_numpy(dtype=np.int32))
#         train_edge_ts = torch.from_numpy(pd.concat([train_edges_data['Timestamp'], train_edges_data['Timestamp']]).to_numpy(dtype=np.int32))
        
        self.train_graph = dgl.graph((train_edge_src, train_edge_dst), num_nodes=self.num_nodes)
        self.train_graph.edata['timestamp'] = train_edge_ts

        ## update train graph for frequency- and temporal- awared contrastive learning
        decay_rate = self.decay_rate
        cur_timestamp = self.train_graph.edata['timestamp'].max()
        decay_by = 30*60*60*24 # month
        def udf_time_function(edges):
            return {'ts_decay_score' : torch.exp(-decay_rate*((cur_timestamp - edges.data['timestamp'])/decay_by))}
        # self.train_graph.edata['timestamp'] = self.train_graph.edata['timestamp'].type(torch.float)
        self.train_graph.update_all(udf_time_function, dgl.function.sum('ts_decay_score', 'ts_sum'))
        self.train_graph.ndata['ts_sum'] = torch.log(1+self.train_graph.ndata['ts_sum'])
        self.train_graph.ndata['ts_sum'] = 1 - self.train_graph.ndata['ts_sum']/(self.train_graph.ndata['ts_sum'].max()+1e-6)
        if self.verb:
            print("ts_sum min: ", self.train_graph.ndata['ts_sum'].min(), "max: ", self.train_graph.ndata['ts_sum'].max())
        ## test dataset
        test_edge_src, test_edge_dst, test_edge_ts = from_edges_data_to_tensor(test_edges_data)
#         test_edge_src = torch.from_numpy(pd.concat([test_edges_data['Src'], test_edges_data['Dst']]).to_numpy(dtype=np.int32))
#         test_edge_dst = torch.from_numpy(pd.concat([test_edges_data['Dst'], test_edges_data['Src']]).to_numpy(dtype=np.int32))
#         test_edge_ts = torch.from_numpy(pd.concat([test_edges_data['Timestamp'], test_edges_data['Timestamp']]).to_numpy(dtype=np.int32))
        
        self.test_graph = dgl.graph((test_edge_src, test_edge_dst), num_nodes=self.num_nodes)
        self.test_graph.edata['timestamp'] = test_edge_ts

        # construct neg dataset
        neg_eid_nums = dgl.to_simple(self.graph).number_of_edges()
        test_neg_eid_nums = dgl.to_simple(self.graph).number_of_edges()
        # neg_u = torch.from_numpy(np.random.choice(graph.nodes(), neg_eid_nums))
        # neg_v = torch.from_numpy(np.random.choice(graph.nodes(), neg_eid_nums))

        # train_neg_graph = dgl.graph((neg_u, neg_v), num_nodes=dataset.num_nodes)

        self.train_neg_graph = dgl.graph(dgl.sampling.global_uniform_negative_sampling(self.graph, 4*neg_eid_nums))
        self.test_neg_graph = dgl.graph(dgl.sampling.global_uniform_negative_sampling(self.graph, neg_eid_nums))

        self.train_neg_graph.edata['timestamp'] = torch.full([self.train_neg_graph.num_edges()], self.train_graph.edata['timestamp'].max())

        if self.verb:
            print("graph nodes: {}".format(self.graph.number_of_nodes()))
            print("splited train edges: {}, test edges: {}".format(self.train_graph.number_of_edges(), self.test_graph.number_of_edges()))
    
    def to(self, device):
        self.train_graph = self.train_graph.to(device)
        self.test_graph = self.test_graph.to(device)
        self.train_neg_graph = self.train_neg_graph.to(device)
        self.test_neg_graph = self.test_neg_graph.to(device)
        return self
        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1