import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import dgl


"""
define reconstruction loss
"""
def compute_loss(pos_score, neg_score, device=torch.torch.device('cpu')):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    return roc_auc_score(labels, scores)

def compute_apscore(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    return average_precision_score(labels, scores)

def compute_loss_with_weight(pos_score, neg_score, weight, class_weight=torch.Tensor([1,1]), device=torch.torch.device('cpu')):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels, weight, pos_weight=class_weight)

def compute_loss_with_weight_temporal_decay(pos_score, neg_score, pos_timestamps, neg_timestamps,
                                            recent_timestamp, decay_rate = 0.02, decay_by = 'month',
                                            class_weight=torch.Tensor([1,1]), device=torch.torch.device('cpu')):
    """
    Parameter:
    decay_by: 30*60*60*24 month (default)
    """
    if decay_by=='month':
        stair_by = 30*60*60*24
    elif decay_by=='day':
        stair_by = 60*60*24
    elif decay_by=='year':
        stair_by = 365*60*60*24
    elif decay_by=='minute':
        stair_by = 60
    elif decay_by=='hour':
        stair_by = 60*60
    elif decay_by=='second':
        stair_by = 1
    else:
        raise Exception("Invalid parameter!", decay_by)
    initial_weight = class_weight[0]
    neg_initial_weight = class_weight[1]
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
#     timestamps = torch.cat([pos_timestamps, neg_timestamps])
    pos_temporal_weight = initial_weight * torch.exp(-decay_rate*((recent_timestamp-pos_timestamps)//stair_by)) # decay by month
    neg_temporal_weight = neg_initial_weight * torch.exp(-decay_rate*((recent_timestamp-neg_timestamps)//stair_by))
    temporal_weight = torch.cat([pos_temporal_weight, 0.5*neg_temporal_weight])
    # temporal_weight = initial_weight * torch.float_power(decay_rate, recent_timestamp - timestamps)
    # temporal_weight = initial_weight / (decay_rate * (recent_timestamp - timestamps))
    # temporal_weight = initial_weight / (1.0 + decay_rate * (recent_timestamp - timestamps))
    # temporal_weight = initial_weight * torch.float_power(decay_rate, 1.0 + (recent_timestamp - timestamps)//decay_every_num) # step decay
    return F.binary_cross_entropy_with_logits(scores, labels, temporal_weight)


"""
define contrastive learning loss
"""
def batch_CL_loss(batch_nodes, graph, h1_norm, h2_norm, temporal_loss=False, neig_positive=True, sym_loss=False, device=torch.torch.device('cpu')):
    # global h1_norm, h2_norm
    pos_nodes = graph.out_edges(batch_nodes,form='all')
    pos_nodes_degrees = graph.out_degrees(batch_nodes)
    pos_nodes_dst = pos_nodes[1]
    batch_size = len(batch_nodes)
    pos_nodes_timestamp = graph.edata['timestamp'][pos_nodes[2].long()]
    
    nodes_edges_mask = torch.zeros((len(batch_nodes), pos_nodes_dst.shape[0])).to(device)
    for i in range(len(batch_nodes)):
        nodes_edges_mask[i][pos_nodes_degrees[0:i].sum():pos_nodes_degrees[0:i+1].sum()] = 1
    if temporal_loss:
        denominator = ((torch.reshape(graph.ndata['ts_sum'][batch_nodes], (-1,1)) @ torch.reshape(graph.ndata['ts_sum'], (-1,1)).T) * 
                        torch.exp(h1_norm[batch_nodes] @ h1_norm.T)).sum(-1)
        if sym_loss:
            sym_denominator = ((torch.reshape(graph.ndata['ts_sum'][batch_nodes], (-1,1)) @ torch.reshape(graph.ndata['ts_sum'], (-1,1)).T) * 
                        torch.exp(h2_norm[batch_nodes] @ h2_norm.T)).sum(-1)
    else:
        denominator = torch.exp((h1_norm[batch_nodes] @ h1_norm.T)).sum(-1)
        if sym_loss:
            sym_denominator = torch.exp((h2_norm[batch_nodes] @ h2_norm.T)).sum(-1)

    numerator = (h1_norm[batch_nodes] * h2_norm[batch_nodes]).sum(-1)
    if neig_positive:
        numerator += ((h1_norm[batch_nodes] @ h1_norm[pos_nodes_dst.long()].T) * nodes_edges_mask).sum(-1)

    # sum(log(exp(numerator)/denominator))
    barched_cl_loss = numerator - (pos_nodes_degrees+1)*torch.log(denominator+1e-7)
    if sym_loss:
        barched_cl_loss += numerator - (pos_nodes_degrees+1)*torch.log(sym_denominator+1e-7)

    # numerator = (torch.exp((h1_norm[batch_nodes] @ h1_norm[pos_nodes_dst.long()].T) *
    #                        nodes_edges_mask).sum(-1) +
    #              torch.exp(h1_norm[batch_nodes] * h2_norm[batch_nodes]).sum(-1))
    
    # barched_cl_loss = -torch.log(numerator/denominator)
    return -barched_cl_loss.mean()



"""
define data augmentation
"""
# feature masking: assume input_feature is Embedding
def random_feature_mask(input_feature, drop_percent=0.5, device=torch.device('cpu')):
    aug_feature = copy.deepcopy(input_feature.weight)
    p = torch.ones(input_feature.weight.shape,dtype=torch.float).bernoulli_(1-drop_percent).to(device)
    aug_feature = aug_feature * p
    aug_feature = nn.Embedding.from_pretrained(aug_feature)
    return aug_feature

# replace: 是否在当前graph中还是另外创建graph
def random_masking_timestamp(dataset, mask_num, timestamp_mask, replace=False, device=torch.device('cpu')):
    """
    Usage:
    aug_graph1 = random_masking_timestamp(dataset, mask_num = int(dataset.train_graph.number_of_edges()*0.2), timestamp_mask=0)
    ------------------ 
    Parameter
    ------------------
    replace: 
        是否在当前graph中masking, 还是另外创建graph, masking
    timestamp_mask:
        一般用当前时间作为masking, mask后的边不会decay
    mask_num: 
        timestamp mask的数量
    ------------------
    """
    (edges_src, edges_dst), edges_timestamp = dataset.train_graph.edges(), dataset.train_graph.edata['timestamp']
    if (not replace):
        edges_src = copy.deepcopy(edges_src)
        edges_dst = copy.deepcopy(edges_dst)
        edges_timestamp = copy.deepcopy(edges_timestamp)
    
    masking_idxs = np.random.choice(len(edges_timestamp), mask_num, replace=False)
    edges_timestamp[masking_idxs] = timestamp_mask
    if (not replace):
        aug_graph = dgl.graph((edges_src, edges_dst), num_nodes=dataset.num_nodes)
        aug_graph.edata['timestamp'] = edges_timestamp
    else:
        aug_graph = dataset.train_graph
    return aug_graph


def random_remove_and_add_edges(dataset, replace_num, timestamp_mask, attempt_times=100, replace=False, device=torch.device('cpu')):
    """
    Usage:
    aug_graph2 = random_remove_and_add_edges(dataset, replace_num=int(dataset.train_graph.number_of_edges()*0.1), timestamp_mask=0)
    ----------------
    """
    (edges_src, edges_dst), edges_timestamp = dataset.train_graph.edges(), dataset.train_graph.edata['timestamp']
    if (not replace):
        edges_src = copy.deepcopy(edges_src)
        edges_dst = copy.deepcopy(edges_dst)
        edges_timestamp = copy.deepcopy(edges_timestamp)
    
    replace_idxs = np.random.choice(len(edges_src), replace_num, replace=False)
    random_src = np.random.choice(dataset.num_nodes, replace_num)
    random_dst = np.random.choice(dataset.num_nodes, replace_num)
    edges_src[replace_idxs] = torch.from_numpy(random_src).to(device)
    edges_src[replace_idxs] = torch.from_numpy(random_dst).to(device)
    edges_timestamp[replace_idxs] = timestamp_mask
    
    aug_graph = dgl.graph((edges_src, edges_dst), num_nodes=dataset.num_nodes)
    aug_graph.edata['timestamp'] = edges_timestamp
    aug_graph = aug_graph.to(device)
    return aug_graph


