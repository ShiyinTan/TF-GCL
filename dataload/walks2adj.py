import networkx as nx
import pandas as pd 
import pickle as pkl
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
from torch.utils.data import Dataset, DataLoader

def walks_to_matrix(walks, max_num, cpu_idx, build_tree=False):
    pbar = tqdm(total=len(walks), desc='Generating walks (CPU: {})'.format(cpu_idx))
    W = sp.lil_matrix((max_num, max_num))
    if build_tree:
        for walk in walks:
            pbar.update(1)
            walk = list(map(int, walk))
            for i in range(len(walk)):
                W[walk[0],walk[i]] += 1
                W[walk[i],walk[0]] += 1
    else:
        for walk in walks:
            pbar.update(1)
            walk = list(map(int, walk))
            for i in range(len(walk)):
                for j in range(i+1, len(walk)):
                    W[walk[i],walk[j]] += 1
                    W[walk[j],walk[i]] += 1
    pbar.close()
    return W.tocsr()


class CustomEdgeDataset(Dataset):
    def __init__(self, walks, max_node_num, num_workers=4, build_tree=False):
        self.num_walks_each_worker = len(walks)//num_workers
        print("------Init Dataset-----")
        time_st = time.time()
        W_each_worker = Parallel(n_jobs=num_workers)(
            delayed(walks_to_matrix)(
                walks[i*self.num_walks_each_worker:(i+1)*self.num_walks_each_worker], 
                max_node_num, i, build_tree=build_tree)
            for i in range(num_workers))
        time_ed = time.time()
        print("-----Init Time Consuming: ", time_ed - time_st, "-----")
        self.W = sp.csr_matrix((max_node_num, max_node_num))
        for W_each in W_each_worker:
            self.W += W_each
        (self.row_ind, self.col_ind), self.weight = self.W.nonzero(), self.W.data
    
    def __len__(self):
        return len(self.weight)

    def __getitem__(self, idx):
        label = self.weight[idx]
        edge = (self.row_ind[idx], self.col_ind[idx])
        return edge, label