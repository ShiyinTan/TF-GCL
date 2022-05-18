import numpy as np
import pandas as pd
import torch

def split_train_test_data(E: pd.DataFrame, split_timestamp):
    train_edges_data = E[E.Timestamp < split_timestamp]
    test_edges_data = E[E.Timestamp >= split_timestamp]
    return train_edges_data, test_edges_data

def from_edges_data_to_tensor(edges_data: pd.DataFrame, source='Src', target='Dst'):
    edge_src = torch.from_numpy(pd.concat([edges_data[source], edges_data[target]]).to_numpy(dtype=np.int32))
    edge_dst = torch.from_numpy(pd.concat([edges_data[target], edges_data[source]]).to_numpy(dtype=np.int32))
    edge_ts = torch.from_numpy(pd.concat([edges_data['Timestamp'], edges_data['Timestamp']]).to_numpy(dtype=np.int32))
    return edge_src, edge_dst, edge_ts