import numpy as np
import scipy.sparse as sp
import torch
import time
import random

def read_data(path):
    
    """read file and return the triples containing its ground truth label (0/1)"""
    
    f = open(path)
    triples_with_label = []
    for line in f:
        triple_with_label = line.strip().split("\t")
        triples_with_label.append(triple_with_label)
    f.close()
    return triples_with_label

def write_dic(write_path, array):
    
    """generate a dictionary"""
    
    f = open(write_path, "w+")
    for i in range(len(array)):
        f.write("{}\t{}\n".format(i, array[i]))
    f.close()
    print("saved dictionary to {}".format(write_path))
    
def dictionary(input_list):
    
    """
    To generate a dictionary.
    Index: item in the array.
    Value: the index of this item.
    """
    
    return dict(zip(input_list, range(len(input_list))))

def normalize(mx):
    
    """Row-normalize sparse matrix"""
    
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx




def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def split_known(triples):
    
    """
    Further split the triples into 2 sets:
    1. an incomplete graph: known
    2. a set of missing facts we want to recover: unknown
    """
    
    DATA_LENGTH = len(triples)
    split_ratio = [0.9, 0.1]
    candidate = np.array(range(DATA_LENGTH))
    np.random.shuffle(candidate)
    idx_known = candidate[:int(DATA_LENGTH * split_ratio[0])]
    idx_unknown = candidate[int(DATA_LENGTH * split_ratio[0]):]
    known = []
    unknown = []
    for i in idx_known:
        known.append(triples[i])
    for i in idx_unknown:
        unknown.append(triples[i])
    return known, unknown