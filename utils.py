
from texttable import Texttable
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
import time
import random
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
# from torch.sparse import coalesce
import math
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    P = metrics.precision_score(labels_true, labels_pred, average='macro')
    R = metrics.recall_score(labels_true, labels_pred, average='macro')
    F1_ma = metrics.f1_score(labels_true, labels_pred, average='macro')
    F1_mi = metrics.f1_score(labels_true, labels_pred, average='micro')

    return ACC, P, R, F1_ma, F1_mi

def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def get_topk_candidates(X, A, k=10):
    # rng = torch.Generator().manual_seed(42)
    N = A.size(0)
    device = X.device
    existing_edges = A.coalesce().indices()

    u, v = existing_edges
    u_sorted = torch.min(u, v)
    v_sorted = torch.max(u, v)
    existing_linear = u_sorted * N + v_sorted
    existing_linear = torch.unique(existing_linear, sorted=True)

    X_norm = X / torch.norm(X, dim=1, keepdim=True)
    S = torch.mm(X_norm, X_norm.T)
    S.fill_diagonal_(-float('inf'))
    _, topk_indices = torch.topk(S, k=k, dim=1)

    rows = torch.arange(N, device=device).repeat_interleave(k)
    cols = topk_indices.flatten()
    candidates = torch.stack([rows, cols], dim=0)

    u, v = candidates
    u_sorted = torch.min(u, v)
    v_sorted = torch.max(u, v)
    candidates_linear = u_sorted * N + v_sorted
    candidates_linear = torch.unique(candidates_linear, sorted=True)

    mask = ~torch.isin(candidates_linear, existing_linear)
    new_linear = candidates_linear[mask]

    u = new_linear // N
    v = new_linear % N
    return torch.stack([u, v], dim=0)



def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    np.random.seed(0)
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx

