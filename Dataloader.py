
import scipy.sparse as ss
from scipy.sparse import csc_matrix
from scipy.io import loadmat
import scipy.io as sio
import torch
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
from torch_sparse import coalesce
import os
import numpy as np
def load_graph_data(args):
    if args.dataset == 'Penn94':
        mat = sio.loadmat('./data/' + args.dataset + '.mat')
        A = mat['A']
        metadata = mat['local_info']

        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
        edge_index, _ = coalesce(edge_index, None, A.shape[0], A.shape[1])

        adj_with_loop = torch.sparse_coo_tensor(
            indices=edge_index,
            values=torch.ones(edge_index.shape[1]),
            size=(A.shape[0], A.shape[1])
        ).coalesce()

        mask_no_loop = edge_index[0] != edge_index[1]
        edge_index_no_loop = edge_index[:, mask_no_loop]

        adj_no_loop = torch.sparse_coo_tensor(
            indices=edge_index_no_loop,
            values=torch.ones(edge_index_no_loop.shape[1]),
            size=(A.shape[0], A.shape[1])
        ).coalesce()

        metadata = metadata.astype(np.int_)
        label = metadata[:, 1] - 1
        feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
        features = np.empty((A.shape[0], 0))
        for col in range(feature_vals.shape[1]):
            feat_col = feature_vals[:, col]
            feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
            features = np.hstack((features, feat_onehot))

        node_feat = torch.tensor(features, dtype=torch.float).to(args.device)
        labels = torch.tensor(label).to(args.device)

        splits_drive_url = {
            'snap-patents': '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
            'pokec': '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
        }
        if not os.path.exists(f'./split/{args.dataset}-splits.npy'):
            assert args.dataset in splits_drive_url.keys()
            gdown.download(
                id=splits_drive_url[args.dataset],
                output=f'./split/{args.dataset}-splits.npy',
                quiet=False
            )

        splits_lst = np.load(f'./split/{args.dataset}-splits.npy', allow_pickle=True)
        for i in range(len(splits_lst)):
            for key in splits_lst[i]:
                if not torch.is_tensor(splits_lst[i][key]):
                    splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
        train_mask = splits_lst[1]['train'].numpy()
        valid_mask = splits_lst[1]['valid'].numpy()
        test_mask = splits_lst[1]['test'].numpy()

        return adj_with_loop.to(args.device), adj_no_loop.to(args.device), node_feat, labels, torch.from_numpy(
            train_mask), torch.from_numpy(valid_mask), torch.from_numpy(test_mask)

    else:
        data = sio.loadmat(args.path_graph + args.dataset + '.mat')
        features = data['X']
        adj = data['adj']
        labels = data['Y'].flatten()
        labels = labels - min(set(labels))

        base = os.path.join('./split/', args.dataset)
        train_mask = torch.from_numpy(loadmat(os.path.join(base, 'train_mask.mat'))['train_mask'].squeeze().astype(bool))
        valid_mask = torch.from_numpy(loadmat(os.path.join(base, 'valid_mask.mat'))['valid_mask'].squeeze().astype(bool))
        test_mask = torch.from_numpy(loadmat(os.path.join(base, 'test_mask.mat'))['test_mask'].squeeze().astype(bool))

        labels = torch.from_numpy(labels).long()

        adj = torch.from_numpy(adj)
        adj_t = adj + adj.t().multiply(adj.t() > adj) - adj.multiply(adj.t() > adj)
        _, adj_t = normalized_adjacency(adj_t + torch.eye(adj.shape[0], adj.shape[0]))
        adj_t = adj_t.tocoo()
        sparse_adj = torch.sparse_coo_tensor(
            torch.tensor([adj_t.row, adj_t.col]),
            torch.tensor(adj_t.data),
            adj_t.shape
        ).to(args.device)

        A = sparse_adj.detach().clone().coalesce()
        edge_indices = A._indices()
        edge_values = A._values() if A._values() is not None else torch.ones(edge_indices.shape[1], device=A.device)

        mask = edge_indices[0] != edge_indices[1]
        filtered_indices = edge_indices[:, mask]
        filtered_values = edge_values[mask]
        A_without_self_loop = torch.sparse_coo_tensor(
            filtered_indices,
            filtered_values,
            A.size()
        ).coalesce().to(A.device)

        feature = torch.from_numpy(features).float().to(args.device)
        feature = F.normalize(feature, p=2.0, dim=1)
        feature = feature.to(args.device)
        print('the number of sample:' + str(feature.shape[0]))
        return sparse_adj, A_without_self_loop, feature, labels, train_mask, valid_mask, test_mask

def normalized_adjacency(adj):
   adj = ss.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = ss.diags(d_inv_sqrt)
   mx = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
   if isinstance(mx, torch.Tensor):
        return mx, d_mat_inv_sqrt
   else:
        sp_mx = mx
        mx = np.array(mx.toarray())
        return torch.from_numpy(mx).float(), sp_mx
