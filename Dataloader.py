
import scipy.sparse as ss
from scipy.sparse import csc_matrix
from utils import *
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale
from ogb.nodeproppred import PygNodePropPredDataset, NodePropPredDataset
from torch_geometric.utils import to_undirected
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from sklearn.preprocessing import label_binarize
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
import gdown
import torch
from sklearn.preprocessing import label_binarize
from torch_geometric.data import Data
from torch_sparse import coalesce
import os.path as osp
import numpy as np
def distance(point1, point2):  # 计算距离（欧几里得距离）
    return np.sqrt(np.sum((point1 - point2) ** 2))


def k_means(data, k, max_iter=100):
    centers = {}  # 初始聚类中心
    # 初始化，随机选k个样本作为初始聚类中心。 random.sample(): 随机不重复抽取k个值
    n_data = data.shape[0]  # 样本个数
    Label = np.array([0]*n_data)
    for idx, i in enumerate(random.sample(range(n_data), k)):
        # idx取值范围[0, k-1]，代表第几个聚类中心;  data[i]为随机选取的样本作为聚类中心
        centers[idx] = data[i]

        # 开始迭代
    for i in range(max_iter):  # 迭代次数
        print("开始第{}次迭代".format(i + 1))
        clusters = {}  # 聚类结果，聚类中心的索引idx -> [样本集合]
        for j in range(k):  # 初始化为空列表
            clusters[j] = []

        for sample_idx, sample in enumerate(data):  # 遍历每个样本
            distances = []  # 计算该样本到每个聚类中心的距离 (只会有k个元素)
            for c in centers:  # 遍历每个聚类中心
                # 添加该样本点到聚类中心的距离
                distances.append(distance(sample, centers[c]))
            idx = np.argmin(distances)  # 最小距离的索引
            clusters[idx].append(sample)  # 将该样本添加到第idx个聚类中心
            Label[sample_idx] = np.argmin(distances)

        pre_centers = centers.copy()  # 记录之前的聚类中心点

        for c in clusters.keys():
            # 重新计算中心点（计算该聚类中心的所有样本的均值）
            centers[c] = np.mean(clusters[c], axis=0)

        is_convergent = True
        for c in centers:
            if distance(pre_centers[c], centers[c]) > 1e-8:  # 中心点是否变化
                is_convergent = False
                break
        if is_convergent == True:
            # 如果新旧聚类中心不变，则迭代停止
            break
    return centers, clusters, Label


def predict(p_data, centers):  # 预测新样本点所在的类
    # 计算p_data 到每个聚类中心的距离，然后返回距离最小所在的聚类。
    distances = [distance(p_data, centers[c]) for c in centers]
    return np.argmin(distances)

def to_sparsetensor(data):
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=N)

    # row, col = data.edge_index
    # adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    # adj = adj.set_diag()
    adj = torch.sparse_coo_tensor(
        indices=data.edge_index,
        values=torch.ones(data.edge_index.shape[1]),
        size=(N, N)
    ).coalesce()
    return adj

def normalize_large_adj(adj, mode='DA'):
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    if mode == 'DA':
        return deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj
    if mode == 'DAD':
        return deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return adj


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0])
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label

def load_ogb(args):
    if args.dataset == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset/')
        graph = dataset[0]
        split_idx = dataset.get_idx_split()
        train_mask = split_idx['train'].numpy()
        valid_mask = split_idx['valid'].numpy()
        test_mask = split_idx['test'].numpy()
        feature = graph.x.float().to(args.device)
        feature = F.normalize(feature, p=2.0, dim=1)
        feature = feature.to(args.device)
        graph.y = graph.y.squeeze()
        adj = to_sparsetensor(graph)
        graph.adj = normalize_large_adj(adj)
        labels = graph.y.squeeze()
        num_classes = len(np.unique(labels.cpu().numpy()))
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(feature.size(0), feature.size(0)))
        adj = adj.coalesce().to(args.device)
        return graph.edge_index, graph.adj, feature, labels, train_mask, valid_mask, test_mask

    elif args.dataset == 'arxiv-year':
        dataset = NCDataset('arxiv-year')
        nclass = 5
        ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
        dataset.graph = ogb_dataset.graph
        dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
        dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])
        # dataset.graph['node_year'] = torch.as_tensor(dataset.graph['node_year'])
        dict_data = {
            'num_nodes': 169343,
            'edge_index': dataset.graph['edge_index'],
            'x': dataset.graph['node_feat'],
            'node_year': dataset.graph['node_year'],
            'y': dataset.graph['node_year'],
        }
        graph = Data(**dict_data)
        graph.x = normalize(graph.x)
        feature = torch.from_numpy(graph.x).float().to(args.device)
        # feature = graph.x.float().to(args.device)
        # adj = to_sparsetensor(graph)
        # row, col = graph.edge_index
        # edge_count = graph.edge_index.shape[1]
        # degree = torch.bincount(row, minlength=edge_count).float()
        # degree_inv_sqrt = 1. / torch.sqrt(degree)
        # edge_weight = degree_inv_sqrt[row] * degree_inv_sqrt[col]
        edge_weight = torch.ones(graph.edge_index.shape[1])
        adj = torch.sparse_coo_tensor(graph.edge_index, edge_weight, size=[feature.size(0), feature.size(0)])
        # graph.adj = normalize_large_adj(adj, 'DAD')
        labels = even_quantile_labels(
            dataset.graph['node_year'].flatten(), nclass, verbose=False)
        # labels = torch.as_tensor(label).reshape(-1, 1)
        train_idx, valid_idx, test_idx = rand_train_test_idx(
            torch.as_tensor(labels).reshape(-1, 1), train_prop=.5, valid_prop=.25)
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
        train_mask = split_idx['train']
        valid_mask = split_idx['valid']
        test_mask = split_idx['test']
        mask_no_loop = dataset.graph['edge_index'][0] != dataset.graph['edge_index'][1]
        edge_index_no_loop = dataset.graph['edge_index'][:, mask_no_loop]

        adj_no_loop = torch.sparse_coo_tensor(
            indices=edge_index_no_loop,
            values=torch.ones(edge_index_no_loop.shape[1]),
            size=(feature.shape[0], feature.shape[0])
        ).coalesce()
        return adj, adj_no_loop.to(args.device), feature, torch.from_numpy(labels).long().to(args.device), train_mask, valid_mask, test_mask

    elif args.dataset == 'genius':
        data = sio.loadmat(args.path_graph + args.dataset + '.mat')
        feature = torch.from_numpy(data['node_feat']).to(args.device)
        # feature = F.normalize(feature, p=2.0, dim=1)
        feature = feature.to(args.device)
        labels = data['label'].flatten()
        labels = labels - min(set(labels))
        train_mask, valid_mask, test_mask = generate_partition(labels, args)
        labels = torch.from_numpy(labels).long().to(args.device)
        edge_index = torch.from_numpy(data['edge_index'])
        mask_no_loop = edge_index[0] != edge_index[1]
        edge_index_no_loop = edge_index[:, mask_no_loop]

        adj_no_loop = torch.sparse_coo_tensor(
            indices=edge_index_no_loop,
            values=torch.ones(edge_index_no_loop.shape[1]),
            size=(feature.shape[0], feature.shape[0])
        ).coalesce()
        edge_count = edge_index.shape[1]
        row, col = edge_index
        degree = torch.bincount(row, minlength=edge_count).float()
        degree_inv_sqrt = 1. / torch.sqrt(degree)
        edge_weight = degree_inv_sqrt[row] * degree_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=[feature.size(0), feature.size(0)])
        return adj.to(args.device), adj_no_loop.to(args.device), feature, labels, train_mask, valid_mask, test_mask

    elif args.dataset == 'Penn94':
        # 读取数据
        mat = sio.loadmat('/gpu-data/hsj/GCN_dataset/' + args.dataset + '.mat')
        A = mat['A']  # scipy sparse matrix (usually csr_matrix)
        metadata = mat['local_info']

        # 创建 PyTorch 格式的稀疏邻接矩阵（edge_index 格式）
        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
        edge_index, _ = coalesce(edge_index, None, A.shape[0], A.shape[1])  # 去重+排序

        # ========== 稀疏邻接矩阵（带自环） ==========
        adj_with_loop = torch.sparse_coo_tensor(
            indices=edge_index,
            values=torch.ones(edge_index.shape[1]),
            size=(A.shape[0], A.shape[1])
        ).coalesce()

        # ========== 稀疏邻接矩阵（不带自环） ==========
        # 去除自环边：条件是 row != col
        mask_no_loop = edge_index[0] != edge_index[1]
        edge_index_no_loop = edge_index[:, mask_no_loop]

        adj_no_loop = torch.sparse_coo_tensor(
            indices=edge_index_no_loop,
            values=torch.ones(edge_index_no_loop.shape[1]),
            size=(A.shape[0], A.shape[1])
        ).coalesce()

        # ========== 特征和标签处理 ==========
        metadata = metadata.astype(np.int_)
        label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

        # 将特征 one-hot 化
        feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
        features = np.empty((A.shape[0], 0))
        for col in range(feature_vals.shape[1]):
            feat_col = feature_vals[:, col]
            feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
            features = np.hstack((features, feat_onehot))

        node_feat = torch.tensor(features, dtype=torch.float).to(args.device)
        labels = torch.tensor(label).to(args.device)

        # ========== 读取 splits ==========
        splits_drive_url = {
            'snap-patents': '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
            'pokec': '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
        }
        if not os.path.exists(f'./data/splits/{args.dataset}-splits.npy'):
            assert args.dataset in splits_drive_url.keys()
            gdown.download(
                id=splits_drive_url[args.dataset],
                output=f'./data/splits/{args.dataset}-splits.npy',
                quiet=False
            )

        splits_lst = np.load(f'./data/splits/{args.dataset}-splits.npy', allow_pickle=True)
        for i in range(len(splits_lst)):
            for key in splits_lst[i]:
                if not torch.is_tensor(splits_lst[i][key]):
                    splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
        train_mask = splits_lst[1]['train'].numpy()
        valid_mask = splits_lst[1]['valid'].numpy()
        test_mask = splits_lst[1]['test'].numpy()

        return adj_no_loop.to(args.device), node_feat, labels, torch.from_numpy(train_mask), torch.from_numpy(valid_mask), torch.from_numpy(test_mask)


def load_multi_data(args):
    data = sio.loadmat(args.path_modal + args.dataset + '.mat')
    if args.dataset == "imdb_":
        labels = data['label'].flatten()
        features = data['feature']
    elif args.dataset in ["HW", "Caltech101-all"]:
        labels = data['truth'].flatten()
        features = data['X']
    elif args.dataset == 'NoisyMNIST-70000':
        features = data['data']
    else:
        features = data['X']
    if args.dataset == 'scene15':
        for i in range(features.shape[1]):
            features[0][i] =features[0][i].T
        labels = data['truth'].flatten()
    else:
        labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    train_mask, valid_mask, test_mask = generate_partition(labels, args)
    labels = torch.from_numpy(labels).long()
    adj_list = []
    fea_list = []

    # ######Construct KNN
    concat_feature = torch.Tensor().to(args.device)
    for i in range(features.shape[1]):
        features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        if ss.isspmatrix(feature):
            feature = feature.todense()
        direction_judge = './adj_matrix/' + args.dataset + '/' + 'v' + str(i) + '/' + str(args.knns) + '_adj.npz'
        if os.path.exists(direction_judge):
            print("Loading laplacian matrix from " + str(i) + "th view of " + args.dataset)
            adj_s = torch.from_numpy(ss.load_npz(direction_judge).todense()).float().to(args.device)
        else:
            print("Constructing the adjacency matrix of "+ 'v' + str(i) + args.dataset)
            adj_s = construct_adjacency_matrix(feature, args.knns, args.pr1, args.pr2, args.common_neighbors)
            adj_s = ss.coo_matrix(adj_s)
            adj_s = adj_s + adj_s.T.multiply(adj_s.T > adj_s) - adj_s.multiply(adj_s.T > adj_s)
            adj_s_hat = construct_adj_hat(adj_s)
            save_direction = './adj_matrix/' + args.dataset + '/' + 'v' + str(i) + '/' + str(args.knns)
            if not os.path.exists(save_direction):
                os.makedirs(save_direction)
            print("Saving the adjacency matrix to " + save_direction)
            ss.save_npz(save_direction + '_adj.npz', adj_s_hat)
            adj_s = torch.from_numpy(adj_s_hat.todense()).float().to(args.device)

        adj_list.append(adj_s)
        feature = torch.from_numpy(feature).float().to(args.device)
        fea_list.append(feature)
        concat_feature = torch.cat((concat_feature, feature), dim=1)
        concat_feature = concat_feature.to(args.device)

    return adj_list, fea_list, labels, train_mask, valid_mask, test_mask

def load_graph_data(args):
    adj_list = []
    data = sio.loadmat(args.path_graph + args.dataset + '.mat')
    if args.dataset == 'deezer-europe':
        features = data['features']
        adj = data['A']
        labels = data['label'].flatten()
        labels = labels - min(set(labels))
        train_mask, valid_mask, test_mask = generate_partition(labels, args)
        labels = torch.from_numpy(labels).long()

        adj_csc = adj  # 传入的 csc_matrix

        # 转成 coo，做 normalize、去自环等一切操作都在 scipy.sparse 上完成
        adj_norm = adj_csc + adj_csc.T.multiply(adj_csc.T > adj_csc) \
                   - adj_csc.multiply(adj_csc.T > adj_csc)
        # 归一化操作（假设 normalized_adjacency 接受 scipy.spmatrix 返回 scipy.spmatrix）
        _, adj_norm = normalized_adjacency(adj_norm + sp.eye(adj_norm.shape[0]))
        adj_norm = adj_norm.tocoo()

        # 构造 PyTorch 稀疏张量
        indices = torch.vstack([
            torch.from_numpy(adj_norm.row).long(),
            torch.from_numpy(adj_norm.col).long()
        ])
        values = torch.from_numpy(adj_norm.data).float()
        sparse_adj = torch.sparse_coo_tensor(indices, values, adj_norm.shape).coalesce().to(args.device)

        # 去掉自环
        mask = indices[0] != indices[1]
        A_without_self_loop = torch.sparse_coo_tensor(
            indices[:, mask],
            values[mask],
            sparse_adj.shape
        ).coalesce().to(args.device)

        # ----------- 2. 稠密化特征并归一化 -----------
        # 假设 features_csc 是 scipy.sparse.csc_matrix
        features_csc = features

        # 先转成 NumPy 的稠密数组（若真的太大导致 OOM，再考虑稀疏处理）
        features_dense = features_csc.toarray()

        # 再转 PyTorch 张量并做 L2 归一化
        feature = torch.from_numpy(features_dense).float().to(args.device)
        feature = F.normalize(feature, p=2.0, dim=1)
    else:
        features = data['X']
        adj = data['adj']
        labels = data['Y'].flatten()
        labels = labels - min(set(labels))
        train_mask, valid_mask, test_mask = generate_partition(labels, args)
        labels = torch.from_numpy(labels).long()

        ##引入攻击
        # file_path = '/gpu-data/hsj/Rewire_graph/attack_adj/graph/{}_meta_adj_{}_{}.npz'.format(args.dataset, 0.25,
        #                                                                                         15)
        # data = np.load(file_path)
        # values = data['data']
        # indices = data['indices']
        # indptr = data['indptr']
        # shape = tuple(data['shape'])
        # adj_matrix = csc_matrix((values, indices, indptr), shape=shape)
        # adj_adv = torch.from_numpy(adj_matrix.toarray().astype(np.float32))
        ############

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
        # 创建掩码过滤自环边 (i == j)
        mask = edge_indices[0] != edge_indices[1]
        filtered_indices = edge_indices[:, mask]
        filtered_values = edge_values[mask]
        # 生成不含自环边的新邻接矩阵
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

def load_data_Isogram(args):
    data = sio.loadmat(args.path_graph + args.dataset + '.mat')
    if args.dataset == "imdb_":
        labels = data['label'].flatten()
        features = data['feature']
    else:
        features = data['X']
        if args.dataset == 'scene15':
            for i in range(features.shape[1]):
                features[0][i] =features[0][i].T
        labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    train_mask, valid_mask, test_mask = generate_partition(labels, args)
    labels = torch.from_numpy(labels).long()
    adj_list = []

    ###########原始的A
    adjs = data['adj']
    # # # ###用攻击后的邻接矩阵替换，仅替换第一个
    # file_path = '/gpu-data/hsj/Attack_code/meta/multiview/{}_meta_adj_v{}_{}_{}.npz'.format(args.dataset, 2, 0.25, 15)
    # data = np.load(file_path)
    # values = data['data']
    # indices = data['indices']
    # indptr = data['indptr']
    # shape = tuple(data['shape'])
    # adj_matrix = csc_matrix((values, indices, indptr), shape=shape)
    # # adjs[0][0] = adj_matrix
    # adjs[0][2] = adj_matrix
    # adjs[0][1] = ss.coo_matrix(adjs[0][1])
    # adjs[0][2] = ss.coo_matrix(adjs[0][2])
    for i in range(adjs.shape[1]):
        adj = adjs[0][i]
        adj = ss.coo_matrix(adj)
        adj_s = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj_s_hat = construct_adj_hat(adj_s)
        adj_s = torch.from_numpy(adj_s_hat.todense()).float().to(args.device)
        # indices = adj_s.nonzero(as_tuple=False).t()
        # values = adj_s[indices[0], indices[1]]
        # sparse_adj = torch.sparse_coo_tensor(indices, values, adj.shape)
        # adj_list.append(sparse_adj)
        adj_list.append(adj_s)
        # hidden_dims = [feature.shape[1]] + args.hdim + [num_classes]
    feature = torch.from_numpy(features).float().to(args.device)
    return adj_list, feature, labels, train_mask, valid_mask, test_mask

def construct_sparse_float_tensor(np_matrix):
    """
        construct a sparse float tensor according a numpy matrix
    :param np_matrix: <class 'numpy.ndarray'>
    :return: torch.sparse.FloatTensor
    """
    sp_matrix = ss.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                             torch.FloatTensor(three_tuple[1]),
                                             torch.Size(three_tuple[2]))
    return sparse_tensor


def feature_normalization(features, normalization_type = 'normalize'):
    for idx, fea in enumerate(features[0]):
        if normalization_type == 'minmax_scale':
            features[0][idx] = minmax_scale(fea)
        elif normalization_type == 'maxabs_scale':
            features[0][idx] = maxabs_scale(fea)
        elif normalization_type == 'normalize':
            features[0][idx] = normalize(fea)
        elif normalization_type == 'robust_scale':
            features[0][idx] = robust_scale(fea)
        elif normalization_type == 'scale':
            features[0][idx] = scale(fea)
        elif normalization_type == '255':
            features[0][idx] = np.divide(fea, 255.)
        elif normalization_type == '50':
            features[0][idx] = np.divide(fea, 50.)
        else:
            print("Please enter a correct normalization type!")
    return features


def preprocess_features(features, sparse=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    # import ipdb; ipdb.set_trace()
    try:
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
    except Exception as e:
        print(e)

    if sparse:
        return sparse_to_tuple(features)
    return features.todense()


def load_adjacency_multiview(multi_view_features, k):
    adj_list = []
    adj_hat_list = []

    for idx, features in enumerate(multi_view_features[0]):
        adj, adj_hat = load_adj(features, k_nearest_neighobrs=k)
        adj_list.append(adj.todense())
        adj_hat_list.append(adj_hat.todense())

    adj_list = np.array(adj_list)
    adj_hat_list = np.array(adj_hat_list)

    return adj_list, adj_hat_list


def load_adj(features, normalization=True, normalization_type='normalize',
              k_nearest_neighobrs=10, prunning_one=False, prunning_two=True , common_neighbors=2):
    if normalization:
        if normalization_type == 'minmax_scale':
            features = minmax_scale(features)
        elif normalization_type == 'maxabs_scale':
            features = maxabs_scale(features)
        elif normalization_type == 'normalize':
            features = normalize(features)
        elif normalization_type == 'robust_scale':
            features = robust_scale(features)
        elif normalization_type == 'scale':
            features = scale(features)
        elif normalization_type == '255':
            features = np.divide(features, 255.)
        elif normalization_type == '50':
            features = np.divide(features, 50.)
        else:
            print("Please enter a correct normalization type!")

    # construct three kinds of adjacency matrix

    adj, adj_wave, adj_hat = construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one,
                                                        prunning_two, common_neighbors)
    return adj, adj_hat



def preprocess_adj(adj, norm=True, sparse=False):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = adj + sp.eye(adj.shape[0])
    if norm:
        adj = normalize_adj(adj)
    if sparse:
        return sparse_to_tuple(adj)
    return adj.todense()


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def get_indice_graph(adj, mask, size, keep_r=1.0):
    indices = mask.nonzero()[0]
    if keep_r < 1.0:
        indices = np.random.choice(indices, int(indices.size*keep_r), False)
    pre_indices = set()
    indices = set(indices)
    while len(indices) < size:
        new_add = indices - pre_indices
        if not new_add:
            break
        pre_indices = indices
        candidates = get_candidates(adj, new_add) - indices
        if len(candidates) > size - len(indices):
            candidates = set(np.random.choice(list(candidates), size-len(indices), False))
        indices.update(candidates)
    print('indices size:-------------->', len(indices))
    return sorted(indices)

def get_candidates(adj, new_add):
    return set(adj[sorted(new_add)].sum(axis=0).nonzero()[1])

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


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


def normalize_adj(adj):    # 这部分就是计算D-1/2AD-1/2
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # 计算每一个节点的度

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.   # d_inv_sqrt是1的话则对应位置置为0，否则不变
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one, prunning_two, common_neighbors):
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs + 1, algorithm='ball_tree').fit(features)
    adj_construct = nbrs.kneighbors_graph(features)  # <class 'scipy.sparse.csr.csr_matrix'>
    adj = ss.coo_matrix(adj_construct)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if prunning_one:
        # Pruning strategy 1
        original_adj = adj.A
        judges_matrix = original_adj == original_adj.T
        adj = original_adj * judges_matrix
        adj = ss.csc_matrix(adj)
    # obtain the adjacency matrix without self-connection
    adj = adj - ss.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        # Pruning strategy 2
        adj = adj.A
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = ss.coo_matrix(adj)
        adj.eliminate_zeros()

    # print("The construction of Laplacian matrix is finished!")
    # print("The time cost of construction: ", time.time() - start_time)
    adj = ss.coo_matrix(adj)
    return adj


def construct_adj_hat(adj):
    """
        construct the Laplacian matrix
    :param adj: original adj matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    # adj = ss.coo_matrix(adj)
    adj_ = ss.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1)) # <class 'numpy.ndarray'> (n_samples, 1)
    print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    # lp = ss.eye(adj.shape[0]) - adj_wave
    return adj_wave


def generate_partition(gnd, args):
    '''
    Generate permutation for training, validating and testing data.
    '''
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    test_ratio = 1 - train_ratio - valid_ratio
    N = gnd.shape[0]
    each_class_num = count_each_class_num(gnd)
    training_each_class_num = {} ## number of labeled samples for each class
    valid_each_class_num = {}

    for label in each_class_num.keys():
        if args.data_split_mode == "Ratio":
            training_each_class_num[label] = max(round(each_class_num[label] * train_ratio), 1) # min is 1
            # valid_each_class_num[label] = max(round(each_class_num[label] * valid_ratio), 1) # min is 1
            valid_num = max(round(N * valid_ratio), 0) # min is 1
            test_num= max(round(N * test_ratio), 1) # min is 1
        else:
            training_each_class_num[label] = args.num_train_per_class
            valid_num = args.num_val
            test_num = args.num_test

    # index of labeled and unlabeled samples
    train_mask = torch.from_numpy(np.full((N), False))
    valid_mask = torch.from_numpy(np.full((N), False))
    test_mask = torch.from_numpy(np.full((N), False))

    # shuffle the data
    data_idx = [i for i in range(len(gnd))]
    # print(index)
    if args.seed >= 0:
        random.seed(args.seed)
        random.shuffle(data_idx)

    # Get training data
    for idx in data_idx:
        label = gnd[idx]
        if (training_each_class_num[label] > 0):
            training_each_class_num[label] -= 1
            train_mask[idx] = True
    for idx in data_idx:
        if train_mask[idx] == True:
            continue
        if (valid_num > 0):
            valid_num -= 1
            valid_mask[idx] = True
        elif (test_num > 0):
            test_num -= 1
            test_mask[idx] = True
    return train_mask, valid_mask, test_mask


def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


class TwitchGamers(InMemoryDataset):
    def __init__(self, root, name='twitch_gamers', transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['twitch_gamers']

        super(TwitchGamers, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['large_twitch_edges.csv', 'large_twitch_features.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        import pandas as pd
        edges = pd.read_csv('/gpu-data/hsj/Rewire_graph/data/twitch_gamers/' + 'large_twitch_edges.csv')
        nodes = pd.read_csv('/gpu-data/hsj/Rewire_graph/data/twitch_gamers/' + 'large_twitch_features.csv')
        edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
        edge_index = to_undirected(edge_index)
        label, features = load_twitch_gamer(nodes, "mature")
        node_feat = torch.tensor(features, dtype=torch.float)
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
        data = Data(x=node_feat, edge_index=edge_index, y=torch.tensor(label))
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding

    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()

    return label, features