
from torch_sparse import SparseTensor, matmul
from torch.nn.parameter import Parameter
from utils import *

CONFIGS = {
    "fast_spmm": None,
    "csrmhspmm": None,
    "csr_edge_softmax": None,
    "fused_gat_func": None,
    "fast_spmm_cpu": None,
    "spmm_flag": False,
    "mh_spmm_flag": False,
    "fused_gat_flag": False,
    "spmm_cpu_flag": False,
}


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, device, activation=None, bias=True):
        super(GraphConvSparse, self).__init__()
        self.device = device
        from torch.nn.parameter import Parameter
        self.weight = Parameter(glorot_init(input_dim, output_dim))
        self.ortho_weight = torch.zeros_like(self.weight)
        self.activation = activation
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, edge_index, edge_weight):
        N, H = inputs.shape[0], inputs.shape[1]
        row, col = edge_index
        value = torch.nan_to_num(edge_weight, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N)).to(self.device)
        output = matmul(adj, inputs)
        return output


def spmm(graph, x, actnn=False, fast_spmm=None, fast_spmm_cpu=None):
    if fast_spmm is None:
        initialize_spmm()
        fast_spmm = CONFIGS["fast_spmm"]
    if fast_spmm_cpu is None:
        initialize_spmm_cpu()
        fast_spmm_cpu = CONFIGS["fast_spmm_cpu"]
    if fast_spmm is not None and str(x.device) != "cpu":
        if graph.out_norm is not None:
            x = graph.out_norm * x

        row_ptr, col_indices = graph.row_indptr, graph.col_indices
        csr_data = graph.raw_edge_weight
        if x.dtype == torch.half:
            csr_data = csr_data.half()
        x = fast_spmm(row_ptr.int(), col_indices.int(), x, csr_data, graph.is_symmetric(), actnn=actnn)

        if graph.in_norm is not None:
            x = graph.in_norm * x
    elif fast_spmm_cpu is not None and str(x.device) == "cpu" and x.requires_grad is False:
        if graph.out_norm is not None:
            x = graph.out_norm * x

        row_ptr, col_indices = graph.row_indptr, graph.col_indices
        csr_data = graph.raw_edge_weight
        x = fast_spmm_cpu(row_ptr.int(), col_indices.int(), csr_data, x)

        if graph.in_norm is not None:
            x = graph.in_norm * x
    else:
        row, col = graph.edge_index
        x = spmm_scatter(row, col, graph.edge_weight, x)
    return x


def spmm_scatter(row, col, values, b):
    r"""
    Args:
        (row, col): Tensor, shape=(2, E)
        values : Tensor, shape=(E,)
        b : Tensor, shape=(N, d)
    """
    output = b.index_select(0, col) * values.unsqueeze(-1).to(b.dtype)
    output = torch.zeros_like(b).scatter_add_(0, row.unsqueeze(-1).expand_as(output), output)
    return output


def initialize_spmm_cpu():
    if CONFIGS["spmm_cpu_flag"]:
        return
    CONFIGS["spmm_cpu_flag"] = True

    from cogdl.operators.spmm import spmm_cpu

    CONFIGS["fast_spmm_cpu"] = spmm_cpu


def initialize_spmm():
    if CONFIGS["spmm_flag"]:
        return
    CONFIGS["spmm_flag"] = True
    if torch.cuda.is_available():
        from cogdl.operators.spmm import csrspmm

        CONFIGS["fast_spmm"] = csrspmm


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.FloatTensor(input_dim, output_dim)*2*init_range - init_range
    return initial


class Model(nn.Module):
    def __init__(self, args, device, num_features, hidden_dims, num_class, dropout, n):
        super(Model, self).__init__()
        self.hidden_dims = [hidden_dims]
        self.dropout = dropout
        self.device = device
        self.args = args
        self.layers = int(args.layers)
        self.num_class = num_class

        self.gc = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(num_features, self.hidden_dims[0] // 2))
        self.fcs.append(nn.Linear(self.hidden_dims[0] // 2, num_class))
        self.lns.append(nn.LayerNorm(self.hidden_dims[0] // 2))
        for i in range(self.layers):
            self.gc.append(GraphConvSparse(self.hidden_dims[0] // 2, self.hidden_dims[0] // 2, self.device))
            self.lns.append(nn.LayerNorm(self.hidden_dims[0] // 2))
        self.layer_norm_first = True
        self.use_ln = True
        self.adj_list = []

        self.weight = nn.Parameter(torch.rand(self.layers + 1))

        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv1, stdv1)

    def forward(self, feature, adj_ori, candidates_):
        self.adj_list = []
        candidates = candidates_.clone()
        feature.requires_grad_(True)
        input = self.fcs[0](feature)
        if self.layer_norm_first:
            input = self.lns[0](input)

        hidden = input
        energies = (self.fcs[-1](input) + 1e-8).logsumexp(dim=1)
        energy_input = input
        for i in range(len(self.gc)):
            if self.args.is_energy:
                with torch.autograd.set_detect_anomaly(True):
                    rewire_adj = Rewire_adj_matrix(adj_ori, hidden, energies, self.args, candidates, i)
                    adj = rewire_adj.detach().clone().requires_grad_(False)
                    self.adj_list.append(adj)  # 0, ..., L-1
                with torch.no_grad():
                    energies, energy_input = self.forward_energy(energy_input, adj, i, 's')
            adj_f = adj.coalesce()
            row = (adj_f.indices())[0].long()
            col = (adj_f.indices())[1].long()
            edge_index = torch.stack([row, col])
            edge_weight = adj_f.values()
            hidden = F.relu(self.gc[i](hidden, edge_index, edge_weight))
            if self.use_ln:
                hidden = self.lns[i + 1](hidden)
            hidden = F.dropout(hidden, self.dropout, training=self.training)

            hidden = self.args.alpha * hidden + (1 - self.args.alpha) * input
        output = self.fcs[-1](hidden)

        return output

    def forward_energy(self, input, adj, i, flag):
        if flag == 'a':
            input.requires_grad_(True)
            input = self.fcs[0](input)
            if self.layer_norm_first:
                input = self.lns[0](input)
            hidden = input
            for l in range(self.layers):
                adj_f = adj[l].coalesce()
                row = (adj_f.indices())[0].long()
                col = (adj_f.indices())[1].long()
                edge_index = torch.stack([row, col])
                edge_weight = adj_f.values()
                hidden = F.relu(self.gc[l](hidden, edge_index, edge_weight))
                if self.use_ln:
                    hidden = self.lns[l + 1](hidden)
                hidden = F.dropout(hidden, self.dropout, training=self.training)
                hidden = self.args.alpha * hidden + (1 - self.args.alpha) * input
            output = self.fcs[-1](hidden) + 1e-8
            p = output.logsumexp(dim=1)
            return p
        else:
            hidden = input
            adj_f = adj.coalesce()
            row = (adj_f.indices())[0].long()
            col = (adj_f.indices())[1].long()
            edge_index = torch.stack([row, col])
            edge_weight = adj_f.values()
            hidden = F.relu(self.gc[i](hidden, edge_index, edge_weight))
            if self.use_ln:
                hidden = self.lns[i + 1](hidden)
            hidden = F.dropout(hidden, self.dropout, training=self.training)
            hidden = self.args.alpha * hidden + (1 - self.args.alpha) * input
            output = self.fcs[-1](hidden) + 1e-8
            p = output.logsumexp(dim=1)
            return p, hidden

    def adjust_bn_layers(self, feature, test_model, args, agu_adj_sparse):
        bn_params = []
        num_nodes = feature.size(0)
        for name, param in test_model.named_parameters():
            if 'lns' in name:
                bn_params.append(param)
        optimizer_ = torch.optim.Adam(bn_params, lr=args.lr, weight_decay=args.weight_decay)
        test_model.train()
        optimizer_.zero_grad()
        p_data = test_model.forward_energy(feature, self.adj_list, 0, 'a')
        shuf_feats = feature[:, torch.randperm(feature.size(1))]  # shuffle features
        p_neigh = test_model.forward_energy(shuf_feats, agu_adj_sparse, 0, 'a')
        energy = p_data - p_neigh / p_data
        feature.requires_grad_(True)
        energy_grad = torch.autograd.grad(energy.sum(), feature, create_graph=True)[0]
        energy_grad_inner = torch.sum(energy_grad ** 2)
        energy_squared_sum = torch.sum(energy ** 2)
        neigh_loss = 1 / num_nodes * (energy_grad_inner + 1 / 2 * energy_squared_sum)
        neigh_loss.backward()
        optimizer_.step()
        del p_data, p_neigh, energy, energy_grad, energy_grad_inner, energy_squared_sum, shuf_feats, self.adj_list
        torch.cuda.empty_cache()

    def thred_proj(self, theta):
        theta_sigmoid = torch.sigmoid(theta)
        theta_sigmoid_mat = theta_sigmoid.repeat(1, theta_sigmoid.shape[0])
        theta_sigmoid_triu = torch.triu(theta_sigmoid_mat)
        theta_sigmoid_diag = torch.diag(theta_sigmoid_triu.diag())
        theta_sigmoid_tri = theta_sigmoid_triu + theta_sigmoid_triu.t() - theta_sigmoid_diag
        return theta_sigmoid_tri


class DifferentiableAdjMask(nn.Module):
    def __init__(self, device, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.kernel = torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 1, 3) / 3

    def forward(self, A, T, D, M):
        edge_indices = A.indices()
        edge_rows, edge_cols = edge_indices
        N = A.size(0)

        sorted_idx = torch.argsort(T, descending=True, stable=True)
        sorted_edges = torch.stack([edge_rows[sorted_idx], edge_cols[sorted_idx]], dim=0)
        sorted_T = T[sorted_idx]

        mask_high = (D >= D.mean()) & (M >= M.mean()) & (T >= T.mean())
        High_l = torch.nonzero(mask_high, as_tuple=False).view(-1)

        mask = torch.isin(sorted_idx, High_l)
        cumulative_counts = torch.cumsum(mask.long(), dim=0)

        mu_k = cumulative_counts / (High_l.numel() + 1e-8)

        total_edges = sorted_T.numel()
        self_loop_ratio = N / total_edges
        adjustment = int(total_edges * self_loop_ratio)
        non_self_T = sorted_T[:total_edges - adjustment]
        mu_k = mu_k[:total_edges - adjustment]
        T_smooth = (torch.cat([non_self_T[:1], non_self_T[:-1]]) + non_self_T + torch.cat(
            [non_self_T[1:], non_self_T[-1:]])) / 3

        T_ratios = T_smooth[:-1] / (T_smooth[1:] + 1e-8)
        T_ratios = torch.cat([T_ratios, torch.ones(1, device=T_smooth.device)])
        TGap = mu_k * T_ratios
        search_ratio = 1
        search_range = max(1, int(TGap.numel() * search_ratio))
        L = torch.argmax(TGap[:search_range]).item()

        delete_num = L
        delete_edges = sorted_edges[:, :delete_num]
        original_linear = edge_rows * N + edge_cols
        delete_linear = delete_edges[0] * N + delete_edges[1]
        retain_mask = ~torch.isin(original_linear, delete_linear)

        retain_indices = edge_indices[:, retain_mask]
        retain_values = A.values()[retain_mask]
        A_new = torch.sparse_coo_tensor(retain_indices, retain_values, size=A.size())

        return A_new


def Rewire_adj_matrix(A_without_self_loop, feature, energy, args, candidates, i):

    x = feature.detach()

    adj = A_without_self_loop.coalesce()

    edge_idx = adj.indices()
    node_i, node_j = edge_idx[0], edge_idx[1]
    D = torch.norm(x[node_i] - x[node_j], p=2, dim=1)
    M = energy[node_i] * energy[node_j]
    T = D * M

    if args.dataset in ['Tolokers', 'Penn94', 'Cora', 'Citeseer', 'Pubmed']:
        adder = add_edges(args)
        A_rewired = adder(adj, x, energy, args, candidates)
    else:
        masker = DifferentiableAdjMask(args.device)
        A_trimmed = masker(adj, T, D, M)

        adder = add_edges(args)
        A_rewired = adder(A_trimmed, x, energy, args, candidates)

    A_rewired = symmetric_normalize(A_rewired, i + 1, args.layers)

    return A_rewired


def symmetric_normalize(A, order, layers):
    N = A.size(0)

    diag_indices = torch.arange(N, device=A.device)
    diag = torch.stack([diag_indices, diag_indices], dim=0)
    self_loop = torch.sparse_coo_tensor(diag, torch.ones(N, device=A.device), A.size(), device=A.device)
    A_hat = (A + self_loop).coalesce()

    deg = torch.sparse.sum(A_hat, dim=1).to_dense() + 1e-8
    deg_inv_sqrt = deg.pow(-0.5)

    indices = A_hat.indices()
    norm_vals = (deg_inv_sqrt[indices[0]] * deg_inv_sqrt[indices[1]]) * ((layers - order + 1) / layers)

    return torch.sparse_coo_tensor(indices, norm_vals, A.size(), device=A.device).coalesce()


class add_edges(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.temperature = 0.1
        self.w_D = nn.Parameter(torch.ones(3))
        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.w_D.size(0))
        self.w_D.data.uniform_(-stdv1, stdv1)

    def forward(self, A, feature, energy, args, candidates):
        x = feature.detach()

        node_i_add, node_j_add = candidates[0], candidates[1]

        D_edges_add = torch.norm(x[node_i_add] - x[node_j_add], p=2, dim=1)
        M_edges_add = energy[node_i_add] * energy[node_j_add]
        T_edges_add = D_edges_add * M_edges_add

        sorted_idx = torch.argsort(T_edges_add, descending=True, stable=True)
        sorted_edges = torch.stack([node_i_add[sorted_idx], node_j_add[sorted_idx]], dim=0)  #
        sorted_T = T_edges_add[sorted_idx]

        epsilon = 1e-8
        T_min = sorted_T.min()
        T_max = sorted_T.max()
        normalized_T = (sorted_T - T_min) / (T_max - T_min + epsilon)
        p = 1.0 - normalized_T
        sampling_rate = self.args.sampling_rate
        scale = sampling_rate / (p.mean() + epsilon)
        p = torch.clamp(p * scale, max=1.0)

        logits = torch.stack([torch.log(1 - p + epsilon), torch.log(p + epsilon)], dim=-1)
        temperature = 0.3
        samples = F.gumbel_softmax(logits, tau=temperature, hard=False)
        soft_selected = samples[:, 1]
        selected_edges = sorted_edges * soft_selected.unsqueeze(0)
        old_indices = A.coalesce().indices()
        new_indices = torch.cat([old_indices, selected_edges], dim=1)
        new_values = torch.ones(new_indices.shape[1], device=A.device)

        N = A.size(0)
        sort_order = torch.argsort(new_indices[0] * N + new_indices[1], stable=True)
        new_indices = new_indices[:, sort_order]
        new_values = new_values[sort_order]

        A_new = torch.sparse_coo_tensor(new_indices, new_values, A.size(), device=A.device).coalesce()
        return A_new
