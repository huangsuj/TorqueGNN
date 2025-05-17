
import warnings
from tqdm import tqdm
from args import parameter_parser
from utils import *
from Dataloader import load_graph_data
import configparser
from TorqueGNN import Model
import numpy as np
# import wandb
import torch.nn.functional as F
import copy
from copy import deepcopy
import torch
def aug_random_edge(nb_nodes, edge_index, perturb_percent=0.2, drop_edge=True, add_edge=True, self_loop=True,
                     use_avg_deg=True, seed=None):

    total_edges = edge_index.shape[1]
    avg_degree = int(total_edges / nb_nodes)

    edge_dict = {}
    for i in range(nb_nodes):
        edge_dict[i] = set()

    for edge in edge_index:
        i, j = edge[0], edge[1]
        i = i.item()
        j = j.item()
        edge_dict[i].add(j)
        edge_dict[j].add(i)

    if drop_edge:
        for i in range(nb_nodes):
            d = len(edge_dict[i])
            if use_avg_deg:
                num_edge_to_drop = avg_degree
            else:
                num_edge_to_drop = int(d * perturb_percent)

            node_list = list(edge_dict[i]) #
            num_edge_to_drop = min(num_edge_to_drop, d)
            sampled_nodes = random.sample(node_list, num_edge_to_drop)

            for j in sampled_nodes:
                edge_dict[i].discard(j)
                edge_dict[j].discard(i)

    node_list = [i for i in range(nb_nodes)]
    add_list = []
    for i in range(nb_nodes):
        if use_avg_deg:
            num_edge_to_add = avg_degree
        else:
            d = len(edge_dict[i])
            num_edge_to_add = int(d * perturb_percent)

        sampled_nodes = random.sample(node_list, num_edge_to_add)
        for j in sampled_nodes:
            add_list.append((i, j))

    if add_edge:
        for edge in add_list:
            u = edge[0]
            v = edge[1]
            edge_dict[u].add(v)
            edge_dict[v].add(u)

    if self_loop:
        for i in range(nb_nodes):
            edge_dict[i].add(i)

    updated_edges = set()
    for i in range(nb_nodes):
        for j in edge_dict[i]:
            updated_edges.add((i, j))
            updated_edges.add((j, i))

    row = []
    col = []
    for edge in updated_edges:
        u = edge[0]
        v = edge[1]
        row.append(u)
        col.append(v)

    aug_edge_index = [row, col]
    aug_edge_index = torch.tensor(aug_edge_index)

    return aug_edge_index

def construct_agu_adj(feature, data_edge_index, device, args):

    clean_edge_index = data_edge_index.to(device)
    num_nodes = feature.size(0)
    aug_clean_edge_index = aug_random_edge(num_nodes, clean_edge_index, perturb_percent=0.2,
                                            drop_edge=True, add_edge=True, self_loop=True, use_avg_deg=True,seed=args.seed)
    aug_clean_edge_index = aug_clean_edge_index.to(device)
    ### 增广图边权为1
    aug_clean_edge_weights = torch.ones(aug_clean_edge_index.size(1), dtype=torch.float, device=device)
    adj_sparse = torch.sparse_coo_tensor(
        indices=aug_clean_edge_index,
        values=aug_clean_edge_weights,
        size=(num_nodes, num_nodes)
    )

    return adj_sparse

def train_start(feature, adj_ori, train_mask, valid_mask, args, labels, best_valid_acc, model, optimizer, best_model, candidates):
        loss_function1 = torch.nn.NLLLoss()
        model.train()
        output = model(feature, adj_ori, candidates)
        output_softmax = F.log_softmax(output, dim=1)
        if args.dataset in ['questions', 'Tolokers']:
            Loss = F.binary_cross_entropy_with_logits(output[train_mask], F.one_hot(labels, num_classes=2).float()[train_mask])
            from sklearn.metrics import roc_auc_score
            pred_labels = F.softmax(output)[train_mask].squeeze()
            score = roc_auc_score(F.one_hot(labels, num_classes=2).float().cpu().detach().numpy()[train_mask.cpu().detach()], pred_labels.detach().cpu())
            Train_ACC = score
        else:
            Loss = loss_function1(output_softmax[train_mask], labels[train_mask].long())
            pred_labels = torch.argmax(output, 1).data.cpu().numpy()
            Train_ACC, _, _, _, _ = get_evaluation_results(
                labels.cpu().detach().numpy()[train_mask.cpu().detach()],
                pred_labels[train_mask.cpu().detach()])
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        pred_labels = torch.argmax(output, 1).data.cpu().numpy()
        Train_ACC, _, _, _, _ = get_evaluation_results(
            labels.cpu().detach().numpy()[train_mask.cpu().detach()],
            pred_labels[train_mask.cpu().detach()])

        with torch.no_grad():
            model.eval()
            output_ = model(feature, adj_ori, candidates)
            if args.dataset in ['questions', 'Tolokers']:
                from sklearn.metrics import roc_auc_score
                pred_labels = F.softmax(output_)[valid_mask].squeeze()
                score = roc_auc_score(
                    F.one_hot(labels, num_classes=2).float().cpu().detach().numpy()[valid_mask.cpu().detach()],
                    pred_labels.detach().cpu())
                Valid_per = score
            else:
                pred_labels = torch.argmax(output_, 1).cpu().detach().numpy()
                Valid_per, _, _, _, _ = get_evaluation_results(
                    labels.cpu().detach().numpy()[valid_mask.cpu().detach()],
                    pred_labels[valid_mask.cpu().detach()])

        if (Valid_per >= best_valid_acc):
            best_valid_acc = Valid_per
            best_model = copy.deepcopy(model)

        return Loss.item(), Valid_per, Train_ACC, best_model, best_valid_acc


def train(args):
    adj_ori, A_without_self_loop, feature, labels, train_mask, valid_mask, test_mask = load_graph_data(args)
    args.ini_edge = A_without_self_loop.coalesce()._indices().size(1)
    num_classes = labels.unique().size(0)
    labels = labels.to(args.device)

    edge_indices = A_without_self_loop.coalesce().indices()
    data_edge_index = edge_indices.to(torch.long).contiguous()
    agu_adj_sparse = construct_agu_adj(feature, data_edge_index, args.device, args)
    agu_adj_sparse_list = [agu_adj_sparse for i in range(int(args.layers))]


    best_valid_acc = 0
    candidates = get_topk_candidates(feature, A_without_self_loop, k=args.add_edge)
    test_model = Model(args, args.device, feature.shape[1], args.hdim, num_classes, args.dropout,
                       feature.shape[0]).to(args.device)
    best_model = copy.deepcopy(test_model)
    optimizer = torch.optim.Adam(test_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            loss, Valid_ACC, Train_ACC, best_model, best_valid_acc = \
                train_start(feature, adj_ori, train_mask, valid_mask, args, labels, best_valid_acc, test_model, optimizer,
                            best_model, candidates)
            if args.is_energy:
                for k in range(args.energy_epochs):
                    test_model.adjust_bn_layers(feature, test_model, args, agu_adj_sparse_list)
                loss, Valid_ACC, Train_ACC, best_model, best_valid_acc = \
                    train_start(feature, adj_ori, train_mask, valid_mask, args, labels, best_valid_acc,
                                test_model, optimizer, best_model, candidates)

            pbar.set_postfix({
                "Epoch": epoch + 1,
                'Loss_train': '{:.4f}'.format(np.mean(loss)),
                'Train_ACC': '{:.2f}'.format(Train_ACC * 100),
            })
            pbar.update(1)
        with torch.no_grad():
            best_model.eval()
            output = best_model(feature, adj_ori, candidates)
            print("Evaluating the model")
            if args.dataset in ['questions', 'Tolokers']:
                from sklearn.metrics import roc_auc_score
                pred_labels = F.softmax(output)[test_mask].squeeze()
                score = roc_auc_score(
                    F.one_hot(labels, num_classes=2).float().cpu().detach().numpy()[test_mask.cpu().detach()],
                    pred_labels.detach().cpu())
                print("------------------------")
                print("Test_score:   {:.2f}".format(score * 100))
                print("------------------------")
                return loss, score
            else:
                pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                ACC, _, _, _, _ = get_evaluation_results(
                    labels.cpu().detach().numpy()[test_mask.cpu().detach()],
                    pred_labels[test_mask.cpu().detach()])
                print("------------------------")
                print("Test_ACC:   {:.2f}".format(ACC * 100))
                print("------------------------")

                return loss, ACC


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    conf = configparser.ConfigParser()
    args = parameter_parser()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    Graph_dataset = ['Penn94']
    args.path_graph = './data/'
    for index, item in enumerate(Graph_dataset):
        args.dataset = item
        print('--------------Graph Datasets: {}--------------------'.format(args.dataset))

        conf = configparser.ConfigParser()
        config_path = './config_demo' + '.ini'
        conf.read(config_path, encoding='utf-8')
        args.num_epoch = int(conf.getfloat(args.dataset, 'epoch'))
        args.hdim = int(conf.getfloat(args.dataset, 'hdim'))
        args.lr = conf.getfloat(args.dataset, 'lr')
        args.layers = conf.getfloat(args.dataset, 'layers')
        args.alpha = conf.getfloat(args.dataset, 'alpha')
        args.weight_decay = conf.getfloat(args.dataset, 'weight_decay')
        args.dropout = conf.getfloat(args.dataset, 'dropout')
        args.add_edge = int(conf.getfloat(args.dataset, 'add_edge'))
        args.valid_ratio = conf.getfloat(args.dataset, 'valid_ratio')
        edge_ = conf.get(args.dataset, 'add_edge').split(',')
        edge_list = [int(item.strip()) for item in edge_]


        args.device = 'cuda:0'
        tab_printer(args)
        all_ACC = []
        for n_num in range(args.n_repeated):

            loss, ACC = train(args)
            all_ACC.append(ACC)
            print("-----------------------")
            print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
            print("-----------------------")
