import argparse
import os
import os.path as osp
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch.nn as nn
import torch_geometric
import torch
import torch.nn.functional as F
from datasets.drugood_dataset import DrugOOD
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from drugood.datasets import build_dataloader, build_dataset
from drugood.models import build_backbone
from GOOD.data.good_datasets.good_twitter import GOODTwitter
from GOOD.data.good_datasets.good_twitch import GOODTwitch
from GOOD.data.good_datasets.good_webkb import GOODWebKB
from GOOD.data.good_datasets.good_sst2 import GOODSST2
from mmcv import Config
from models.mymodel import mymodel
from sklearn.metrics import roc_auc_score, recall_score, f1_score, confusion_matrix, average_precision_score, matthews_corrcoef,accuracy_score
from models.losses import get_contrast_loss, get_irm_loss, get_contrastive_loss_with_positive_and_negative_batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader, Data
from tqdm import tqdm
from utils.logger import Logger
from utils.util import args_print, set_seed
from sklearn.neighbors import KernelDensity
import json
from MIOOD.datasets.mymodel_dataset import GraphDataset_Classification, GraphDataLoader_Classification
from MIOOD.datasets.mymodel_dataset import DILIDataset
from sklearn import metrics
from sklearn.metrics import mutual_info_score
import warnings
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import dgl
with warnings.catch_warnings():
    warnings.simplefilter("ignore")


def init_args():
    parser = argparse.ArgumentParser('graph Mutual Information for OOD')
    # base config
    parser.add_argument('--device', default=0, type=int, help='cuda device')
    parser.add_argument('--root', default='/data_1/wxl22/MIOOD/data',type=str, help='root for datasets')
    parser.add_argument('--data_config', default='/data_1/wxl22/MIOOD/configs', type=str, help='root for data config')
    # parser.add_argument('--seed', default='[2019]', help=' random seed')
    parser.add_argument('--seed', default='[2019,2020,2021,2022]', help=' random seed')
    parser.add_argument('--dataset_type', default='ADMEOOD', type=str)
    parser.add_argument('--dataset', default='label_ec50_core', type=str, help='name for datasets')
    # parser.add_argument('--cuda', default=0, type=int, help='select cuda id')
    parser.add_argument('--log_dir', default='/data_1/wxl22/MIOOD/logs', type=str, help='root for logs')


    #mol represent config
    parser.add_argument('--mol_FP',type=str,choices=['atom','ss','both','none'],default='both',help='cat mol FingerPrint to Motif or Atom representation')
    parser.add_argument('--atom_in_dim', type=int, default=37, help='atom feature init dim')
    parser.add_argument('--bond_in_dim', type=int, default=13, help='bond feature init dim')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate in MLP')
    parser.add_argument('--hid_dim', type=int, default=96, help='node, edge, fg hidden dims in Net')
    parser.add_argument('--mol_in_dim', type=int, default=167, help='molecule fingerprint init dim')
    parser.add_argument('--ss_node_in_dim', type=int, default=50, help='func group node feature init dim')
    parser.add_argument('--ss_edge_in_dim', type=int, default=37, help='func group edge feature init dim')
    parser.add_argument('--step',type=int,default=4,help='message passing steps')
    parser.add_argument('--agg_op',type=str,choices=['max','mean','sum'],default='mean',help='aggregations in local augmentation')
    parser.add_argument('--resdual',type=bool,default=False,help='resdual choice in message passing')
    parser.add_argument('--attention',type=bool,default=True,help='whether to use global attention pooling')
    parser.add_argument('--heads',type=int,default=4,help='Multi-head num')


    # training config
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=50, type=int, help='training iterations')
    parser.add_argument('--lr', default=1e-4 , type=float, help='learning rate for the predictor')

    # model config
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--r', default=0.75, type=float, help='selected ratio')
    # emb dim of classifier
    parser.add_argument('-c_dim', '--classifier_emb_dim', default=128, type=int)
    # inputs of classifier
    # raw:  raw feat
    # feat: hidden feat from featurizer
    parser.add_argument('-c_in', '--classifier_input_feat', default='raw', type=str)
    parser.add_argument('--model', default='gin', type=str)
    parser.add_argument('--pooling', default='attention', type=str)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--virtual_node', action='store_true')
    parser.add_argument('--eval_metric',default='rocauc',type=str,help='specify a particular eval metric, e.g., mat for MatthewsCoef')

    # contrasting summary from the classifier or featurizer
    # rep:  classifier rep
    # feat: featurizer rep
    # conv: featurizer rep + 1L GNNConv
    parser.add_argument('-c_rep', '--contrast_rep', default='rep', type=str)
    # pooling method for the last two options in c_rep
    parser.add_argument('-c_pool', '--contrast_pooling', default='add', type=str)

    # spurious rep for maximizing I(G_S;Y)
    # rep:  classifier rep
    # conv: featurizer rep + 1L GNNConv
    parser.add_argument('-s_rep', '--spurious_rep', default='rep', type=str)
    # strength of the hinge reg, \beta in the paper
    parser.add_argument('--cl_coe', default=1, type=float)
    parser.add_argument('--mx_coe', default = 4, type=float)
    parser.add_argument('--T', default=2.75, type=float)
    parser.add_argument('--do', default="do", type=str)
    # misc
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--save_model', default='False', type=bool)  # save pred to ./pred if not empty

    args = parser.parse_args()
    return args


def evaluate(pred, gt, metric='auc', dataset='admeood'):
    if isinstance(metric, str):
        metric = [metric]
    allowed_metric = ['auc', 'acc', 'mcc', 'f1', 'pr_auc', 'recall','mul_roc']
    invalid_metric = set(metric) - set(allowed_metric)
    if len(invalid_metric) != 0:
        raise ValueError(f'Invalid Value {invalid_metric}')
    result = {}
    for M in metric:
        if M == 'auc':
            all_prob = pred[:, 0] + pred[:, 1]
            assert torch.all(torch.abs(all_prob - 1) < 1e-2), \
                "Input should be a binary distribution"
            score = pred[:, 1]
            result[M] = roc_auc_score(gt, score)
        elif M == 'acc':
            if dataset == 'admeood':
                pred = pred.argmax(dim=-1)
                # total, correct = len(pred), torch.sum(pred.long() == gt.long())
                result[M] = accuracy_score(gt, pred)
            else:
                # 使用torch.max函数获取每个样本预测的类别
                pred_classes = torch.argmax(pred, dim=1)
                # 使用torch.eq函数检查预测类别是否与真实标签匹配
                correct = torch.eq(pred_classes, gt)
                # 使用torch.mean计算准确率
                accuracy = torch.mean(correct.float())
                result[M] = accuracy
        elif M == 'recall':
            result[M] = recall_score(gt, pred, average='weighted', zero_division=0)
        elif M == 'f1':
            result[M] = f1_score(gt, pred, average='weighted', zero_division=0)
        elif M == 'pr_auc':
            result[M] = average_precision_score(gt, pred)
        elif M == 'mcc':
            result[M] = matthews_corrcoef(gt, pred)
        elif M == 'mul_roc':
            pass
    return result

def eval_one_epoch(model, dl, device, labels, verbose=True):
    model = model.eval()
    traYAll = []
    result_all, gt_all = [], []
    for step, (gs, labels, pygs) in tqdm(enumerate(dl)):
        labels = labels.to(device).long()
        pygs = pygs.to(device)
        gs = gs.to(device)
        af = gs.nodes['atom'].data['feat']
        bf = gs.edges[('atom', 'interacts', 'atom')].data['feat']
        fnf = gs.nodes['func_group'].data['feat']
        fef=gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
        molf=gs.nodes['molecule'].data['feat']

        traYAll += labels.detach().cpu().numpy().tolist()
        with torch.no_grad():
            result = model(gs, af, bf,fnf,fef,molf,pygs)
            result = torch.softmax(result, dim=-1)
            result_all.append(result.detach().cpu())
    result_all = torch.cat(result_all, dim=0)
    # return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'acc', 'mcc', 'f1', 'recall', 'pr_auc'])
    return evaluate(pred=result_all, gt=traYAll, metric=['auc', 'acc', 'mcc', 'f1', 'recall', 'pr_auc'], dataset='admeood')

def filter_data(gs, ls, pyg):
    filtered_train_gs = []
    filtered_train_ls = []
    filtered_train_pygs = []
    # singal_pyg = torch_geometric.data.Batch.to_data_list(pyg[0].Data)
    for g, l, p in zip(gs, ls, pyg):
        if g is not None:
            filtered_train_gs.append(g)
            filtered_train_ls.append(l)
            filtered_train_pygs.append(p)
    return filtered_train_gs, filtered_train_ls, filtered_train_pygs

if __name__ == "__main__":
    args = init_args()
    args.seed = eval(args.seed)
    data_config_path = os.path.join(args.data_config, args.dataset+'.py')
    # dataset_config = Config.fromfile(data_config_path)
    # 导入数据集
    root = os.path.join(args.root, args.dataset)
    if not os.path.exists(root):
        os.mkdir(root)
    if args.dataset_type.lower() == 'admeood':
        train_dataset = DILIDataset(root = root, mode="train", name=args.dataset)
        train_gs, train_ls, train_pyg = filter_data(train_dataset.gt_data_gs, train_dataset.gt_data_ls, train_dataset.pygGraphs)
        valid_dataset = DILIDataset(root = root, mode="ood_val", name=args.dataset)
        valid_gs, valid_ls, valid_pyg = filter_data(valid_dataset.gt_data_gs, valid_dataset.gt_data_ls,valid_dataset.pygGraphs)
        test_dataset = DILIDataset(root = root, mode="ood_test", name=args.dataset)
        test_gs, test_ls, test_pyg = filter_data(test_dataset.gt_data_gs, test_dataset.gt_data_ls,valid_dataset.pygGraphs)

    # 设置dataloader
    if args.dataset_type.lower() == 'admeood':
        train_ds = GraphDataset_Classification(train_gs, train_ls, train_pyg)
        train_dl = GraphDataLoader_Classification(train_ds, num_workers=args.num_workers, batch_size=args.batch_size,
                                       shuffle=True, drop_last = True)
        valid_ds = GraphDataset_Classification(valid_gs, valid_ls, valid_pyg)
        valid_dl = GraphDataLoader_Classification(valid_ds, num_workers=args.num_workers, batch_size=args.batch_size,
                                       shuffle=True, drop_last = True)
        test_ds = GraphDataset_Classification(test_gs, test_ls, test_pyg)
        test_dl = GraphDataLoader_Classification(test_ds, num_workers=args.num_workers, batch_size=args.batch_size,
                                       shuffle=True, drop_last = True)


        train_gs = dgl.batch(train_gs).to(args.device)
        train_labels=torch.tensor(train_ls).to(args.device)

        val_gs = dgl.batch(valid_gs).to(args.device)
        val_labels=torch.tensor(valid_ls).to(args.device)

        test_gs=dgl.batch(test_gs).to(args.device)
        test_labels=torch.tensor(test_ls).to(args.device)



    input_dim = 39
    num_classes = 2



    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device)

    def ce_loss(a, b, reduction='mean'):
        return F.cross_entropy(a, b, reduction=reduction)

    criterion = ce_loss
    edge_dim = -1

    # log 设置
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")

    all_seed_info = {
        'test_acc': [],
        'train_acc': [],
        'val_acc': [],
    }
    experiment_name = f'{args.dataset}-{args.mx_coe}-{args.cl_coe}-{datetime_now}'
    exp_dir = os.path.join(args.log_dir, experiment_name)
    os.mkdir(exp_dir)
    logger = Logger.init_logger(filename=exp_dir + '/log.log')
    args_print(args, logger)
    logger.info(f"Using criterion {criterion}")
    logger.info(f"#Train: {len(train_dl.dataset)}  #Val: {len(valid_dl.dataset)} #Test: {len(test_dl.dataset)} ")
    best_weights = None
    all_info = {
        'test_auc': [],
        'train_auc': [],
        'val_auc': [],
        'test_acc': [],
        'train_acc': [],
        'val_acc': [],
        'test_mcc': [],
        'train_mcc': [],
        'val_mcc': [],
        'test_f1': [],
        'train_f1': [],
        'val_f1': [],
        'test_recall': [],
        'train_recall': [],
        'val_recall': [],
        'test_pr_auc': [],
        'train_pr_auc': [],
        'val_pr_auc': [],
        }
    for seed in args.seed:
        set_seed(seed)

        model = mymodel(  args=args,
                        ratio=args.r,
                        input_dim=input_dim,
                        edge_dim=edge_dim,
                        out_dim=num_classes,
                        gnn_type=args.model,
                        num_layers=args.num_layers,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.dropout,
                        graph_pooling=args.pooling,
                        virtual_node=args.virtual_node,
                        c_dim=args.classifier_emb_dim,
                        c_in=args.classifier_input_feat,
                        c_rep=args.contrast_rep,
                        c_pool=args.contrast_pooling,
                        s_rep=args.spurious_rep,
                        do = args.do
                    ).to(device)

        model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        evaluator = Evaluator('ogbg-molhiv')
        eval_metric = args.eval_metric = 'rocauc'
        # print(model)
        last_train_acc, last_test_acc, last_val_acc = 0, 0, 0
        cnt = 0



        valid_curv, train_curv ={}, {}
        for epoch in range(args.epoch):
            all_loss, n_bw = 0, 0
            all_losses = {}
            contrast_loss, all_contrast_loss = torch.zeros(1).to(device), 0.

            spu_pred_loss = torch.zeros(1).to(device)
            model.train()
            # 启用了PyTorch的异常检测功能。
            torch.autograd.set_detect_anomaly(True)
            traYAll = []
            num_batch = (len(train_dl.dataset) // args.batch_size)

            for step, (gs, labels, pygs) in tqdm(enumerate(train_dl), total=num_batch, desc=f'Epoch[{epoch}] >> ', disable=args.no_tqdm, ncols=60):

                traYAll += labels.detach().cpu().numpy().tolist()
                gs = gs.to(args.device)
                pygs = pygs.to(args.device)
                labels = labels.to(args.device).long()
                af=gs.nodes['atom'].data['feat']
                bf = gs.edges[('atom', 'interacts', 'atom')].data['feat']
                fnf = gs.nodes['func_group'].data['feat']
                fef=gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                molf=gs.nodes['molecule'].data['feat']

                n_bw += 1
                gs.to(device)
                # ignore nan targets
                # https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/main_pyg.py

                (causal_pred, spu_pred), causal_rep, spu_rep, graph_rep, mx_pred = model(gs, af, bf,fnf,fef,molf, pygs, return_data="rep", return_spu = True)

                if args.dataset_type.lower() == 'admeood':
                    pred_loss = criterion(causal_pred, labels, reduction='none').mean()



                if args.do.lower() == "do":
                    y_label = labels[:len(causal_pred)]  # 假设graph.y是一个张量，并且我们想要取前len(causal_pred)个元素
                    mx_y = y_label.unsqueeze(1).expand(-1, len(spu_pred)).reshape(-1)
                    mx_pred_loss = criterion(mx_pred, mx_y, reduction='none')
                    mx_pred_loss = mx_pred_loss.mean()



                cl_loss = get_contrastive_loss_with_positive_and_negative_batch(graph_rep, causal_rep, spu_rep, temperature=args.T)

                # ********************************************LOSE
                if args.do.lower()=="do":
                    batch_loss = pred_loss + args.mx_coe * mx_pred_loss + args.cl_coe * cl_loss
                else:
                    batch_loss = pred_loss + args.cl_coe * cl_loss
                model_optimizer.zero_grad()
                batch_loss.backward()
                model_optimizer.step()
                all_loss += batch_loss.item()


            all_loss /= n_bw
            model.eval()

            val_perf = eval_one_epoch(model, valid_dl, device, val_labels)
            # test_perf = eval_one_epoch(model, test_dl, device, test_labels)
            train_perf = eval_one_epoch(model, train_dl, device, train_labels)
            if args.dataset_type.lower() == 'admeood':
                if val_perf['auc'] >= last_val_acc:
                    last_train_acc = train_perf['auc']
                    last_val_acc = val_perf['auc']
                    # last_test_acc = test_perf['auc']
                    if args.save_model:
                        best_weights = deepcopy(model.state_dict())
            elif args.dataset_type.lower() == 'good':
                if val_perf['acc'] >= last_val_acc:
                    last_train_acc = train_perf['acc']
                    last_val_acc = val_perf['acc']
                    # last_test_acc = test_perf['acc']
                    if args.save_model:
                        best_weights = deepcopy(model.state_dict())
            for k, v in val_perf.items():
                if k not in valid_curv:
                    valid_curv[k], train_curv[k] = [], []
                valid_curv[k].append(val_perf[k])
                # test_curv[k].append(test_perf[k])
                train_curv[k].append(train_perf[k])
            logger.info('[INFO] seed: {}'.format(seed))
            logger.info('[INFO] valid: {}'.format(val_perf))
            # logger.info('[INFO] test: {}'.format(test_perf))
            logger.info('[INFO] train: {}'.format(train_perf))

        best_result = {}
        for k, v in valid_curv.items():
            if k == 'auc':
                pos = int(np.argmax(v))
            best_result[k] = [pos, v[pos], train_curv[k][pos]]
        if args.save_model:
            print("Saving best weights..")
            model_path = os.path.join(exp_dir, args.dataset) + f"_{seed}.pt"
            for k, v in best_weights.items():
                best_weights[k] = v.cpu()
            torch.save(best_weights, model_path)
            print("Done..")
        logger.info('[INFO] best results: {}'.format(best_result))
        if args.dataset_type.lower() == 'admeood':
            # all_info['test_auc'].append(best_result['auc'][2])
            all_info['train_auc'].append(best_result['auc'][2])
            all_info['val_auc'].append(best_result['auc'][1])
            # all_info['test_acc'].append(best_result['acc'][2])
            all_info['train_acc'].append(best_result['acc'][2])
            all_info['val_acc'].append(best_result['acc'][1])
            # all_info['test_mcc'].append(best_result['mcc'][2])
            all_info['train_mcc'].append(best_result['mcc'][2])
            all_info['val_mcc'].append(best_result['mcc'][1])
            # all_info['test_f1'].append(best_result['f1'][2])
            all_info['train_f1'].append(best_result['f1'][2])
            all_info['val_f1'].append(best_result['f1'][1])
            # all_info['test_recall'].append(best_result['recall'][2])
            all_info['train_recall'].append(best_result['recall'][2])
            all_info['val_recall'].append(best_result['recall'][1])
            # all_info['test_pr_auc'].append(best_result['pr_auc'][2])
            all_info['train_pr_auc'].append(best_result['pr_auc'][2])
            all_info['val_pr_auc'].append(best_result['pr_auc'][1])
    if args.dataset_type.lower() == 'admeood':
        logger.info("[INFO] all_info: {}".format(all_info))
        logger.info("\nTrain AUC:{:.6f}-+-{:.6f}\nVal AUC:{:.6f}-+-{:.6f}\nTrain ACC:{:.6f}-+-{:.6f}\nVal ACC:{:.6f}-+-{:.6f}\nVal MCC:{:.6f}-+-{:.6f}\nVal F1:{:.6f}-+-{:.6f}\nVal RECALL:{:.6f}-+-{:.6f}\nVal PR_AUC:{:.6f}-+-{:.6f} ".format(
        torch.tensor(all_info['train_auc']).mean(),
        torch.tensor(all_info['train_auc']).std(),
        torch.tensor(all_info['val_auc']).mean(),
        torch.tensor(all_info['val_auc']).std(),
        torch.tensor(all_info['train_acc']).mean(),
        torch.tensor(all_info['train_acc']).std(),
        torch.tensor(all_info['val_acc']).mean(),
        torch.tensor(all_info['val_acc']).std(),
        torch.tensor(all_info['val_mcc']).mean(),
        torch.tensor(all_info['val_mcc']).std(),
        torch.tensor(all_info['val_f1']).mean(),
        torch.tensor(all_info['val_f1']).std(),
        torch.tensor(all_info['val_recall']).mean(),
        torch.tensor(all_info['val_recall']).std(),
        torch.tensor(all_info['val_pr_auc']).mean(),
        torch.tensor(all_info['val_pr_auc']).std()))



    print("\n\n\n")
    torch.cuda.empty_cache()