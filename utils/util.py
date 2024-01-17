import torch
import numpy as np

import random
from texttable import Texttable
import torch
import numpy as np
import dgl
from drugood.utils import smile2graph
from functools import reduce
from sklearn.metrics import roc_auc_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_partition(len_dataset, device, seed, p=[0.5,0.5]):
    '''
        group the graph randomly

        Input:   len_dataset   -> [int]
                 the number of data to be groupped
                 
                 device        -> [torch.device]
                
                 p             -> [list]
                 probabilities of the random assignment for each group
        Output: 
                 vec           -> [torch.LongTensor]
                 group assignment for each data
    '''
    assert abs(np.sum(p) - 1) < 1e-4
    
    vec = torch.tensor([]).to(device)
    for idx, idx_p in enumerate(p):
        vec = torch.cat([vec, torch.ones(int(len_dataset * idx_p)).to(device) * idx])
        
    vec = torch.cat([vec, torch.ones(len_dataset - len(vec)).to(device) * idx])
    perm = torch.randperm(len_dataset, generator=torch.Generator().manual_seed(seed))
    return vec.long()[perm]


def args_print(args, logger):
    print('\n')
    _dict = vars(args)
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, _dict[k]])
    logger.info(table.draw())


def PrintGraph(graph):

    if graph.name:
        print("Name: %s" % graph.name)
    print("# Nodes:%6d      | # Edges:%6d |  Class: %2d" \
          % (graph.num_nodes, graph.num_edges, graph.y))

    print("# Node features: %3d| # Edge feature(s): %3d" \
          % (graph.num_node_features, graph.num_edge_features))

def evaluate(pred, gt, metric='auc'):
    if isinstance(metric, str):
        metric = [metric]
    allowed_metric = ['auc', 'accuracy']
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
        else:
            pred = pred.argmax(dim=-1)
            total, correct = len(pred), torch.sum(pred.long() == gt.long())
            result[M] = (correct / total).item()
    return result