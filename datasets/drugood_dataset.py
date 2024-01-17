import os.path as osp
import pickle as pkl
import random

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops, remove_self_loops
from tqdm import tqdm


class   DrugOOD(InMemoryDataset):

    def __init__(self, root, dataset, name, mode, transform=None, pre_transform=None, pre_filter=None):
        """
        init from original DrugOOD dataset in the form of dgl.heterograph.DGLHeteroGraph
        """
        super(DrugOOD, self).__init__(root, transform, pre_transform, pre_filter)
        self.load_data(root, dataset, name, mode)

    def load_data(self, root, dataset, name, mode):
        data_path = osp.join(root, name + "_" + mode + ".pt")
        if not osp.exists(data_path):
            data_list = []
            # for data in dataset:
            for step, data in tqdm(enumerate(dataset), total=len(dataset), desc="Converting"):
                graph = data['input']
                y = data['gt_label']
                group = data['group']

                edge_index = graph.edges()
                edge_attr = graph.edata['x']  #.long()
                node_attr = graph.ndata['x']  #.long()
                node_num = graph.num_nodes()
                edge_num = graph.num_edges()

                new_data = Data(edge_index=torch.stack(list(edge_index), dim=0),
                                edge_attr=edge_attr,
                                x=node_attr,
                                y=y,
                                node_num=node_num,
                                edge_num=edge_num,
                                group=group)
                data_list.append(new_data)
            all_data_list = self.collate(data_list)


            torch.save(self.collate(data_list), data_path)

        self.data, self.slices = torch.load(data_path)
