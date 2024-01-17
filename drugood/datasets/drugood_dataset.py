# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv import build_from_cfg

from drugood.core.utils.data_collect import Collater
from drugood.datasets.base_dataset import BaseDataset
from drugood.datasets.builder import DATASETS, PIPELINES

__all__ = ['DrugOODDataset', 'LBAPDataset', 'SBAPDataset', 'mySBAPDataset']


@DATASETS.register_module()
class DrugOODDataset(BaseDataset):
    def __init__(self,
                 split="train",
                 label_key="cls_label",
                 **kwargs):
        self.split = split
        self.label_key = label_key

        super(DrugOODDataset, self).__init__(**kwargs)
        self.sort_domain()
        self.groups = self.get_groups()
        self._collate = self.initial_collater()

    def initial_collater(self):
        return Collater()

    def sort_domain(self):
        unique_domains = torch.unique(torch.FloatTensor([case['domain_id'] for case in self.data_infos]))
        for case in self.data_infos:
            case['domain_id'] = torch.searchsorted(unique_domains, case['domain_id'])

    def get_groups(self):
        groups = torch.FloatTensor([case['domain_id'] for case in self.data_infos]).long().unsqueeze(-1)
        return groups

    def get_gt_labels(self):
        gt_labels = np.array([int(data[self.label_key]) for data in self.data_infos])
        return gt_labels

    def load_annotations(self):
        data = mmcv.load(self.ann_file)
        return data["split"][self.split]


@DATASETS.register_module()
class LBAPDataset(DrugOODDataset):
    def __init__(self, **kwargs):
        super(LBAPDataset, self).__init__(**kwargs)

    def prepare_data(self, idx):
        case = self.data_infos[idx]
        input = case["smiles"]
        results = {'input': input,
                   'gt_label': int(case[self.label_key]),
                   'group': case['domain_id']}
        return self.pipeline(results)


@DATASETS.register_module()
class SBAPDataset(DrugOODDataset):
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = build_from_cfg(tokenizer, PIPELINES)
        super(SBAPDataset, self).__init__(**kwargs)

    def prepare_data(self, idx):
        case = self.data_infos[idx]
        input = case["smiles"]
        aux_input = case["protein"]
        results = {'input': input,
                   'aux_input': aux_input,
                   'gt_label': int(case[self.label_key]),
                   'group': case['domain_id']}
        return self.pipeline(results)

    def initial_collater(self):
        return Collater(convert_fn=partial(self.tokenizer.__call__))


@DATASETS.register_module()
class mySBAPDataset(DrugOODDataset):
    def __init__(self, tokenizer, exclude_keys, **kwargs):
        self.tokenizer = build_from_cfg(tokenizer, PIPELINES)
        self.exclude_keys = exclude_keys
        super(mySBAPDataset, self).__init__(**kwargs)


    def prepare_data(self, idx):
        case = self.data_infos[idx]
        input = case["protein"]
        results = {'smiles':case["smiles"],
                    'gt_label': int(case[self.label_key]),
                    'affinity': case['reg_label'],
                    'protein': input,
                    'subs': case['substructure'],
                    'group': case['domain_id'],
                    'mw': case['mw'],
                    'hbd': case['hbd'],
                    'hba': case['hba'],
                    'logp': case['logp'],
                    'rob': case['rob'],
                    'qed': case['qed'],
                    'chiral_center': case['chiral_center'],
                    'tpsa': case['tpsa']
                    }
        return self.pipeline(results)

    def initial_collater(self):
        return Collater(convert_fn=partial(self.tokenizer.__call__), exclude_keys=self.exclude_keys)


@DATASETS.register_module()
class myLBAPDataset(DrugOODDataset):
    def __init__(self, **kwargs):
        super(myLBAPDataset, self).__init__(**kwargs)

    def prepare_data(self, idx):
        case = self.data_infos[idx]
        input = case["smiles"]
        results = {'input':input,
                    'gt_label': int(case[self.label_key]),
                    'subs': case['substructure'],
                    'group': case['domain_id'],
                    'alogp': case['alogp']}
        return self.pipeline(results)