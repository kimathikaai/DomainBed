# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""Unit tests."""

import os
import csv
from typing import List, Tuple

from domainbed import datasets
from domainbed import hparams_registry

os.environ['DATA_DIR'] = '/home/s42hossa/scratch/dev/data'
csv_file_path = '/home/s42hossa/scratch/dev/DomainBed/label_distribution.csv'

def get_overlap_params() -> List[Tuple[str, str, int, List[int]]]:
    num_domains = {"PACS": 4, "VLCS": 4, "OfficeHome": 4}

    params = []
    for dataset, N_s in num_domains.items():
        for overlap in datasets.OVERLAP_TYPES:
            for test_env in range(N_s):
                id = f"{dataset}_{overlap}_test_{test_env}"
                params.append((id, dataset, overlap, test_env))

    return params

def get_noc_info(_, dataset_name, overlap, test_env):
    """
    Test that class filters remove classes from enviroment datasets
    """
    hparams = hparams_registry.default_hparams("ERM", dataset_name)
    dataset = datasets.get_dataset_class(dataset_name)(
        os.environ["DATA_DIR"], [test_env], hparams, overlap, overlap_seed = 0
    )

    ts = dict()
    noc = list()

    for env in dataset:
        targets = set(env.targets)

        if not env.is_test_env:
            for target in targets:
                if target not in ts.keys():
                    ts.update({target: 1})
                else:
                    ts[target] += 1

    for target in ts.keys():
        if ts[target] <= 1:
            noc.append(target)
    
    train_label_distribution = dict()
    test_label_distribution = dict()
    train_label_distribution.update({'noc': 0, 'total': 0})
    test_label_distribution.update({'noc': 0, 'total': 0})

    for env in dataset:

        for target in env.targets:
            if target in noc:
                if env.is_test_env:
                    test_label_distribution['noc'] += 1
                else:
                    train_label_distribution['noc'] += 1
            
            if env.is_test_env:
                test_label_distribution['total'] += 1
            else:
                train_label_distribution['total'] += 1

    print(dataset_name)
    print(overlap)
    print(train_label_distribution)
    print(test_label_distribution)

    return train_label_distribution, test_label_distribution
    

params = get_overlap_params()
label_distribution = {
    'dataset' : list(),
    'overlap' : list(),
    'test_env' : list(),
    'n_oc_train' : list(),
    'n_noc_train' : list(),
    'n_oc_test' : list(),
    'n_noc_test' : list(),
    'total_train' : list(),
    'total_test' : list(),
    'noc_percent_train': list(),
    'noc_percent_test': list()
}

for param in params:
    id, dataset_name, overlap, test_env = param
    train_label_distribution, test_label_distribution = get_noc_info(id, dataset_name, overlap, test_env)

    label_distribution['dataset'].append(dataset_name)
    label_distribution['overlap'].append(overlap)
    label_distribution['test_env'].append(test_env)
    label_distribution['n_oc_train'].append(
        train_label_distribution['total'] - train_label_distribution['noc'])
    label_distribution['n_noc_train'].append(train_label_distribution['noc'])
    label_distribution['n_oc_test'].append(
        test_label_distribution['total'] - test_label_distribution['noc'])
    label_distribution['n_noc_test'].append(test_label_distribution['noc'])
    label_distribution['total_train'].append(train_label_distribution['total'])
    label_distribution['total_test'].append(test_label_distribution['total'])
    label_distribution['noc_percent_train'].append(
        train_label_distribution['noc']*100/train_label_distribution['total']
    )
    label_distribution['noc_percent_test'].append(
        test_label_distribution['noc']*100/test_label_distribution['total']
    )

with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(label_distribution.keys())  
    for row in zip(*label_distribution.values()):
        writer.writerow(row)
