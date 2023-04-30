# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""Unit tests."""

import os
from typing import List, Tuple

from domainbed import datasets
from domainbed import hparams_registry

os.environ['DATA_DIR'] = '/home/s42hossa/scratch/dev/data'

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
    

params = get_overlap_params()

for param in params:
    _, dataset_name, overlap, test_env = param
    get_noc_info(_, dataset_name, overlap, test_env)