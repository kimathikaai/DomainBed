# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""Unit tests."""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import unittest
import uuid
from typing import List, Tuple

import torch

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed import networks

from parameterized import parameterized

from domainbed.test import helpers


def get_overlap_params() -> List[Tuple[str, str, int, List[int]]]:
    overlapping_classes = {
        "PACS": {0: [], 33: [2, 4], 66: [0, 2, 3, 4, 5], 100: list(range(7))},
        "VLCS": {0: [], 33: [2, 3], 66: [0, 2, 3, 4], 100: list(range(5))},
    }
    num_domains = {"PACS": 4, "VLCS": 4}

    params = []
    for dataset in ["PACS", "VLCS"]:
        for overlap in [0, 33, 66, 100]:
            for test_env in range(num_domains[dataset]):
                id = dataset + str(overlap)+f"_test_{test_env}"
                params.append((id, dataset, overlap, test_env, overlapping_classes[dataset][overlap]))

    return params


class TestDatasets(unittest.TestCase):
    @parameterized.expand(itertools.product(datasets.DATASETS))
    @unittest.skipIf(
        "DATA_DIR" not in os.environ, "needs DATA_DIR environment " "variable"
    )
    def test_dataset_erm(self, dataset_name):
        """
        Test that ERM can complete one step on a given dataset without raising
        an error.
        Also test that num_environments() works correctly.
        """
        batch_size = 8
        hparams = hparams_registry.default_hparams("ERM", dataset_name)
        dataset = datasets.get_dataset_class(dataset_name)(
            os.environ["DATA_DIR"], [], hparams
        )
        self.assertEqual(datasets.num_environments(dataset_name), len(dataset))
        algorithm = algorithms.get_algorithm_class("ERM")(
            dataset.input_shape, dataset.num_classes, len(dataset), hparams
        ).cuda()
        minibatches = helpers.make_minibatches(dataset, batch_size)
        algorithm.update(minibatches)


class TestOverlapDatasets(unittest.TestCase):
    @parameterized.expand(get_overlap_params())
    @unittest.skipIf(
        "DATA_DIR" not in os.environ, "needs DATA_DIR environment " "variable"
    )
    def test_overlap_datasets(
        self, _, dataset_name, class_overlap_id, test_env, overlapping_classes
    ):
        """
        Test that class filters remove classes from enviroment datasets
        """
        hparams = hparams_registry.default_hparams("ERM", dataset_name)
        dataset = datasets.get_dataset_class(dataset_name)(
            os.environ["DATA_DIR"], [test_env], hparams, class_overlap_id
        )
        self.assertEqual(datasets.num_environments(dataset_name), len(dataset))
        self.assertEqual(set(dataset.overlapping_classes), set(overlapping_classes))

        for env in dataset:
            targets = set(env.targets)
            print(
                f"{dataset_name} {class_overlap_id}%, ",
                f"env:{env.env_name}, test:{env.is_test_env}, ",
                f"allowed:{env.allowed_classes}, targets:{targets}, ",
                f"remove_classes:{env.remove_classes}",
            )
            self.assertEqual(
                set(env.allowed_classes),
                targets,
                (
                    f"{dataset_name} {class_overlap_id}%, "
                    f"env:{env.env_name}, test:{env.is_test_env}, "
                    f"allowed:{env.allowed_classes}, targets:{targets}, "
                    f"remove_classes:{env.remove_classes}"
                ),
            )
            if env.is_test_env:
                self.assertEqual(len(targets), dataset.num_classes)
