#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 DATA_DIR=/pub2/data/ python -m unittest domainbed.test.test_datasets.TestOverlapDatasets
CUDA_VISIBLE_DEVICES=2 DATA_DIR=/pub2/data/ python -m unittest domainbed.test.test_datasets.TestOverlapDatasets
CUDA_VISIBLE_DEVICES=2 DATA_DIR=/pub2/data/ python -m unittest domainbed.test.test_model_selection.TestSelectionMethod
CUDA_VISIBLE_DEVICES=2 DATA_DIR=/pub2/data/ python -m unittest domainbed.test.test_model_selection.TestIIDAccuracySelectionMethod
