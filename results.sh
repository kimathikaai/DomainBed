#!/bin/bash

dataset='All'

for overlap in 0 33 66 100
do
    input_dir='/Users/kimathikaai/scratch/saved/domainbed_results/'$overlap
    selec_metric='acc'
    eval_metric='acc'
    python -m domainbed.scripts.collect_results\
        --input_dir=$input_dir \
        --eval_metric=$eval_metric \
        --selec_metric=$selec_metric \
        --overlap=$overlap \
        --dataset=$dataset \
        --latex

    selec_metric='acc'
    eval_metric='f1'
    python -m domainbed.scripts.collect_results\
        --input_dir=$input_dir \
        --eval_metric=$eval_metric \
        --selec_metric=$selec_metric \
        --overlap=$overlap \
        --dataset=$dataset \
        --latex

    selec_metric='acc'
    eval_metric='oacc'
    python -m domainbed.scripts.collect_results\
        --input_dir=$input_dir \
        --eval_metric=$eval_metric \
        --selec_metric=$selec_metric \
        --overlap=$overlap \
        --dataset=$dataset \
        --latex

    selec_metric='acc'
    eval_metric='nacc'
    python -m domainbed.scripts.collect_results\
        --input_dir=$input_dir \
        --eval_metric=$eval_metric \
        --selec_metric=$selec_metric \
        --overlap=$overlap \
        --dataset=$dataset \
        --latex
done
