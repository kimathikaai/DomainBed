#!/bin/bash

dataset='All'

for overlap in 0 33 66 100
do
    input_dir='/Users/kimathikaai/scratch/saved/domainbed_results/'$overlap

    for selec_metric in acc nacc oacc f1
    do
        for eval_metric in acc nacc oacc f1
        do
            python -m domainbed.scripts.collect_results\
                --input_dir=$input_dir \
                --eval_metric=$eval_metric \
                --selec_metric=$selec_metric \
                --overlap=$overlap \
                --dataset=$dataset \
                --latex
        done
    done
done
