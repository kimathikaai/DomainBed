#!/bin/bash

dataset='All'

input_dir='/Users/kimathikaai/scratch/saved/domainbed_results/tsne'

for selec_metric in macc vacc acc nacc oacc f1
do
    for eval_metric in macc vacc acc nacc oacc f1
    do
        python -m domainbed.scripts.collect_results\
            --input_dir=$input_dir \
            --eval_metric=$eval_metric \
            --selec_metric=$selec_metric \
            --overlap=$overlap \
            --dataset=$dataset
    done
done
