#!/bin/bash

dataset='All'

# for overlap in mid high low 33 66
for overlap in 33 66
do
    input_dir='/Users/kimathikaai/scratch/saved/domainbed_results/'$overlap

    for selec_metric in macc vacc acc nacc oacc f1
    do
        for eval_metric in macc vacc acc nacc oacc f1
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
