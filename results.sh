#!/bin/bash

dataset='Track'

# # for overlap in mid high low 33 66
# for overlap in 33 66
# do
#     input_dir='/Users/kimathikaai/scratch/saved/domainbed_results/rerun/'${overlap}'_pacs'
#
#     for selec_metric in nacc
#     do
#         for class in {0..6}
#         do
#             python -m domainbed.scripts.collect_results\
#                 --input_dir=$input_dir \
#                 --eval_metric=accC$class \
#                 --selec_metric=$selec_metric \
#                 --overlap=$overlap \
#                 --dataset=$dataset \
#                 --latex
#         done
#         for eval_metric in nacc acc macc
#         do
#             python -m domainbed.scripts.collect_results\
#                 --input_dir=$input_dir \
#                 --eval_metric=$eval_metric \
#                 --selec_metric=$selec_metric \
#                 --overlap=$overlap \
#                 --dataset=$dataset \
#                 --latex
#         done
#     done
# done
#
# for overlap in 33 66
# do
#     input_dir='/Users/kimathikaai/scratch/saved/domainbed_results/rerun/'${overlap}'_vlcs'
#
#     for selec_metric in nacc
#     do
#         for class in {0..4}
#         do
#             python -m domainbed.scripts.collect_results\
#                 --input_dir=$input_dir \
#                 --eval_metric=accC$class \
#                 --selec_metric=$selec_metric \
#                 --overlap=$overlap \
#                 --dataset=$dataset \
#                 --latex
#         done
#         for eval_metric in nacc acc macc
#         do
#             python -m domainbed.scripts.collect_results\
#                 --input_dir=$input_dir \
#                 --eval_metric=$eval_metric \
#                 --selec_metric=$selec_metric \
#                 --overlap=$overlap \
#                 --dataset=$dataset \
#                 --latex
#         done
#     done
# done

for overlap in 33
do
    input_dir='/Users/kimathikaai/scratch/saved/domainbed_results/rerun/'${overlap}'_officehome'

    for selec_metric in nacc
    do
        for class in {0..64}
        do
            python -m domainbed.scripts.collect_results\
                --input_dir=$input_dir \
                --eval_metric=accC$class \
                --selec_metric=$selec_metric \
                --overlap=$overlap \
                --dataset=$dataset \
                --latex
        done
        for eval_metric in nacc acc macc
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
