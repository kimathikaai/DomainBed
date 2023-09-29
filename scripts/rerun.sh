#!/bin/bash

# cp -r 66/vlcs/iclr2024_* rerun/66_vlcs
# cp -r 66/pacs/iclr2024_* rerun/66_pacs
# cp -r 66/officehome/iclr2024_* rerun/66_officehome
# cp -r 33/vlcs/iclr2024_* rerun/33_vlcs
# cp -r 33/pacs/iclr2024_* rerun/33_pacs
# cp -r 33/officehome/iclr2024_* rerun/33_officehome


for overlap in 33 66
do
    # get per dataset
    dataset=officehome
    input_dir='/Users/kimathikaai/scratch/saved/domainbed_results/rerun/'${overlap}_$dataset
    for class in {0..64}
    do
        python -m domainbed.scripts.collect_results\
            --input_dir=${input_dir}\
            --eval_metric=accC${class} \
            --selec_metric=nacc \
            --overlap=$overlap \
            --dataset=$dataset \
            --latex
    done

    dataset=vlcs
    input_dir='/Users/kimathikaai/scratch/saved/domainbed_results/rerun/'${overlap}_$dataset
    for class in {0..4}
    do
        python -m domainbed.scripts.collect_results\
            --input_dir=${input_dir}\
            --eval_metric=accC${class} \
            --selec_metric=nacc \
            --overlap=$overlap \
            --dataset=$dataset \
            --latex
    done

    dataset=pacs
    input_dir='/Users/kimathikaai/scratch/saved/domainbed_results/rerun/'${overlap}_$dataset
    for class in {0..6}
    do
        python -m domainbed.scripts.collect_results\
            --input_dir=${input_dir}\
            --eval_metric=accC${class} \
            --selec_metric=nacc \
            --overlap=$overlap \
            --dataset=$dataset \
            --latex
    done
done
