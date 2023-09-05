#!/bin/bash

# local variables
datadir=/pub2/data
outputdir=/pub2/podg
n_hparams=5
steps=5001
trial=3
gpu=2
algorithm=CausIRL_MMD
jobid=iclr2024

for dataset in OfficeHome VLCS PACS
do
    for overlap in low high
    do
        curr_outdir=${outputdir}/${jobid}_${algorithm}_${dataset}_${overlap}
        echo starting ${curr_outdir}
        mkdir -p ${curr_outdir}

        # Remove incomplete runs
        python -m domainbed.scripts.sweep delete_incomplete\
           --data_dir=${datadir} \
           --algorithms $algorithm \
           --output_dir $curr_outdir\
           --command_launcher local \
           --overlap $overlap \
           --steps ${steps} \
           --single_test_envs \
           --datasets=${dataset} \
           --n_hparams ${n_hparams} \
           --n_trials ${trial} \
           --skip_confirmation

        CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.sweep launch\
           --data_dir=${datadir} \
           --algorithms $algorithm \
           --output_dir $curr_outdir\
           --command_launcher local \
           --overlap $overlap \
           --steps ${steps} \
           --single_test_envs \
           --datasets=${dataset} \
           --n_hparams ${n_hparams} \
           --n_trials ${trial} \
           --skip_confirmation
    done
done

