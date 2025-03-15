#!/bin/bash

# local variables
datadir=/home/kkaai/scratch/data
outputdir=/home/kkaai/scratch/saved/fond-logs
n_hparams=5
steps=5001
trial=3
gpu=2
jobid=tpami

for overlap in 1 2
do
    for dataset in WILDSCamelyon
    do
        for algorithm in ERM CORAL
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
done

