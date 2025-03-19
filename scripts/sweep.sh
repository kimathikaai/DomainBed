#!/bin/bash

# local variables
datadir=/home/kkaai/scratch/data
outputdir=/home/kkaai/scratch/saved/fond-logs
n_hparams=3
n_hparams_from=0
steps=3001
trial=3
test_env=4
gpu=2
jobid=tpami

for overlap in 1
do
    for dataset in WILDSCamelyon
    do
        for algorithm in ERM XDomBetaError EQRM MLDG
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
               --n_hparams_from ${n_hparams_from} \
               --n_trials ${trial} \
               --test_env ${test_env} \
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
               --n_hparams_from ${n_hparams_from} \
               --n_trials ${trial} \
               --test_env ${test_env} \
               --skip_confirmation
        done
    done
done
