#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32000M
#SBATCH --time=1-00:00:00
#SBATCH --account=def-sirisha

module load python/3.8 cuda cudnn

# Prepare virtualenv
source /home/jkurien/workspace/envs/domainbed/bin/activate

# local variables
datadir=/home/jkurien/scratch/data
outputdir=/home/jkurien/scratch/saved
n_hparams=5
steps=5001
trial=3
jobid=iclr2024

algorithm=NOC

for dataset in VLCS PACS OfficeHome
do
    for overlap in low_linked_only high_linked_only
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

        # Run 
        python -m domainbed.scripts.sweep launch\
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