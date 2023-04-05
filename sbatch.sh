#!/bin/bash

#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32000M
#SBATCH --time=3-00:00:00
#SBATCH --account=rrg-pfieguth

module load python/3.8 cuda cudnn

# Prepare virtualenv
source ~/envs/domainbed/bin/activate

# local variables
outputdir=~/scratch/saved
datadir=~/scratch/data
datasets=PACS
n_hparams=5
steps=5001
trial=3
gpu=0
overlap=66

# TODO make sure using nonlinear classifier
# jobid=${SLURM_JOBID}_${datasets}_${overlap}
# If restarting use the previous jobid
jobid=${SLURM_JOBID}_${datasets}_${overlap}
#for algorithm in XDom MLDG Transfer
for algorithm in XDom MLDG Transfer
do
    current_output_dir=${outputdir}/${jobid}/${algorithm}_${datasets}_o${overlap}_h${n_hparams}_s${steps}_t${trial}_${jobid}
    echo starting ${current_output_dir}
    mkdir -p ${current_output_dir}

    python -m domainbed.scripts.sweep delete_incomplete\
           --data_dir=${datadir} \
           --output_dir=${current_output_dir} \
           --command_launcher local \
           --overlap ${overlap} \
           --steps ${steps} \
           --single_test_envs \
           --algorithms ${algorithm} \
           --datasets ${datasets} \
           --n_hparams ${n_hparams} \
           --n_trials ${trial} \
           --skip_confirmation

    python -m domainbed.scripts.sweep launch\
           --data_dir=${datadir} \
           --output_dir=${current_output_dir} \
           --command_launcher local \
           --overlap ${overlap} \
           --steps ${steps} \
           --single_test_envs \
           --algorithms ${algorithm} \
           --datasets ${datasets} \
           --n_hparams ${n_hparams} \
           --n_trials ${trial} \
           --skip_confirmation
done
