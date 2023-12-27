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
test_env=0
teacher_checkpoint_path=/home/jkurien/scratch/saved/teacher_networks/iclr2024_FOND_PACS_high/d291d33f53df94d84ba0e904dd0a2fe2/model_step300.pkl
teacher_algorithm=FOND
student_algorithm=FOND_Distillation_Separate_Projector

for dataset in PACS
do
    for overlap in high
    do
        curr_outdir=${outputdir}/${jobid}_${student_algorithm}_${dataset}_${overlap}
        echo starting ${curr_outdir}
        mkdir -p ${curr_outdir}

        # Remove incomplete runs
        python -m domainbed.scripts.sweep delete_incomplete\
           --data_dir=${datadir} \
           --algorithms $student_algorithm \
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
        python3 -m domainbed.scripts.train_distillation\
            --data_dir=${datadir} \
            --output_dir $curr_outdir\
            --teacher_checkpoint_path $teacher_checkpoint_path \
            --dataset ${dataset} \
            --teacher_algorithm $teacher_algorithm \
            --student_algorithm $student_algorithm \
            --test_env $test_env \
            --overlap $overlap \
            --skip_model_save
    done
done

