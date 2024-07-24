#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH -N 1
# #SBATCH --nodelist=node436
#SBATCH --exclude=node430
#SBATCH --gres=gpu:1

source activate vit_adapt

srun python segmentation/mados_evaluation.py

#srun python segmentation/train.py \
#--load_from "/var/node433/local/ryan_a/data/leo_missing/leo_new_queue/ckp-epoch=14_mod.ckpt" \
#--work_dir "/var/node433/local/ryan_a/data/dfc2020_vit_adapter/leo_new_queue_e14"

