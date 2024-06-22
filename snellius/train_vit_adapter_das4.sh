#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH -N 1
# #SBATCH --nodelist=node436
#SBATCH --exclude=node430
#SBATCH --gres=gpu:1

source activate vit_adapt

srun python segmentation/train.py


