#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1

# Execute program located in $HOME

source activate vit_adapt

srun sh make.sh




#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=40:00:00
#SBATCH --output=vit_adapter.out
#SBATCH --job-name=vit_adapter
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

source activate vit_adapt

srun python segmentation/train.py
