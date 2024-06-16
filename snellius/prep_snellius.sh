#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH --time=40:00:00
#SBATCH --output=prep_segmunich.out
#SBATCH --job-name=prep_segmunich
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

source activate vit_adapt

srun python SegMunich_prep.py
