#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH -N 1

source activate vit_adapt

srun python SegMunich_prep.py

