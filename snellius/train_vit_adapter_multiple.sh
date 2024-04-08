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

srun python

srun python segmentation/train.py --load-from /gpfs/work5/0/prjs0790/data/modified_checkpoints/ssl4eo_odin_run_2024-04-04_17-16_ckpt0_vit_adapter_online_network.pth & pid1=$!
wait $pid1
srun python segmentation/train.py --load-from /gpfs/work5/0/prjs0790/data/modified_checkpoints/ssl4eo_odin_run_2024-04-04_17-16_ckpt1_vit_adapter_online_network.pth & pid2=$!
wait $pid2
srun python segmentation/train.py --load-from /gpfs/work5/0/prjs0790/data/modified_checkpoints/ssl4eo_odin_run_2024-04-04_17-16_ckpt2_vit_adapter_online_network.pth & pid3=$!
wait $pid3
srun python segmentation/train.py --load-from /gpfs/work5/0/prjs0790/data/modified_checkpoints/ssl4eo_odin_run_2024-04-04_17-16_ckpt3_vit_adapter_online_network.pth & pid4=$!
wait $pid4
srun python segmentation/train.py --load-from /gpfs/work5/0/prjs0790/data/modified_checkpoints/ssl4eo_odin_run_2024-04-04_17-16_ckpt4_vit_adapter_online_network.pth & pid5=$!
wait $pid5

