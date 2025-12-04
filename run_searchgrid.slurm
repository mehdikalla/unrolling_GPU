#!/bin/bash
#SBATCH --job-name=GS_2D
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:30:00
#SBATCH --array=0-24         

#SBATCH --output=/scratch/mkalla/unrolling_GPU/Results/logs/gs2d_%A_%a.out
#SBATCH --error=/scratch/mkalla/unrolling_GPU/Results/logs/gs2d_%A_%a.err

set -euo pipefail
module purge
module load cuda/12.5
export MPLBACKEND=Agg
cd /scratch/mkalla/unrolling_GPU
source gpu_env/bin/activate

# Paramètres de la grille 10x10
# Python fera le mapping: task_id 0 -> (L[0], T[0]), task_id 99 -> (L[9], T[9])
python main.py \
    --function grid_search \
    --device cuda \
    --gs_min_exp -7.0 \
    --gs_max_exp -1.0 \
    --gs_min_tau 0.01 \
    --gs_max_tau 1.99 \
    --gs_num_points 25 \
    --random_search  