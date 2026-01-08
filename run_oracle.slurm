#!/bin/bash
#SBATCH --job-name=Oracle_Random
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

#SBATCH --output=/scratch/mkalla/unrolling_GPU/Results/logs/oracle_rand_%j.out
#SBATCH --error=/scratch/mkalla/unrolling_GPU/Results/logs/oracle_rand_%j.err

set -euo pipefail
module purge
module load cuda/12.5
export MPLBACKEND=Agg

cd /scratch/mkalla/unrolling_GPU
source gpu_env/bin/activate

echo "Lancement de l'Oracle (Random Search)..."

# Exemple : 1000 couples aléatoires
# Lambda entre 10^-7 et 10^-1
# Tau entre 0.1 et 1.9
# 2000 itérations par couple pour assurer la convergence

python main.py \
    --function oracle \
    --device cuda \
    --oracle_random \
    --oracle_num_points 1000 \
    --oracle_min_exp -7.0 \
    --oracle_max_exp -1.0 \
    --oracle_min_tau 0.1 \
    --oracle_max_tau 1.9 \
    --oracle_iter 2000

echo "Fin Oracle Random."