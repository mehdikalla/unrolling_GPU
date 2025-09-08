#!/bin/bash
#SBATCH --job-name=main_gpu
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/mkalla/unrolling_GPU/logs/%x_%j.out
#SBATCH --error=/scratch/mkalla/unrolling_GPU/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_DIR=/scratch/mkalla/unrolling_GPU
VENV_DIR=$PROJECT_DIR/gpu_env

module purge
module load python/3.10
module load cuda/12.5

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

# Vérif GPU (court, utile)
nvidia-smi || { echo "GPU non visible sur ce nœud"; exit 1; }

# Lancement — ajoute tes arguments si besoin
python main.py
