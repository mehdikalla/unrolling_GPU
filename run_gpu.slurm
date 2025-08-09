#!/bin/bash
#SBATCH --job-name=main_gpu
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/mkalla/unrolling_GPU/logs/%j.out
#SBATCH --error=/scratch/mkalla/unrolling_GPU/logs/%j.err
set -euo pipefail

module purge
module load cuda/12.5
export MPLBACKEND=Agg   # Pour matplotlib sans écran

cd /scratch/mkalla/unrolling_GPU
source gpu_env/bin/activate

# Petit check GPU
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

# Lancer ton programme
python main.py
