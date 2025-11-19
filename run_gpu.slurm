#!/bin/bash
#SBATCH --job-name=main_gpu
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/mkalla/unrolling_GPU/logs/%j.out
#SBATCH --error=/scratch/mkalla/unrolling_GPU/logs/%j.err
set -euo pipefail
module purge
module load cuda/12.5
export MPLBACKEND=Agg   

# matplotlib sans écran
cd /scratch/mkalla/unrolling_GPU
source gpu_env/bin/activate

# logs rapides pour vérifier l'env:
python -V
which python
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

python main.py
