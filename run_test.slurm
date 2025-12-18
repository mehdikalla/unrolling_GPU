#!/bin/bash
#
# ===============================================
# Configuration SLURM pour le Test (UNROLLED)
# ===============================================
#SBATCH --job-name=Test_Unrolled
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00              # Suffisant pour une phase de test
#SBATCH --output=/scratch/mkalla/unrolling_GPU/Results/logs/test_%j.out
#SBATCH --error=/scratch/mkalla/unrolling_GPU/Results/logs/test_%j.err

# -------------------------------------------------------------
# Configuration ENVIRONNEMENT
# -------------------------------------------------------------
set -euo pipefail
module purge
module load cuda/12.5
export MPLBACKEND=Agg   

cd /scratch/mkalla/unrolling_GPU
source gpu_env/bin/activate

echo "--- Démarrage du Test Unrolled ---"

# -------------------------------------------------------------
# EXÉCUTION DU CODE PYTHON
# -------------------------------------------------------------

# IMPORTANT : Remplacer le chemin dans --resume par le vrai chemin de ton modèle .pth
# Exemple : Results/unrolled/run-20241218-140000/best_model.pth

python main.py \
    --function test \
    --device cuda \
    --number_layers 10 \
    --number_pd_layers 10 \
    --test_batch_size 10 \
    --resume "./Results/unrolled/run-20251208-104453/checkpoints/checkpoint_epoch50.pt"

echo "--- Test Unrolled terminé ---"