#!/bin/bash
#
# ===============================================
# Configuration SLURM pour l'Entraînement (UNROLLED)
# ===============================================
#SBATCH --job-name=Train_Unrolled
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:30:00              # Temps long pour l'entraînement (ajuster)
#SBATCH --output=/scratch/mkalla/unrolling_GPU/Results/logs/unrolled_%j.out
#SBATCH --error=/scratch/mkalla/unrolling_GPU/Results/logs/unrolled_%j.err

# -------------------------------------------------------------
# Configuration ENVIRONNEMENT
# -------------------------------------------------------------
set -euo pipefail
module purge
module load cuda/12.5
export MPLBACKEND=Agg   

cd /scratch/mkalla/unrolling_GPU
source gpu_env/bin/activate

echo "--- Démarrage de l'entraînement Unrolled ---"

# -------------------------------------------------------------
# EXÉCUTION DU CODE PYTHON
# -------------------------------------------------------------

# Lancement de main.py en mode train.
# Les résultats iront dans ./Results/unrolled/run-XXXXX/
python main.py \
    --function train \
    --device cuda \
    --number_layers 5 \
    --number_pd_layers 5 \
    --epochs 5 \
    --lr 8e-3 \
    --train_batch_size 10 

echo "--- Entraînement Unrolled terminé ---"