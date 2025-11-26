#!/bin/bash
#
# ===============================================
# Configuration SLURM pour la Grid Search
# ===============================================
#SBATCH --job-name=GS_P3MG         # Nom de la tâche
#SBATCH --partition=gpu-best       # Partition utilisée
#SBATCH --gres=gpu:1               # Allouer 1 GPU par tâche
#SBATCH --cpus-per-task=2          # 2 CPUs par tâche
#SBATCH --mem=8G                   # 8 Go de RAM par tâche
#SBATCH --time=01:00:00            # Temps limite raisonnable par test (1h)

# Définit la plage d'indices pour le tableau de tâches.
# Doit correspondre au nombre de lambdas à tester (5 ici).
#SBATCH --array=0-4

# Redirection des logs Slurm (utilise l'ID du job %A et l'index de la tâche %a)
# Le dossier 'logs' doit exister dans /scratch/mkalla/unrolling_GPU/
# Ces logs capturent la sortie console de CHAQUE test individuel.
#SBATCH --output=/scratch/mkalla/unrolling_GPU/logs/gs_%A_%a.out
#SBATCH --error=/scratch/mkalla/unrolling_GPU/logs/gs_%A_%a.err

# -------------------------------------------------------------
# Configuration ENVIRONNEMENT
# -------------------------------------------------------------
set -euo pipefail
module purge
module load cuda/12.5
export MPLBACKEND=Agg   

cd /scratch/mkalla/unrolling_GPU
source gpu_env/bin/activate

# -------------------------------------------------------------
# DÉFINITION DE LA GRILLE LAMBDA
# -------------------------------------------------------------
# Liste des valeurs de lambda à tester (doit correspondre à la taille du --array)
LAMBDA_VALUES=(1e-4 5e-5 1e-5 5e-6 1e-6)

# Récupération de l'index du job actuel dans le tableau Slurm (0, 1, 2, 3, 4)
JOB_INDEX=$SLURM_ARRAY_TASK_ID

# Sélection de la valeur de lambda correspondant à cet index
CURRENT_LAMBDA=${LAMBDA_VALUES[$JOB_INDEX]}

echo "--- Démarrage de la tâche Slurm $JOB_INDEX pour Lambda = $CURRENT_LAMBDA ---"

# -------------------------------------------------------------
# EXÉCUTION DU CODE PYTHON
# -------------------------------------------------------------

# Lancement de main.py en mode grid_search avec un seul lambda.
# Chaque job créera son propre dossier de résultats sous ./Results/baseline/
python main.py \
    --function grid_search \
    --gs_lambdas "$CURRENT_LAMBDA" \
    --number_layers 10 \
    --number_pd_layers 10 \
    --epochs 1 \
    --train_batch_size 1 # Utiliser une petite configuration pour le test rapide

echo "--- Tâche terminée pour Lambda = $CURRENT_LAMBDA ---"