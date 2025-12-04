# main.py (Version Random Search 2D - Uniform Sampling)
import argparse
import os
import sys
import torch
from datetime import datetime
from src.network import U_P3MG
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(
        description="P3MG: entraînement ou test (GPU auto si dispo)"
    )
    # Mode
    p.add_argument("--function", type=str, choices=["train", "test", "grid_search"], default="train")

    # Données
    p.add_argument("--dataset_dir", type=str, default="./Dataset")
    p.add_argument("--train", type=str, default="train.pt")
    p.add_argument("--val",   type=str, default="val.pt")
    p.add_argument("--test",  type=str, default="test.pt")

    # Modèle
    p.add_argument("--number_layers", type=int, default=10)
    p.add_argument("--number_pd_layers", type=int, default=10)

    # Paramètres statiques P3MG
    p.add_argument("--alpha", type=float, default=1e-5)
    p.add_argument("--beta",  type=float, default=1e-5)
    p.add_argument("--eta",   type=float, default=1e-2)

    # Entraînement
    p.add_argument("--epochs",           type=int,   default=25)
    p.add_argument("--lr",               type=float, default=5e-3)
    p.add_argument("--train_batch_size", type=int,   default=10)
    p.add_argument("--val_batch_size",   type=int,   default=1)
    p.add_argument("--test_batch_size",  type=int,   default=1)
    
    # Grid/Random Search Arguments
    # 1. Lambda (Log-Uniform)
    p.add_argument("--gs_min_exp", type=float, default=-6.0, help="Log10 min lambda")
    p.add_argument("--gs_max_exp", type=float, default=-2.0, help="Log10 max lambda")
    p.add_argument("--gs_num_points", type=int, default=10, help="Nombre total d'échantillons (si mode random) ou taille grille Lambda (si mode grid)")
    p.add_argument("--gs_lambdas", nargs='+', type=float, default=None, help="Liste manuelle Lambda")
    
    # 2. Tau (Uniform)
    p.add_argument("--gs_min_tau", type=float, default=0.1, help="Valeur min Tau")
    p.add_argument("--gs_max_tau", type=float, default=1.9, help="Valeur max Tau")
    p.add_argument("--gs_num_tau", type=int, default=10,    help="Taille grille Tau (si mode grid - Ignoré en mode random pur)")
    
    # Mode Random Search
    p.add_argument("--random_search", action="store_true", help="Si activé, tire des paires (Lambda, Tau) aléatoirement au lieu d'une grille.")
    
    p.add_argument("--gs_ckpt",    type=str, default=None, help="Checkpoint base pour GS")

    # I/O & device
    p.add_argument("--save_dir", type=str, default="./Results")
    p.add_argument("--device",   type=str, default="cuda")
    p.add_argument("--resume",   type=str, default=None)
    p.add_argument("--seed",     type=int, default=42)

    return p.parse_args()


def choose_device(pref: str) -> torch.device:
    if pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(pref)
    return torch.device("cpu")


def main():
    args = parse_args()

    # Seed
    # Pour le random search, on veut que la seed soit la même pour que la liste générée soit identique sur tous les nœuds,
    # MAIS que chaque nœud puisse choisir son échantillon de manière déterministe via son Task ID.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = choose_device(args.device)
    if device.type == "cuda":
        print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(device)}")
    else:
        print("[INFO] CUDA non dispo — bascule sur CPU.")

    # 1. DÉTERMINER LE TYPE DE RUN
    if args.function == "train":
        run_type_folder = "unrolled"
    elif args.function in ["test", "grid_search"]:
        run_type_folder = "baseline"
    else:
        run_type_folder = "misc" 

    # 2. CRÉER LE CHEMIN RACINE
    os.makedirs(args.save_dir, exist_ok=True)
    run_type_path = os.path.join(args.save_dir, run_type_folder)
    os.makedirs(run_type_path, exist_ok=True) 

    # 3. LOGIQUE DE CONSOLIDATION DES DOSSIERS
    if args.function == "grid_search":
        job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
        if job_id:
             # Mode Slurm
             job_name = os.environ.get('SLURM_JOB_NAME', 'GS_2D')
             unique_id = f"{job_name}-{job_id}"
        else:
             # Mode Local
             unique_id = datetime.now().strftime("local-gs2d-%Y%m%d-%H%M%S")
             
        run_dir = os.path.join(run_type_path, unique_id)
        os.makedirs(run_dir, exist_ok=True)
        
    else:
        run_dir = os.path.join(run_type_path, datetime.now().strftime("run-%Y%m%d-%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)


    path_train = os.path.join(args.dataset_dir, args.train)
    path_val   = os.path.join(args.dataset_dir, args.val)
    path_test  = os.path.join(args.dataset_dir, args.test)

    # Sanity check
    for pth, name in [(path_train, "train"), (path_val, "val"), (path_test, "test")]:
        if not os.path.isfile(pth):
            print(f"[ERREUR] Fichier {name} introuvable: {pth}")
            sys.exit(1)

    initial_x0 = None

    static_params = (args.alpha, args.beta, args.eta)
    train_params  = (args.epochs, args.lr, args.train_batch_size, args.val_batch_size, args.test_batch_size)
    
    # 4. GESTION DES LISTES 2D (Lambda & Tau)
    cur_l = None
    cur_t = None
    
    if args.function == "grid_search":
        # Génération des points
        
        # Cas 1: Random Search (Échantillonnage uniforme aléatoire)
        # On génère N paires (Lambda, Tau) uniques
        if args.random_search:
            num_samples = args.gs_num_points # Nombre total d'échantillons à tester
            
            # Génération déterministe (seed fixée plus haut)
            # Lambda: 10^(uniform(min_exp, max_exp))
            exponents = np.random.uniform(args.gs_min_exp, args.gs_max_exp, num_samples)
            lambdas = (10 ** exponents).tolist()
            
            # Tau: uniform(min_tau, max_tau)
            taus = np.random.uniform(args.gs_min_tau, args.gs_max_tau, num_samples).tolist()
            
            print(f"[INFO] Mode Random Search: {num_samples} paires générées.")
            
            # Dans ce mode, lambdas[i] est couplé avec taus[i]. Ce ne sont pas des axes de grille.
            # On a besoin d'une liste de paires.
            search_space_is_grid = False

        # Cas 2: Grid Search (Produit cartésien)
        else:
            if args.gs_lambdas is None:
                lambdas = np.logspace(args.gs_min_exp, args.gs_max_exp, args.gs_num_points).tolist()
            else:
                lambdas = args.gs_lambdas
            
            taus = np.linspace(args.gs_min_tau, args.gs_max_tau, args.gs_num_tau).tolist()
            print(f"[INFO] Mode Grid Search: {len(lambdas)} Lambdas x {len(taus)} Taus.")
            search_space_is_grid = True
        
        
        # Logique de sélection Slurm
        task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        
        if task_id is not None:
             # Mode Slurm
             tid = int(task_id)
             
             if search_space_is_grid:
                 # Mapping 2D Grid -> 1D Index
                 n_tau = len(taus)
                 total_tasks = len(lambdas) * n_tau
                 
                 if tid < total_tasks:
                     i_l = tid // n_tau
                     i_t = tid % n_tau
                     cur_l = [lambdas[i_l]]
                     cur_t = [taus[i_t]]
                     print(f"[INFO] Slurm Task {tid}: GRID -> L={cur_l[0]:.2e}, T={cur_t[0]:.2f}")
                 else:
                      print(f"[ERREUR] ID Slurm ({tid}) hors limites (Max {total_tasks-1}).")
                      sys.exit(1)
             else:
                 # Mapping 1D Random List -> 1D Index
                 # Ici, lambdas et taus sont des listes de même longueur appariées (paires i)
                 total_tasks = len(lambdas)
                 
                 if tid < total_tasks:
                     cur_l = [lambdas[tid]]
                     cur_t = [taus[tid]]
                     print(f"[INFO] Slurm Task {tid}: RANDOM -> L={cur_l[0]:.2e}, T={cur_t[0]:.2f}")
                 else:
                      print(f"[ERREUR] ID Slurm ({tid}) hors limites (Max {total_tasks-1}).")
                      sys.exit(1)

        else:
             # NOTE: Pour que ça marche en local avec Random Search, il faudrait appeler manager pour chaque paire.
             # Comme l'usage principal est Slurm (où cur_l et cur_t sont de taille 1), ça marche.
             # En local, on va supposer Grid Search classique par défaut.
             if args.random_search:
                 print("[WARN] Le mode Random Search en local séquentiel n'est pas optimisé. Utilisation du premier point.")
                 cur_l = [lambdas[0]]
                 cur_t = [taus[0]]
             else:
                 cur_l = lambdas
                 cur_t = taus


    # 5. INSTANCIATION DU MANAGER
    manager = U_P3MG(
        num_layers=args.number_layers,
        num_pd_layers=args.number_pd_layers,
        static_params=static_params,
        initial_x0=initial_x0,
        train_params=train_params,
        paths=(path_train, path_val, path_test, run_dir),
        device=str(device),
        args_dict=vars(args)
    )

    ckpt = args.resume if (args.resume and os.path.isfile(args.resume)) else None

    # 6. LOGIQUE DE LANCEMENT
    if args.function == "train":
        manager.train(need_names=False, checkpoint_path=ckpt, args_dict=vars(args))
        
    elif args.function == "test":
        manager.test(need_names=False, checkpoint_path=ckpt)
        
    elif args.function == "grid_search":
        gs_ckpt_path = args.gs_ckpt if (args.gs_ckpt and os.path.isfile(args.gs_ckpt)) else ckpt
        # Appel de la fonction
        manager.grid_search_2d(cur_l, cur_t, need_names=False, checkpoint_path=gs_ckpt_path)


if __name__ == "__main__":
    main()