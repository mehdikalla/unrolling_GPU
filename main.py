# main.py (Version Corrigée: Random Search sans tri, Argument Mode ajouté)
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
    # Mode : Uniquement les 3 modes principaux
    p.add_argument("--function", type=str, choices=["train", "test", "grid_search", "oracle"], default="train")
    
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
    p.add_argument("--Loss", type=str, default="MSE")
    p.add_argument("--epochs",           type=int,   default=25)
    p.add_argument("--lr",               type=float, default=5e-3)
    p.add_argument("--train_batch_size", type=int,   default=10)
    p.add_argument("--val_batch_size",   type=int,   default=1)
    p.add_argument("--test_batch_size",  type=int,   default=1)
    
    # Grid/Random Search Params
    p.add_argument("--gs_min_exp", type=float, default=-6.0, help="Log10 min lambda")
    p.add_argument("--gs_max_exp", type=float, default=-2.0, help="Log10 max lambda")
    p.add_argument("--gs_num_points", type=int, default=10, help="Nb points Lambda (Grid) ou Nb Paires (Random)")
    p.add_argument("--gs_lambdas", nargs='+', type=float, default=None, help="Liste manuelle Lambda")
    
    p.add_argument("--gs_min_tau", type=float, default=0.1, help="Valeur min Tau")
    p.add_argument("--gs_max_tau", type=float, default=1.9, help="Valeur max Tau")
    p.add_argument("--gs_num_tau", type=int, default=10,    help="Nb points Tau (Ignoré en Random)")
    
    p.add_argument("--random_search", action="store_true", help="Si activé, tire des paires (L, T) aléatoires sans grille.")
    p.add_argument("--gs_ckpt",    type=str, default=None, help="Checkpoint base pour GS")

    # I/O & device
    p.add_argument("--save_dir", type=str, default="./Results")
    p.add_argument("--device",   type=str, default="cuda") 
    p.add_argument("--resume",   type=str, default=None)
    p.add_argument("--seed",     type=int, default=42)

    # Paramètres spécifiques Oracle
    p.add_argument("--oracle_random", action="store_true", help="Active le mode Random Search pour l'Oracle")
    p.add_argument("--oracle_num_points", type=int, default=100, help="Nombre de couples (L, T) aléatoires à tester")
    # Grille Lambda
    p.add_argument("--oracle_min_exp", type=float, default=-5.0)
    p.add_argument("--oracle_max_exp", type=float, default=-1.0)
    p.add_argument("--oracle_n_lambda", type=int, default=15)
    # Grille Tau
    p.add_argument("--oracle_min_tau", type=float, default=0.1)
    p.add_argument("--oracle_max_tau", type=float, default=1.9)
    p.add_argument("--oracle_n_tau", type=int, default=15)
    # Iterations
    p.add_argument("--oracle_iter", type=int, default=10000, help="Nb itérations algo itératif")

    return p.parse_args()


def choose_device(pref: str) -> torch.device:
    if pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(pref)
    return torch.device("cpu")


def main():
    args = parse_args()

    # Seed (Crucial pour cohérence Slurm en mode Random)
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
    elif args.function == "grid_search":
        run_type_folder = "baseline"
    elif args.function == "test":
        run_type_folder = "network_test"
    elif args.function == "oracle":
        
        if args.oracle_random:
            print(f"[INFO] Mode Oracle Random Search activé.")
            # 1. Génération Lambda (Log-Uniforme)
            # On tire l'exposant uniformément entre min_exp et max_exp
            raw_exps = np.random.uniform(args.oracle_min_exp, args.oracle_max_exp, args.oracle_num_points)
            lambdas = (10**raw_exps).tolist()
            
            # 2. Génération Tau (Uniforme Linéaire)
            taus = np.random.uniform(args.oracle_min_tau, args.oracle_max_tau, args.oracle_num_points).tolist()
            
            mode = "random"
        else:
            print(f"[INFO] Mode Oracle Grid Search activé.")
            lambdas = np.logspace(args.oracle_min_exp, args.oracle_max_exp, args.oracle_n_lambda).tolist()
            taus = np.linspace(args.oracle_min_tau, args.oracle_max_tau, args.oracle_n_tau).tolist()
            mode = "grid"
            
            if args.gs_lambdas is not None:
                lambdas = args.gs_lambdas

        manager.run_oracle_analysis(lambdas, taus, max_iter=args.oracle_iter, mode=mode)

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
    search_mode = "grid" # Par défaut
    
    if args.function == "grid_search":
        # A. Lambda
        if args.gs_lambdas is None:
            if args.random_search:
                search_mode = "random"
                # Tirage Log-Uniforme : 10^(uniform)
                # IMPORTANT: On NE TRIE PAS (sorted) pour préserver l'aléatoire des paires
                raw_exps = np.random.uniform(args.gs_min_exp, args.gs_max_exp, args.gs_num_points)
                lambdas = (10**raw_exps).tolist()
                print(f"[INFO] Random Search: Génération de {len(lambdas)} lambdas log-uniformes (non triés).")
            else:
                search_mode = "grid"
                # Grille régulière Log (toujours triée par nature)
                lambdas = np.logspace(args.gs_min_exp, args.gs_max_exp, args.gs_num_points).tolist()
                print(f"[INFO] Grid Search: Génération de {len(lambdas)} lambdas log-espacés.")
        else:
            lambdas = args.gs_lambdas
            
        # B. Tau
        if args.random_search:
            # Tirage Uniforme
            # IMPORTANT: On NE TRIE PAS pour préserver l'aléatoire des paires (Lambda_i, Tau_i)
            raw_taus = np.random.uniform(args.gs_min_tau, args.gs_max_tau, args.gs_num_points)
            taus = raw_taus.tolist()
            print(f"[INFO] Random Search: Génération de {len(taus)} taus uniformes (non triés).")
        else:
            # Grille régulière Linéaire
            taus = np.linspace(args.gs_min_tau, args.gs_max_tau, args.gs_num_tau).tolist()
            print(f"[INFO] Grid Search: Génération de {len(taus)} taus espacés.")
        
        
        # Logique de sélection Slurm
        task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        
        if task_id is not None:
             tid = int(task_id)
             
             if args.random_search:
                 # Mode Random : Mapping 1D (Paires)
                 total_tasks = len(lambdas)
                 if tid < total_tasks:
                     cur_l = [lambdas[tid]]
                     cur_t = [taus[tid]]
                     print(f"[INFO] Tâche Slurm {tid} (Random): L={cur_l[0]:.2e}, T={cur_t[0]:.2f}")
                 else:
                      print(f"[ERREUR] ID de tâche Slurm ({tid}) hors limites (Max {total_tasks-1}).")
                      sys.exit(1)
             else:
                 # Mode Grid : Mapping 2D (Produit Cartésien)
                 n_tau = len(taus)
                 total_tasks = len(lambdas) * n_tau
                 
                 if tid < total_tasks:
                     # tid = i_l * n_tau + i_t
                     i_l = tid // n_tau
                     i_t = tid % n_tau
                     cur_l = [lambdas[i_l]]
                     cur_t = [taus[i_t]]
                     print(f"[INFO] Tâche Slurm {tid} (Grid): L={cur_l[0]:.2e}, T={cur_t[0]:.2f}")
                 else:
                      print(f"[ERREUR] ID de tâche Slurm ({tid}) hors limites (Max {total_tasks-1}).")
                      sys.exit(1)
        else:
             # Mode Local : On passe tout
             if args.random_search:
                 print(f"[INFO] Mode local Random : Test de {len(lambdas)} paires (L, T).")
             else:
                 print(f"[INFO] Mode local Grid : Test de {len(lambdas)}L x {len(taus)}T combinaisons.")
             
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
        args_dict=vars(args),
        criterion=args.Loss
    )

    ckpt = args.resume if (args.resume and os.path.isfile(args.resume)) else None

    # 6. LOGIQUE DE LANCEMENT
    if args.function == "train":
        manager.train(need_names=False, checkpoint_path=ckpt, args_dict=vars(args))
        
    # --- MODE TEST ---
    elif args.function == "test":
        # EN FORCE : Si pas de resume ou fichier introuvable -> ERREUR
        if not args.resume or not os.path.isfile(args.resume):
            print(f"[ERREUR CRITIQUE] En mode 'test', vous DEVEZ fournir un chemin valide via --resume.")
            print(f"Chemin reçu : {args.resume}")
            sys.exit(1)
            
        print(f"[INFO] Mode Test : Chargement du modèle depuis {args.resume}")
        manager.test(need_names=False, checkpoint_path=args.resume)

    elif args.function == "grid_search":
        gs_ckpt_path = args.gs_ckpt if (args.gs_ckpt and os.path.isfile(args.gs_ckpt)) else ckpt
        # Appel avec le mode pour que network.py sache s'il doit zipper ou croiser
        manager.grid_search_2d(cur_l, cur_t, need_names=False, checkpoint_path=gs_ckpt_path, mode=search_mode)


if __name__ == "__main__":
    main()