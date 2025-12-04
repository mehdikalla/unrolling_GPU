# main.py
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
    p.add_argument("--function", type=str, choices=["train", "test", "grid_search"], default="train")
    p.add_argument("--dataset_dir", type=str, default="./Dataset")
    p.add_argument("--train", type=str, default="train.pt")
    p.add_argument("--val",   type=str, default="val.pt")
    p.add_argument("--test",  type=str, default="test.pt")
    p.add_argument("--number_layers", type=int, default=10)
    p.add_argument("--number_pd_layers", type=int, default=10)
    p.add_argument("--alpha", type=float, default=1e-5)
    p.add_argument("--beta",  type=float, default=1e-5)
    p.add_argument("--eta",   type=float, default=1e-2)
    p.add_argument("--epochs",           type=int,   default=25)
    p.add_argument("--lr",               type=float, default=5e-3)
    p.add_argument("--train_batch_size", type=int,   default=10)
    p.add_argument("--val_batch_size",   type=int,   default=1)
    p.add_argument("--test_batch_size",  type=int,   default=1)
    
    # Grid Search Arguments
    p.add_argument("--gs_min_exp", type=float, default=-6.0, help="Log10 min lambda")
    p.add_argument("--gs_max_exp", type=float, default=-2.0, help="Log10 max lambda")
    p.add_argument("--gs_num_points", type=int, default=10, help="Nb points Lambda (Random)")
    p.add_argument("--gs_lambdas", nargs='+', type=float, default=None, help="Liste manuelle de lambdas.")
    
    p.add_argument("--gs_min_tau", type=float, default=0.1, help="Valeur min Tau")
    p.add_argument("--gs_max_tau", type=float, default=1.9, help="Valeur max Tau")
    p.add_argument("--gs_num_tau", type=int, default=10,    help="Nb points Tau (Random)")
    
    p.add_argument("--gs_ckpt",    type=str, default=None, help="Checkpoint base pour GS")
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

    # Seed (Crucial pour que les listes random soient identiques sur tous les jobs Slurm)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    if device.type == "cuda":
        print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(device)}")
    else:
        print("[INFO] CUDA non dispo — bascule sur CPU.")

    if args.function == "train":
        run_type_folder = "unrolled"
    elif args.function in ["test", "grid_search"]:
        run_type_folder = "baseline"
    else:
        run_type_folder = "misc" 

    os.makedirs(args.save_dir, exist_ok=True)
    run_type_path = os.path.join(args.save_dir, run_type_folder)
    os.makedirs(run_type_path, exist_ok=True) 

    if args.function == "grid_search":
        job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
        if job_id:
             job_name = os.environ.get('SLURM_JOB_NAME', 'grid_search')
             unique_id = f"{job_name}-{job_id}"
        else:
             unique_id = datetime.now().strftime("local-gs-%Y%m%d-%H%M%S")
        run_dir = os.path.join(run_type_path, unique_id)
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = os.path.join(run_type_path, datetime.now().strftime("run-%Y%m%d-%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)

    path_train = os.path.join(args.dataset_dir, args.train)
    path_val   = os.path.join(args.dataset_dir, args.val)
    path_test  = os.path.join(args.dataset_dir, args.test)

    for pth, name in [(path_train, "train"), (path_val, "val"), (path_test, "test")]:
        if not os.path.isfile(pth):
            print(f"[ERREUR] Fichier {name} introuvable: {pth}")
            sys.exit(1)

    initial_x0 = None
    static_params = (args.alpha, args.beta, args.eta)
    train_params  = (args.epochs, args.lr, args.train_batch_size, args.val_batch_size, args.test_batch_size)
    
    # 4. GESTION DES LISTES 2D (Mode Aléatoire Uniforme)
    cur_l = None
    cur_t = None
    
    if args.function == "grid_search":
        # A. Lambda : 10^(Uniform)
        if args.gs_lambdas is None:
            # On tire dans l'espace des exposants
            raw_exp = np.random.uniform(args.gs_min_exp, args.gs_max_exp, args.gs_num_points)
            # On trie pour la cohérence Slurm
            lambdas = sorted((10**raw_exp).tolist())
        else:
            lambdas = sorted(args.gs_lambdas)
            
        # B. Tau : Uniform
        raw_tau = np.random.uniform(args.gs_min_tau, args.gs_max_tau, args.gs_num_tau)
        taus = sorted(raw_tau.tolist())
        
        # Logique de sélection Slurm (Mapping 2D -> 1D)
        task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        
        if task_id is not None:
             tid = int(task_id)
             n_tau = len(taus)
             total_tasks = len(lambdas) * n_tau
             
             if tid < total_tasks:
                 i_l = tid // n_tau
                 i_t = tid % n_tau
                 cur_l = [lambdas[i_l]]
                 cur_t = [taus[i_t]]
                 print(f"[INFO] Tâche Slurm {tid}: Lambda[{i_l}]={cur_l[0]:.2e}, Tau[{i_t}]={cur_t[0]:.2f}")
             else:
                  print(f"[ERREUR] ID de tâche Slurm ({tid}) hors limites (Max {total_tasks-1}).")
                  sys.exit(1)
        else:
             print(f"[INFO] Mode local : Test de {len(lambdas)} Lambdas x {len(taus)} Taus ({len(lambdas)*len(taus)} combinaisons).")
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
        # Appel de la fonction 2D
        manager.grid_search_2d(cur_l, cur_t, need_names=False, checkpoint_path=gs_ckpt_path)

if __name__ == "__main__":
    main()