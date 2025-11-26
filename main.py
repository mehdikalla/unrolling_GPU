# main.py
import argparse
import os
import sys
import torch
from datetime import datetime
from src.network import U_P3MG # Changement d'importation vers src/network


def parse_args():
    p = argparse.ArgumentParser(
        description="P3MG: entraînement ou test (GPU auto si dispo)"
    )
    # Mode
    p.add_argument("--function", type=str, choices=["train", "test", "grid_search", "plot_params"], default="train")

    # Données
    p.add_argument("--dataset_dir", type=str, default="./Dataset")
    p.add_argument("--train", type=str, default="train.pt")
    p.add_argument("--val",   type=str, default="val.pt")
    p.add_argument("--test",  type=str, default="test.pt")

    # Modèle
    p.add_argument("--number_layers", type=int, default=2)
    p.add_argument("--number_pd_layers", type=int, default=2)

    # Paramètres statiques P3MG
    p.add_argument("--alpha", type=float, default=1e-5)
    p.add_argument("--beta",  type=float, default=1e-5)
    p.add_argument("--eta",   type=float, default=1e-2)

    # Entraînement
    p.add_argument("--epochs",           type=int,   default=2)
    p.add_argument("--lr",               type=float, default=5e-3)
    p.add_argument("--train_batch_size", type=int,   default=10)
    p.add_argument("--val_batch_size",   type=int,   default=1)
    p.add_argument("--test_batch_size",  type=int,   default=1)
    
    p.add_argument("--gs_lambdas", nargs='+', type=float, default=[1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
                   help="Liste des valeurs de lambda (régularisation) à tester pour la grid search.")
    p.add_argument("--gs_ckpt",    type=str, default=None,
                   help="Chemin d'un checkpoint .pt à utiliser comme base pour la grid search.")


    # I/O & device
    p.add_argument("--save_dir", type=str, default="./Results")
    p.add_argument("--device",   type=str, default="cuda")  # "cuda" ou "cpu"
    p.add_argument("--resume",   type=str, default=None,    # chemin d’un checkpoint .pt
                   help="Chemin d'un checkpoint .pt à reprendre (train/test) ou à analyser (plot_params).")
    p.add_argument("--seed",     type=int, default=42)

    return p.parse_args()


def choose_device(pref: str) -> torch.device:
    if pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(pref)
    return torch.device("cpu")


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = choose_device(args.device)
    if device.type == "cuda":
        print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(device)}")
    else:
        print("[INFO] CUDA non dispo — bascule sur CPU.")

    # 1. DÉTERMINER LE TYPE DE RUN POUR LA NOUVELLE ARBORESCENCE
    if args.function == "train" or args.function == "plot_params":
        run_type_folder = "unrolled"
    elif args.function == "test" or args.function == "grid_search":
        run_type_folder = "baseline" # Pour classer les résultats de tests non-appris/classiques
    else:
        run_type_folder = "misc" 

    # 2. CRÉER LE CHEMIN RACINE (e.g., ./Results/unrolled)
    os.makedirs(args.save_dir, exist_ok=True)
    run_type_path = os.path.join(args.save_dir, run_type_folder)
    os.makedirs(run_type_path, exist_ok=True) 

    # 3. CRÉER LE DOSSIER D'EXÉCUTION UNIQUE (run_dir)
    run_dir = os.path.join(run_type_path, datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    path_train = os.path.join(args.dataset_dir, args.train)
    path_val   = os.path.join(args.dataset_dir, args.val)
    path_test  = os.path.join(args.dataset_dir, args.test)

    # Sanity sur les fichiers
    for pth, name in [(path_train, "train"), (path_val, "val"), (path_test, "test")]:
        if not os.path.isfile(pth):
            print(f"[ERREUR] Fichier {name} introuvable: {pth}")
            sys.exit(1)

    initial_x0 = None

    static_params = (args.alpha, args.beta, args.eta)
    train_params  = (args.epochs, args.lr, args.train_batch_size, args.val_batch_size, args.test_batch_size)
    
    # 4. INSTANCIATION DU MANAGER (Passage de tous les arguments pour la sauvegarde de la config)
    manager = U_P3MG(
        num_layers=args.number_layers,
        num_pd_layers=args.number_pd_layers,
        static_params=static_params,
        initial_x0=initial_x0,
        train_params=train_params,
        paths=(path_train, path_val, path_test, run_dir), # run_dir est le chemin de sauvegarde
        device=str(device),
        args_dict=vars(args) # Conversion en dictionnaire pour la sauvegarde
    )

    ckpt = args.resume if (args.resume and os.path.isfile(args.resume)) else None

    # 5. LOGIQUE DE LANCEMENT
    if args.function == "train":
        # Le train sauvegarde automatiquement la config et les plots à la fin.
        manager.train(need_names=False, checkpoint_path=ckpt)
        
    elif args.function == "test":
        # Utilise le chemin de resume/ckpt si fourni
        manager.test(need_names=False, checkpoint_path=ckpt)
        
    elif args.function == "grid_search":
        gs_ckpt_path = args.gs_ckpt if (args.gs_ckpt and os.path.isfile(args.gs_ckpt)) else ckpt
        # La Grid Search effectue son propre plotting des résultats.
        manager.grid_search_lambda(args.gs_lambdas, need_names=False, checkpoint_path=gs_ckpt_path)
    
    elif args.function == "plot_params":
        plot_path = args.resume
        if not plot_path or not os.path.isfile(plot_path):
             print(f"[ERREUR] Le mode plot_params nécessite un chemin valide vers un checkpoint via --resume. Fichier introuvable: {plot_path}")
             sys.exit(1)
        
        # Le manager est réinitialisé avec les paramètres de base, puis charge le modèle
        manager.plot_learned_params_evolution(plot_path)


if __name__ == "__main__":
    main()