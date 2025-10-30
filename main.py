# main.py
import argparse
import os
import sys
import torch
from datetime import datetime
from modules.P3MG_network import U_P3MG


def parse_args():
    p = argparse.ArgumentParser(
        description="P3MG: entraînement ou test (GPU auto si dispo)"
    )
    # Mode
    p.add_argument("--function", type=str, choices=["train", "test"], default="train")

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
    p.add_argument("--sub",   type=int,   default=1)
    p.add_argument("--sup",   type=int,   default=10)

    # Entraînement
    p.add_argument("--epochs",           type=int,   default=50)
    p.add_argument("--lr",               type=float, default=5e-3)
    p.add_argument("--train_batch_size", type=int,   default=10)
    p.add_argument("--val_batch_size",   type=int,   default=1)
    p.add_argument("--test_batch_size",  type=int,   default=1)

    # I/O & device
    p.add_argument("--save_dir", type=str, default="./Results")
    p.add_argument("--device",   type=str, default="cuda")  # "cuda" ou "cpu"
    p.add_argument("--resume",   type=str, default=None)    # chemin d’un checkpoint .pt
    p.add_argument("--seed",     type=int, default=42)

    return p.parse_args()


def choose_device(pref: str) -> torch.device:
    if pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(pref)
    return torch.device("cpu")


def main():
    args = parse_args()

    # Seed (comportement reproductible raisonnable)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = choose_device(args.device)
    if device.type == "cuda":
        print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(device)}")
    else:
        print("[INFO] CUDA non dispo — bascule sur CPU.")

    # Dossiers et chemins
    os.makedirs(args.save_dir, exist_ok=True)
    run_dir = os.path.join(args.save_dir, datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    path_train = os.path.join(args.dataset_dir, args.train)
    path_val   = os.path.join(args.dataset_dir, args.val)
    path_test  = os.path.join(args.dataset_dir, args.test)

    # Sanity sur les fichiers
    for pth, name in [(path_train, "train"), (path_val, "val"), (path_test, "test")]:
        if not os.path.isfile(pth):
            print(f"[ERREUR] Fichier {name} introuvable: {pth}")
            sys.exit(1)

    # Tu forces le float64 dans tout le pipeline (réseau et fonctions).
    # Si tu veux passer en float32 plus tard, il faudra adapter les .double() dans le code.
    initial_x0 = None

    static_params = (args.alpha, args.beta, args.eta, args.sub, args.sup)
    train_params  = (args.epochs, args.lr, args.train_batch_size, args.val_batch_size, args.test_batch_size)

    manager = U_P3MG(
        num_layers=args.number_layers,
        num_pd_layers=args.number_pd_layers,
        static_params=static_params,
        initial_x0=initial_x0,
        train_params=train_params,
        paths=(path_train, path_val, path_test, run_dir),
        device=str(device)
    )

    # Option reprise
    ckpt = args.resume if (args.resume and os.path.isfile(args.resume)) else None

    if args.function == "train":
        manager.train(need_names=False, checkpoint_path=ckpt)
    else:
        # S’assure que les loaders test existent
        manager.create_loaders(need_names=False)
        manager.test(need_names=False, checkpoint_path=ckpt)


if __name__ == "__main__":
    main()
