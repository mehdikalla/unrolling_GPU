# src/visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import shutil

def plot_gs_loss_params(path_save: str, path_logs: str, path_plots: str):
    """
    Lit tous les fichiers JSON du Grid Search, trouve l'optimum,
    et génère les deux graphes de projection (Loss vs Lambda et Loss vs Tau).
    """
    print(f"[INFO] Consolidation des résultats GS depuis : {path_logs}")
    data = []
    jsons = glob.glob(os.path.join(path_logs, 'gs_result_*.json'))
    if not jsons: 
        print("[WARN] Aucun fichier JSON trouvé pour la consolidation.")
        return

    for j in jsons:
        try:
            with open(j, 'r') as f: data.append(json.load(f))
        except Exception as e:
            # Souvent dû à un fichier partiellement écrit lors d'un crash Slurm
            print(f"[WARN] Erreur de lecture/parsing du JSON {j}: {e}")
            pass
    if not data: return

    # Extraction Brute
    raw_l = np.array([d['lambda'] for d in data])
    raw_t = np.array([d['tau'] for d in data])
    raw_loss = np.array([d['loss'] for d in data])

    # Meilleur Point Global
    best_idx = np.argmin(raw_loss)
    best_l, best_t, best_loss = raw_l[best_idx], raw_t[best_idx], raw_loss[best_idx]

    # --- Plot 1: Projection Loss vs Lambda (Scatter) ---
    plt.figure(figsize=(10, 6))
    plt.scatter(raw_l, raw_loss, c=raw_t, cmap='viridis', label='Points', alpha=0.7)
    plt.colorbar(label=r'$\tau$')
    plt.plot(best_l, best_loss, '*', markersize=15, label='Optimum')
    plt.xscale('log')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Loss')
    plt.ylim(0, 0.0001)
    plt.title('Projection : Loss vs Lambda')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(path_save, 'GLOBAL_GS_SCATTER_LAMBDA.png'))
    plt.close()

    # --- Plot 2: Projection Loss vs Tau (Scatter) ---
    plt.figure(figsize=(10, 6))
    plt.scatter(raw_t, raw_loss, c=np.log10(raw_l), cmap='plasma', label='Points', alpha=0.7)
    plt.colorbar(label=r'$\log_{10}(\lambda)$')
    plt.plot(best_t, best_loss, '*', markersize=15, label='Optimum')
    plt.xlabel(r'$\tau$')
    plt.ylabel('Loss')
    plt.ylim(0, 0.0001)
    plt.title('Projection : Loss vs Tau')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(path_save, 'GLOBAL_GS_SCATTER_TAU.png'))
    plt.close()

    # Copie Signal
    pat_l = f"{best_l:.2e}".replace('+','')
    pat_t = f"{best_t:.2f}"
    sig_glob = os.path.join(path_plots, f"signal_L{pat_l}*T{pat_t}*.png")
    found = glob.glob(sig_glob)
    if found:
        shutil.copy(found[0], os.path.join(path_save, 'GLOBAL_BEST_SIGNAL.png'))
        print("[SUCCESS] Signal optimal copié.")

    print(f"[SUCCESS] Courbes générées. Best: L={best_l:.2e}, T={best_t:.2f}")


def plot_signals_gs(true, pred, l_val, t_val, path):
    """ Plot un signal unique pour un point (Lambda, Tau) donné. """
    pat_l = f"{l_val:.2e}".replace('+','')
    pat_t = f"{t_val:.2f}"
    fname = f"signal_L{pat_l}_T{pat_t}.png"
    plt.figure(figsize=(10,3)); plt.plot(true[0].cpu().numpy(), label='True')
    plt.plot(pred[0].detach().cpu().numpy(), '--', label='Pred')
    plt.grid()
    plt.title(f'L={l_val:.2e} T={t_val:.2f}')
    plt.legend()
    plt.savefig(os.path.join(path, fname))
    plt.close()