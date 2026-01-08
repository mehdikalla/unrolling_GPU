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

def plot_oracle_analysis(best_params, best_mses, oracle_dir):
    """
    Generates plots for the Oracle analysis.
    - best_params: Array (N_samples, 2) containing [Lambda_opt, Tau_opt] for each signal.
    - best_mses: Array (N_samples,) containing the optimal MSE for each signal.
    - oracle_dir: Output directory.
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    os.makedirs(oracle_dir, exist_ok=True)
    
    lambdas = best_params[:, 0]
    taus = best_params[:, 1]
    
    # 1. Histogram of optimal Lambdas
    plt.figure(figsize=(8, 5))
    plt.hist(np.log10(lambdas), bins=20, color='teal', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Optimal Lambdas (per signal)")
    plt.xlabel("Log10(Lambda)")
    plt.ylabel("Number of signals")
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(oracle_dir, "distrib_best_lambdas.png"))
    plt.close()

    # 2. Histogram of optimal Taus
    plt.figure(figsize=(8, 5))
    plt.hist(taus, bins=20, color='orange', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Optimal Taus (per signal)")
    plt.xlabel("Tau")
    plt.ylabel("Number of signals")
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(oracle_dir, "distrib_best_taus.png"))
    plt.close()
    
    # 3. 2D Scatter Plot (Winning parameter pairs)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(taus, np.log10(lambdas), c=best_mses, cmap='viridis', alpha=0.6, s=15)
    plt.colorbar(sc, label='Best MSE')
    plt.title("Map of Best Parameters (Lambda vs Tau)")
    plt.xlabel("Tau")
    plt.ylabel("Log10(Lambda)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(oracle_dir, "scatter_best_params.png"))
    plt.close()
    
def plot_oracle_reconstruction(x_true, y_input, x_oracle, best_l, best_t, oracle_dir, index):
    """
    Plots a comparison: Noisy Signal vs Ground Truth vs Oracle Reconstruction
    """
    plt.figure(figsize=(12, 5))
    # Plot only a part if it's too long, otherwise everything
    plt.plot(x_true, 'k', label='Ground Truth', linewidth=1.5, alpha=0.8)
    plt.plot(y_input, 'lightgray', label='Input (Noisy)', alpha=0.5)
    plt.plot(x_oracle, 'r--', label=f'Oracle (L={best_l:.1e}, T={best_t:.2f})', linewidth=1.5)
    
    plt.title(f"Oracle Reconstruction - Signal #{index}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(oracle_dir, f"oracle_signal_example_{index}.png"))
    plt.close()