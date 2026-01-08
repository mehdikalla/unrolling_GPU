# Fichier: src/network.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import json 
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
from Dataset.module import MyDataset

from src.models.P3MG_model import P3MG_model
from src.models.P3MG_func import P3MGNet

from src.utils.functions import snr_loss, tsnr_loss
from src.utils.visualization import plot_signals_gs, plot_signals_gs 
from src.utils.plotting_manager import PlottingManager

class U_P3MG(nn.Module):
    def __init__(self, num_layers, num_pd_layers, static_params, initial_x0, train_params, paths, device="cuda", args_dict=None, criterion="MSE"):
        super().__init__()
        self.num_layers = num_layers
        self.num_pd_layers = num_pd_layers
        self.static_params = static_params
        self.num_epochs, self.lr, self.train_bs, self.val_bs, self.test_bs = train_params
        
        self.device = torch.device(device if isinstance(device, str) else device)
        
        self.model = P3MG_model(self.num_layers, self.num_pd_layers).double().to(self.device)
        self.initial_x0 = initial_x0.double() if initial_x0 is not None else None
        self.p3mg_tmp = P3MGNet(self.num_pd_layers).to(self.device).double()

        (self.path_train, self.path_val, self.path_test, self.path_save) = paths
        self.path_checkpoints = os.path.join(self.path_save, 'checkpoints')
        self.path_plots       = os.path.join(self.path_save, 'plots')
        self.path_logs        = os.path.join(self.path_save, 'logs')
        
        for p in [self.path_save, self.path_checkpoints, self.path_plots, self.path_logs]:
            os.makedirs(p, exist_ok=True)
        
        if args_dict: self.save_config(args_dict)

        self.train_loader, self.val_loader, self.test_loader = None, None, None

        if criterion == 'MSE':
            self.criterion = nn.MSELoss(reduction='mean')
        elif criterion == 'SNR':
            self.criterion = snr_loss()
        elif criterion == 'TSNR':
            self.criterion = tsnr_loss()
        else:
            raise ValueError(f"Loss '{criterion}' non reconnue. Choisir parmi 'MSE', 'SNR', 'TSNR'.")
    
        tau_params = [p for n, p in self.model.named_parameters() if 'tau_k' in n]
        other_params = [p for n, p in self.model.named_parameters() if 'tau_k' not in n]
        self.optimizer = optim.Adam([
            {'params': other_params, 'lr': self.lr},
            {'params': tau_params, 'lr': self.lr * 5.0}
        ], lr=self.lr)
        self.N_dim, self.M_dim = 100, 100 

        self.plot_manager = PlottingManager(
            model=self.model, 
            p3mg_tmp=self.p3mg_tmp, 
            val_loader=None, 
            N_dim=self.N_dim, 
            M_dim=self.M_dim, 
            static_params=self.static_params, 
            device=self.device, 
            path_plots=self.path_plots
        )

    def create_loaders(self, need_names: bool = False):
        try:
            train_ds = MyDataset(self.path_train, self.initial_x0.numpy() if self.initial_x0 is not None else None, return_name=need_names)
            val_ds = MyDataset(self.path_val, self.initial_x0.numpy() if self.initial_x0 is not None else None, return_name=need_names)
            test_ds = MyDataset(self.path_test, self.initial_x0.numpy() if self.initial_x0 is not None else None, return_name=need_names)
            
            self.train_loader = DataLoader(train_ds, batch_size=self.train_bs, shuffle=True, num_workers=4)
            self.val_loader = DataLoader(val_ds, batch_size=self.val_bs, shuffle=False, num_workers=4)
            self.test_loader = DataLoader(test_ds, batch_size=self.test_bs, shuffle=False, num_workers=4)
            
            if hasattr(train_ds, 'X_true'): self.N_dim = train_ds.X_true.shape[1]
            if hasattr(train_ds, 'Y'): self.M_dim = train_ds.Y.shape[1]

            self.plot_manager.val_loader = self.val_loader
            self.plot_manager.N_dim = self.N_dim
            self.plot_manager.M_dim = self.M_dim
            
        except Exception as e:
            print(f"[WARN] Loaders init: {e}")

    def save_config(self, args_dict, filename='run_config.json'):
        try:
            clean = {k: str(v) if isinstance(v, torch.device) else v for k, v in args_dict.items()}
            with open(os.path.join(self.path_logs, filename), 'w') as f: json.dump(clean, f, indent=4)
        except: pass

    def load_checkpoint(self, checkpoint_path):
        """
        Charge les poids du modèle de manière robuste.
        Gère les cas CPU/GPU et les structures de dictionnaire.
        """
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("[WARNING] Aucun checkpoint valide fourni. Le modèle utilise ses poids d'initialisation (aléatoires).")
            return

        print(f"[LOADING] Chargement du checkpoint : {checkpoint_path}")
        
        try:
            # 1. Charger le fichier en gérant le device (évite les erreurs CUDA out of memory ou device mismatch)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # 2. Extraire le state_dict
            # Parfois on sauvegarde tout le dict {'epoch': 10, 'state_dict': ...}, parfois juste les poids.
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Optionnel : Récupérer l'époque si besoin
                # start_epoch = checkpoint.get('epoch', 0)
                # print(f"   -> Checkpoint de l'époque {start_epoch}")
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                 state_dict = checkpoint['state_dict'] # Convention PyTorch Lightning parfois
            else:
                # Supposons que c'est directement le state_dict
                state_dict = checkpoint

            # 3. Chargement strict dans le modèle
            # strict=True (défaut) crashe si les clés ne correspondent pas exactement. C'est ce qu'on veut pour être sûr !
            self.model.load_state_dict(state_dict, strict=True)
            
            print(f"[SUCCESS] Poids chargés avec succès !")

        except Exception as e:
            print(f"\n[ERREUR FATALE] Impossible de charger le checkpoint.\nErreur : {e}")
            sys.exit(1)

    def _unpack(self, batch):
        if len(batch) == 3: return batch[0], batch[1], batch[2]
        return batch[0], batch[1], None

    def train(self, need_names=False, checkpoint_path=None, args_dict=None):
        self.create_loaders(need_names)
        start_ep = self.load_checkpoint(checkpoint_path)
        dx = torch.zeros(1, self.N_dim).double().to(self.device)
        dy = torch.zeros(1, self.M_dim).double().to(self.device)
        static = self.p3mg_tmp.init_P3MG(list(self.static_params), dx, dy)
        tr_loss, val_loss = [], []
        best_vloss = float('inf')
        current_ckpt_path = None
        
        for ep in range(start_ep, self.num_epochs):
            self.model.train()
            losses = []
            for batch in self.train_loader:
                xt, y, x0 = self._unpack(batch)
                xt, y = xt.to(self.device).double(), y.to(self.device).double()
                if x0 is not None: x0 = x0.to(self.device).double()
                else: x0 = y.sum(1, keepdim=True).repeat(1, self.N_dim)/(self.M_dim*self.N_dim)
                self.optimizer.zero_grad()
                xp, _, _ = self.model(static, None, x0, y)
                loss = self.criterion(xp, xt)
                loss.backward(); self.optimizer.step()
                losses.append(loss.item())
            
            t_loss = sum(losses)/len(losses) if losses else 0
            tr_loss.append(t_loss)
            self.model.eval()
            v_losses = []
            with torch.no_grad():
                for batch in self.val_loader:
                    xt, y, x0 = self._unpack(batch)
                    xt, y = xt.to(self.device).double(), y.to(self.device).double()
                    if x0 is not None: x0 = x0.to(self.device).double()
                    else: x0 = y.sum(1, keepdim=True).repeat(1, self.N_dim)/(self.M_dim*self.N_dim)
                    xp, _, _ = self.model(static, None, x0, y)
                    v_losses.append(self.criterion(xp, xt).item())
            v_loss = sum(v_losses)/len(v_losses) if v_losses else 0
            val_loss.append(v_loss)
            print(f"Ep {ep+1}: Train={t_loss:.4e} Val={v_loss:.4e}")
            current_ckpt_path = os.path.join(self.path_checkpoints, f'checkpoint_epoch{ep+1}.pt')
            torch.save({'epoch': ep+1, 'model_state_dict': self.model.state_dict(), 'train_losses': tr_loss}, current_ckpt_path)
            if v_loss < best_vloss:
                best_vloss = v_loss
                torch.save({'epoch': ep+1, 'model_state_dict': self.model.state_dict()}, os.path.join(self.path_checkpoints, 'best_model.pt'))
            
            self.plot_manager.plot_signals(xt, xp, ep)

        self.plot_manager.plot_losses(tr_loss, val_loss)
        best_path = os.path.join(self.path_checkpoints, 'best_model.pt')
        if not os.path.exists(best_path) and current_ckpt_path: best_path = current_ckpt_path
        if os.path.exists(best_path):
            self.plot_manager.plot_learned_params_evolution(best_path)
            self.plot_manager.plot_best_signals(best_path)
        return tr_loss, val_loss

    def test(self, need_names=False, checkpoint_path=None):
        """
        Teste le modèle, affiche les stats et plot le MEILLEUR et le PIRE signal.
        """
        self.load_checkpoint(checkpoint_path)
        self.create_loaders(need_names)
        
        dx = torch.zeros(1, self.N_dim).double().to(self.device)
        dy = torch.zeros(1, self.M_dim).double().to(self.device)
        static = self.p3mg_tmp.init_P3MG(list(self.static_params), dx, dy)
        
        self.model.eval()
        all_losses = []
        
        # Variables pour traquer le meilleur et le pire
        best_mse = float('inf')
        worst_mse = -1.0
        
        best_pair = None  # (xt, xp)
        worst_pair = None # (xt, xp)
        
        print(f"[INFO] Démarrage du test sur {len(self.test_loader.dataset)} échantillons...")

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                xt, y, x0 = self._unpack(batch)
                xt, y = xt.to(self.device).double(), y.to(self.device).double()
                if x0 is not None: x0=x0.to(self.device).double()
                else: x0 = y.sum(1, keepdim=True).repeat(1, self.N_dim)/(self.M_dim*self.N_dim)
                
                xp, _, _ = self.model(static, None, x0, y)
                
                # Calcul de la MSE par échantillon (batch_size,)
                # (batch, N) -> (xp - xt)**2 -> mean sur dim 1
                sample_losses = torch.mean((xp - xt)**2, dim=1)
                
                # --- LOGIQUE DE RECHERCHE MIN/MAX ---
                
                # 1. Trouver le min et max dans ce batch
                batch_min_val, batch_min_idx = torch.min(sample_losses, dim=0)
                batch_max_val, batch_max_idx = torch.max(sample_losses, dim=0)
                
                # 2. Mettre à jour le BEST global
                if batch_min_val.item() < best_mse:
                    best_mse = batch_min_val.item()
                    # On clone et détache pour stocker en mémoire CPU sans casser le graphe ou saturer le GPU
                    best_pair = (
                        xt[batch_min_idx].unsqueeze(0).cpu(), 
                        xp[batch_min_idx].unsqueeze(0).cpu()
                    )

                # 3. Mettre à jour le WORST global (Utile pour debugger !)
                if batch_max_val.item() > worst_mse:
                    worst_mse = batch_max_val.item()
                    worst_pair = (
                        xt[batch_max_idx].unsqueeze(0).cpu(), 
                        xp[batch_max_idx].unsqueeze(0).cpu()
                    )

                # Stockage pour les stats globales
                all_losses.extend(sample_losses.cpu().numpy().tolist())

        # --- FIN DE BOUCLE : PLOTTING ---
        
        if best_pair is not None:
            print(f"[RESULT] Meilleure MSE trouvée : {best_mse:.2e}")
            # On renvoie sur le device pour le plot (si ta fonction plot attend du GPU)
            # Sinon laisse en CPU selon ta fonction plot_manager
            xt_best, xp_best = best_pair
            self.plot_manager.plot_signals(xt_best, xp_best, 'test_BEST_mse_{:.2e}'.format(best_mse))
            
        if worst_pair is not None:
            print(f"[RESULT] Pire MSE trouvée : {worst_mse:.2e}")
            xt_worst, xp_worst = worst_pair
            self.plot_manager.plot_signals(xt_worst, xp_worst, 'test_WORST_mse_{:.2e}'.format(worst_mse))
        # Calcul des Statistiques
        losses_arr = np.array(all_losses)
        mean_val = np.mean(losses_arr)
        std_val  = np.std(losses_arr)
        min_val  = np.min(losses_arr)
        max_val  = np.max(losses_arr)
        count    = len(losses_arr)

        # Création du Tableau (String)
        table_str = (
            f"\n"
            f"+-----------------------------------------+\n"
            f"|        RESULTATS DU TEST SET            |\n"
            f"+-----------------------+-----------------+\n"
            f"| Metrique              | Valeur          |\n"
            f"+-----------------------+-----------------+\n"
            f"| Nombre d'echantillons | {count:<15} |\n"
            f"| Moyenne (MSE)         | {mean_val:<15.4e} |\n"
            f"| Ecart-Type (Std)      | {std_val:<15.4e} |\n"
            f"| Minimum (MSE)         | {min_val:<15.4e} |\n"
            f"| Maximum (MSE)         | {max_val:<15.4e} |\n"
            f"+-----------------------+-----------------+\n"
        )
        
        # 1. Affichage console
        print(table_str)
        
        # 2. Sauvegarde fichier texte local
        table_path = os.path.join(self.path_logs, 'test_results_table.txt')
        with open(table_path, 'w') as f:
            f.write(table_str)
        print(f"[INFO] Tableau sauvegardé dans : {table_path}")

        # 3. Plot de l'histogramme
        self.plot_manager.plot_test_error_distribution(losses_arr)
        
        return mean_val

    def run_test_2d(self, l_val, t_val, loader, ckpt, plot_path, save_signal=False):
        self.load_checkpoint(ckpt); self.model.eval()
        dx = torch.zeros(1, self.N_dim).double().to(self.device)
        dy = torch.zeros(1, self.M_dim).double().to(self.device)
        static = self.p3mg_tmp.init_P3MG(list(self.static_params), dx, dy)
        l_over = torch.tensor(l_val).double().to(self.device)
        t_over = torch.tensor(t_val).double().to(self.device)
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                xt, y, x0 = self._unpack(batch)
                xt, y = xt.to(self.device).double(), y.to(self.device).double()
                if x0 is not None: x0=x0.to(self.device).double()
                else: x0 = y.sum(1, keepdim=True).repeat(1, self.N_dim)/(self.M_dim*self.N_dim)
                xp, _, _ = self.model(static, None, x0, y, lmbd_override=l_over, tau_override=t_over)
                losses.append(self.criterion(xp, xt).item())
                if save_signal and i == 0: plot_signals_gs(xt, xp, l_val, t_val, plot_path) 
        return sum(losses)/len(losses) if losses else 0

    def grid_search_2d(self, lambdas, taus, need_names=False, checkpoint_path=None, mode="grid"):
        self.create_loaders(need_names); out_dir = self.path_plots
        is_distributed = (len(lambdas) == 1 and len(taus) == 1)
        
        to_test = []
        if mode == "random" and not is_distributed:
            for l, t in zip(lambdas, taus): to_test.append((l, t))
        else:
            for l in lambdas:
                for t in taus: to_test.append((l, t))

        print(f"[INFO] Lancement GS (Mode={mode}, {len(to_test)} points)")

        for l, t in to_test:
            loss = self.run_test_2d(l, t, self.test_loader, checkpoint_path, out_dir, save_signal=is_distributed)
            print(f"L={l:.2e} T={t:.2f} : {loss:.4e}")
            task_id = os.environ.get('SLURM_ARRAY_TASK_ID', 'local')
            safe_l = f"{l:.2e}".replace('+',''); safe_t = f"{t:.2f}"
            fname = f'gs_result_L{safe_l}_T{safe_t}_tid{task_id}.json'
            with open(os.path.join(self.path_logs, fname), 'w') as f:
                json.dump({"lambda": l, "tau": t, "loss": loss}, f, indent=4)

        plot_signals_gs(self.path_save, self.path_logs, self.path_plots)

    def solve_iterative_p3mg(self, y, x_true, lmbd_val, tau_val, max_iter=500):
        """
        Résout P3MG itératif sur GPU pour un batch complet.
        Utilise les hyperparamètres fixes lmbd_val et tau_val.
        """
        device = self.device
        
        # Init variables statiques
        # On utilise des dummies pour dx/dy car init_P3MG n'en a besoin que pour le type/device
        dx_dummy = torch.zeros(1, self.N_dim).double().to(device)
        dy_dummy = torch.zeros(1, self.M_dim).double().to(device)
        
        # Init x0 (ici x0 = y pour l'iteratif classique)
        x = y.clone()
        
        # Calcul des matrices constantes (Hmat etc.) via P3MG_func
        static_vars = self.p3mg_tmp.init_P3MG(list(self.static_params), dx_dummy, dy_dummy)
        
        # Formatage des hyperparamètres en tenseurs
        current_lmbd = torch.tensor(lmbd_val, device=device).double()
        # Le modèle attend une liste de Taus (un par 'couche' virtuelle), on duplique la valeur
        tau_params_list = [torch.tensor(tau_val, device=device).double() for _ in range(self.num_pd_layers)]

        # Boucle itérative (pas d'apprentissage, juste application de l'algo)
        with torch.no_grad():
            # Initialisation dynamique (itération 1)
            x, dynamic = self.p3mg_tmp.iter_P3MG_base(static_vars, x, y, current_lmbd, tau_params_list)
            
            # Itérations suivantes
            for k in range(1, max_iter):
                x, dynamic = self.p3mg_tmp.iter_P3MG(static_vars, dynamic, x, y, current_lmbd, tau_params_list)
        
        return x

def run_oracle_analysis(self, lambdas, taus, max_iter=500, mode="grid"):
        """
        Exécute l'analyse Oracle.
        mode="grid"   : Teste toutes les combinaisons itertools.product(lambdas, taus)
        mode="random" : Teste les paires zip(lambdas, taus)
        """
        from src.utils.visualization import plot_oracle_analysis, plot_oracle_reconstruction
        import itertools
        from tqdm import tqdm

        # 1. Préparation
        oracle_dir = os.path.join(self.path_save, "oracle")
        os.makedirs(oracle_dir, exist_ok=True)
        
        # Définition de la liste des couples à tester
        if mode == "random":
            if len(lambdas) != len(taus):
                raise ValueError(f"En mode random, len(lambdas) ({len(lambdas)}) doit égaler len(taus) ({len(taus)})")
            # On forme les couples (L_i, T_i)
            grid = list(zip(lambdas, taus))
            desc_str = f"Random Search : {len(grid)} couples (L, T)"
        else:
            # Mode Grid classique
            grid = list(itertools.product(lambdas, taus))
            desc_str = f"Grid Search : {len(lambdas)} L x {len(taus)} T ({len(grid)} points)"

        print(f"\n[ORACLE] Démarrage de l'analyse.")
        print(f"         Sortie : {oracle_dir}")
        print(f"         Mode   : {desc_str}")
        print(f"         Iters  : {max_iter}")

        # 2. Chargement Données (Batch Processing sur GPU)
        if self.test_loader is None: self.create_loaders()
        
        all_X_true, all_Y = [], []
        for batch in self.test_loader:
            xt, y, _ = self._unpack(batch)
            all_X_true.append(xt)
            all_Y.append(y)
            
        X_true_all = torch.cat(all_X_true, dim=0).to(self.device).double()
        Y_all      = torch.cat(all_Y, dim=0).to(self.device).double()
        N_samples  = X_true_all.shape[0]

        # 3. Trackers
        best_mse = torch.full((N_samples,), float('inf'), device=self.device, dtype=torch.double)
        # On initialise best_params à -1 pour repérer s'il y a un souci
        best_params = torch.full((N_samples, 2), -1.0, device=self.device, dtype=torch.double) 

        # 4. Boucle sur les couples
        for (l_val, t_val) in tqdm(grid, desc="Oracle Loop"):
            
            # A. Résolution (Batch complet)
            X_hat = self.solve_iterative_p3mg(Y_all, X_true_all, lmbd_val=l_val, tau_val=t_val, max_iter=max_iter)
            
            # B. Calcul MSE par signal
            mse_curr = torch.mean((X_hat - X_true_all)**2, dim=1)
            
            # C. Mise à jour "Best Per Sample"
            improved = mse_curr < best_mse
            
            if improved.any():
                best_mse[improved] = mse_curr[improved]
                best_params[improved, 0] = l_val
                best_params[improved, 1] = t_val

        # 5. Métriques Finales
        sig_energy = torch.mean(X_true_all**2, dim=1)
        best_snr_list = 10 * torch.log10(sig_energy / (best_mse + 1e-12))
        
        final_mse = torch.mean(best_mse).item()
        final_snr = torch.mean(best_snr_list).item()

        print("\n" + "="*40)
        print(f" RÉSULTATS ORACLE ({mode.upper()})")
        print("="*40)
        print(f" MSE Mean : {final_mse:.4e}")
        print(f" SNR Mean : {final_snr:.2f} dB")
        print("="*40)

        # 6. Sauvegardes
        results_dict = {
            "mode": mode,
            "global_mse": final_mse,
            "global_snr": final_snr,
            "best_params_L": best_params[:, 0].cpu().tolist(),
            "best_params_T": best_params[:, 1].cpu().tolist(),
            "best_mses": best_mse.cpu().tolist()
        }
        torch.save(results_dict, os.path.join(oracle_dir, "oracle_data.pt"))
        
        # Plots
        plot_oracle_analysis(best_params.cpu().numpy(), best_mse.cpu().numpy(), oracle_dir)
        
        # Reconstruction exemple (Best + Median)
        best_idx_global = torch.argmin(best_mse).item()
        median_idx_global = torch.argsort(best_mse)[N_samples // 2].item()
        
        for idx, lbl in zip([best_idx_global, median_idx_global], ["best_case", "median_case"]):
            l_opt = best_params[idx, 0].item()
            t_opt = best_params[idx, 1].item()
            
            # Relance uniquement pour ce signal
            if l_opt > 0: # Check validité
                y_s = Y_all[idx:idx+1]; xt_s = X_true_all[idx:idx+1]
                x_o = self.solve_iterative_p3mg(y_s, xt_s, lmbd_val=l_opt, tau_val=t_opt, max_iter=max_iter)
                plot_oracle_reconstruction(
                    xt_s.squeeze().cpu().numpy(), y_s.squeeze().cpu().numpy(), x_o.squeeze().cpu().numpy(),
                    l_opt, t_opt, oracle_dir, f"{lbl}_idx{idx}"
                )
                
        print(f"[SUCCESS] Analyse Oracle terminée. Voir {oracle_dir}")