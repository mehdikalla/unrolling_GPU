# src/network.py (Copier-coller intégral avec Optimisation du Taux d'Apprentissage pour Tau)
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json 
from datetime import datetime
from torch.utils.data import DataLoader

from Dataset.module import MyDataset  

from src.models.P3MG_model import P3MG_model
from src.models.P3MG_func import P3MGNet
import torch.nn.functional as F

# ==============================================================================
# Manager de la boucle d'apprentissage et de test
# ==============================================================================
class U_P3MG(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_pd_layers: int,
        static_params: tuple,
        initial_x0: torch.Tensor,
        train_params: tuple,
        paths: tuple,
        device: str = "cuda",
        args_dict: dict = None
    ):
        super().__init__()

        # 1) Hyperparamètres
        self.num_layers = num_layers       # N
        self.num_pd_layers = num_pd_layers # M
        self.static_params = static_params
        self.num_epochs, self.lr, self.train_bs, self.val_bs, self.test_bs = train_params

        # 2) Device
        if isinstance(device, str):
            if device.startswith("cuda") and torch.cuda.is_available():
                self.device = torch.device(device)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        # 3) Modèle
        self.model = P3MG_model(self.num_layers, self.num_pd_layers).double().to(self.device)
        self.initial_x0 = initial_x0.double() if initial_x0 is not None else None
        
        # Instance temporaire pour calculs statiques (init_P3MG)
        self.p3mg_tmp = P3MGNet(self.num_pd_layers).to(self.device).double()


        # 4) Chemins et Structure
        (self.path_train, self.path_val, self.path_test, self.path_save) = paths
        
        self.path_checkpoints = os.path.join(self.path_save, 'checkpoints')
        self.path_plots       = os.path.join(self.path_save, 'plots')
        self.path_logs        = os.path.join(self.path_save, 'logs')
        
        os.makedirs(self.path_save, exist_ok=True)
        os.makedirs(self.path_checkpoints, exist_ok=True)
        os.makedirs(self.path_plots, exist_ok=True)
        os.makedirs(self.path_logs, exist_ok=True)
        
        if args_dict is not None:
             self.save_config(args_dict)

        # 5) Variables internes
        self.train_loader, self.val_loader, self.test_loader = None, None, None
        self.criterion = nn.MSELoss(reduction="mean")
        self.N_dim = 100 
        self.M_dim = 100

        # 6) Optimiseur (MODIFICATION CLÉ : Groupes de paramètres pour le taux d'apprentissage)
        tau_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            # Les paramètres Tau sont nommés 'tau_k' dans les couches P3MG
            if 'tau_k' in name:
                tau_params.append(param)
            else:
                other_params.append(param)
        
        # Définition des groupes:
        optimizer_params = [
            {'params': other_params, 'lr': self.lr},
            # Le LR pour Tau est 10 fois supérieur pour prévenir le gradient évanescent
            {'params': tau_params, 'lr': self.lr * 10} 
        ]
        
        self.optimizer = optim.Adam(optimizer_params, lr=self.lr)


    def create_loaders(self, need_names: bool = False):
        try:
            train_ds = MyDataset(self.path_train, self.initial_x0.numpy() if self.initial_x0 is not None else None, return_name=need_names)
            val_ds =  MyDataset(self.path_val, self.initial_x0.numpy() if self.initial_x0 is not None else None, return_name=need_names)
            test_ds = MyDataset(self.path_test, self.initial_x0.numpy() if self.initial_x0 is not None else None, return_name=need_names)
            
            self.train_loader = DataLoader(train_ds, batch_size=self.train_bs, shuffle=True, num_workers=4, pin_memory=True)
            self.val_loader = DataLoader(val_ds, batch_size=self.val_bs, shuffle=False, num_workers=4, pin_memory=True)
            self.test_loader = DataLoader(test_ds, batch_size=self.test_bs, shuffle=False, num_workers=4, pin_memory=True)
            
            self.N_dim = train_ds.X_true.shape[1] 
            self.M_dim = train_ds.Y.shape[1] 
            
        except Exception as e:
             print(f"[ERREUR] Création des loaders: {e}")
             self.N_dim, self.M_dim = 100, 100

    def save_config(self, args_dict: dict, filename: str = 'run_config.json'):
        config_path = os.path.join(self.path_logs, filename)
        try:
            clean_args = {k: str(v) if isinstance(v, torch.device) else v for k, v in args_dict.items()}
            with open(config_path, 'w') as f:
                json.dump(clean_args, f, indent=4)
        except Exception as e:
            print(f"[ERREUR] Config save: {e}")
            
    def load_checkpoint(self, checkpoint_path: str):
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print(f"[INFO] Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                 # Le chargement est compatible avec les groupes de paramètres de l'optimiseur
                 self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            return ckpt.get('epoch', 0)
        return 0

    def _unpack_batch(self, batch, need_names: bool):
        if need_names:
            if len(batch) == 4: return batch[1], batch[2], batch[3]
            else: return batch[1], batch[2], None
        else:
            if len(batch) == 3: return batch[0], batch[1], batch[2]
            else: return batch[0], batch[1], None

    def train(self, need_names: bool = False, checkpoint_path: str = None, args_dict: dict = None):
        self.create_loaders(need_names)
        current_ckpt_path = None
        start_epoch = self.load_checkpoint(checkpoint_path)
        
        # Init static params (simulation)
        dummy_x0 = torch.zeros(1, self.N_dim).double().to(self.device)
        dummy_y = torch.zeros(1, self.M_dim).double().to(self.device)
        static = self.p3mg_tmp.init_P3MG(list(self.static_params), dummy_x0, dummy_y) 
        
        train_losses, val_losses = [], []
        best_val_loss = float('inf')

        for epoch in range(start_epoch, self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # --- TRAIN ---
            self.model.train()
            batch_losses = []
            for batch in self.train_loader:
                X_true, Y, X0 = self._unpack_batch(batch, need_names)
                X_true = X_true.to(self.device, non_blocking=True).double()
                Y = Y.to(self.device, non_blocking=True).double()
                X0 = X0.to(self.device, non_blocking=True).double() if X0 is not None else None
                
                if X0 is None:
                    sumY = Y.sum(dim=1, keepdim=True)
                    X0 = sumY.repeat(1, self.N_dim) / (self.M_dim * self.N_dim)
                
                self.optimizer.zero_grad()
                X_pred, _, _ = self.model(static, None, X0, Y) 
                loss = self.criterion(X_pred, X_true)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
            
            epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
            train_losses.append(epoch_loss)

            # --- VAL ---
            self.model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for batch in self.val_loader:
                    X_true, Y, X0 = self._unpack_batch(batch, need_names)
                    X_true = X_true.to(self.device).double()
                    Y = Y.to(self.device).double()
                    X0 = X0.to(self.device).double() if X0 is not None else None
                    if X0 is None:
                        sumY = Y.sum(dim=1, keepdim=True)
                        X0 = sumY.repeat(1, self.N_dim) / (self.M_dim * self.N_dim)
                    
                    X_pred, _, _ = self.model(static, None, X0, Y)
                    val_batch_losses.append(self.criterion(X_pred, X_true).item())
                    
            val_loss = sum(val_batch_losses) / len(val_batch_losses) if val_batch_losses else 0
            val_losses.append(val_loss)
            
            print(f" Train: {epoch_loss:.4e} | Val: {val_loss:.4e}")

            # Checkpoint
            current_ckpt_path = os.path.join(self.path_checkpoints, f'checkpoint_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }, current_ckpt_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.path_checkpoints, 'best_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict()
                }, best_model_path)

            # Signal plot (epoch end)
            self.plot_signals(X_true, X_pred, epoch)

        # Courbes
        self.plot_losses(train_losses, val_losses)
        
        # --- NOUVEAU : PLOT DES PARAMÈTRES APPRIS À LA FIN DE L'ENTRAÎNEMENT ---
        final_plot_path = os.path.join(self.path_checkpoints, 'best_model.pt')
        if not os.path.exists(final_plot_path):
             final_plot_path = current_ckpt_path 
             
        if final_plot_path:
            self.plot_learned_params_evolution(final_plot_path)
        else:
            print("[AVERTISSEMENT] Aucun checkpoint disponible pour le plotting des paramètres.")
                 
        return train_losses, val_losses

    def test(self, need_names: bool = False, checkpoint_path: str = None):
        self.load_checkpoint(checkpoint_path)
        self.create_loaders(need_names)
        
        dummy_x0 = torch.zeros(1, self.N_dim).double().to(self.device)
        dummy_y = torch.zeros(1, self.M_dim).double().to(self.device)
        static = self.p3mg_tmp.init_P3MG(list(self.static_params), dummy_x0, dummy_y) 
        
        self.model.eval()
        test_losses = []
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                X_true, Y, X0 = self._unpack_batch(batch, need_names)
                X_true = X_true.to(self.device).double()
                Y = Y.to(self.device).double()
                X0 = X0.to(self.device).double() if X0 is not None else None
                if X0 is None:
                    sumY = Y.sum(dim=1, keepdim=True)
                    X0 = sumY.repeat(1, self.N_dim) / (self.M_dim * self.N_dim)

                X_pred, _, _ = self.model(static, None, X0, Y)
                test_losses.append(self.criterion(X_pred, X_true).item())
                if i == 0: self.plot_signals(X_true, X_pred, 'test')

        avg_loss = sum(test_losses)/len(test_losses) if test_losses else 0
        print(f"[TEST] Loss: {avg_loss:.4e}")
        return avg_loss

    # ========================================================================
    # PLOTTING
    # ========================================================================

    def plot_losses(self, train, val):
        plt.figure(figsize=(10,6))
        plt.plot(train, label='Train'); plt.plot(val, label='Val')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(self.path_plots, 'loss_curve.png')); plt.close()

    def plot_signals(self, true, pred, epoch):
        for i in range(min(3, true.size(0))):
            plt.figure(figsize=(10,4))
            plt.plot(true[i].cpu().numpy(), '-', label='True')
            plt.plot(pred[i].detach().cpu().numpy(), '--', label='Pred')
            plt.title(f'Sample {i} (Epoch {epoch})'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(self.path_plots, f'signal_ep{epoch}_s{i}.png')); plt.close()

    def plot_signals_grid_search(self, true, pred, lmbd_val, path_save):
        for i in range(min(1, true.size(0))): 
            plt.figure(figsize=(10,4))
            plt.plot(true[i].cpu().numpy(), '-', label='True')
            plt.plot(pred[i].detach().cpu().numpy(), '--', label='Pred')
            plt.title(f'Reconstruction Signal (Lambda {lmbd_val:.2e})'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(path_save, f'signal_recons_lambda_{lmbd_val:.2e}.png')); plt.close()

    def plot_grid_search(self, results):
        l, v = list(results.keys()), list(results.values())
        plt.figure(figsize=(10,6))
        plt.semilogx(l, v, 'o-')
        plt.xlabel(r'$\lambda$ (Valeur constante)'); plt.ylabel('Loss (MSE)')
        plt.title('Grid Search: Loss vs Paramètre Lambda')
        plt.grid(True, which="both", ls="-", alpha=0.6)
        plt.savefig(os.path.join(self.path_plots, 'grid_search_loss_vs_lambda.png')); plt.close()
        print("[INFO] Plot Loss vs Lambda généré.")


    def plot_learned_params_evolution(self, path_model: str):
        if not os.path.isfile(path_model): return
        print(f"[INFO] Plotting params from: {path_model}")
        
        ckpt = torch.load(path_model, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        
        base = os.path.dirname(os.path.dirname(path_model))
        out_dir = os.path.join(base, 'plots')
        if not os.path.isdir(out_dir): out_dir = os.path.dirname(path_model)

        S = nn.Softplus()
        N = self.model.num_layers
        M = self.model.num_pd_layers

        # --- Extraction ---
        lmbd_vals = []
        tau_rows = []

        dummy_y_input = torch.ones(1, self.M_dim).double().to(self.device)
        dummy_x0 = torch.zeros(1, self.N_dim).double().to(self.device)
        
        dummy_static = self.p3mg_tmp.init_P3MG(list(self.static_params), dummy_x0, dummy_y_input) 

        with torch.no_grad():
            _, _, dynamic_lambdas_tensors = self.model(dummy_static, None, dummy_x0, dummy_y_input)

        for k, layer in enumerate(self.model.Layers):
            if k < len(dynamic_lambdas_tensors):
                lmbd_vals.append(dynamic_lambdas_tensors[k].squeeze().item())
            
            if hasattr(layer, 'tau_k'):
                tau_rows.append(S(layer.tau_k).detach().cpu().numpy().flatten())


        # --- Plot Lambda ---
        if lmbd_vals:
            plt.figure(figsize=(10,6))
            plt.plot(range(1, N+1), lmbd_vals, 'o-')
            plt.xlabel('Couche k'); plt.ylabel(r'$\lambda_k$ (Valeur calculée)')
            plt.title('Évolution du paramètre Lambda par couche')
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, 'learnt_lambda_curve.png')); plt.close()

        # --- Plot Tau (Heatmap) ---
        if tau_rows and all(len(row) == M for row in tau_rows):
            tau_mat = np.array(tau_rows) # (N, M)
            plt.figure(figsize=(12, 6))
            
            im = plt.imshow(tau_mat, aspect='auto', cmap='viridis', origin='lower',
                       extent=[0.5, M+0.5, 0.5, N+0.5])
            plt.colorbar(im, label=r'$\tau_{k,j}$ (appris)')
            
            plt.xlabel('Sous-couche j (PD)'); plt.ylabel('Couche k (P3MG)')
            plt.title(r'Heatmap des pas $\tau$ (N couches x M sous-couches)')
            
            plt.xticks(np.arange(1, M+1)); plt.yticks(np.arange(1, N+1))
            plt.savefig(os.path.join(out_dir, 'learnt_tau_heatmap.png')); plt.close()
            print("[INFO] Heatmap Tau générée.")
        
        print("[INFO] Plotting des paramètres appris terminé.")


    def run_test_for_lambda(self, val, loader, ckpt, plot_path, is_first_val):
        self.load_checkpoint(ckpt)
        self.model.eval()
        
        dummy_x0 = torch.zeros(1, self.N_dim).double().to(self.device)
        dummy_y = torch.zeros(1, self.M_dim).double().to(self.device)
        static = self.p3mg_tmp.init_P3MG(list(self.static_params), dummy_x0, dummy_y)
        
        override = torch.tensor(val).double().to(self.device)
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                xt, y, x0 = self._unpack_batch(batch, False)
                xt, y = xt.to(self.device).double(), y.to(self.device).double()
                if x0 is not None: x0=x0.to(self.device).double()
                else: x0 = y.sum(1, keepdim=True).repeat(1, self.N_dim)/(self.M_dim*self.N_dim)
                
                xp, _, _ = self.model(static, None, x0, y, lmbd_override=override)
                total_loss += self.criterion(xp, xt).item()

                if is_first_val and i == 0:
                     self.plot_signals_grid_search(xt, xp, val, plot_path)
                     
        return total_loss/len(loader)

    def grid_search_lambda(self, vals, need_names=False, checkpoint_path=None):
        self.create_loaders(need_names)
        res = {}
        out_dir = self.path_plots
        
        for idx, v in enumerate(vals):
            is_first = (idx == 0)
            res[v] = self.run_test_for_lambda(v, self.test_loader, checkpoint_path, out_dir, is_first)
            print(f"Lambda {v}: Loss {res[v]:.4e}")

        self.plot_grid_search(res) 
        
        log_path = os.path.join(self.path_logs, 'grid_search_results.json')
        try:
            with open(log_path, 'w') as f:
                json.dump({str(k): v for k, v in res.items()}, f, indent=4)
            print(f"[INFO] Résultats Grid Search sauvegardés dans {log_path}")
        except Exception as e:
            print(f"[ERREUR] Échec de la sauvegarde des résultats du Grid Search : {e}")
        
        return res