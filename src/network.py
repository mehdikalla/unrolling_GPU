# src/network.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json 
import glob
import shutil
from torch.utils.data import DataLoader
from Dataset.module import MyDataset
from src.models.P3MG_model import P3MG_model
from src.models.P3MG_func import P3MGNet

class U_P3MG(nn.Module):
    def __init__(self, num_layers, num_pd_layers, static_params, initial_x0, train_params, paths, device="cuda", args_dict=None):
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
        self.criterion = nn.MSELoss(reduction="mean")
        
        tau_params = [p for n, p in self.model.named_parameters() if 'tau_k' in n]
        other_params = [p for n, p in self.model.named_parameters() if 'tau_k' not in n]
        self.optimizer = optim.Adam([
            {'params': other_params, 'lr': self.lr},
            {'params': tau_params, 'lr': self.lr * 5.0}
        ], lr=self.lr)
        self.N_dim, self.M_dim = 100, 100 

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
        except Exception as e:
            print(f"[WARN] Loaders init: {e}")

    def save_config(self, args_dict, filename='run_config.json'):
        try:
            clean = {k: str(v) if isinstance(v, torch.device) else v for k, v in args_dict.items()}
            with open(os.path.join(self.path_logs, filename), 'w') as f: json.dump(clean, f, indent=4)
        except: pass

    def load_checkpoint(self, path):
        if path and os.path.isfile(path):
            print(f"[INFO] Loading: {path}")
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt: self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            return ckpt.get('epoch', 0)
        return 0

    def _unpack(self, batch):
        if len(batch) == 3: return batch[0], batch[1], batch[2]
        if len(batch) == 4: return batch[1], batch[2], batch[3]
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
                loss.backward()
                self.optimizer.step()
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
            self.plot_signals(xt, xp, ep)

        self.plot_losses(tr_loss, val_loss)
        best_path = os.path.join(self.path_checkpoints, 'best_model.pt')
        if not os.path.exists(best_path) and current_ckpt_path: best_path = current_ckpt_path
        if os.path.exists(best_path):
            self.plot_learned_params_evolution(best_path)
            self.plot_best_signals(best_path)
        return tr_loss, val_loss

    def test(self, need_names=False, checkpoint_path=None):
        self.load_checkpoint(checkpoint_path)
        self.create_loaders(need_names)
        dx = torch.zeros(1, self.N_dim).double().to(self.device)
        dy = torch.zeros(1, self.M_dim).double().to(self.device)
        static = self.p3mg_tmp.init_P3MG(list(self.static_params), dx, dy)
        self.model.eval()
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                xt, y, x0 = self._unpack(batch)
                xt, y = xt.to(self.device).double(), y.to(self.device).double()
                if x0 is not None: x0=x0.to(self.device).double()
                else: x0 = y.sum(1, keepdim=True).repeat(1, self.N_dim)/(self.M_dim*self.N_dim)
                xp, _, _ = self.model(static, None, x0, y)
                losses.append(self.criterion(xp, xt).item())
                if i==0: self.plot_signals(xt, xp, 'test')
        avg = sum(losses)/len(losses) if losses else 0
        print(f"[TEST] Loss: {avg:.4e}")
        return avg

    def run_test_2d(self, l_val, t_val, loader, ckpt, plot_path, save_signal=False):
        self.load_checkpoint(ckpt)
        self.model.eval()
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
                if save_signal and i == 0:
                    self.plot_signals_gs_2d(xt, xp, l_val, t_val, plot_path)
        return sum(losses)/len(losses) if losses else 0

    def grid_search_2d(self, lambdas, taus, need_names=False, checkpoint_path=None):
        self.create_loaders(need_names)
        out_dir = self.path_plots
        is_distributed = (len(lambdas) == 1 and len(taus) == 1)
        
        for l in lambdas:
            for t in taus:
                loss = self.run_test_2d(l, t, self.test_loader, checkpoint_path, out_dir, save_signal=is_distributed)
                print(f"L={l:.2e} T={t:.2f} : {loss:.4e}")
                
                task_id = os.environ.get('SLURM_ARRAY_TASK_ID', 'local')
                safe_l = f"{l:.2e}".replace('+','')
                safe_t = f"{t:.2f}"
                fname = f'gs_result_L{safe_l}_T{safe_t}_tid{task_id}.json'
                
                with open(os.path.join(self.path_logs, fname), 'w') as f:
                    json.dump({"lambda": l, "tau": t, "loss": loss}, f, indent=4)

        self.consolidate_and_plot_2d_gs()

    def consolidate_and_plot_2d_gs(self):
        print(f"[INFO] Consolidation 2D avec arrondi visuel...")
        data = []
        jsons = glob.glob(os.path.join(self.path_logs, 'gs_result_*.json'))
        if not jsons: return

        for j in jsons:
            try:
                with open(j, 'r') as f: data.append(json.load(f))
            except: pass
        if not data: return

        # Extraction brute
        raw_l = np.array([d['lambda'] for d in data])
        raw_t = np.array([d['tau'] for d in data])
        raw_loss = np.array([d['loss'] for d in data])

        # === CORRECTION "Heatmap Blanche" : Arrondi pour l'affichage ===
        # On arrondit pour que les valeurs proches tombent dans la même case
        # Lambda (Log scale): on arrondit le log10
        l_log = np.log10(raw_l)
        l_rounded = np.round(l_log, decimals=4)
        
        # Tau (Linear scale): on arrondit la valeur
        t_rounded = np.round(raw_t, decimals=4)
        
        # Création des axes uniques basés sur l'arrondi
        unique_l = np.sort(np.unique(l_rounded))
        unique_t = np.sort(np.unique(t_rounded))
        
        # Si trop peu de points uniques, on ne peut pas faire de meshgrid
        if len(unique_l) < 2 or len(unique_t) < 2:
            return

        # Remplissage de la grille
        grid = np.zeros((len(unique_t), len(unique_l))) + np.nan
        best_loss, best_pair = float('inf'), (None, None)

        for i in range(len(data)):
            loss = raw_loss[i]
            if loss < best_loss:
                best_loss = loss
                best_pair = (raw_l[i], raw_t[i])
            
            # Mapping vers la grille arrondie
            idx_l = np.searchsorted(unique_l, l_rounded[i])
            idx_t = np.searchsorted(unique_t, t_rounded[i])
            
            # Protection d'index (normalement inutile avec unique)
            if idx_l < len(unique_l) and idx_t < len(unique_t):
                grid[idx_t, idx_l] = loss

        # Plot Heatmap
        plt.figure(figsize=(10, 8))
        # Reconversion en valeurs réelles pour les axes (10^L)
        X, Y = np.meshgrid(10**unique_l, unique_t)
        
        # Log10(Loss) pour le contraste
        # On remplit les trous (NaN) avec le max pour ne pas faire de trous blancs
        valid_max = np.nanmax(grid)
        grid_filled = np.where(np.isnan(grid), valid_max, grid)
        
        plt.pcolormesh(X, Y, np.log10(grid_filled), shading='auto', cmap='viridis')
        plt.colorbar(label='Log10(Loss)')
        
        plt.xscale('log')
        plt.xlabel(r'$\lambda$ (Log)')
        plt.ylabel(r'$\tau$ (Linear)')
        plt.title(f'GS 2D (Best: L={best_pair[0]:.2e}, T={best_pair[1]:.2f})')
        plt.plot(best_pair[0], best_pair[1], 'r*', markersize=15, markeredgecolor='white')
        
        plt.savefig(os.path.join(self.path_save, 'GLOBAL_GS_HEATMAP.png'))
        plt.close()
        print("[SUCCESS] Heatmap générée.")
        
        if best_pair[0] is not None:
            pat_l = f"{best_pair[0]:.2e}".replace('+','')
            pat_t = f"{best_pair[1]:.2f}"
            # On cherche avec un pattern large car l'arrondi affichage != float exact fichier
            # On cherche le fichier qui correspond à ce best pair
            possible_files = glob.glob(os.path.join(self.path_plots, f"signal_L{pat_l}*T{pat_t}*.png"))
            if possible_files:
                shutil.copy(possible_files[0], os.path.join(self.path_save, 'GLOBAL_BEST_SIGNAL.png'))
                print("[SUCCESS] Signal optimal copié.")

    # --- PLOT UTILS ---
    def plot_losses(self, tr, val):
        plt.figure(); plt.plot(tr, label='T'); plt.plot(val, label='V'); plt.legend()
        plt.savefig(os.path.join(self.path_plots, 'loss.png')); plt.close()

    def plot_signals(self, true, pred, epoch):
        for i in range(min(3, true.size(0))):
            plt.figure(figsize=(10,3)); plt.plot(true[i].cpu().numpy()); plt.plot(pred[i].detach().cpu().numpy(), '--')
            plt.title(f'Ep {epoch} S {i}'); plt.savefig(os.path.join(self.path_plots, f'sig_ep{epoch}_s{i}.png')); plt.close()

    def plot_best_signals(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        if self.val_loader is None: self.create_loaders(False)
        for batch in self.val_loader:
            xt, y, x0 = self._unpack(batch)
            xt, y = xt.to(self.device).double(), y.to(self.device).double()
            if x0 is None: x0 = y.sum(1,keepdim=True).repeat(1,self.N_dim)/(self.M_dim*self.N_dim)
            dx = torch.zeros(1,self.N_dim).double().to(self.device); dy=torch.zeros(1,self.M_dim).double().to(self.device)
            st = self.p3mg_tmp.init_P3MG(list(self.static_params), dx, dy)
            with torch.no_grad(): xp, _, _ = self.model(st, None, x0, y)
            plt.figure(figsize=(10,3)); plt.plot(xt[0].cpu().numpy()); plt.plot(xp[0].detach().cpu().numpy(), '--')
            plt.title('Best Model'); plt.savefig(os.path.join(self.path_plots, 'best_sig.png')); plt.close()
            break

    def plot_signals_gs_2d(self, true, pred, l_val, t_val, path):
        pat_l = f"{l_val:.2e}".replace('+','')
        pat_t = f"{t_val:.2f}"
        fname = f"signal_L{pat_l}_T{pat_t}.png"
        plt.figure(figsize=(10,3)); plt.plot(true[0].cpu().numpy()); plt.plot(pred[0].detach().cpu().numpy(), '--')
        plt.title(f'L={l_val:.2e} T={t_val:.2f}'); plt.savefig(os.path.join(path, fname)); plt.close()

    def plot_learned_params_evolution(self, path):
        if not os.path.isfile(path): return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        out_dir = os.path.join(os.path.dirname(os.path.dirname(path)), 'plots')
        if not os.path.isdir(out_dir): out_dir = os.path.dirname(path)
        S = nn.Softplus()
        N, M = self.model.num_layers, self.model.num_pd_layers
        lmbd_vals, tau_rows = [], []
        dummy_y = torch.ones(1, self.M_dim).double().to(self.device)
        dummy_x = torch.zeros(1, self.N_dim).double().to(self.device)
        dummy_static = self.p3mg_tmp.init_P3MG(list(self.static_params), dummy_x, dummy_y) 
        with torch.no_grad(): _, _, dl = self.model(dummy_static, None, dummy_x, dummy_y)
        for k, layer in enumerate(self.model.Layers):
            if k < len(dl): lmbd_vals.append(dl[k].squeeze().item())
            if hasattr(layer, 'tau_k'): tau_rows.append(S(layer.tau_k).detach().cpu().numpy().flatten())
        if lmbd_vals:
            plt.figure(figsize=(10,6)); plt.plot(range(1, N+1), lmbd_vals, 'o-'); plt.savefig(os.path.join(out_dir, 'learnt_lambda_curve.png')); plt.close()
        if tau_rows:
            tau_mat = np.array(tau_rows)
            plt.figure(figsize=(12, 6)); plt.imshow(tau_mat, aspect='auto', cmap='viridis', origin='lower', extent=[0.5, M+0.5, 0.5, N+0.5])
            plt.colorbar(); plt.savefig(os.path.join(out_dir, 'learnt_tau_heatmap.png')); plt.close()