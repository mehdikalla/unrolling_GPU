import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from modules.module import MyDataset
from modules.P3MG_model import P3MG_model
from modules.P3MG_func import P3MGNet
import torch.nn.functional as F


class U_P3MG(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_pd_layers: int,
        static_params: tuple,
        initial_x0: torch.Tensor,
        train_params: tuple,
        paths: tuple,
        device: str = "cuda"
    ):
        super().__init__()

        # 1) Normaliser le device en premier
        if isinstance(device, str):
            if device.startswith("cuda") and torch.cuda.is_available():
                self.device = torch.device(device)
            else:
                self.device = torch.device("cpu")
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device("cpu")


        # 2) Paramètres du modèle
        self.num_layers    = num_layers
        self.num_pd_layers = num_pd_layers
        self.static_params = static_params
        if initial_x0 is not None and not isinstance(initial_x0, torch.Tensor):
            initial_x0 = torch.from_numpy(initial_x0)
        self.initial_x0    = initial_x0.double() if initial_x0 is not None else None
        self.p3mg_tmp = P3MGNet().to(self.device).double()
        

        # 3) Hyper-paramètres d'entraînement
        (self.num_epochs,
         self.lr,
         self.train_bs,
         self.val_bs,
         self.test_bs) = train_params

        # 4) Chemins Dataset et sauvegarde
        (self.path_train,
         self.path_val,
         self.path_test,
         self.path_save) = paths
        os.makedirs(self.path_save, exist_ok=True)

        # 5) Device, modèle et loss
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model     = P3MG_model(self.num_layers, self.num_pd_layers).to(self.device).double()
        self.criterion = nn.MSELoss(reduction="mean")

        # 6) Placeholders
        self.train_loader = None
        self.val_loader   = None
        self.test_loader  = None
        self.optimizer    = None
        self.scheduler    = None


    def create_loaders(self, need_names: bool = False):
        """
        Crée les DataLoaders et extrait dynamiquement N et M.
        """
        # Train
        train_ds = MyDataset(
            self.path_train,
            self.initial_x0.numpy() if self.initial_x0 is not None else None,
            return_name=need_names
        )
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        # Dimensions
        self.N = train_ds.X_true.shape[1]
        self.M = train_ds.Y.shape[1]

        # Val
        val_ds = MyDataset(
            self.path_val,
            self.initial_x0.numpy() if self.initial_x0 is not None else None,
            return_name=need_names
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.val_bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Test
        test_ds = MyDataset(
            self.path_test,
            self.initial_x0.numpy() if self.initial_x0 is not None else None,
            return_name=need_names
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.test_bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def plot_losses(self, train_losses, val_losses):
        plt.figure()
        epochs = list(range(1, len(train_losses) + 1))
        plt.plot(epochs, train_losses, label='Train')
        plt.plot(epochs, val_losses,   label='Val')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend(); plt.title('Loss Curve')
        plt.savefig(os.path.join(self.path_save, 'loss_curve.png'))
        plt.close()

    def plot_signals(self, X_true, X_pred, epoch):
        num = min(3, X_true.size(0))
        for i in range(num):
            plt.figure()
            plt.plot(X_true[i].cpu().numpy(),  '-', label='True')
            plt.plot(X_pred[i].detach().cpu().numpy(), '--', label='Pred')
            plt.title(f'Epoch {epoch+1} – Sample {i}')
            plt.legend()
            plt.savefig(os.path.join(self.path_save, f'signal_epoch{epoch+1}_sample{i}.png'))
            plt.close()

    def _unpack_batch(self, batch, need_names: bool):
        if need_names:
            if len(batch) == 4:
                _, X_true, Y, X0 = batch
            else:
                _, X_true, Y = batch; X0 = None
        else:
            if len(batch) == 3:
                X_true, Y, X0 = batch
            else:
                X_true, Y = batch; X0 = None
        return X_true, Y, X0

    def train(self, need_names: bool = False, checkpoint_path: str = None):
        # Loaders + dimensions N/M
        self.create_loaders(need_names)
        # Optimizer
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Optional checkpoint load
        start_epoch = 0
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epoch', -1) + 1

        train_losses, val_losses = [], []

        for epoch in range(start_epoch, self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            # ——— Phase Train ———
            self.model.train()
            # torch.autograd.set_detect_anomaly(True)  # <-- activer uniquement en debug, très lent

            batch_losses = []
            for batch_idx, batch in enumerate(self.train_loader, 1):
                X_true, Y, X0 = self._unpack_batch(batch, need_names)

                # to(device) + double + transferts asynchrones
                X_true = X_true.to(self.device, non_blocking=True).double()
                Y      = Y.to(self.device,      non_blocking=True).double()
                X0     = X0.to(self.device,     non_blocking=True).double() if X0 is not None else None

                # Init X0 si nécessaire
                if X0 is None:
                    sumY = Y.sum(dim=1, keepdim=True)
                    X0   = sumY.repeat(1, self.N) / (self.M * self.N)

                # Forward statique
                static = self.p3mg_tmp.init_P3MG(list(self.static_params), X0, Y)

                # Forward réseau
                X_pred, _ = self.model(static, None, X0, Y)

                # Sanity
                if X_pred.shape != X_true.shape:
                    raise ValueError(f"Shape mismatch: {X_pred.shape} vs {X_true.shape}")

                loss = self.criterion(X_pred, X_true)
                batch_losses.append(loss.item())

                #self.optimizer.zero_grad(set_to_none=True)
                #loss.backward()
                #self.optimizer.step()

                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(self.train_loader)} – Loss: {loss.item():.4e}")

            epoch_loss = sum(batch_losses) / len(batch_losses)
            train_losses.append(epoch_loss)

            # ——— Phase Validation ———
            self.model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for batch in self.val_loader:
                    X_true, Y, X0 = self._unpack_batch(batch, need_names)

                    X_true = X_true.to(self.device, non_blocking=True).double()
                    Y      = Y.to(self.device,      non_blocking=True).double()
                    X0     = X0.to(self.device,     non_blocking=True).double() if X0 is not None else None

                    if X0 is None:
                        sumY = Y.sum(dim=1, keepdim=True)
                        X0   = sumY.repeat(1, self.N) / (self.M * self.N)

                    static = self.p3mg_tmp.init_P3MG(list(self.static_params), X0, Y)
                    X_pred, _ = self.model(static, None, X0, Y)
                    val_batch_losses.append(self.criterion(X_pred, X_true).item())

            val_loss = sum(val_batch_losses) / len(val_batch_losses)
            val_losses.append(val_loss)
            print(f" Train Loss: {epoch_loss:.4e} | Val Loss: {val_loss:.4e}")

            # Checkpoint save
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
            }, os.path.join(self.path_save, f'checkpoint_epoch{epoch}.pt'))
            # Quelques signaux
            self.plot_signals(X_true, X_pred, epoch)

        # Courbes
        self.plot_losses(train_losses, val_losses)
        return train_losses, val_losses

    def test(self, need_names: bool = False, checkpoint_path: str = None):
        # Load checkpoint
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])

        if self.test_loader is None:
            self.create_loaders(need_names)

        self.model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in self.test_loader:
                X_true, Y, X0 = self._unpack_batch(batch, need_names)

                X_true = X_true.to(self.device, non_blocking=True).double()
                Y      = Y.to(self.device,      non_blocking=True).double()
                X0     = X0.to(self.device,     non_blocking=True).double() if X0 is not None else None

                if X0 is None:
                    sumY = Y.sum(dim=1, keepdim=True)
                    X0   = sumY.repeat(1, self.N) / (self.M * self.N)

                static   = static = self.p3mg_tmp.init_P3MG(list(self.static_params), X0, Y)
                X_pred, _ = self.model(static, None, X0, Y)
                test_losses.append(self.criterion(X_pred, X_true).item())

        test_loss = sum(test_losses) / len(test_losses)
        print(f"Test Loss: {test_loss:.4e}")
        return test_loss

    def plot_stepsizes(self, path_model: str):
        """
        Sauvegarde l'évolution des pas ν_k (Softplus des paramètres `nu`).
        """
        checkpoint = torch.load(path_model, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Récupère directement les paramètres des couches
        list_nu = []
        for i, layer in enumerate(self.model.Layers):
            if hasattr(layer, "nu"):
                val = F.softplus(layer.nu).detach().cpu().numpy()
                list_nu.append(val)

        plt.figure()
        plt.plot(list_nu, marker='o', linestyle='solid', linewidth=0.5)
        plt.ylabel(r'$\nu_k$'); plt.xlabel(r'layer $k$')
        out = os.path.join(os.path.dirname(path_model), "learnt_nu_k.png")
        plt.savefig(out); plt.close()
