# modules/P3MG_model.py
import torch
import torch.nn as nn
from modules.P3MG_func import P3MGNet  # ta classe OO avec iter_P3MG_base et iter_P3MG

R = nn.Softplus()

# --------------------
# P3MG model layers
# --------------------
class layer_0(nn.Module):
    """ 
    First layer of the P3MG model.
    Performs the first iteration of P3MG without dynamic variables.
    """
    def __init__(self, N_pd):
        super().__init__()
        self.lmbd = nn.Parameter(torch.DoubleTensor([8e-5]), requires_grad=True)
        self.p3mg = P3MGNet(num_pd_layers=N_pd)

    def forward(self, static, dynamic, x, y):
        lmbd = R(self.lmbd)
        # tau n'est pas utilisé ici, contrairement à layer_k
        x_new, dynamic_new = self.p3mg.iter_P3MG_base(static, x, y, lmbd)
        return x_new, dynamic_new


class layer_k(nn.Module):
    """
    Generic layer for k > 0 in the P3MG model.
    Performs the P3MG iterations using dynamic variables.
    """
    def __init__(self, N_pd):
        super().__init__()
        self.lmbd = nn.Parameter(torch.DoubleTensor([8e-5]), requires_grad=True)
        self.PD_tau = nn.Parameter(torch.DoubleTensor([0.5]), requires_grad=True)
        self.p3mg = P3MGNet(num_pd_layers=N_pd)

    def forward(self, static, dynamic, x, y):
        lmbd = R(self.lmbd)
        x_new, dynamic_new = self.p3mg.iter_P3MG(static, dynamic, x, y, lmbd)
        return x_new, dynamic_new


# --------------------
# P3MG model container
# --------------------
class P3MG_model(nn.Module):
    def __init__(self, num_layers, num_pd_layers):
        """
        num_layers : nombre total de couches (itérations) du modèle P3MG.
        num_pd_layers : nombre d'itérations (couches) du modèle primal-dual utilisé à l'intérieur de P3MG.
        """
        super().__init__()
        self.Layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.Layers.append(layer_0(num_pd_layers))
            else:
                self.Layers.append(layer_k(num_pd_layers))

    def forward(self, static, dynamic, x0, y, x_true=None):
        x, dyn = x0, dynamic
        for l in self.Layers:
            x, dyn = l(static, dyn, x, y)
        return x, dyn
# --------------------

