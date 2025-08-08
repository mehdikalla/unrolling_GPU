# modules/P3MG_model.py
import torch
import torch.nn as nn
from modules.P3MG_func import iter_P3MG_base, iter_P3MG

R = nn.Softplus()

# P3MG model layers
class layer_0(nn.Module):
    """ 
    First layer of the P3MG model.
    This layer initializes the parameters and performs the first iteration of P3MG.
    It does not use dynamic variables.
    """
    def __init__(self):
        super().__init__()
        self.lmbd = nn.Parameter(torch.DoubleTensor([8e-5]), requires_grad=True)
        self.PD_tau = nn.Parameter(torch.DoubleTensor([0.5]), requires_grad=True)

    def forward(self, static, dynamic, x, y):
        lmbd  = R(self.lmbd)
        tau = R(self.PD_tau)
        x_new, dynamic_new = iter_P3MG_base(static, x, y, lmbd, tau)
        return x_new, dynamic_new

class layer_k(nn.Module):
    """
    Generic layer for k > 0 in the P3MG model.
    This layer performs the P3MG iterations using dynamic variables.
    It uses the static parameters and the previous dynamic state.
    """
    def __init__(self):
        super().__init__()
        self.lmbd = nn.Parameter(torch.DoubleTensor([8e-5]), requires_grad=True)
        self.PD_tau = nn.Parameter(torch.DoubleTensor([0.5]), requires_grad=True)

    def forward(self, static, dynamic, x, y):
        lmbd  = R(self.lmbd)
        tau = R(self.PD_tau)
        x_new, dynamic_new = iter_P3MG(static, dynamic, x, y, lmbd, tau)
        return x_new, dynamic_new


# P3MG model
class P3MG_model(nn.Module):
    def __init__(self, lmbdm_layers):
        super().__init__()
        self.Layers = nn.ModuleList([layer_0()] + [layer_k() for _ in range(lmbdm_layers-1)])

    def forward(self, static, dynamic, x0, y, x_true=None):
        x, dyn = x0, dynamic
        for l in self.Layers:
            x, dyn = l(static, dyn, x, y)
        return x, dyn

# Loss function for P3MG model
class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_pred, x_true):
        diff = x_pred - x_true
        return torch.mean(diff.pow(2))
