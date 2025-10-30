# modules/P3MG_model.py
import torch
import torch.nn as nn
from modules.PD_func import PrimalDualNet  

class BoundedSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # f(x) = 0.5 + sigmoid(x) ∈ (0.5, 1.5)
        return 0.5 + self.sigmoid(x)

R=nn.Softplus()
SB=BoundedSigmoid()
S = nn.Sigmoid()
# -------------------------
# Primal-Dual model layers
# -------------------------
class PD_layer(nn.Module):
    """
    Generic layer for k > 0 in the Primal-Dual model.
    Uses the class-based method pd.iterPD(...) instead of the free function.
    """
    def __init__(self):
        super().__init__()
        self.pd = PrimalDualNet()

    def forward(self, sub_static, u_new):
        u_new = self.pd.iter_PD(sub_static, u_new)
        return u_new


class PD_model(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.Layers = nn.ModuleList([PD_layer() for _ in range(num_layers)])

    def forward(self, sub_static, u0):
        u = u0
        for l in self.Layers:
            u = l(sub_static, u)
        return u
# -------------------
