# modules/P3MG_model.py
import torch
import torch.nn as nn
from modules.PD_func import PrimalDualNet  
S = nn.Softplus()

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
        self.tau = nn.Parameter(torch.DoubleTensor([0.5]), requires_grad=True)

    def forward(self, sub_static, w_new):
        tau = S(self.tau)
        w_new = self.pd.iter_PD(sub_static, w_new, tau)
        return w_new


class PD_model(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.Layers = nn.ModuleList([PD_layer() for _ in range(num_layers)])

    def forward(self, sub_static, w0):
        w = w0
        for l in self.Layers:
            w = l(sub_static, w)
        return w
# -------------------
