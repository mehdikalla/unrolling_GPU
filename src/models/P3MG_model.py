# modules/P3MG_model.py
import torch as tc
import torch.nn as nn
from models.P3MG_func import P3MGNet 
from models.FC_block import FC_block
S = nn.Softplus()

# --------------------
# P3MG model layers
# --------------------
class layer_0(nn.Module):
    """ 
    First layer of the P3MG model.
    Performs the first iteration of P3MG without dynamic variables.
    """
    def __init__(self, p3mg):
        super().__init__()
        self.p3mg = p3mg
        self.f_act = FC_block([100, 50, 25, 12, 1])

    def forward(self, static, dynamic, x, y, lmbd_override=None):
        device = 'cuda' if tc.cuda.is_available() else 'cpu'
        y = y.to(device)
        
        if lmbd_override is not None:
            lmbd = lmbd_override.to(x.device).double()
        else:
            lmbd = S(self.f_act(tc.pow(y,2)))
            
        x_new, dynamic_new = self.p3mg.iter_P3MG_base(static, x, y, lmbd)
        return x_new, dynamic_new


class layer_k(nn.Module):
    """
    Generic layer for k > 0 in the P3MG model.
    Performs the P3MG iterations using dynamic variables.
    """
    def __init__(self, p3mg):
        super().__init__()
        self.p3mg = p3mg
        self.f_act = FC_block([100, 50, 25, 12, 1])

    def forward(self, static, dynamic, x, y, lmbd_override=None):
        device = 'cuda' if tc.cuda.is_available() else 'cpu'
        y = y.to(device)
        
        if lmbd_override is not None:
            lmbd = lmbd_override.to(x.device).double()
        else:
            lmbd = S(self.f_act(tc.pow(y,2)))
            
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
        self.p3mg = P3MGNet(num_pd_layers)
        for i in range(num_layers):
            if i == 0:
                self.Layers.append(layer_0(self.p3mg))
            else:
                self.Layers.append(layer_k(self.p3mg))

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None):
        x, dyn = x0, dynamic
        for l in self.Layers:
            x, dyn = l(static, dyn, x, y, lmbd_override=lmbd_override) 
        return x, dyn
# --------------------