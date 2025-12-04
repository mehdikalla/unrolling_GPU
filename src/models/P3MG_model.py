import torch as tc
import torch.nn as nn
from .P3MG_func import P3MGNet
from .FC_block import FC_block
S = nn.Softplus()

def S2(x):
    return 2*tc.sigmoid(x)

# --------------------
# P3MG model layers
# --------------------
class layer_0(nn.Module):
    def __init__(self, num_pd_layers: int):
        super().__init__()
        self.p3mg_func = P3MGNet(num_pd_layers)
        
        # 1. Lambda (Dynamique)
        self.f_act = FC_block([100, 50, 25, 12, 1])
        # 2. Tau (Explicite - Vecteur de M valeurs)
        self.tau_k = nn.Parameter(tc.empty(num_pd_layers).double().fill_(0.5), requires_grad=True) 

    def forward(self, static, dynamic, x, y, lmbd_override=None, tau_override=None):
        device = 'cuda' if tc.cuda.is_available() else 'cpu'
        x = x.to(device)
        y = y.to(device)
        
        # Gestion Lambda
        if lmbd_override is not None:
             lmbd = lmbd_override.to(x.device).double()
        else:
             lmbd = S(self.f_act(tc.pow(y,2)))
        
        # Gestion Tau (Override ou Appris)
        if tau_override is not None:
             # Si override (scalaire), on l'étend pour correspondre au nombre de sous-couches
             tau_val = tau_override.to(x.device).double()
             tau_params = tau_val.expand(self.tau_k.shape)
        else:
             tau_params = S2(self.tau_k) 

        x_new, dynamic_new = self.p3mg_func.iter_P3MG_base(static, x, y, lmbd, tau_params)
        return x_new, dynamic_new, lmbd


class layer_k(nn.Module):
    def __init__(self, num_pd_layers: int):
        super().__init__()
        self.p3mg_func = P3MGNet(num_pd_layers)
        self.f_act = FC_block([100, 50, 25, 12, 1])
        self.tau_k = nn.Parameter(tc.empty(num_pd_layers).double().fill_(0.5), requires_grad=True)

    def forward(self, static, dynamic, x, y, lmbd_override=None, tau_override=None):
        device = 'cuda' if tc.cuda.is_available() else 'cpu'
        x = x.to(device)
        y = y.to(device)
        
        if lmbd_override is not None:
             lmbd = lmbd_override.to(x.device).double()
        else:
             lmbd = S(self.f_act(tc.pow(y,2)))
             
        if tau_override is not None:
             tau_val = tau_override.to(x.device).double()
             tau_params = tau_val.expand(self.tau_k.shape)
        else:
             tau_params = S2(self.tau_k)
            
        x_new, dynamic_new = self.p3mg_func.iter_P3MG(static, dynamic, x, y, lmbd, tau_params)
        return x_new, dynamic_new, lmbd


# --------------------
# P3MG model container
# --------------------
class P3MG_model(nn.Module):
    def __init__(self, num_layers, num_pd_layers):
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.num_pd_layers = num_pd_layers
        
        for i in range(num_layers):
            if i == 0: self.Layers.append(layer_0(num_pd_layers))
            else: self.Layers.append(layer_k(num_pd_layers))

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None, tau_override=None):
        x, dyn = x0, dynamic
        dynamic_lambdas = []
        
        for l in self.Layers:
            # On passe les deux overrides
            x, dyn, lmbd_k = l(static, dyn, x, y, lmbd_override=lmbd_override, tau_override=tau_override) 
            dynamic_lambdas.append(lmbd_k)
        
        return x, dyn, dynamic_lambdas