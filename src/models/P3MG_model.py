import torch as tc
import torch.nn as nn
from .P3MG_func import P3MGNet
from .FC_block import FC_block
S = nn.Softplus()

# --------------------
# P3MG model layers
# --------------------
class layer_0(nn.Module):
    """ 
    Première couche : possède ses paramètres dynamiques lambda (via FCNet) et tau (M valeurs).
    """
    def __init__(self, num_pd_layers: int):
        super().__init__()
        self.p3mg_func = P3MGNet(num_pd_layers)
        self.f_act = FC_block([100, 50, 25, 12, 1]) 
        self.tau_k = nn.Parameter(tc.rand(num_pd_layers).double() * 1e-4, requires_grad=True) 

    def forward(self, static, dynamic, x, y, lmbd_override=None):
        device = 'cuda' if tc.cuda.is_available() else 'cpu'
        x = x.to(device)
        y = y.to(device)

        if lmbd_override is not None:
             lmbd = lmbd_override.to(x.device).double()
        else:
             lmbd = S(self.f_act(tc.pow(y,2)))
        
        tau_params = S(self.tau_k) 

        x_new, dynamic_new = self.p3mg_func.iter_P3MG_base(static, x, y, lmbd, tau_params)

        return x_new, dynamic_new, lmbd


class layer_k(nn.Module):
    """
    Couche générique (k > 0) : possède ses paramètres dynamiques lambda et tau.
    """
    def __init__(self, num_pd_layers: int):
        super().__init__()
        self.p3mg_func = P3MGNet(num_pd_layers)
        self.f_act =  FC_block([100, 50, 25, 12, 1])
        self.tau_k = nn.Parameter(tc.rand(num_pd_layers).double() * 1e-4, requires_grad=True)

    def forward(self, static, dynamic, x, y, lmbd_override=None):
        device = 'cuda' if tc.cuda.is_available() else 'cpu'
        x = x.to(device)
        y = y.to(device)
        
        if lmbd_override is not None:
             lmbd = lmbd_override.to(x.device).double()
        else:
             lmbd = S(self.f_act(tc.pow(y,2)))
             
        tau_params = S(self.tau_k)
            
        x_new, dynamic_new = self.p3mg_func.iter_P3MG(static, dynamic, x, y, lmbd, tau_params)
        
        return x_new, dynamic_new, lmbd


# --------------------
# P3MG model container
# --------------------
class P3MG_model(nn.Module):
    def __init__(self, num_layers, num_pd_layers):
        """
        num_layers : N couches (itérations P3MG)
        num_pd_layers : M sous-couches (itérations PD)
        """
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.num_pd_layers = num_pd_layers
        
        # Création des N couches, chacune ayant ses propres paramètres
        for i in range(num_layers):
            if i == 0:
                self.Layers.append(layer_0(num_pd_layers))
            else:
                self.Layers.append(layer_k(num_pd_layers))

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None):
        x, dyn = x0, dynamic
        lambdas_list= []
        
        for l in self.Layers:
            x, dyn, lmbd_k = l(static, dyn, x, y, lmbd_override=lmbd_override) 
            lambdas_list.append(lmbd_k)
        
        return x, dyn, lambdas_list