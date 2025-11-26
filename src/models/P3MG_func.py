import torch as tc
import torch.nn as nn
from src.utils import*
from src.models.PD_func import PrimalDualNet
from src.models.PD_model import PD_model # NOTE: Doit être utilisé comme un callable

class P3MGNet(nn.Module):
    def __init__(self, num_pd_layers: int = 3):
        super().__init__()
        self.pd_net = PrimalDualNet()
        self.num_pd_layers = num_pd_layers
        
    # -------------------------
    # init_P3MG 
    # -------------------------
    def init_P3MG(self, static_input, x0, y):
        # ... (Logique inchangée pour initialiser les paramètres statiques Hmat, Vprec, etc.) ...
        P, N, M = x0.size(0), x0.size(1), y.size(1)
        alpha, beta, eta = static_input

        # génération de Hmat sur CPU
        T, Hmat = dosy_mat(
            int(N), int(M), 0, 1.5, 1, 1000, dtype=x0.dtype
        )

        # Constantes
        Cg2    = 9 / (N * 8 * eta**2)
        Hnorm2 = tc.norm(Hmat, 2) ** 2
        gamma  = 1.9

        # SVD préconditionnement
        rankprec = 10
        Uprec, Sprec, VprecT = tc.linalg.svd(Hmat, full_matrices=False)
        Vprec    = VprecT.t()[:, :rankprec]
        Sprec    = Sprec[:rankprec]
        Sprec_2  = 1.0 / (Sprec ** 2)

        return Hmat, alpha, beta, eta, gamma, Cg2, Vprec, Sprec_2, Hnorm2

    # --------------------------------
    # iter_P3MG_base (Correction pour Tau)
    # --------------------------------
    def iter_P3MG_base(self, static, x, y, lmbd, tau_params): # <-- AJOUT DE tau_params
        """
        Première itération P3MG.
        """
        # NOTE: PD_model est instancié ici, mais les paramètres tau sont passés de l'extérieur.
        pd_model_instance = PD_model(self.num_pd_layers) 
        
        P, N, M = x.size(0), x.size(1), y.size(1)
        Hmat, alpha, beta, eta, gamma, Cg2, Vprec, Sprec_2, Hnorm2 = static

        LipsMaj = lmbd * (1 / (alpha * beta) + Cg2) + Hnorm2

        device = "cuda" if tc.cuda.is_available() else "cpu"
        dtype  = x.dtype

        # Statics -> device
        Hmat    = Hmat.to(device=device, dtype=dtype)
        Vprec   = Vprec.to(device=device, dtype=dtype)
        Sprec_2 = Sprec_2.to(device=device, dtype=dtype)

        # Gradient
        gradx, l1 = gradient_x(x, y, Hmat, alpha, beta, eta, lmbd)

        # Pas de descente + projection
        forward = x - (gamma / LipsMaj) * gradx
        xb      = proj_simplex(forward)
        Pgradx  = xb - x

        gradxb, l1b = gradient_x(xb, y, Hmat, alpha, beta, eta, lmbd)

        # Sous-espace (itération 1)
        Dx_list = []
        Ad_list = []
        subspace_iter1 = [1, 0]
        if subspace_iter1[0]:
            Dx_list.append(Pgradx)
        if subspace_iter1[1]:
            Adiag0     = majorante_x_diag(x, alpha, l1, beta, Cg2, lmbd, 0)
            Delta_inv  = 1.0 / Adiag0
            Vprec_T    = Vprec.t().unsqueeze(0).expand(P, -1, -1)
            weighted_V = Vprec_T * Delta_inv.unsqueeze(1)
            temp_batch = tc.bmm(weighted_V, Vprec.unsqueeze(0).expand(P, -1, -1))
            diag_S     = tc.diag_embed(Sprec_2).unsqueeze(0).expand(P, -1, -1)
            temp       = diag_S + temp_batch
            Prec_gradx = Delta_inv * gradx
            temp_vec   = tc.matmul(Prec_gradx, Vprec)
            sol        = tc.linalg.solve(temp, temp_vec.unsqueeze(-1)).squeeze(-1)
            correction = tc.bmm(Vprec.unsqueeze(0).expand(P, -1, -1), sol.unsqueeze(-1)).squeeze(-1)
            Prec_gradx = Prec_gradx - Delta_inv * correction
            temp_x     = x - Prec_gradx
            proj_temp  = proj_simplex(temp_x)
            PgradxP    = proj_temp - x
            Dx_list.append(PgradxP)

        for d in Dx_list:
            Ad_temp = Majorante_x(d, xb, alpha, l1b, beta, Cg2, Hmat, lmbd)
            Ad_list.append(Ad_temp)
        
        Dx = tc.stack(Dx_list, dim=1)  # (P, L, N)
        
        if  len(Ad_list) > 0:
            Ad = tc.stack(Ad_list, dim=1)  # (P, L, N)
        else :
            Ad = tc.zeros((P, 0, N), dtype=dtype, device=device)

        Bx = tc.bmm(Dx, Ad.transpose(1, 2))  # (P, L, L)

        # --------- Appel du MODÈLE PD ---------
        sub_static_input = [xb, Dx, Bx, gradxb, P, N]

        # 1) init via PDInit_layer
        w0, sub_static = self.pd_net.init_PD(sub_static_input)

        # 2) unrolling PD via PD_model (PASSAGE DES PARAMÈTRES TAU)
        # NOTE: PD_model doit accepter tau_params et les utiliser.
        w_final = pd_model_instance(sub_static, w0, tau_params) 
        un, vn = w_final

        # 3) reconstruire dx et x
        dx_new = tc.bmm(un.unsqueeze(1), Dx).squeeze(1)
        x_new  = xb + dx_new

        dynamic = [dx_new, Pgradx]
        return x_new, dynamic

    # -----------------------------
    # iter_P3MG 
    # -----------------------------
    def iter_P3MG(self, static, dynamic, x, y, lmbd, tau_params): # <-- AJOUT DE tau_params
        """
        Itérations P3MG génériques (k>1).
        """
        pd_model_instance = PD_model(self.num_pd_layers) 
        
        P, N, M = x.size(0), x.size(1), y.size(1)

        Hmat, alpha, beta, eta, gamma, Cg2, Vprec, Sprec_2, Hnorm2 = static
        LipsMaj = lmbd * (1 / (alpha * beta) + Cg2) + Hnorm2
        dx_old, Pgradx_old = dynamic

        device = "cuda" if tc.cuda.is_available() else "cpu"
        dtype  = x.dtype

        # Statics -> device
        Hmat    = Hmat.to(device=device, dtype=dtype)
        Vprec   = Vprec.to(device=device, dtype=dtype)
        Sprec_2 = Sprec_2.to(device=device, dtype=dtype)

        # Gradient et projection
        gradx, l1 = gradient_x(x, y, Hmat, alpha, beta, eta, lmbd)
        fwd  = x - (gamma / LipsMaj) * gradx
        xb   = proj_simplex(fwd)
        Pgradx = xb - x
        gradxb, l1b = gradient_x(xb, y, Hmat, alpha, beta, eta, lmbd)

        # Sous-espace (général)
        Dx_list = []
        Ad_list = []
        subspace_gen = [1, 1, 1, 1]
        if subspace_gen[0]:
            Dx_list.append(Pgradx)
        if subspace_gen[1]:
            Adiag0     = majorante_x_diag(x, alpha, l1, beta, Cg2, lmbd, 0)
            Delta_inv  = 1.0 / Adiag0
            Vprec_T    = Vprec.t().unsqueeze(0).expand(P, -1, -1)
            weighted_V = Vprec_T * Delta_inv.unsqueeze(1)
            temp_batch = tc.bmm(weighted_V, Vprec.unsqueeze(0).expand(P, -1, -1))
            diag_S     = tc.diag_embed(Sprec_2).unsqueeze(0).expand(P, -1, -1)
            temp       = diag_S + temp_batch
            Prec_gradx = Delta_inv * gradx
            temp_vec   = tc.matmul(Prec_gradx, Vprec)
            sol        = tc.linalg.solve(temp, temp_vec.unsqueeze(-1)).squeeze(-1)
            correction = tc.bmm(Vprec.unsqueeze(0).expand(P, -1, -1), sol.unsqueeze(-1)).squeeze(-1)
            Prec_gradx = Prec_gradx - Delta_inv * correction
            temp_x     = x - Prec_gradx
            proj_temp  = proj_simplex(temp_x)
            PgradxP    = proj_temp - x
            Dx_list.append(PgradxP)
        if subspace_gen[2]:
            Dx_list.append(dx_old)
        if subspace_gen[3]:
            Dx_list.append(Pgradx_old)

        if len(Dx_list) == 0:
            Dx_list.append(Pgradx)

        for d in Dx_list:
            Ad_temp = Majorante_x(d, xb, alpha, l1b, beta, Cg2, Hmat, lmbd)
            Ad_list.append(Ad_temp)
        
        Dx = tc.stack(Dx_list, dim=1)
        
        if  len(Ad_list) > 0:
            Ad = tc.stack(Ad_list, dim=1)
        else :
            Ad = tc.zeros((P, 0, N), dtype=dtype, device=device)

        Bx = tc.bmm(Dx, Ad.transpose(1, 2))
    
        # --------- Appel du MODÈLE PD ---------
        sub_static_input = [xb, Dx, Bx, gradxb, P, N]

        # 1) init via PDInit_layer
        w0, sub_static = self.pd_net.init_PD(sub_static_input)

        # 2) unrolling PD via PD_model (PASSAGE DES PARAMÈTRES TAU)
        w_final = pd_model_instance(sub_static, w0, tau_params) # <-- CORRECTION ICI
        un, vn = w_final

        # 3) reconstruire dx et x
        dx_new = tc.bmm(un.unsqueeze(1), Dx).squeeze(1)
        x_new  = xb + dx_new

        dynamic_new = [dx_new, Pgradx]
        return x_new, dynamic_new