# modules/P3MG_func.py
import torch as tc
import torch.nn as nn
from modules.utils import*
from modules.PD_func import PrimalDualNet
from modules.PD_model import PD_model

class P3MGNet(nn.Module):
    def __init__(self, num_pd_layers: int = 3):
        """
        num_pd_layers : nombre d'itérations (couches) du modèle primal-dual utilisé à l'intérieur de P3MG.
        """
        super().__init__()
        self.pd_net = PrimalDualNet()
        self.pd_model = PD_model(num_pd_layers)
    # -------------------------
    # init_P3MG (inchangée)
    # -------------------------
    def init_P3MG(self, static_input, x0, y):
        """
        Étape 0 : Initialisation, chargement des données, définition des variables
        et calculs préliminaires.
        """
        P, N, M = x0.size(0), x0.size(1), y.size(1)
        alpha, beta, eta, sub, sup = static_input

        # génération de Hmat sur CPU
        T, Hmat = dosy_mat(
            int(N),   # nombre de D-values
            int(M),   # nombre de points temporels
            0,        # tmin
            1.5,      # tmax
            1,        # Dmin
            1000,     # Dmax
            dtype=x0.dtype
        )

        # Constantes
        Cg2    = 9 / (N * 8 * eta**2)
        Hnorm2 = tc.norm(Hmat, 2) ** 2
        gamma  = 1.9

        # SVD préconditionnement
        rankprec = 10
        Uprec, Sprec, VprecT = tc.linalg.svd(Hmat, full_matrices=False)
        Vprec    = VprecT.t()[:, :rankprec]          # (N, r)
        Sprec    = Sprec[:rankprec]                   # (r,)
        Sprec_2  = 1.0 / (Sprec ** 2)                 # (r,)

        return Hmat, alpha, beta, eta, sub, sup, gamma, Cg2, Vprec, Sprec_2, Hnorm2

    # --------------------------------
    # iter_P3MG_base (utilise PD_model)
    # --------------------------------
    def iter_P3MG_base(self, static, x, y, lmbd):
        """
        Première itération P3MG.
        NOTE: on appelle le *modèle* PD (PDInit_layer + PD_model) et non la classe PrimalDualNet.
        """
        P, N, M = x.size(0), x.size(1), y.size(1)
        Hmat, alpha, beta, eta, sub, sup, gamma, Cg2, Vprec, Sprec_2, Hnorm2 = static

        LipsMaj = lmbd * (1 / (alpha * beta) + Cg2) + Hnorm2

        device = x.device
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
        subspace_iter1 = [1, 0]
        if subspace_iter1[0]:
            Dx_list.append(Pgradx)
        if subspace_iter1[1]:
            # Majorant diagonal (batch)
            Adiag0     = majorante_x_diag(x, alpha, l1, beta, Cg2, lmbd, 0)   # (P, N)
            Delta_inv  = 1.0 / Adiag0                                       # (P, N)
            Vprec_T    = Vprec.t().unsqueeze(0).expand(P, -1, -1)           # (P, r, N)
            weighted_V = Vprec_T * Delta_inv.unsqueeze(1)                   # (P, r, N)
            temp_batch = tc.bmm(weighted_V, Vprec.unsqueeze(0).expand(P, -1, -1))  # (P, r, r)
            diag_S     = tc.diag_embed(Sprec_2).unsqueeze(0).expand(P, -1, -1)     # (P, r, r)
            temp       = diag_S + temp_batch                                 # (P, r, r)

            # Préconditionnement du gradient
            Prec_gradx = Delta_inv * gradx                                   # (P, N)
            temp_vec   = tc.matmul(Prec_gradx, Vprec)                         # (P, r)
            sol        = tc.linalg.solve(temp, temp_vec.unsqueeze(-1)).squeeze(-1) # (P, r)
            correction = tc.bmm(Vprec.unsqueeze(0).expand(P, -1, -1), sol.unsqueeze(-1)).squeeze(-1)  # (P,N)
            Prec_gradx = Prec_gradx - Delta_inv * correction

            temp_x     = x - Prec_gradx
            proj_temp  = proj_simplex(temp_x)
            PgradxP    = proj_temp - x
            Dx_list.append(PgradxP)

        if len(Dx_list) == 0:
            Dx_list.append(Pgradx)

        Dx = tc.stack(Dx_list, dim=1)  # (P, L, N)

        # Bx "placeholder" (comme dans ta version d’origine)
        Ad = tc.zeros((P, Dx.size(1), N), dtype=dtype, device=device)
        Bx = tc.bmm(Dx, Ad.transpose(1, 2))  # (P, L, L)

        # --------- Appel du MODÈLE PD ---------
        # Construire l'input attendu par init_PD: [xb, D, B, grad, sub, sup, P, N, tau]
        # NB: tau passé ici n'est pas utilisé par init_PD ; le PD_model apprendra ses propres τ via Softplus
        
        sub_static_input = [xb, Dx, Bx, gradxb, sub, sup, P, N]

        # 1) init via PDInit_layer
        u0, sub_static = self.pd_net.init_PD(sub_static_input)     # u0 = [un, vn]

        # 2) unrolling PD via PD_model (utilise ses τ internes)
        u_final = self.pd_model(sub_static, u0)             # [un, vn] après K couches
        un, vn = u_final

        # 3) reconstruire dx et x
        dx_new = tc.bmm(un.unsqueeze(1), Dx).squeeze(1)     # (P, N)
        x_new  = xb + dx_new

        dynamic = [dx_new, Pgradx]
        return x_new, dynamic

    # -----------------------------
    # iter_P3MG (utilise PD_model)
    # -----------------------------
    def iter_P3MG(self, static, dynamic, x, y, lmbd):
        """
        Itérations P3MG génériques (k>1).
        NOTE: on appelle le *modèle* PD (PDInit_layer + PD_model) et non la classe PrimalDualNet.
        """
        P, N, M = x.size(0), x.size(1), y.size(1)
        
        Hmat, alpha, beta, eta, sub, sup, gamma, Cg2, Vprec, Sprec_2, Hnorm2 = static
        LipsMaj = lmbd * (1 / (alpha * beta) + Cg2) + Hnorm2
        dx_old, Pgradx_old = dynamic

        device = x.device
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

        Dx = tc.stack(Dx_list, dim=1)  # (P, L, N)

        # Bx "placeholder" (comme dans la version d’origine)
        Ad = tc.zeros((P, Dx.size(1), N), dtype=dtype, device=device)
        Bx = tc.bmm(Dx, Ad.transpose(1, 2))  # (P, L, L)

        # --------- Appel du MODÈLE PD ---------
        sub_static_input = [xb, Dx, Bx, gradxb, sub, sup, P, N]

        # 1) init via PDInit_layer
        u0, sub_static = self.pd_net.init_PD(sub_static_input)     # [un, vn]

        # 2) unrolling PD via PD_model
        u_final = self.pd_model(sub_static, u0)             # [un, vn]
        un, vn = u_final

        # 3) reconstruire dx et x
        dx_new = tc.bmm(un.unsqueeze(1), Dx).squeeze(1)     # (P, N)
        x_new  = xb + dx_new

        dynamic_new = [dx_new, Pgradx]
        return x_new, dynamic_new
