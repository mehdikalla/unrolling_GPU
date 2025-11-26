# p3mg_classes.py
import torch as tc
import torch.nn as nn
from src.utils import*


class PrimalDualNet(nn.Module):
    def __init__(self):
        super().__init__()

    def init_PD(self, sub_static_input):
        """
        Initialisation des variables et calculs préliminaires pour le primal-dual.
        """
        xb, D, B, grad, P, N = sub_static_input

        device = xb.device
        dtype  = xb.dtype

        D    = D.to(device)
        B    = B.to(device)
        grad = grad.to(device)

        L = D.size(1)

        B = 0.5 * (B + B.transpose(1, 2))  # (P, L, L)

        un = tc.zeros((P, L), dtype=dtype, device=device)
        vn = tc.zeros((P, N), dtype=dtype, device=device)

        norm_D = tc.norm(D.reshape(P, -1), p=2, dim=1).clamp_min(1e-12)  # (P,)
        delta0 = 1.0 / norm_D
        gamma0 = 1.0 / norm_D

        w_new = [un, vn]
        sub_static = [xb, grad, P, D, B, L, delta0, gamma0]

        return w_new, sub_static

    def iter_PD(self, sub_static, w_new, tau, q_d = 1, q_g=1):
        """
        Itération générique du modèle Primal-Dual
        """
        xb, grad, P, D, B, L, delta0, gamma0 = sub_static

        delta = q_d*delta0
        gamma = q_g*gamma0

        device = xb.device
        dtype  = xb.dtype

        eye = tc.eye(L, dtype=dtype, device=device).unsqueeze(0).expand(P, L, L)
        eyemu = eye + delta.view(P, 1, 1) * B
        inv_eyemu = tc.linalg.inv(eyemu)
        un, vn = w_new

        # u = un - μ * (vn @ D^T)
        u = un - delta.view(P, 1) * tc.bmm(vn.unsqueeze(1), D.transpose(1, 2)).squeeze(1)  # (P,L)

        # prox step
        temp = tc.bmm(grad.unsqueeze(1), D.transpose(1, 2)).squeeze(1)  # (P,L)
        pn   = u - delta.view(P, 1) * temp                               # (P,L)
        pn   = tc.bmm(pn.unsqueeze(1), inv_eyemu).squeeze(1)             # (P,L)

        # v update
        v  = vn + gamma.view(P, 1) * tc.bmm((2 * pn - un).unsqueeze(1), D).squeeze(1)  # (P,N)
        qn = v + gamma.view(P, 1) * xb - gamma.view(P, 1) * proj_simplex(v / gamma.view(P, 1) + xb)

        un = un + tau * (pn - un)
        vn = vn + tau * (qn - vn)

        w_new = [un, vn]
        return w_new

