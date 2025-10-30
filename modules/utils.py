from typing import Union, Optional
import torch as tc
import torch.nn as nn

class FCNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Max:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:

                        f(x) = max( x1, ..., xn)

    'gamma' is the scale factor

     INPUTS
    ========
     x         - scalar or ND tensor
     gamma     - positive, scalar or tensor compatible with the blocks of 'x'
                 [default: gamma=1]
     axis      - None or int, axis of block-wise processing [default: axis=None]
                  axis = None --> 'x' is processed as a single vector (default).
                  axis >=0   --> 'x' is processed block-wise along the specified axis.
    """

    def __init__(self, axis: Optional[int] = None):
        if (axis is not None) and (axis < 0):
            axis = None
        self.axis = axis

    def prox(self, x: tc.Tensor, gamma: Union[float, tc.Tensor] = 1.0) -> tc.Tensor:
        # --- FIX MINIMAL: forcer gamma/scale sur le même device/dtype que x,
        #     même si gamma est déjà un Tensor (ex: Simplex(eta=1) sur CPU) ---
        if isinstance(gamma, tc.Tensor):
            scale = gamma.to(dtype=x.dtype, device=x.device)
        else:
            scale = tc.tensor(gamma, dtype=x.dtype, device=x.device)
        # ---------------------------------------------------------------------

        self._check(x, scale)
        axis = self.axis
        sz = x.shape

        # If x is a single element, force it to be a vector.
        if x.numel() <= 1:
            x = x.reshape(-1)

        # Build shape for block-wise operations.
        sz0 = list(x.shape)
        if axis is not None:
            sz0[axis] = 1
        if scale.numel() > 1:
            scale = scale.reshape(sz0)

        # 0. Linearize if axis is None.
        if axis is None:
            x = x.reshape(-1)
            scale = scale.reshape(-1)
            sort_x = -tc.sort(-x).values
            ones_ = tc.ones_like(sort_x)
            cum_sum = (tc.cumsum(sort_x, dim=0) - scale) / tc.cumsum(ones_, dim=0)

        else:
            sort_x = -tc.sort(-x, dim=axis).values
            ones_ = tc.ones_like(sort_x)
            cum_sum = (tc.cumsum(sort_x, dim=axis) - scale) / tc.cumsum(ones_, dim=axis)

        # 3. Find the index: n = max{ j : s(j) > c(j) }.
        mask = sort_x > cum_sum
        mat = tc.arange(mask.numel(), device=x.device, dtype=x.dtype).reshape(mask.shape)
        mask = mask.to(x.dtype) * mat
        if axis is None:
            ind_max = tc.argmax(mask)
        else:
            ind_max = tc.argmax(mask, dim=axis)

        # 4. Retrieve the threshold value.
        if ind_max.numel() <= 1:
            prox_threshold = cum_sum[ind_max].reshape(scale.shape)
        else:
            if axis is None:
                prox_threshold = tc.minimum(cum_sum[ind_max], x)
            else:
                ind_max_unsq = ind_max.unsqueeze(axis)
                gathered = tc.gather(cum_sum, dim=axis, index=ind_max_unsq)
                prox_threshold = tc.minimum(gathered, x)
                prox_threshold = prox_threshold.reshape(x.shape)

        # 5. Compute the prox: the pointwise minimum between x and the threshold.
        prox_x = tc.minimum(x, prox_threshold)

        # 6. Zero out entries when all entries satisfy s(j) > c(j) along the axis.
        if axis is None:
            if tc.all(mask.bool()):
                prox_x = prox_x * 0
        else:
            all_mask = tc.all(mask.bool(), dim=axis, keepdim=True)
            prox_x = prox_x * (1 - all_mask.to(x.dtype))

        # 7. Restore original shape.
        prox_x = prox_x.reshape(sz)
        return prox_x

    def __call__(self, x: tc.Tensor) -> tc.Tensor:
        if self.axis is None:
            return tc.sum(tc.max(x))
        else:
            return tc.sum(tc.max(x, dim=self.axis).values)

    def _check(self, x: tc.Tensor, gamma: tc.Tensor):
        # Use torch.any instead of numpy.any.
        if tc.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is a tensor) must be strictly positive"
            )
        if self.axis is None and gamma.numel() > 1:
            raise ValueError(
                "'gamma' must be a scalar when the parameter 'axis' is equal to None"
            )
        if gamma.numel() <= 1:
            return
        sz = x.shape
        if len(sz) <= 1:
            self.axis = None
        if len(sz) <= 1:
            raise ValueError("'gamma' must be scalar when 'x' is one dimensional")
        if len(sz) > 1 and (self.axis is not None):
            sz0 = list(sz)
            sz0[self.axis] = 1
            prod = 1
            for d in sz0:
                prod *= d
            if gamma.numel() > 1 and (prod != gamma.numel()):
                raise ValueError(
                    "The dimension of 'gamma' is not compatible with the blocks of 'x'"
                )

class Simplex:
    """Compute the projection and the indicator of the simplex.

    Recall: every vector x belonging to the simplex verifies:
    
                    x >= 0 and  (1,...,1).T * x = eta

    where (1,...,1) is an ND tensor with all components equal to one,
    and (1,...,1).T its transpose.

    INPUTS
    ========
     x    - ND tensor
     eta  - positive, scalar or tensor compatible with the blocks of 'x'
     axis - int or None, direction of block-wise processing [DEFAULT: axis=None]
            When the input 'x' is an array, the computation can vary as follows:
            - axis = None --> 'x' is processed as a single vector.
            - axis >= 0 --> 'x' is processed block-wise along the specified axis
                          (axis=0 -> rows, axis=1 -> columns, etc.).
    """

    def __init__(self, eta: Union[float, tc.Tensor], axis: Optional[int] = None):
        # Convert eta to a torch tensor if necessary.
        if not isinstance(eta, tc.Tensor):
            eta = tc.tensor(eta, dtype=tc.float64)
        if tc.any(eta <= 0):
            raise Exception(
                "'eta' (or all of its components if it is a tensor) must be positive"
            )
        self.eta = eta
        self.axis = axis

    # Proximal operator (i.e. projection on the simplex)
    def prox(self, x: tc.Tensor) -> tc.Tensor:
        # NB: Max.prox gère déjà le cast device/dtype de gamma → rien d'autre à changer ici.
        return x - Max(self.axis).prox(x, self.eta)

    # Indicator of the simplex
    def __call__(self, x: tc.Tensor) -> float:
        """
        Indicate if the input 'x' is in the constraint set or not.
        
        Returns:
            0      if the input 'x' is in the set,
            +inf   otherwise.
        """
        # Compute the sum along the specified axis (or over all elements if axis is None).
        if self.axis is None:
            scalar_prod = tc.sum(x)
        else:
            scalar_prod = tc.sum(x, dim=self.axis)
        tol = 1e-10
        if tc.all(x >= 0) and tc.all(tc.abs(scalar_prod - self.eta) < tol):
            return 0
        return float('inf')



def LipschitzSOOT(alpha, beta, eta, N):
    # Tous les paramètres sont supposés scalaires.
    return 1/(alpha*beta) + 1/(2*alpha**2) * max(1, (N*alpha/beta)**2) + 1/eta**2

def gradient_x(x, y, Hmat, alpha, beta, eta, nu):
    """
    Calcule le gradient par rapport à x pour un batch de signaux.
    
    x    : tenseur de forme (K, N)
    y    : tenseur de forme (K, M)
    Hmat : tenseur de forme (M, N)
    
    Renvoie :
       grad : tenseur de forme (K, N)
       l1   : tenseur de forme (K, 1) contenant le terme L1 par signal.
    """
    # Calculer l1 et l2 par signal (somme sur la dimension temporelle : dim=1)
    l1 = tc.sum(tc.sqrt(x**2 + alpha**2) - alpha, dim=1, keepdim=True)  # forme (K, 1)
    l2 = tc.sqrt(tc.sum(x**2 + eta**2, dim=1, keepdim=True))             # forme (K, 1)
    
    # Terme de fidélité : x @ Hmat.T a une forme (K, M)
    Hx_y = tc.matmul(x, Hmat.t()) - y
    gradfid = tc.matmul(Hx_y, Hmat)  # (K, M) @ (M, N) = (K, N)
    
    # Gradient de régularisation (broadcasting avec (K, 1) se fait automatiquement)
    gradl1l2 = nu * (x / (tc.sqrt(x**2 + alpha**2) * (l1 + beta)) - x / (l2**2))
    grad = gradfid + gradl1l2
    return grad, l1

def proj_simplex(x):
    """
    Projette chaque ligne de x sur le simplexe.
    
    x : tenseur de forme (K, N) ou (K, 1)
    Renvoie un tenseur de même forme.
    """
    test_simplex = Simplex(eta=1)
    if x.ndim == 2 and x.shape[0] > 1:
        projected = tc.empty_like(x)
        for i in range(x.shape[0]):
            # Traiter chaque signal (ligne) individuellement.
            xi = x[i, :].unsqueeze(0)  # forme (1, N)
            projected[i, :] = test_simplex.prox(xi).squeeze(0)
        return projected
    else:
        return test_simplex.prox(x)

def compute_proj_grad_norm(x, gradx, L):
    """
    Calcule la norme de Frobenius du gradient projeté.
    
    x, gradx : tenseurs de forme (K, N)
    L        : scalaire (constante de Lipschitz)
    
    Renvoie une norme scalaire sur le batch.
    """
    alpha_val = 1  # interprété comme 1/L (en supposant que L est déjà calculé)
    temp = x - alpha_val * gradx
    # La fonction proj_simplex gère désormais les batches.
    Pgradx = proj_simplex(temp) - x
    return tc.norm(Pgradx)

def majorante_x_diag(x, alpha, l1, beta, Cg2, nu, Hnorm2):
    """
    Calcule la diagonale du majorant ponctuel pour un batch de signaux.
    
    x  : tenseur de forme (K, N)
    l1 : tenseur de forme (K, 1) (valeur par signal)
    
    Renvoie un tenseur de forme (K, N)
    """
    Al1l2 = nu * (1.0 / (tc.sqrt(x**2 + alpha**2) * (l1 + beta)) + Cg2)
    return Al1l2 + Hnorm2

def Majorante_x(d, x, alpha, l1, beta, Cg2, Hmat, nu):
    """
    Calcule le majorant pour la direction d (en batch).
    
    d et x : tenseurs de forme (K, N)
    Renvoie un tenseur de forme (K, N)
    """
    Al1l2 = nu * (1.0 / (tc.sqrt(x**2 + alpha**2) * (l1 + beta)) + Cg2)
    # Pour chaque signal, on applique (Hmat.T @ Hmat) : équivalent à d @ (Hmat.T @ Hmat)
    Ad = Al1l2 * d + tc.matmul(d, tc.matmul(Hmat.t(), Hmat))
    return Ad

def Criterion(x, Hmat, y, sigma, beta, eta, nu):
    """
    Calcule le critère objectif pour un batch de signaux.
    
    x    : tenseur de forme (K, N)
    y    : tenseur de forme (K, M)
    
    Renvoie :
       crit : tenseur de forme (K, 1) contenant le critère par signal.
    """
    # Calculer l1 et l2 par signal (somme sur la dimension temporelle : dim=1)
    l1 = tc.sum(tc.sqrt(x**2 + sigma**2) - sigma, dim=1, keepdim=True)  # (K, 1)
    l2 = tc.sqrt(tc.sum(x**2 + eta**2, dim=1, keepdim=True))             # (K, 1)
    Hx_y = tc.matmul(x, Hmat.t()) - y     # (K, M)
    fid = 0.5 * tc.sum(Hx_y**2, dim=1, keepdim=True)  # (K, 1)
    l1l2 = nu * tc.log((l1 + beta) / l2)
    crit = fid + l1l2
    return crit

def signal_noise(x, xtrue):
    """
    Calcule les erreurs l1 et l2 par signal.
    
    x, xtrue : tenseurs de forme (K, N)
    
    Renvoie :
      norm1, norm2 : chacun de forme (K, 1)
    """
    norm1 = tc.sum(tc.abs(x - xtrue), dim=1, keepdim=True) / xtrue.shape[1]
    norm2 = tc.sqrt(tc.sum((x - xtrue)**2, dim=1, keepdim=True) / xtrue.shape[1])
    return norm1, norm2

def display_datas(iter, Crit, l1x, l2x, time_val):
    """
    Affiche les informations d'itération. Pour un batch de signaux, affiche la métrique pour le premier signal.
    
    Crit : tenseur de forme (K, 1) ou scalaire.
    l1x, l2x : tenseurs de forme (K, 1) ou scalaires.
    """
    print("\n----------------------------------")
    print(f"Iteration = {iter}")
    if Crit.ndim == 2:
        crit_val = Crit[0, 0].item()  # premier signal, premier élément
    else:
        crit_val = Crit.item()
    print(f"Crit = {crit_val}")
    print(f"Time = {time_val}")
    print("----------------------------")
    if l1x.ndim == 2:
        l1_val = l1x[0, 0].item()
    else:
        l1_val = l1x.item()
    if l2x.ndim == 2:
        l2_val = l2x[0, 0].item()
    else:
        l2_val = l2x.item()
    print(f"x : l1  = {l1_val}, l2  = {l2_val}")
    print("----------------------------------")

def dosy_mat(N, M, tmin, tmax, Dmin, Dmax, dtype=tc.float64, device='cpu'):
    """
    Builds a matrix (Hmat) used for DOSY (Diffusion-Ordered Spectroscopy) simulations
    and a vector of discrete D-values (T).

    Args:
        N (int): Dimension for the D-values.
        M (int): Number of rows in the output matrix Hmat.
        tmin (float): Minimum value for the time axis.
        tmax (float): Maximum value for the time axis.
        Dmin (float): Minimum value for D.
        Dmax (float): Maximum value for D.
        dtype (torch.dtype): PyTorch data type (default: torch.float64).
        device (str): Device ('cpu' or 'cuda').

    Returns:
        (torch.Tensor, torch.Tensor):
          - T: A tensor of shape (N,) containing the discrete D-values (log-spaced between Dmin and Dmax).
          - Hmat: A tensor of shape (M, N) created via a Kronecker-like operation with an exponential decay.
    """
    # Compute the log-space increment between Dmin and Dmax.
    
    D = (tc.log(tc.tensor(Dmin, device=device, dtype=dtype)) -
         tc.log(tc.tensor(Dmax, device=device, dtype=dtype))) / (N - 1)

    # Build the D-values vector T of shape (N,).
    T = (tc.tensor(Dmin, device=device, dtype=dtype) *
         tc.exp(-D * tc.arange(1, N + 1, device=device, dtype=dtype)))

    # Create the time vector t of shape (M, 1).
    t = tc.linspace(tmin, tmax, M, device=device, dtype=dtype).unsqueeze(1)

    # Reshape T for broadcasting: (1, N).
    T_row = T.unsqueeze(0)

    # Kronecker-like product of shape (M, N).
    kron_t_T = tc.kron(t, T_row)

    # Exponential decay for DOSY.
    Hmat = tc.exp(-kron_t_T)
    return T, Hmat

