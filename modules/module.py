import numpy as np
import torch as tc
from torch.utils.data import Dataset
from typing import Optional, Union

TensorLike = Union[tc.Tensor, np.ndarray]

class MyDataset(Dataset):
    """
    Dataset PyTorch à partir d’un .pt contenant :
      - xtrue : (K, N)
      - yblur : (K, M)  (ou transposé ; auto-corrigé)
    Args:
      dataset_path : chemin vers le .pt
      initial_x0   : vecteur initial (numpy ou Tensor) ou None
      return_name  : si True, renvoie aussi un identifiant de signal
    """

    def __init__(
        self,
        dataset_path: str,
        initial_x0: Optional[TensorLike] = None,
        return_name: bool = False
    ):
        super().__init__()

        data  = tc.load(dataset_path)
        raw_X = data["xtrue"]
        raw_Y = data["yblur"]

        # Si premières dimensions différentes, corriger par transposition
        if raw_X.shape[0] != raw_Y.shape[0]:
            raw_X = raw_X.T
            raw_Y = raw_Y.T
        assert raw_X.shape[0] == raw_Y.shape[0], \
            f"Incohérence K: {raw_X.shape[0]} vs {raw_Y.shape[0]}"

        # Stockage en double + contiguous (CPU)
        self.X_true = raw_X.double().contiguous()
        self.Y      = raw_Y.double().contiguous()

        # x0 optionnel : un seul vecteur réutilisé pour tous les échantillons
        if initial_x0 is not None:
            x0 = initial_x0 if isinstance(initial_x0, tc.Tensor) else tc.from_numpy(initial_x0)
            self.initial_x0 = x0.double().contiguous()
        else:
            self.initial_x0 = None

        self.return_name = return_name

    def __len__(self):
        return self.X_true.shape[0]

    def __getitem__(self, idx: int):
        x  = self.X_true[idx]
        y  = self.Y[idx]
        x0 = self.initial_x0.clone() if self.initial_x0 is not None else None

        if self.return_name:
            name = f"signal_{idx}"
            return (name, x, y, x0) if x0 is not None else (name, x, y)
        else:
            return (x, y, x0) if x0 is not None else (x, y)
