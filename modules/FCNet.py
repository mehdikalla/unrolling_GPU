import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self, layer_sizes):
        """
        layer_sizes : liste d'entiers [in_dim, hidden1, hidden2, ..., out_dim]
        """
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
