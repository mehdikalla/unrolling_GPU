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
        # Propagation à travers toutes les couches sauf la dernière avec ReLU
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # Dernière couche sans activation
        x = self.layers[-1](x)

        return x
