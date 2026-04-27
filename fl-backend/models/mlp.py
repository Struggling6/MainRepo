import torch.nn as nn
from .base import BaseModel


class MLPModel(BaseModel):
    def __init__(
        self,
        in_channels:  int,
        hidden_size:  int   = 128,
        num_layers:   int   = 2,
        num_classes:  int   = 1,
        dropout:      float = 0.3,
    ):
        super().__init__()

        layers = []
        in_dim = in_channels
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)