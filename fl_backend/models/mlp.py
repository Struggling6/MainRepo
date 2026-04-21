import torch
import torch.nn as nn
from .base import BaseModel


class MLPModel(BaseModel):
    def __init__(self, model_config: dict, data_metadata: dict):
        super().__init__()

        input_dim = data_metadata["input_dim"]
        num_classes = data_metadata["num_classes"]

        hidden_dim = model_config.get("hidden_dim", 64)
        dropout = model_config.get("dropout", 0.2)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)