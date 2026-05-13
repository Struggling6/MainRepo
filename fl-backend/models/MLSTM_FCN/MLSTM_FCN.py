import torch
import torch.nn as nn

from .LSTM import LSTM
from .FCN import FCN


class MLSTM_FCN(nn.Module):
    """
    Multivariate LSTM-FCN: parallel LSTM and FCN feature branches whose
    outputs are concatenated and passed through a shared linear classifier.

    Args:
        input_dim:         dimensionality of the input features.
        num_timesteps:     number of time steps in each window.
        num_classes:       number of output classes.
        dimension_shuffle: use the MLSTM-FCN dimension shuffle in the LSTM branch.
        lstm_units:        LSTM hidden dimensionality.
        dropout:           dropout after the LSTM.
        se_reduction:      SE-block reduction ratio inside the FCN branch.
        num_layers:        number of stacked LSTM layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_timesteps: int,
        num_classes: int,
        dimension_shuffle: bool = True,
        lstm_units: int = 8,
        dropout: float = 0.8,
        se_reduction: int = 16,
        num_layers: int = 1,
    ):
        super().__init__()

        self.lstm = LSTM(
            input_dim=input_dim,
            num_timesteps=num_timesteps,
            lstm_units=lstm_units,
            dropout=dropout,
            dimension_shuffle=dimension_shuffle,
            num_layers=num_layers,
        )

        self.fcn = FCN(
            num_variables=input_dim,
            use_se=True if se_reduction > 0 else False,
            se_reduction=se_reduction,
        )

        self.classifier = nn.Linear(lstm_units + FCN.OUTPUT_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_timesteps, input_dim)
        Returns:
            Logits of shape (batch, num_classes), or (batch,) when num_classes=1.
        """
        return self.classifier(torch.cat([self.lstm(x), self.fcn(x)], dim=1)).squeeze(-1)
