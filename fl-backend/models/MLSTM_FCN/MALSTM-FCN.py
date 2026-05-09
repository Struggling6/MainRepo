import torch
import torch.nn as nn
import torch.nn.functional as F

from .SqueezeExciteBlock import SqueezeExciteBlock
from .LSTM import LSTM
from .FCN import FCN


class MLSTM_FCN(nn.Module):
    """
    Configurable Multivariate LSTM-FCN.

    The two branches are constructed only when their flag is enabled, so a
    branch-disabled variant has zero parameter cost for that branch. The
    classifier head is sized automatically based on which branches are active.

    Args:
        num_variables:     number of input variables.
        num_timesteps:     number of time steps in each window.
        num_classes:       number of output classes.
        use_lstm:          include the LSTM branch.
        use_fcn:           include the FCN branch.
        use_se:            include SE blocks in the FCN branch
                           (no effect when use_fcn=False).
        dimension_shuffle: use the MLSTM-FCN dimension shuffle in the LSTM branch
                           (no effect when use_lstm=False).
        lstm_units:        LSTM hidden dimensionality.
        dropout:           dropout after the LSTM.
        se_reduction:      SE-block reduction ratio.

    Raises:
        ValueError if both use_lstm and use_fcn are False.
    """

    def __init__(
        self,
        num_variables: int,
        num_timesteps: int,
        num_classes: int,
        use_lstm: bool = True,
        use_fcn: bool = True,
        use_se: bool = True,
        dimension_shuffle: bool = True,
        lstm_units: int = 8,
        dropout: float = 0.8,
        se_reduction: int = 16,
    ):
        super().__init__()
        if not (use_lstm or use_fcn):
            raise ValueError("At least one of use_lstm or use_fcn must be True.")

        self.use_lstm = use_lstm
        self.use_fcn = use_fcn

        feature_dim = 0
        if use_lstm:
            self.lstm_branch = LSTM(
                num_variables=num_variables,
                num_timesteps=num_timesteps,
                lstm_units=lstm_units,
                dropout=dropout,
                dimension_shuffle=dimension_shuffle,
            )
            feature_dim += lstm_units

        if use_fcn:
            self.fcn_branch = FCN(
                num_variables=num_variables,
                use_se=use_se,
                se_reduction=se_reduction,
            )
            feature_dim += FCN.OUTPUT_DIM

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_timesteps, num_variables)
        Returns:
            Logits of shape (batch, num_classes).
        """
        features = []
        if self.use_lstm:
            features.append(self.lstm_branch(x))
        if self.use_fcn:
            features.append(self.fcn_branch(x))

        combined = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        return self.classifier(combined)