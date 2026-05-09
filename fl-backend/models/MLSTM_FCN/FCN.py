import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    """
    Fully Convolutional Network branch.

    Three Conv1d layers with batch normalisation and ReLU, optionally
    augmented with Squeeze-and-Excitation blocks after the first two convs
    (matching MLSTM-FCN). Ends with global average pooling, producing a
    fixed-size feature vector.

    Args:
        num_variables: number of input variables (channels) per time step.
        use_se:        whether to insert Squeeze-and-Excitation blocks.
                       True reproduces MLSTM-FCN; False reproduces the
                       original Wang et al. (2017) FCN / the convolutional
                       branch of the 2018 LSTM-FCN.
        se_reduction:  reduction ratio for the SE blocks.

    Output:
        Feature vector of shape (batch, FCNBranch.OUTPUT_DIM).
    """

    OUTPUT_DIM = 128

    def __init__(self, num_variables: int, use_se: bool = True, se_reduction: int = 16):
        super().__init__()
        self.use_se = use_se

        self.conv1 = nn.Conv1d(num_variables, 128, kernel_size=8, padding="same")
        self.bn1 = nn.BatchNorm1d(128)
        self.se1 = SqueezeExciteBlock(128, se_reduction) if use_se else nn.Identity()

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(256)
        self.se2 = SqueezeExciteBlock(256, se_reduction) if use_se else nn.Identity()

        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm1d(128)
        # No SE block after the final conv, per the paper.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_timesteps, num_variables)
        Returns:
            (batch, 128)
        """
        # Conv1d expects (batch, channels, time).
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        return x.mean(dim=2)  # global average pool over time