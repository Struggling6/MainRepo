import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExciteBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).

    The MLSTM-FCN paper uses this on top of the temporal convolution outputs
    to model inter-channel dependencies, with a default reduction ratio of 16.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced = max(channels // reduction_ratio, 1)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        z = x.mean(dim=2)                       # squeeze: global avg pool over time
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        z = z.unsqueeze(2)                      # (batch, channels, 1)
        return x * z                            # excite: scale each channel