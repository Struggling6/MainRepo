import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, d_model, dropout):
        super().__init__()

        self.input_norm = nn.InstanceNorm1d(in_channels)

        # Used when we have real temporal windows, e.g. seq_len = 168
        self.temporal_features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout),

            nn.Conv1d(128, d_model, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(d_model),
            nn.ReLU(),
        )

        # Used when temporal windows are disabled, e.g. seq_len = 1
        # No pooling and no InstanceNorm, because both are unsafe for length 1.
        self.non_temporal_features = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=1),
            nn.ReLU(),
            nn.Dropout1d(dropout),
        )

    def forward(self, x):
        # x arrives as (batch, timesteps, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, timesteps)

        # No-window mode: sequence length = 1
        if x.size(-1) < 4:
            return self.non_temporal_features(x)

        # Normal temporal mode
        x = self.input_norm(x)
        return self.temporal_features(x)