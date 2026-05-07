import torch
import torch.nn as nn
from .base import BaseModel


class LSTMModel(BaseModel):
    """
    Simple LSTM model for time series classification.
    Takes a sequence of shape (batch, timesteps, features) and
    outputs a single value per sample for binary classification.

    The LSTM processes the sequence step by step, and the hidden
    state from the final timestep is passed to a classifier.
    """

    def __init__(
        self,
        in_channels:  int,
        hidden_size:  int   = 128,
        num_layers:   int   = 2,
        num_classes:  int   = 1,
        dropout:      float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,   # features per timestep
            hidden_size=hidden_size,  # size of the hidden state
            num_layers=num_layers,    # how many LSTM layers to stack
            batch_first=True,         # expects (batch, timesteps, features)
            dropout=dropout if num_layers > 1 else 0.0,
            # dropout is only applied between LSTM layers, not after the last one
            # so it has no effect with num_layers=1 — set to 0 to avoid a warning
        )
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: (batch, timesteps, features)

        # lstm_out: (batch, timesteps, hidden_size) — output at every timestep
        # hidden:   tuple of (h_n, c_n) — final hidden and cell states
        lstm_out, _ = self.lstm(x)

        # Take only the last timestep's output — it has seen the full sequence
        last_timestep = lstm_out[:, -1, :]  # (batch, hidden_size)

        last_timestep = self.dropout(last_timestep)

        return self.classifier(last_timestep).squeeze(-1)  # (batch,)