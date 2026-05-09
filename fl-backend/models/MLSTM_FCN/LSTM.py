import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    LSTM branch with optional dimension shuffle.

    With dimension_shuffle=True (the MLSTM-FCN default), the LSTM treats
    variables as the sequence dimension and timesteps as the per-step
    feature. The LSTM thus acts as a learned dimension reducer rather than
    a temporal model. With dimension_shuffle=False, the LSTM operates
    conventionally: timesteps as the sequence dimension, variables as the
    per-step feature.

    Args:
        num_variables:     number of input variables.
        num_timesteps:     number of time steps in each window.
        lstm_units:        hidden dimensionality of the LSTM.
        dropout:           dropout applied to the LSTM output.
        dimension_shuffle: see above.

    Output:
        Feature vector of shape (batch, lstm_units).
    """

    def __init__(
        self,
        num_variables: int,
        num_timesteps: int,
        lstm_units: int = 8,
        dropout: float = 0.8,
        dimension_shuffle: bool = True,
    ):
        super().__init__()
        self.dimension_shuffle = dimension_shuffle
        self.lstm_units = lstm_units

        # With shuffle:    input_size = num_timesteps, seq_len = num_variables
        # Without shuffle: input_size = num_variables, seq_len = num_timesteps
        input_size = num_timesteps if dimension_shuffle else num_variables
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_timesteps, num_variables)
        Returns:
            (batch, lstm_units)
        """
        if self.dimension_shuffle:
            x = x.transpose(1, 2)  # (B, T, V) -> (B, V, T)
        # else: (B, T, V) is fed directly, with V as the per-step feature.

        out, _ = self.lstm(x)
        out = out[:, -1, :]        # take last step
        return self.dropout(out)
