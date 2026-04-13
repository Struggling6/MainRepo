import torch.nn as nn #neural network module

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, d_model, dropout):
        super().__init__()

        # Normalize each feature across the sequence
        # Instead of using scaling manually using StandardScaler
        self.input_norm = nn.InstanceNorm1d(in_channels)  

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # reduce the sequence length by half
            nn.Dropout1d(dropout),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2), 
            nn.Dropout1d(dropout),

            nn.Conv1d(128, d_model, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(d_model),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x arrives as (batch, timesteps, features) = (batch, 168, 45)
        x = x.permute(0, 2, 1)        # → (batch, 45, 168) required by Conv1d
        x = self.input_norm(x)        # ← normalize raw sensor values
        return self.features(x)
    
