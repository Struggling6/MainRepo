
import torch
import torch.nn as nn
import torch.nn.functional as F

#TO-DO 
    #Put config værdier ind i dette istedet
    #Er outputtet korrekt?? Ser der en specifik form for output som modellen skal have
    
class CNNTransformer(nn.Module):
    def __init__(self, model_config: dict, data_metadata: dict):
        super().__init__()

        # Hent værdier fra config
        in_channels = model_config["in_channels"]           # 1
        embed_dim = model_config["embed_dim"]               # 128
        num_heads = model_config["num_heads"]               # 4
        num_layers = model_config["num_layers"]             # 2
        dropout = model_config["dropout"]  

        # Hent fra metadata
        num_features = data_metadata["num_features"]        # f.eks. 45
        
        # Intermediær channel dimension
        mid_channels = 64

        self.encoder = nn.Sequential(
            # Input: [batch, 1, num_features]
            nn.Conv1d(
                in_channels,    
                mid_channels,     
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 

            nn.Conv1d(mid_channels, mid_channels * 2, kernel_size=5, padding=2),  # 64 → 128
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  

            nn.Conv1d(mid_channels * 2, embed_dim, kernel_size=5, padding=2),  # 128 → embed_dim
            nn.ReLU(),
        )
        # Output: [batch, embed_dim, features//4]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=256,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.decoder = nn.Sequential(
            # Input: [batch, embed_dim, features//4]
            nn.Conv1d(embed_dim, mid_channels * 2, kernel_size=5, padding=2), 
            nn.ReLU(),

            nn.Conv1d(mid_channels * 2, mid_channels, kernel_size=5, padding=2), 
            nn.ReLU(),

            nn.Conv1d(mid_channels, in_channels, kernel_size=5, padding=2), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [batch, 1, num_features]
        z = self.encoder(x)
        # Output: [batch, embed_dim, features//4]

        z = z.transpose(1, 2)  # [batch, features//4, embed_dim] → [batch, features//4, embed_dim]
        z = self.transformer(z)
        # Output: [batch, features//4, embed_dim]

        z = z.transpose(1, 2)  # [batch, embed_dim, features//4]

        x_hat = self.decoder(z)
        # Output: [batch, 1, features//4]

        # Interpolate tilbage til original størrelse
        x_hat = F.interpolate(
            x_hat,
            size=x.shape[-1],
            mode="linear",
            align_corners=False,
        )
        # Output: [batch, 1, num_features]

        return x_hat
