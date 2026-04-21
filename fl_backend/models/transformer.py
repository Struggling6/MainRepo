import torch
import torch.nn as nn

from .base import BaseModel
from .supervised_utils.Classifier import Classifier


class Transformer(BaseModel):
    """
    Pure Transformer Encoder model (no CNN feature extraction).
    
    Architecture:
      1. Input Normalization (InstanceNorm1d)
      2. Linear projection: in_channels → d_model
      3. Positional embeddings for 168 timesteps
      4. Transformer encoder stack
      5. Mean pooling over sequence
      6. Classifier head (MLP)
    """
    
    def __init__(
        self, 
        in_channels,      # number of input features (e.g. 45)
        d_model,          # size of embedding dimension (e.g. 128)
        nhead,            # number of attention heads
        num_layers,       # number of transformer encoder layers
        seq_len=24,      # sequence length of each input window (full 24, no reduction)
        num_classes=1,    # output classes (1 for binary, >1 for multi-class)
        dropout=0.3,      # dropout rate
    ):
        super().__init__()
        
        # Input normalization (normalizes each feature across the sequence)
        self.input_norm = nn.InstanceNorm1d(in_channels)
        
        # Project from in_channels to d_model. Because transformer excepcts vectors of size d_model
        # input:  (batch, 168, 45)
        # output: (batch, 168, d_model)
        self.input_projection = nn.Linear(in_channels, d_model)
        
        # Positional embeddings for each timestep
        # Tells the transformer "where" in the sequence each timestep is
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        
        # Transformer encoder layer (single layer definition)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, #Embedding size
            nhead=nhead, #Splits attention into multiple perspectives.
            dim_feedforward=d_model * 4,  # standard expansion ratio
            dropout=dropout,
            batch_first=True,  # input: (batch, seq, features)
        )
        
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers, # how many times to repeat the encoder layer
        )
        
        # Final classification head (MLP)
        self.classifier = Classifier(d_model, num_classes, dropout=dropout) 
    
    def forward(self, x): #The forward() method defines the full data flow of the model. It includes preprocessing, the transformer encoder, and the classification head. The transformer itself is only part of this pipeline, specifically the transformer_encoder call.
        """
        Forward pass. 
        
        Args:
            x: input tensor of shape (batch_size, seq_len, in_channels)
               typically (batch, 168, 45)
        
        Returns:
            output tensor of shape (batch_size,) for binary or
                                     (batch_size, num_classes) for multi-class
        """
        # x shape: (batch, 168, 45)
        
        # Normalize: InstanceNorm1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)  # (batch, 45, 168)
        x = self.input_norm(x)
        x = x.permute(0, 2, 1)  # (batch, 168, 45)
        
        # Project to d_model
        x = self.input_projection(x)
        # x shape: (batch, 168, d_model)
        
        # Create position indices: [0, 1, 2, ..., 167]
        positions = torch.arange(x.size(1), device=x.device)  # (168,)
        positions = positions.unsqueeze(0)  # (1, 168) add batch dimension
        
        # Add positional information
        x = x + self.pos_embedding(positions)
        # x shape: (batch, 168, d_model)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x) 
        # x shape: (batch, 168, d_model)
        
        # Mean pooling: aggregate across all timesteps. Converts sequence → single vector
        # turns (batch, 168, d_model) → (batch, d_model)
        x = x.mean(dim=1)
        
        # Pass through classifier (MLP head)
        x = self.classifier(x)
        # x shape: (batch, num_classes)
        
        # Squeeze last dimension if it's size 1 (for binary classification)
        # (batch, 1) → (batch,)
        return x.squeeze(-1)
