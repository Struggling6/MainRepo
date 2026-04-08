
# Machine learning libraries
import torch
import torch.nn as nn #neural network module
from supervised_utils.featureExtractor import FeatureExtractor
from supervised_utils.Classifier import Classifier
from base import BaseModel

class SupervisedTansformerCNN(BaseModel):
    def __init__(self, in_channels, d_model, nhead, num_layers, num_classes=1, dropout=0.3):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(in_channels, d_model, dropout=dropout)
        self.pos_embedding = nn.Embedding(42, d_model)  # 168 → MaxPool → 84 → MaxPool → 42

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = Classifier(d_model, num_classes, dropout=dropout)

    def forward(self, x):
        # x: (batch, 168, 45)
        x = self.feature_extractor(x)                          # (batch, d_model, 42) 
        x = x.permute(0, 2, 1)                                 # (batch, 42, d_model)

        positions = torch.arange(x.size(1), device=x.device)   # [0, 1, ..., 41]
        x = x + self.pos_embedding(positions)                  # (batch, 42, d_model)

        x = self.transformer_encoder(x)                        # (batch, 42, d_model)
        x = x.mean(dim=1)                                      # (batch, d_model)
        x = self.classifier(x)                                 # (batch, num_classes)

        return x.squeeze(-1)
        