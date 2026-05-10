import torch
import torch.nn as nn

from FeatureExtractor import FeatureExtractor
from .Classifier import Classifier


class CNN(nn.Module):
    def __init__(self, in_channels, d_model, num_classes=1, dropout=0.3):
        super().__init__()

        # CNN extracts temporal patterns from raw input
        # We keep it unchanged to isolate the effect of removing the Transformer
        self.feature_extractor = FeatureExtractor(in_channels, d_model, dropout=dropout)

        # Classifier converts learned features into final predictions
        # Input size must match d_model (output of CNN)
        self.classifier = Classifier(d_model, num_classes, dropout=dropout)

    def forward(self, x):
        # x shape: (batch, 168, 45)
        # 168 = time steps, 45 = input features

        x = self.feature_extractor(x)
        # After CNN:
        # shape becomes (batch, d_model, 42)
        # → d_model = number of learned features (channels)
        # → 42 = reduced time dimension after pooling


        x = x.mean(dim=2)
        #The only different line: Because the tensor shape changes—after the Transformer it is (batch, time, features) so you average over dim=1 (time), while after the CNN it is (batch, features, time) so you average over dim=2 (time).
        # Global average pooling over time dimension
        # Converts (batch, d_model, 42) → (batch, d_model)
        # This is required because the classifier expects a fixed-size vector
        # It summarizes how active each feature is by averaging its values over all timesteps, producing a single number that represents its overall strength in the sequence.

        x = self.classifier(x)
        # Maps feature vector to output predictions
        # shape: (batch, num_classes)

        return x.squeeze(-1)
        # For binary classification:
        # converts (batch, 1) → (batch,)