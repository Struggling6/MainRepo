import torch.nn as nn #neural network module

class Classifier(nn.Module):
    def __init__(self, d_model, num_classes=1, dropout=0.4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes) # output for multi-class classification
        )

    def forward(self, x):
        return self.classifier(x)