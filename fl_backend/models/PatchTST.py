from transformers import set_seed
from .base import BaseModel
from transformers import (
    PatchTSTForClassification,
)
set_seed(42)

class PatchTST(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.model = PatchTSTForClassification(config)

    def forward(self, x):
        # x: (batch, seq_len, num_channels) — float32 from _build_dataloader
        # Returns: (batch,) raw logits — sigmoid is applied in _val_epoch
        return self.model(past_values=x).prediction_logits.squeeze(-1)