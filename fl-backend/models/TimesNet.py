from .base import BaseModel
import pypots as pp
from pypots.classification import TimesNet as TimesNetModel

class TimesNet(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.model = TimesNetModel(config)

    def forward(self, x):
        return self.model(past_values=x).prediction_logits.squeeze(-1)