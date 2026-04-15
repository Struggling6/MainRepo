import os

# Third Party
from transformers import (
    EarlyStoppingCallback,
    PatchTSTForClassification,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
)
from fl_backend.config import PatchTSTConfig
#from config import PatchTSTConfig
import numpy as np
import pandas as pd
from transformers import set_seed

set_seed(42)

config = PatchTSTConfig()

print(config)


