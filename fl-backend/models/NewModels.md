# Making new models
When making new models one should make a file like ```SomeModel.py``` and make the model logic inside that file

The file should look something like the following:
```
import torch.nn as nn
from .base import BaseModel


class SomeModel(BaseModel):
    def __init__(self, model_config: dict, data_metadata: dict):
        super().__init__()

        # Read only what this model actually needs
        # from model_config and data_metadata

    def forward(self, x):
        # return logits
        return ...

```

When that file has been completed and the internal logic is complete, it is important to register it in ```registry.py ``` like so:

```
MODEL_REGISTRY = {
    "mlp": MLPModel,
    "someModel": SomeModel, # <------ this one here
}

```