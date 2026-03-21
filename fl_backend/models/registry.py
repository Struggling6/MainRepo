#import models from the different files
#something like this:
# from .cnn.py import CNNModel
#from .transformer.py import TransformerModel
from .mlp import MLPModel


MODEL_REGISTRY = {
    #"cnn": CNNModel,
    #"transformer": TransformerModel,
    "mlp": MLPModel,
}

def create_model(model_config: dict, data_metadata: dict):
    model_name = model_config["name"].lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")
    
    model_cls = MODEL_REGISTRY[model_name]

    return model_cls(
        model_config=model_config,
        data_metadata=data_metadata
    )