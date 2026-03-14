#import models from the different files
#something like this:
# from .cnn.py import CNNModel
#from .transformer.py import TransformerModel

MODEL_REGISTRY = {
    "cnn": CNNModel,
    "transformer": TransformerModel,
}

def create_model(model_config: dict, data_config: dict):
    model_name = model_config["name"]

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")
    
    model_cls = MODEL_REGISTRY[model_name]

    return model_cls(
        input_dim=data_config["num_features"],
        seq_length=data_config["sequence_length"],
        num_classes=data_config["num_classes"],
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"],
    )