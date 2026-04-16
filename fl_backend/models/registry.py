#import models from the different files
#something like this:
# from .cnn.py import CNNModel
#from .transformer.py import TransformerModel
from .mlp import MLPModel
from .supervised_cnn_transformer import SupervisedTransformerCNN
from .transformer import Transformer

MODEL_REGISTRY = {
    "mlp":                        MLPModel,
    "supervised_cnn_transformer": SupervisedTransformerCNN,
    "transformer":                Transformer,
}

def create_model(model_config, metadata):
    model_cls = MODEL_REGISTRY.get(model_config.name)
    if model_cls is None:
        raise ValueError(f"Model '{model_config.name}' is not registered.")

    return model_cls(
        in_channels=metadata["input_dim"],  # comes from dataset, not model config
        d_model=model_config.d_model,
        nhead=model_config.nhead,
        num_layers=model_config.num_layers,
        num_classes=model_config.num_classes,
        dropout=model_config.dropout,
    )