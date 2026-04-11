#import models from the different files
#something like this:
# from .cnn.py import CNNModel
#from .transformer.py import TransformerModel
from .mlp import MLPModel
from .supervised_cnn_transformer import SupervisedTransformerCNN

MODEL_REGISTRY = {
    "mlp":                        MLPModel,
    "supervised_cnn_transformer": SupervisedTransformerCNN,
}

def create_model(model_config, data_metadata):
    model_cls = MODEL_REGISTRY.get(model_config.name)
    if model_cls is None:
        raise ValueError(f"Model '{model_config.name}' is not registered.")

    return model_cls(
        in_channels=data_metadata["input_dim"],  # comes from dataset, not model config
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,        # note: model uses num_heads not nhead
        num_layers=model_config.num_layers,
        num_classes=model_config.num_classes,
        dropout=model_config.dropout,
    )