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

def create_model(model_config, data_metadata: dict):
    model_name = model_config.name.lower()  # .name not ["name"] — dataclass attribute

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_name]

    return model_cls(
        model_config=model_config,
        data_metadata=data_metadata,
    )