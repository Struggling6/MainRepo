import numpy as np
import torch


def get_model_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {
        k: torch.tensor(v) for k, v in params_dict
    }
    model.load_state_dict(state_dict, strict=True)