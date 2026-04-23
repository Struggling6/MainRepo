import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def get_model_parameters(model):
    params = [v.detach().cpu().numpy() for _, v in model.state_dict().items()]
    if len(params) == 0:
        raise RuntimeError(
            f"No tensors found in state_dict for model {type(model).__name__}"
        )
    return params


def set_model_parameters(model, parameters):
    state_dict = model.state_dict()
    keys = list(state_dict.keys())

    if len(keys) != len(parameters):
        raise RuntimeError(
            f"Parameter count mismatch: model has {len(keys)}, got {len(parameters)}"
        )

    new_state = {}
    for key, new_param in zip(keys, parameters):
        ref = state_dict[key]
        new_state[key] = torch.tensor(
            new_param,
            dtype=ref.dtype,
            device=ref.device,
        )

    model.load_state_dict(new_state, strict=True)