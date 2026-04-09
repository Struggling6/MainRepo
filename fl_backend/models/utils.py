import numpy as np
import torch


def get_device():
    """
    Returns the best available device in priority order:
      1. ROCm (AMD GPU via HIP) — detected through torch.cuda, which ROCm mirrors
      2. CUDA (Nvidia GPU)      — same API, included for completeness
      3. CPU                    — fallback if no GPU is available
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU available: {gpu_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon detected — using MPS")
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU")
    
    return device

print("GPU available:", torch.cuda.is_available())

def get_model_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {
        k: torch.tensor(v) for k, v in params_dict
    }
    model.load_state_dict(state_dict, strict=True)