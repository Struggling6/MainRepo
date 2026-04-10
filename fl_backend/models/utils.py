import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


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

def save_model(
    model:      nn.Module,
    path:       Path,
    config:     object = None,
    threshold:  float  = 0.5,
    metrics:    dict   = None,
):
    """
    Save model weights, architecture config, best threshold,
    and training metrics to a single checkpoint file.

    Parameters
    ----------
    model     : nn.Module — the trained model
    path      : Path      — where to save the checkpoint (.pt file)
    config    : dataclass — model config (CNNTransformerConfig etc.)
    threshold : float     — best decision threshold found during training
    metrics   : dict      — final training metrics to store alongside weights
    """
    
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)  # create checkpoints/ dir if it doesn't exist

        checkpoint = {
            "model_state_dict": model.state_dict(), # PyTorch convention for saving/loading weights
            "model_config":     config,
            "threshold":        threshold,
            "metrics":          metrics or {},
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    else:
        raise ValueError("Path must be a pathlib.Path object")


def load_model(model: nn.Module, path: Path):
    """
    Load model weights from a checkpoint file.
    It takes an uninitialised model with the correct architecture and populates it with saved weights.


    Parameters
    ----------
    model  : nn.Module — an uninitialised model with the correct architecture
    path   : Path      — path to the checkpoint file

    Returns
    -------
    model      : nn.Module — model with loaded weights
    threshold  : float     — best decision threshold saved during training
    metrics    : dict      — training metrics stored at save time
    """
    device     = get_device()
    checkpoint = torch.load(path, map_location=device) # load to CPU first, then move model to GPU if available

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval() # set to the model to evaluation mode (disables dropout, batchnorm updates, etc.)

    threshold = checkpoint.get("threshold", 0.5)
    metrics   = checkpoint.get("metrics",   {})

    print(f"Model loaded from {path}  (threshold={threshold:.2f})")
    return model, threshold, metrics