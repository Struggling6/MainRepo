from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_model_from_config(
    model_config: Any,
    input_dim: int,
    context_length: int | None = None,
) -> nn.Module:
    try:
        return model_config.build(
            input_dim=input_dim,
            context_length=context_length,
        )
    except TypeError:
        return model_config.build(input_dim=input_dim)


def load_checkpoint_model(
    checkpoint_path: Path,
    config: Any,
    metadata: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    context_length = (
        metadata.get("context_length")
        or getattr(config.model, "context_length", None)
        or getattr(config.data, "context_length", None)
        or 168
    )

    model = build_model_from_config(
        model_config=config.model,
        input_dim=metadata["input_dim"],
        context_length=context_length,
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return model, checkpoint


def summarize_attributions(attributions: torch.Tensor) -> dict[str, np.ndarray]:
    attr = attributions.detach().cpu()
    abs_attr = attr.abs()

    result = {
        "sample_attributions": attr.numpy(),
        "mean_abs_attribution": abs_attr.mean(dim=0).numpy(),
    }

    if attr.ndim == 3:
        result["feature_importance"] = abs_attr.mean(dim=(0, 1)).numpy()
        result["time_importance"] = abs_attr.mean(dim=(0, 2)).numpy()

    elif attr.ndim == 2:
        result["feature_importance"] = abs_attr.mean(dim=0).numpy()

    return result


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    clean = {}

    for key, value in arrays.items():
        if value is None:
            continue

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        elif isinstance(value, (dict, list, tuple, str, int, float, bool)):
            value = np.array(value, dtype=object)

        clean[key] = value

    np.savez_compressed(path, **clean)