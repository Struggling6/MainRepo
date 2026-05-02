from __future__ import annotations

import torch


def make_baseline(
    inputs: torch.Tensor,
    mode: str = "zero",
    reference_batch: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(inputs)

    if mode == "mean":
        ref = reference_batch if reference_batch is not None else inputs
        mean = ref.mean(dim=0, keepdim=True)
        return mean.expand_as(inputs).clone()

    if mode == "sample":
        ref = reference_batch if reference_batch is not None else inputs
        return ref[:1].expand_as(inputs).clone()

    raise ValueError(f"Unknown baseline mode: {mode}")