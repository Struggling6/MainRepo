#!/usr/bin/env python3
"""
Standalone evaluation: load a saved checkpoint and run the Evaluator on the
held-out test set without training a model first.

By default uses the checkpoint at checkpoints/<model>_<data>.pt and the
threshold saved inside the checkpoint. Both can be overridden via CLI.

Examples:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/patchtst_lead_csv.pt
    python evaluate.py --threshold 0.5
"""

import argparse
import os
from pathlib import Path

import torch

from local_experiment import CONFIG
from training.training_utils.Evaluator import Evaluator

BASE_DIR = Path(__file__).resolve().parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to .pt checkpoint (default: checkpoints/<model>_<data>.pt).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (default: value saved in checkpoint).",
    )
    return parser.parse_args()


def resolve_checkpoint_path(arg: Path | None) -> Path:
    if arg is not None:
        return arg.resolve()

    base_dir = Path(
        os.getenv("CHECKPOINT_DIR", Path(__file__).resolve().parent / "checkpoints")
    )
    return (base_dir / f"{CONFIG.model.name}_{CONFIG.data.name}.pt").resolve()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train a model first or pass --checkpoint."
        )

    threshold = args.threshold
    if threshold is None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        threshold = float(ckpt.get("threshold", 0.5))

    print(f"Evaluating: {checkpoint_path}")
    print(f"Threshold:  {threshold:.4f}")

    CONFIG.evaluation.model_path = checkpoint_path
    
    Evaluator.evaluate(model_path=checkpoint_path, threshold=threshold)


if __name__ == "__main__":
    main()
