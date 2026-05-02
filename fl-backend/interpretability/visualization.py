from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance(
    npz_path: Path,
    output_path: Path | None = None,
    top_k: int = 20,
) -> Path:
    data = np.load(npz_path, allow_pickle=True)

    values = data["feature_importance"]

    if "feature_names" in data:
        names = np.array(data["feature_names"], dtype=str)
    else:
        names = np.array([f"feature_{i}" for i in range(len(values))])

    order = np.argsort(values)[-top_k:]

    if output_path is None:
        output_path = npz_path.with_name("ig_feature_importance.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, max(4, top_k * 0.35)))
    plt.barh(names[order], values[order])
    plt.xlabel("Mean absolute Integrated Gradients attribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Feature importance plot saved to: {output_path}")
    return output_path


def plot_time_importance(
    npz_path: Path,
    output_path: Path | None = None,
) -> Path:
    data = np.load(npz_path, allow_pickle=True)

    if "time_importance" not in data:
        raise ValueError(
            "No time_importance found. This only exists for sequence-shaped inputs."
        )

    values = data["time_importance"]

    if output_path is None:
        output_path = npz_path.with_name("ig_time_importance.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(values)), values)
    plt.xlabel("Timestep")
    plt.ylabel("Mean absolute attribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Time importance plot saved to: {output_path}")
    return output_path