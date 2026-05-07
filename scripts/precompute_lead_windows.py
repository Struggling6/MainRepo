#!/usr/bin/env python3
"""
Precompute LEAD local-mode temporal windows into per-client .npz artifacts.

This moves the expensive CSV read/preprocess/window/scale/undersample pipeline
out of Flower client startup. Run this after creating data1.csv ... dataN.csv
with scripts/leadCsv_splitter.py and before `flwr run`.

Example:
    python scripts/precompute_lead_windows.py --num-clients 5
    python scripts/precompute_lead_windows.py --num-clients 5 --overwrite --oversampling-method random_over --oversampling-ratio 1.0

Output:
    fl-backend/datasets/LEAD/windowed/client1.npz
    ...
    fl-backend/datasets/LEAD/windowed/client5.npz
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "fl-backend"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-clients",
        type=int,
        default=None,
        help="Number of client artifacts to create. Defaults to CONFIG.federation.num_clients.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing data1.csv ... dataN.csv. Defaults to CONFIG.data.data_dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for client1.npz ... clientN.npz. Defaults to CONFIG.data.precomputed_dir.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed. Smaller files, slower precompute/load.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing client artifacts.",
    )
    parser.add_argument(
        "--oversampling-method",
        choices=("none", "random_over", "smote"),
        default=None,
        help=(
            "Optional oversampling after undersampling. "
            "Overrides CONFIG.data.oversampling_method."
        ),
    )
    parser.add_argument(
        "--oversampling-ratio",
        type=float,
        default=None,
        help=(
            "Target positive:negative ratio for oversampling. "
            "1.0 means balance positives and negatives."
        ),
    )
    parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=None,
        help="Number of SMOTE neighbors. Automatically capped by available positives.",
    )
    parser.add_argument(
        "--undersample-val",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override CONFIG.data.undersample_val for paper-like validation balancing.",
    )
    parser.add_argument(
        "--oversample-val",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply the selected oversampler to validation artifacts too.",
    )
    return parser.parse_args()


def _dataset_to_numpy(loader):
    dataset = loader.dataset

    if hasattr(dataset, "tensors"):
        X, y = dataset.tensors
        return X.cpu().numpy(), y.cpu().numpy()

    X_parts, y_parts = [], []
    for X_batch, y_batch in loader:
        X_parts.append(X_batch.cpu().numpy())
        y_parts.append(y_batch.cpu().numpy())

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def _resolve_backend_path(path: Path) -> Path:
    return path if path.is_absolute() else BACKEND_DIR / path


def _apply_oversampling(
    X: np.ndarray,
    y: np.ndarray,
    *,
    split_name: str,
    method: str,
    target_ratio: float,
    smote_k_neighbors: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | str]]:
    y = y.astype(np.int64, copy=False)
    num_before = int(len(y))
    positives_before = int(y.sum())
    negatives_before = int(num_before - positives_before)

    metadata: dict[str, int | float | str] = {
        "oversampling_method": method,
        "oversampling_ratio": float(target_ratio),
        "smote_k_neighbors": int(smote_k_neighbors),
        f"num_{split_name}_windows_before_oversampling": num_before,
        f"num_{split_name}_anomalies_before_oversampling": positives_before,
        f"num_{split_name}_windows_after_oversampling": num_before,
        f"num_{split_name}_anomalies_after_oversampling": positives_before,
    }

    if method == "none":
        return X, y, metadata

    if positives_before == 0 or negatives_before == 0:
        print(
            "[LEAD] WARNING: Oversampling requires both classes; "
            "keeping training data unchanged"
        )
        return X, y, metadata

    current_ratio = positives_before / negatives_before
    if target_ratio <= current_ratio:
        print(
            "[LEAD] Oversampling skipped: "
            f"current positive:negative ratio={current_ratio:.4f} "
            f"is already >= target={target_ratio:.4f}"
        )
        return X, y, metadata

    X_shape = X.shape
    X_flat = X.reshape(X_shape[0], -1)

    if method == "random_over":
        from imblearn.over_sampling import RandomOverSampler

        sampler = RandomOverSampler(
            sampling_strategy=target_ratio,
            random_state=seed,
        )
    elif method == "smote":
        if positives_before < 2:
            print(
                "[LEAD] WARNING: SMOTE requires at least two positive samples; "
                "keeping training data unchanged"
            )
            return X, y, metadata

        from imblearn.over_sampling import SMOTE

        effective_k = min(smote_k_neighbors, positives_before - 1)
        metadata["smote_k_neighbors"] = int(effective_k)
        sampler = SMOTE(
            sampling_strategy=target_ratio,
            random_state=seed,
            k_neighbors=effective_k,
        )
    else:
        raise ValueError(f"Unknown oversampling method: {method}")

    X_resampled, y_resampled = sampler.fit_resample(X_flat, y)
    X_resampled = X_resampled.reshape((-1, *X_shape[1:])).astype(
        np.float32,
        copy=False,
    )
    y_resampled = y_resampled.astype(np.int64, copy=False)

    metadata[f"num_{split_name}_windows_after_oversampling"] = int(len(y_resampled))
    metadata[f"num_{split_name}_anomalies_after_oversampling"] = int(y_resampled.sum())

    print(
        f"[LEAD] Applied {split_name} oversampling: "
        f"method={method}, target_pos_neg={target_ratio}:1, "
        f"before={num_before} pos={positives_before}, "
        f"after={len(y_resampled)} pos={int(y_resampled.sum())}, "
        f"anomaly_rate={float(y_resampled.mean()):.4f}"
    )

    return X_resampled, y_resampled, metadata


def main() -> None:
    args = parse_args()

    sys.path.insert(0, str(BACKEND_DIR))

    from local_experiment import CONFIG

    config = copy.deepcopy(CONFIG)
    config.federation.partition_mode = "local"
    config.data.use_precomputed_windows = False

    if not hasattr(config.data, "oversampling_method"):
        config.data.oversampling_method = "none"
    if not hasattr(config.data, "oversampling_ratio"):
        config.data.oversampling_ratio = 1.0
    if not hasattr(config.data, "smote_k_neighbors"):
        config.data.smote_k_neighbors = 5
    if not hasattr(config.data, "oversample_val"):
        config.data.oversample_val = False

    if args.oversampling_method is not None:
        config.data.oversampling_method = args.oversampling_method
    if args.oversampling_ratio is not None:
        config.data.oversampling_ratio = args.oversampling_ratio
    if args.smote_k_neighbors is not None:
        config.data.smote_k_neighbors = args.smote_k_neighbors
    if args.undersample_val is not None:
        config.data.undersample_val = args.undersample_val
    if args.oversample_val is not None:
        config.data.oversample_val = args.oversample_val

    if not 0 < config.data.oversampling_ratio <= 1:
        raise ValueError("--oversampling-ratio must be in the range (0, 1]")
    if config.data.smote_k_neighbors < 1:
        raise ValueError("--smote-k-neighbors must be at least 1")

    if args.num_clients is not None:
        config.federation.num_clients = args.num_clients

    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
    else:
        config.data.data_dir = _resolve_backend_path(Path(config.data.data_dir))

    if args.output_dir is not None:
        config.data.precomputed_dir = args.output_dir
    else:
        config.data.precomputed_dir = _resolve_backend_path(
            Path(config.data.precomputed_dir)
        )

    output_dir = Path(config.data.precomputed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handler = config.data.build_handler(config)

    save_fn = np.savez_compressed if args.compressed else np.savez
    num_clients = config.federation.num_clients

    print(f"Precomputing LEAD windows for {num_clients} client(s)")
    print(f"Input split directory: {Path(config.data.data_dir)}")
    print(f"Output artifact directory: {output_dir}")
    print(
        "Train oversampling: "
        f"method={config.data.oversampling_method}, "
        f"target_pos_neg={config.data.oversampling_ratio}:1, "
        f"smote_k_neighbors={config.data.smote_k_neighbors}, "
        f"undersample_val={config.data.undersample_val}, "
        f"oversample_val={config.data.oversample_val}"
    )

    for partition_id in range(num_clients):
        client_index = partition_id + 1
        output_path = output_dir / config.data.precomputed_pattern.format(
            client_index=client_index
        )

        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing artifact: {output_path}")
            continue

        print(f"\n=== Client {client_index}/{num_clients} ===")
        trainloader, valloader = handler.get_dataloaders(partition_id)

        X_train, y_train = _dataset_to_numpy(trainloader)
        X_val, y_val = _dataset_to_numpy(valloader)
        X_train, y_train, train_oversampling_metadata = _apply_oversampling(
            X_train,
            y_train,
            split_name="train",
            method=config.data.oversampling_method,
            target_ratio=config.data.oversampling_ratio,
            smote_k_neighbors=config.data.smote_k_neighbors,
            seed=config.data.seed + 20_000 + partition_id,
        )
        if config.data.oversample_val:
            X_val, y_val, val_oversampling_metadata = _apply_oversampling(
                X_val,
                y_val,
                split_name="val",
                method=config.data.oversampling_method,
                target_ratio=config.data.oversampling_ratio,
                smote_k_neighbors=config.data.smote_k_neighbors,
                seed=config.data.seed + 30_000 + partition_id,
            )
        else:
            val_oversampling_metadata = {
                "num_val_windows_before_oversampling": int(len(y_val)),
                "num_val_anomalies_before_oversampling": int(y_val.sum()),
                "num_val_windows_after_oversampling": int(len(y_val)),
                "num_val_anomalies_after_oversampling": int(y_val.sum()),
            }

        oversampling_metadata = {
            **train_oversampling_metadata,
            **val_oversampling_metadata,
        }
        aggregation_weight = int(handler.aggregation_weight)
        num_train_windows_before_undersampling = int(
            handler.num_train_windows_before_undersampling
        )
        num_train_anomalies_before_undersampling = int(
            handler.num_train_anomalies_before_undersampling
        )
        num_train_windows_after_undersampling = int(
            handler.num_train_windows_after_undersampling
        )

        print(
            "Saving "
            f"X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_val={X_val.shape}, y_val={y_val.shape}, "
            f"aggregation_weight={aggregation_weight}"
        )

        save_fn(
            output_path,
            X_train=X_train.astype(np.float32, copy=False),
            y_train=y_train.astype(np.int64, copy=False),
            X_val=X_val.astype(np.float32, copy=False),
            y_val=y_val.astype(np.int64, copy=False),
            input_dim=np.array([X_train.shape[-1]], dtype=np.int64),
            window_size=np.array([X_train.shape[1]], dtype=np.int64),
            aggregation_weight=np.array([aggregation_weight], dtype=np.int64),
            num_train_windows_before_undersampling=np.array(
                [num_train_windows_before_undersampling],
                dtype=np.int64,
            ),
            num_train_anomalies_before_undersampling=np.array(
                [num_train_anomalies_before_undersampling],
                dtype=np.int64,
            ),
            num_train_windows_after_undersampling=np.array(
                [num_train_windows_after_undersampling],
                dtype=np.int64,
            ),
            oversampling_method=np.array(
                [oversampling_metadata["oversampling_method"]],
            ),
            oversampling_ratio=np.array(
                [oversampling_metadata["oversampling_ratio"]],
                dtype=np.float32,
            ),
            smote_k_neighbors=np.array(
                [oversampling_metadata["smote_k_neighbors"]],
                dtype=np.int64,
            ),
            oversample_val=np.array([config.data.oversample_val], dtype=np.bool_),
            num_train_windows_before_oversampling=np.array(
                [oversampling_metadata["num_train_windows_before_oversampling"]],
                dtype=np.int64,
            ),
            num_train_anomalies_before_oversampling=np.array(
                [oversampling_metadata["num_train_anomalies_before_oversampling"]],
                dtype=np.int64,
            ),
            num_train_windows_after_oversampling=np.array(
                [oversampling_metadata["num_train_windows_after_oversampling"]],
                dtype=np.int64,
            ),
            num_train_anomalies_after_oversampling=np.array(
                [oversampling_metadata["num_train_anomalies_after_oversampling"]],
                dtype=np.int64,
            ),
            num_val_windows_before_oversampling=np.array(
                [oversampling_metadata["num_val_windows_before_oversampling"]],
                dtype=np.int64,
            ),
            num_val_anomalies_before_oversampling=np.array(
                [oversampling_metadata["num_val_anomalies_before_oversampling"]],
                dtype=np.int64,
            ),
            num_val_windows_after_oversampling=np.array(
                [oversampling_metadata["num_val_windows_after_oversampling"]],
                dtype=np.int64,
            ),
            num_val_anomalies_after_oversampling=np.array(
                [oversampling_metadata["num_val_anomalies_after_oversampling"]],
                dtype=np.int64,
            ),
        )

        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
