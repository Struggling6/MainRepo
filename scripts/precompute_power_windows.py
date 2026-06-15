#!/usr/bin/env python3
"""
Precompute PowerConsumptionAnomaly local-mode temporal windows.

This is the power-dataset equivalent of scripts/precompute_lead_windows.py.
It expects already split local client files, usually produced by:

    python scripts/powerCsv_splitter.py --num-parts 5

Example:
    python scripts/precompute_power_windows.py --num-clients 5 --overwrite

Output:
    fl-backend/datasets/PowerConsumptionAnomaly/windowed/client1.npz
    ...
    fl-backend/datasets/PowerConsumptionAnomaly/windowed/client5.npz
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "fl-backend"
DEFAULT_SPLIT_DIR = Path("datasets/PowerConsumptionAnomaly/partitions")
DEFAULT_OUTPUT_DIR = Path("datasets/PowerConsumptionAnomaly/windowed")
DEFAULT_FILE_PATTERN = "data{client_index}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute power anomaly windows into per-client .npz files."
    )

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
        help="Directory containing data1.csv ... dataN.csv. Defaults to the power partitions directory.",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default=DEFAULT_FILE_PATTERN,
        help="Pattern for split CSVs. Defaults to data{client_index}.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for client1.npz ... clientN.npz. Defaults to the power windowed directory.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override CONFIG.data.window_size.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override CONFIG.data.stride.",
    )
    parser.add_argument(
        "--gap-hours",
        type=int,
        default=None,
        help="Override CONFIG.data.gap_hours.",
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
        "--use-undersampling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override CONFIG.data.use_undersampling.",
    )
    parser.add_argument(
        "--undersampling-ratio",
        type=float,
        default=None,
        help="Override CONFIG.data.undersampling_ratio.",
    )
    parser.add_argument(
        "--undersample-val",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override CONFIG.data.undersample_val.",
    )
    parser.add_argument(
        "--use-oversampling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override CONFIG.data.use_oversampling.",
    )
    parser.add_argument(
        "--oversampling-method",
        choices=(
            "none",
            "random_over",
            "smote",
            "time_series_augment",
            "ts_augment",
        ),
        default=None,
        help="Override CONFIG.data.oversampling_method.",
    )
    parser.add_argument(
        "--oversampling-ratio",
        type=float,
        default=None,
        help="Override CONFIG.data.oversampling_ratio. 1.0 means positives = negatives.",
    )
    parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=None,
        help="Override CONFIG.data.smote_k_neighbors.",
    )
    parser.add_argument(
        "--oversample-val",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override CONFIG.data.oversample_val.",
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


def _count_labels(y: np.ndarray) -> tuple[int, int, float]:
    y = y.astype(np.int64, copy=False)
    positives = int(y.sum())
    total = int(len(y))
    negatives = total - positives
    rate = float(positives / total) if total > 0 else 0.0
    return positives, negatives, rate


def _build_power_config():
    sys.path.insert(0, str(BACKEND_DIR))

    from config import PowerConsumptionAnomalyConfig
    from local_experiment import CONFIG

    config = copy.deepcopy(CONFIG)

    if getattr(config.data, "name", None) != "power_consumption_anomaly":
        config.data = PowerConsumptionAnomalyConfig()

    return config


def _apply_cli_overrides(config, args: argparse.Namespace) -> None:
    if args.num_clients is not None:
        config.federation.num_clients = args.num_clients

    config.federation.partition_mode = "local"
    config.data.use_precomputed_windows = False

    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
    else:
        config.data.data_dir = DEFAULT_SPLIT_DIR

    config.data.file_pattern = args.file_pattern

    if args.output_dir is not None:
        config.data.precomputed_dir = args.output_dir
    else:
        config.data.precomputed_dir = DEFAULT_OUTPUT_DIR

    config.data.data_dir = _resolve_backend_path(Path(config.data.data_dir))
    config.data.precomputed_dir = _resolve_backend_path(
        Path(config.data.precomputed_dir)
    )

    if args.window_size is not None:
        config.data.window_size = args.window_size

    if args.stride is not None:
        config.data.stride = args.stride

    if args.gap_hours is not None:
        config.data.gap_hours = args.gap_hours

    if args.use_undersampling is not None:
        config.data.use_undersampling = args.use_undersampling

    if args.undersampling_ratio is not None:
        config.data.undersampling_ratio = args.undersampling_ratio

    if args.undersample_val is not None:
        config.data.undersample_val = args.undersample_val

    if args.use_oversampling is not None:
        config.data.use_oversampling = args.use_oversampling

    if args.oversampling_method is not None:
        config.data.oversampling_method = args.oversampling_method

    if args.oversampling_ratio is not None:
        config.data.oversampling_ratio = args.oversampling_ratio

    if args.smote_k_neighbors is not None:
        config.data.smote_k_neighbors = args.smote_k_neighbors

    if args.oversample_val is not None:
        config.data.oversample_val = args.oversample_val

    if not 0 < config.data.oversampling_ratio <= 1:
        raise ValueError("--oversampling-ratio must be in the range (0, 1]")

    if config.data.smote_k_neighbors < 1:
        raise ValueError("--smote-k-neighbors must be at least 1")


def main() -> None:
    args = parse_args()
    config = _build_power_config()
    _apply_cli_overrides(config, args)

    output_dir = Path(config.data.precomputed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handler = config.data.build_handler(config)
    save_fn = np.savez_compressed if args.compressed else np.savez
    num_clients = config.federation.num_clients

    print(f"Precomputing power artifacts for {num_clients} client(s)")
    print(f"Input split directory: {Path(config.data.data_dir)}")
    print(f"Input file pattern: {config.data.file_pattern}")
    print(f"Output artifact directory: {output_dir}")
    print(
        "Window config: "
        f"window_size={config.data.window_size}, "
        f"stride={config.data.stride}, "
        f"gap_hours={config.data.gap_hours}"
    )
    print(
        "Balancing config: "
        f"use_undersampling={config.data.use_undersampling}, "
        f"undersampling_ratio={config.data.undersampling_ratio}, "
        f"undersample_val={config.data.undersample_val}, "
        f"use_oversampling={config.data.use_oversampling}, "
        f"oversampling_method={config.data.oversampling_method}, "
        f"oversampling_ratio={config.data.oversampling_ratio}, "
        f"smote_k_neighbors={config.data.smote_k_neighbors}, "
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

        train_pos, train_neg, train_rate = _count_labels(y_train)
        val_pos, val_neg, val_rate = _count_labels(y_val)

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
            f"train_pos={train_pos}, train_neg={train_neg}, train_rate={train_rate:.4f}, "
            f"X_val={X_val.shape}, y_val={y_val.shape}, "
            f"val_pos={val_pos}, val_neg={val_neg}, val_rate={val_rate:.4f}, "
            f"aggregation_weight={aggregation_weight}"
        )

        save_fn(
            output_path,
            X_train=X_train.astype(np.float32, copy=False),
            y_train=y_train.astype(np.int64, copy=False),
            X_val=X_val.astype(np.float32, copy=False),
            y_val=y_val.astype(np.int64, copy=False),
            feature_cols=np.array(handler.feature_cols),
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
            use_undersampling=np.array([config.data.use_undersampling], dtype=np.bool_),
            undersampling_ratio=np.array(
                [config.data.undersampling_ratio],
                dtype=np.float32,
            ),
            undersample_val=np.array([config.data.undersample_val], dtype=np.bool_),
            use_oversampling=np.array([config.data.use_oversampling], dtype=np.bool_),
            oversampling_method=np.array([config.data.oversampling_method]),
            oversampling_ratio=np.array(
                [config.data.oversampling_ratio],
                dtype=np.float32,
            ),
            smote_k_neighbors=np.array(
                [config.data.smote_k_neighbors],
                dtype=np.int64,
            ),
            oversample_val=np.array([config.data.oversample_val], dtype=np.bool_),
            num_train_windows_after_oversampling=np.array(
                [len(y_train)],
                dtype=np.int64,
            ),
            num_train_anomalies_after_oversampling=np.array(
                [train_pos],
                dtype=np.int64,
            ),
            num_val_windows_after_oversampling=np.array(
                [len(y_val)],
                dtype=np.int64,
            ),
            num_val_anomalies_after_oversampling=np.array(
                [val_pos],
                dtype=np.int64,
            ),
        )

        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
