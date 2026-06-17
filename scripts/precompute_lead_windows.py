#!/usr/bin/env python3
"""
Precompute LEAD local-mode temporal windows into per-client .npz artifacts.

Now supports the 60/20/20 split:
  - train split: used for training, may be undersampled/oversampled
  - val split: used during training/validation, may optionally be sampled
  - eval split: held-out final evaluation, never sampled

Output:
    datasets/LEAD/windowed/client1.npz
    ...
    datasets/LEAD/windowed/clientN.npz
    datasets/LEAD/windowed/eval.npz
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

    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--use-undersampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--undersampling-ratio", type=float, default=None)
    parser.add_argument("--undersample-val", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--use-oversampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--oversampling-method",
        choices=("none", "random_over", "smote", "borderline_smote", "time_series_augment", "ts_augment"),
        default=None,
    )
    parser.add_argument("--oversampling-ratio", type=float, default=None)
    parser.add_argument("--smote-k-neighbors", type=int, default=None)
    parser.add_argument("--oversample-val", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument(
        "--eval-filename",
        type=str,
        default="eval.npz",
        help="Name of combined held-out eval artifact.",
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


def _apply_cli_overrides(config, args: argparse.Namespace) -> None:
    if args.num_clients is not None:
        config.federation.num_clients = args.num_clients

    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
    else:
        config.data.data_dir = _resolve_backend_path(Path(config.data.data_dir))

    if args.output_dir is not None:
        config.data.precomputed_dir = args.output_dir
    else:
        config.data.precomputed_dir = _resolve_backend_path(Path(config.data.precomputed_dir))

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

    sys.path.insert(0, str(BACKEND_DIR))

    from local_experiment import CONFIG

    config = copy.deepcopy(CONFIG)

    config.federation.partition_mode = "local"
    config.data.use_precomputed_windows = False

    _apply_cli_overrides(config, args)

    output_dir = Path(config.data.precomputed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_path = output_dir / args.eval_filename

    if eval_output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Combined eval artifact already exists: {eval_output_path}. "
            "Use --overwrite to replace it."
        )

    handler = config.data.build_handler(config)

    save_fn = np.savez_compressed if args.compressed else np.savez
    num_clients = config.federation.num_clients

    all_X_eval = []
    all_y_eval = []

    print(f"Precomputing LEAD artifacts for {num_clients} client(s)")
    print(f"Input split directory: {Path(config.data.data_dir)}")
    print(f"Output artifact directory: {output_dir}")
    print(f"Combined eval artifact: {eval_output_path}")

    print(
        "Split config: "
        f"train_split={config.data.train_split}, "
        f"val_split={config.data.val_split}, "
        f"eval_split={config.data.eval_split}"
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

        if not hasattr(handler, "get_eval_loader"):
            raise AttributeError(
                "LeadCSVHandler must define get_eval_loader() for precomputing eval windows."
            )

        evalloader = handler.get_eval_loader()

        X_train, y_train = _dataset_to_numpy(trainloader)
        X_val, y_val = _dataset_to_numpy(valloader)
        X_eval, y_eval = _dataset_to_numpy(evalloader)

        all_X_eval.append(X_eval)
        all_y_eval.append(y_eval)

        train_pos, train_neg, train_rate = _count_labels(y_train)
        val_pos, val_neg, val_rate = _count_labels(y_val)
        eval_pos, eval_neg, eval_rate = _count_labels(y_eval)

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
            f"X_eval={X_eval.shape}, y_eval={y_eval.shape}, "
            f"eval_pos={eval_pos}, eval_neg={eval_neg}, eval_rate={eval_rate:.4f}, "
            f"aggregation_weight={aggregation_weight}"
        )

        save_fn(
            output_path,
            X_train=X_train.astype(np.float32, copy=False),
            y_train=y_train.astype(np.int64, copy=False),
            X_val=X_val.astype(np.float32, copy=False),
            y_val=y_val.astype(np.int64, copy=False),
            X_eval=X_eval.astype(np.float32, copy=False),
            y_eval=y_eval.astype(np.int64, copy=False),

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
            undersampling_ratio=np.array([config.data.undersampling_ratio], dtype=np.float32),
            undersample_val=np.array([config.data.undersample_val], dtype=np.bool_),

            use_oversampling=np.array([config.data.use_oversampling], dtype=np.bool_),
            oversampling_method=np.array([config.data.oversampling_method]),
            oversampling_ratio=np.array([config.data.oversampling_ratio], dtype=np.float32),
            smote_k_neighbors=np.array([config.data.smote_k_neighbors], dtype=np.int64),
            oversample_val=np.array([config.data.oversample_val], dtype=np.bool_),

            num_train_windows_after_oversampling=np.array([len(y_train)], dtype=np.int64),
            num_train_anomalies_after_oversampling=np.array([train_pos], dtype=np.int64),
            num_val_windows_after_oversampling=np.array([len(y_val)], dtype=np.int64),
            num_val_anomalies_after_oversampling=np.array([val_pos], dtype=np.int64),

            num_eval_windows=np.array([len(y_eval)], dtype=np.int64),
            num_eval_anomalies=np.array([eval_pos], dtype=np.int64),
        )

        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    if not all_X_eval:
        raise RuntimeError("No eval windows were generated.")

    X_eval_all = np.concatenate(all_X_eval, axis=0)
    y_eval_all = np.concatenate(all_y_eval, axis=0)

    eval_pos, eval_neg, eval_rate = _count_labels(y_eval_all)

    print(
        "\nSaving combined held-out eval artifact "
        f"X_eval={X_eval_all.shape}, y_eval={y_eval_all.shape}, "
        f"eval_pos={eval_pos}, eval_neg={eval_neg}, eval_rate={eval_rate:.4f}"
    )

    save_fn(
        eval_output_path,
        X_eval=X_eval_all.astype(np.float32, copy=False),
        y_eval=y_eval_all.astype(np.int64, copy=False),
        input_dim=np.array([X_eval_all.shape[-1]], dtype=np.int64),
        window_size=np.array([X_eval_all.shape[1]], dtype=np.int64),
        num_eval_windows=np.array([len(y_eval_all)], dtype=np.int64),
        num_eval_anomalies=np.array([eval_pos], dtype=np.int64),
    )

    size_mb = eval_output_path.stat().st_size / 1024 / 1024
    print(f"Saved combined eval: {eval_output_path} ({size_mb:.1f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()