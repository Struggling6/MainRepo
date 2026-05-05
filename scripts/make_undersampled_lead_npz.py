import argparse
from pathlib import Path

import numpy as np

from data.lead_csv import LeadCSVHandler
from local_experiment import CONFIG


def undersample_normals(X, y, normal_to_anomaly_ratio=1.0, seed=42):
    rng = np.random.default_rng(seed)

    y = y.astype(int)

    anomaly_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    n_anomalies = len(anomaly_idx)
    n_normals_to_keep = int(n_anomalies * normal_to_anomaly_ratio)

    if n_anomalies == 0:
        raise ValueError("No anomalies found.")

    if len(normal_idx) < n_normals_to_keep:
        raise ValueError(
            f"Not enough normal samples. Need {n_normals_to_keep}, "
            f"but only found {len(normal_idx)}."
        )

    sampled_normal_idx = rng.choice(
        normal_idx,
        size=n_normals_to_keep,
        replace=False,
    )

    selected_idx = np.concatenate([anomaly_idx, sampled_normal_idx])
    rng.shuffle(selected_idx)

    return X[selected_idx], y[selected_idx]


def main():
    parser = argparse.ArgumentParser(
        description="Create undersampled LEAD train/val NPZ dataset."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output NPZ path.",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help=(
            "Normal-to-anomaly ratio. "
            "1.0 = 1 normal per anomaly, "
            "2.0 = 2 normals per anomaly, "
            "5.0 = 5 normals per anomaly."
        ),
    )

    parser.add_argument(
        "--balance-val",
        action="store_true",
        help="Also undersample validation set. Use for paper-style balanced evaluation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    print("[UNDER] Loading LEAD data through LeadCSVHandler...")
    handler = LeadCSVHandler(CONFIG)

    train_loader, val_loader = handler.get_dataloaders(partition_id=0)

    X_train = train_loader.dataset.tensors[0].numpy()
    y_train = train_loader.dataset.tensors[1].numpy().astype(int)

    X_val = val_loader.dataset.tensors[0].numpy()
    y_val = val_loader.dataset.tensors[1].numpy().astype(int)

    print("\n[UNDER] Original class balance:")
    print(f"  X_train: {X_train.shape}")
    print(f"  train anomaly rate: {y_train.mean():.4f}")
    print(f"  train anomalies: {y_train.sum()} / {len(y_train)}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  val anomaly rate:   {y_val.mean():.4f}")
    print(f"  val anomalies:   {y_val.sum()} / {len(y_val)}")

    print(f"\n[UNDER] Undersampling train set with ratio {args.ratio}:1")
    X_train_under, y_train_under = undersample_normals(
        X_train,
        y_train,
        normal_to_anomaly_ratio=args.ratio,
        seed=args.seed,
    )

    if args.balance_val:
        print(f"[UNDER] Undersampling validation set with ratio {args.ratio}:1")
        X_val_out, y_val_out = undersample_normals(
            X_val,
            y_val,
            normal_to_anomaly_ratio=args.ratio,
            seed=args.seed + 1,
        )
    else:
        print("[UNDER] Keeping validation set realistic/unchanged")
        X_val_out, y_val_out = X_val, y_val

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X_train=X_train_under.astype(np.float32),
        y_train=y_train_under.astype(np.int64),
        X_val=X_val_out.astype(np.float32),
        y_val=y_val_out.astype(np.int64),
    )

    print("\n[UNDER] Saved:")
    print(f"  {output_path}")

    print("\n[UNDER] New class balance:")
    print(f"  X_train: {X_train_under.shape}")
    print(f"  train anomaly rate: {y_train_under.mean():.4f}")
    print(f"  train anomalies: {y_train_under.sum()} / {len(y_train_under)}")
    print(f"  X_val:   {X_val_out.shape}")
    print(f"  val anomaly rate:   {y_val_out.mean():.4f}")
    print(f"  val anomalies:   {y_val_out.sum()} / {len(y_val_out)}")


if __name__ == "__main__":
    main()