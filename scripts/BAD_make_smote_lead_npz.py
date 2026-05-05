import argparse
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE

from data.lead_csv import LeadCSVHandler
from local_experiment import CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="datasets/LEAD/lead_smote_windows.npz",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=float,
        default=1.0,
        help="Minority/majority ratio after SMOTE. 1.0 = fully balanced.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[SMOTE] Loading LEAD data through LeadCSVHandler...")
    handler = LeadCSVHandler(CONFIG)

    train_loader, val_loader = handler.get_dataloaders(partition_id=0)

    X_train = train_loader.dataset.tensors[0].numpy()
    y_train = train_loader.dataset.tensors[1].numpy().astype(int)

    X_val = val_loader.dataset.tensors[0].numpy()
    y_val = val_loader.dataset.tensors[1].numpy().astype(int)

    print("[SMOTE] Original shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  y_val:   {y_val.shape}")

    print("[SMOTE] Original class balance:")
    print(f"  train anomaly rate: {y_train.mean():.4f}")
    print(f"  val anomaly rate:   {y_val.mean():.4f}")

    original_shape = X_train.shape
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    smote = SMOTE(
        sampling_strategy=args.sampling_strategy,
        random_state=args.seed,
        k_neighbors=5,
    )

    print("[SMOTE] Applying SMOTE to training set only...")
    X_train_smote_flat, y_train_smote = smote.fit_resample(
        X_train_flat,
        y_train,
    )

    X_train_smote = X_train_smote_flat.reshape(
        X_train_smote_flat.shape[0],
        original_shape[1],
        original_shape[2],
    ).astype(np.float32)

    y_train_smote = y_train_smote.astype(np.int64)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X_train=X_train_smote,
        y_train=y_train_smote,
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(np.int64),
    )

    print("\n[SMOTE] Saved:")
    print(f"  {output_path}")

    print("\n[SMOTE] New class balance:")
    print(f"  X_train: {X_train_smote.shape}")
    print(f"  y_train: {y_train_smote.shape}")
    print(f"  train anomaly rate: {y_train_smote.mean():.4f}")
    print(f"  val anomaly rate:   {y_val.mean():.4f}")


if __name__ == "__main__":
    main()