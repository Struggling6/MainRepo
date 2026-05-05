import argparse
from pathlib import Path

import pandas as pd


def make_balanced_dataset(
    input_path: str,
    output_path: str,
    target_col: str = "anomaly",
    seed: int = 42,
    normal_multiplier: float = 1.0,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    normal_df = df[df[target_col] == 0]
    anomaly_df = df[df[target_col] == 1]

    n_anomalies = len(anomaly_df)
    n_normals = int(n_anomalies * normal_multiplier)

    if n_anomalies == 0:
        raise ValueError("No anomalies found in dataset.")

    if len(normal_df) < n_normals:
        raise ValueError(
            f"Not enough normal samples. Need {n_normals}, "
            f"but only found {len(normal_df)}."
        )

    sampled_normal_df = normal_df.sample(
        n=n_normals,
        random_state=seed,
        replace=False,
    )

    balanced_df = pd.concat(
        [sampled_normal_df, anomaly_df],
        axis=0,
    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(output_path, index=False)

    print("\n=== Balanced dataset created ===")
    print(f"Output:          {output_path}")
    print(f"Original rows:   {len(df)}")
    print(f"Balanced rows:   {len(balanced_df)}")
    print(f"Normal rows:     {(balanced_df[target_col] == 0).sum()}")
    print(f"Anomaly rows:    {(balanced_df[target_col] == 1).sum()}")
    print(f"Anomaly rate:    {balanced_df[target_col].mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to original CSV file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to balanced output CSV",
    )
    parser.add_argument(
        "--target",
        default="anomaly",
        help="Target column name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--normal-multiplier",
        type=float,
        default=1.0,
        help="How many normal samples per anomaly. 1.0 gives 50/50 balance.",
    )

    args = parser.parse_args()

    make_balanced_dataset(
        input_path=args.input,
        output_path=args.output,
        target_col=args.target,
        seed=args.seed,
        normal_multiplier=args.normal_multiplier,
    )


if __name__ == "__main__":
    main()