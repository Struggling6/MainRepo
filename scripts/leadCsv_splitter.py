#!/usr/bin/env python3
"""
Split the LEAD training CSV into multiple client CSV files by whole building_id groups.

Default input is `train_features_clean.csv` (produced by
scripts/clean_lead_features.py). Override with --input-file to operate on the
raw `train_features.csv` if needed.

Assumptions:
- input file is in the same directory as this script unless --input-file is given
- output folder is provided by the user
- each building_id is placed in exactly one output file

Example:
    python scripts/leadCsv_splitter.py --num-parts 10 --output-dir .scripts/datasets/LEAD

Result:
    ./datasets/LEAD/data1.csv
    ./datasets/LEAD/data2.csv
    ...
    ./datasets/LEAD/data10.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILENAME = Path("fl-backend/datasets/LEAD/train_features_clean.csv")
OUTPUT_DIR = Path("fl-backend/datasets/LEAD/partitions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-parts",
        type=int,
        required=True,
        help="Number of output CSV files to create",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where data1.csv ... dataN.csv will be saved",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=INPUT_FILENAME,
        help="Optional explicit path to input CSV. Defaults to train_features_clean.csv in the same folder as this script.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="anomaly",
        help="Target column used to balance anomaly counts across partitions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used only to break ties between otherwise similar buildings.",
    )
    return parser.parse_args()

def assign_buildings_to_partitions(
    df: pd.DataFrame,
    num_parts: int,
    target_col: str,
    seed: int,
) -> list[pd.DataFrame]:
    """
    Assign each building_id to exactly one partition.

    Strategy: sort buildings by anomaly rate descending, then round-robin
    assign through partitions (0,1,...,K-1,0,1,...). Each partition gets
    every K-th building in the sorted order, balancing both building counts
    and anomaly rates across partitions.

    Mirrors the shared-mode partitioning in fl-backend/data/lead_csv.py so
    local-mode shards behave the same as shared-mode partitions.
    """
    if "building_id" not in df.columns:
        print("Error: 'building_id' column not found in input CSV.", file=sys.stderr)
        raise SystemExit(1)

    if target_col not in df.columns:
        print(f"Error: '{target_col}' column not found in input CSV.", file=sys.stderr)
        raise SystemExit(1)

    rng = np.random.default_rng(seed)

    building_stats = (
        df.groupby("building_id")
        .agg(
            num_rows=(target_col, "size"),
            num_anomalies=(target_col, "sum"),
        )
        .reset_index()
    )

    building_stats["num_rows"] = building_stats["num_rows"].astype(int)
    building_stats["num_anomalies"] = building_stats["num_anomalies"].astype(int)
    building_stats["anomaly_rate"] = (
        building_stats["num_anomalies"] / building_stats["num_rows"]
    ).fillna(0.0)
    building_stats["_tie_break"] = rng.random(len(building_stats))

    building_stats = building_stats.sort_values(
        by=["anomaly_rate", "num_rows", "_tie_break"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    total_rows = int(building_stats["num_rows"].sum())
    total_anomalies = int(building_stats["num_anomalies"].sum())
    total_buildings = int(len(building_stats))

    if num_parts > total_buildings:
        print(
            f"Error: --num-parts={num_parts} is greater than the number of "
            f"available buildings={total_buildings}. This would create empty files.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    partitions = [
        {
            "buildings": [],
            "num_rows": 0,
            "num_anomalies": 0,
            "num_buildings": 0,
        }
        for _ in range(num_parts)
    ]

    # Round-robin by anomaly rate: walk buildings from highest anomaly rate to
    # lowest, cycling through partitions (0,1,...,K-1,0,1,...). Each partition
    # gets every K-th building in the sorted order, guaranteeing balanced
    # building counts and a representative mix of rates.
    for i, (_, row) in enumerate(building_stats.iterrows()):
        partition_idx = i % num_parts
        partitions[partition_idx]["buildings"].append(row["building_id"])
        partitions[partition_idx]["num_rows"] += int(row["num_rows"])
        partitions[partition_idx]["num_anomalies"] += int(row["num_anomalies"])
        partitions[partition_idx]["num_buildings"] += 1

    print("Partition summary:")
    print(
        f"total_buildings={total_buildings}, "
        f"total_rows={total_rows}, total_anomalies={total_anomalies}"
    )

    parts: list[pd.DataFrame] = []
    for i, partition in enumerate(partitions, start=1):
        building_ids = partition["buildings"]
        part_df = df[df["building_id"].isin(building_ids)].copy()
        parts.append(part_df)

        rows = partition["num_rows"]
        anomalies = partition["num_anomalies"]
        anomaly_rate = anomalies / rows if rows > 0 else 0.0
        print(
            f"Partition {i}: "
            f"{rows} rows, "
            f"{partition['num_buildings']} building_ids, "
            f"{anomalies} anomalies, "
            f"anomaly_rate={anomaly_rate:.4f}"
        )

    zero_anomaly_parts = [
        idx
        for idx, partition in enumerate(partitions, start=1)
        if partition["num_anomalies"] == 0
    ]

    if zero_anomaly_parts:
        print(
            "WARNING: Some partitions received zero anomaly rows: "
            f"{zero_anomaly_parts}. This may hurt federated anomaly detection."
        )

    return parts


def write_parts(parts: list[pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove old split files first
    for old_file in output_dir.glob("data*.csv"):
        old_file.unlink()

    for idx, part in enumerate(parts, start=1):
        output_file = output_dir / f"data{idx}.csv"
        part.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")


def verify_no_building_overlap(parts: list[pd.DataFrame]) -> None:
    """
    Verify that no building_id appears in more than one part.
    """
    seen: dict[int, int] = {}

    for part_idx, part in enumerate(parts, start=1):
        if part.empty:
            continue

        for building_id in part["building_id"].unique():
            if building_id in seen:
                print(
                    f"Error: building_id {building_id} appears in both "
                    f"part {seen[building_id]} and part {part_idx}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            seen[building_id] = part_idx

    print("Verification passed: each building_id appears in exactly one file.")


def main() -> None:
    args = parse_args()

    if args.num_parts < 1:
        print("--num-parts must be at least 1", file=sys.stderr)
        raise SystemExit(1)

    input_path = args.input_file.resolve()
    output_dir = args.output_dir.resolve()

    print(f"Reading input file: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Loaded shape: {df.shape}")
    print(f"Saving split files to: {output_dir}")

    parts = assign_buildings_to_partitions(
        df,
        args.num_parts,
        target_col=args.target_col,
        seed=args.seed,
    )
    verify_no_building_overlap(parts)
    write_parts(parts, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
