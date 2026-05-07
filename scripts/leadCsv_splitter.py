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
    python split_lead.py --num-parts 10 --output-dir ./datasets/LEAD

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


INPUT_FILENAME = "train_features_clean.csv"


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
        required=True,
        help="Directory where data1.csv ... dataN.csv will be saved",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
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
    parser.add_argument(
        "--row-weight",
        type=float,
        default=3.0,
        help="Weight for balancing row counts across partitions.",
    )
    parser.add_argument(
        "--anomaly-weight",
        type=float,
        default=1.0,
        help="Weight for balancing anomaly counts across partitions.",
    )
    parser.add_argument(
        "--building-weight",
        type=float,
        default=0.5,
        help="Weight for balancing building counts across partitions.",
    )
    return parser.parse_args()


def resolve_input_file(input_file_arg: Path | None) -> Path:
    if input_file_arg is not None:
        input_path = input_file_arg.resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        input_path = script_dir / INPUT_FILENAME

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        raise SystemExit(1)

    return input_path


def assign_buildings_to_partitions(
    df: pd.DataFrame,
    num_parts: int,
    target_col: str,
    seed: int,
    row_weight: float = 3.0,
    anomaly_weight: float = 1.0,
    building_weight: float = 0.5,
) -> list[pd.DataFrame]:
    """
    Assign each building_id to exactly one partition.

    Strategy:
    - Count rows and anomalies per building_id
    - Sort anomaly-heavy/larger buildings first
    - Seed each partition with one building
    - Greedily place remaining buildings into the currently smallest row-count
      partitions, using anomalies/building counts as tie-breakers

    This follows the shared-mode LEAD partitioning idea, but gives row balance
    more weight so local-mode shards have comparable memory/preprocessing cost.
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
    building_stats["_tie_break"] = rng.random(len(building_stats))

    building_stats = building_stats.sort_values(
        by=["num_anomalies", "num_rows", "_tie_break"],
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

    ideal_rows = total_rows / num_parts
    ideal_anomalies = total_anomalies / num_parts if total_anomalies > 0 else 0.0
    ideal_buildings = total_buildings / num_parts

    partitions = [
        {
            "buildings": [],
            "num_rows": 0,
            "num_anomalies": 0,
            "num_buildings": 0,
        }
        for _ in range(num_parts)
    ]

    def score_partition_after_assignment(partition, rows_to_add, anomalies_to_add):
        new_rows = partition["num_rows"] + rows_to_add
        new_anomalies = partition["num_anomalies"] + anomalies_to_add
        new_buildings = partition["num_buildings"] + 1

        if total_anomalies > 0:
            anomaly_score = (
                (new_anomalies - ideal_anomalies)
                / max(ideal_anomalies, 1.0)
            ) ** 2
        else:
            anomaly_score = 0.0

        building_score = (
            (new_buildings - ideal_buildings)
            / max(ideal_buildings, 1.0)
        ) ** 2

        return (
            row_weight * (new_rows / max(ideal_rows, 1.0))
            + anomaly_weight * anomaly_score
            + building_weight * building_score
        )

    remaining_buildings = building_stats.copy()

    for partition_idx in range(num_parts):
        row = remaining_buildings.iloc[0]
        remaining_buildings = remaining_buildings.iloc[1:].reset_index(drop=True)

        building_id = row["building_id"]
        rows = int(row["num_rows"])
        anomalies = int(row["num_anomalies"])

        partitions[partition_idx]["buildings"].append(building_id)
        partitions[partition_idx]["num_rows"] += rows
        partitions[partition_idx]["num_anomalies"] += anomalies
        partitions[partition_idx]["num_buildings"] += 1

    for _, row in remaining_buildings.iterrows():
        building_id = row["building_id"]
        rows = int(row["num_rows"])
        anomalies = int(row["num_anomalies"])

        min_rows = min(partition["num_rows"] for partition in partitions)
        row_candidates = [
            idx
            for idx, partition in enumerate(partitions)
            if partition["num_rows"] == min_rows
        ]

        best_partition_idx = min(
            row_candidates,
            key=lambda idx: score_partition_after_assignment(
                partitions[idx],
                rows,
                anomalies,
            ),
        )

        partitions[best_partition_idx]["buildings"].append(building_id)
        partitions[best_partition_idx]["num_rows"] += rows
        partitions[best_partition_idx]["num_anomalies"] += anomalies
        partitions[best_partition_idx]["num_buildings"] += 1

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

    input_path = resolve_input_file(args.input_file)
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
        row_weight=args.row_weight,
        anomaly_weight=args.anomaly_weight,
        building_weight=args.building_weight,
    )
    verify_no_building_overlap(parts)
    write_parts(parts, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
