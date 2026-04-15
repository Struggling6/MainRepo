#!/usr/bin/env python3
"""
Split train_features.csv into multiple client CSV files by whole building_id groups.

Assumptions:
- train_features.csv is in the same directory as this script
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

import pandas as pd


INPUT_FILENAME = "train_features.csv"


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
        help="Optional explicit path to input CSV. Defaults to train_features.csv in the same folder as this script.",
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


def assign_buildings_to_partitions(df: pd.DataFrame, num_parts: int) -> list[pd.DataFrame]:
    """
    Assign each building_id to exactly one partition.

    Strategy:
    - Count rows per building_id
    - Sort largest buildings first
    - Greedily place each building into the currently smallest partition

    This keeps building data intact while making total row counts reasonably balanced.
    """
    if "building_id" not in df.columns:
        print("Error: 'building_id' column not found in input CSV.", file=sys.stderr)
        raise SystemExit(1)

    building_counts = df["building_id"].value_counts()
    sorted_buildings = building_counts.sort_values(ascending=False)

    partition_buildings: list[list[int]] = [[] for _ in range(num_parts)]
    partition_sizes = [0] * num_parts

    for building_id, row_count in sorted_buildings.items():
        smallest_partition_idx = min(range(num_parts), key=lambda i: partition_sizes[i])
        partition_buildings[smallest_partition_idx].append(building_id)
        partition_sizes[smallest_partition_idx] += int(row_count)

    parts: list[pd.DataFrame] = []
    for i, building_ids in enumerate(partition_buildings, start=1):
        part_df = df[df["building_id"].isin(building_ids)].copy()
        parts.append(part_df)

        unique_buildings = part_df["building_id"].nunique() if not part_df.empty else 0
        print(
            f"Partition {i}: "
            f"{len(part_df)} rows, "
            f"{unique_buildings} building_ids"
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

    parts = assign_buildings_to_partitions(df, args.num_parts)
    verify_no_building_overlap(parts)
    write_parts(parts, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()