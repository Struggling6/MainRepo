#!/usr/bin/env python3
"""
Temporally split the LEAD CSV into a train_val pool and a held-out test set.

Mirrors the cutoff logic in fl-backend/data/time_series_utils.py
(``temporal_grouped_split``) so the test set sits cleanly *after* the
train_val pool in time, with a gap to prevent lag features from leaking
across the boundary.

The pipeline still carves train/val out of the train_val pool internally,
so the train_val file is *not* a final training set — it is the pool the
federated trainer windows and splits further.

Example:
    python scripts/lead_train_test_splitter.py
    python scripts/lead_train_test_splitter.py --train-ratio 0.75 --gap-hours 168
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("fl-backend/datasets/LEAD/train_features_clean.csv")
DEFAULT_TRAIN_OUTPUT = Path("fl-backend/datasets/LEAD/LEAD_train_val.csv")
DEFAULT_TEST_OUTPUT = Path("fl-backend/datasets/LEAD/LEAD_test.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--gap-hours", type=int, default=73)
    parser.add_argument("--time-col", type=str, default="timestamp")
    parser.add_argument("--node-col", type=str, default="building_id")
    parser.add_argument("--target-col", type=str, default="anomaly")
    return parser.parse_args()


def compute_cutoff(df: pd.DataFrame, time_col: str, train_ratio: float) -> pd.Timestamp:
    # Matches temporal_grouped_split: take the timestamp at the train_ratio
    # percentile of all rows sorted by time, regardless of building.
    all_times = df[time_col].sort_values()
    idx = int(len(all_times) * train_ratio)
    return pd.Timestamp(all_times.iloc[idx])


def split_by_building(
    df: pd.DataFrame,
    *,
    time_col: str,
    node_col: str,
    cutoff: pd.Timestamp,
    gap_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gap_end = cutoff + pd.Timedelta(hours=gap_hours)

    train_mask = df[time_col] <= cutoff
    test_mask = df[time_col] > gap_end

    train_df = (
        df.loc[train_mask]
        .sort_values([node_col, time_col])
        .reset_index(drop=True)
    )
    test_df = (
        df.loc[test_mask]
        .sort_values([node_col, time_col])
        .reset_index(drop=True)
    )

    return train_df, test_df


def per_month_summary(
    df: pd.DataFrame, time_col: str, target_col: str
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "rows", "anomalies", "anomaly_rate"])

    months = df[time_col].dt.to_period("M")
    summary = (
        df.groupby(months)
        .agg(rows=(target_col, "size"), anomalies=(target_col, "sum"))
        .reset_index()
        .rename(columns={time_col: "month"})
    )
    summary["anomaly_rate"] = summary["anomalies"] / summary["rows"]
    return summary


def print_file_summary(
    name: str,
    df: pd.DataFrame,
    *,
    time_col: str,
    node_col: str,
    target_col: str,
) -> None:
    print(f"\n[{name}]")
    if df.empty:
        print("  (empty)")
        return

    rows = len(df)
    buildings = df[node_col].nunique()
    anomalies = int(df[target_col].sum())
    anomaly_rate = anomalies / rows if rows else 0.0
    t_min = df[time_col].min()
    t_max = df[time_col].max()

    print(f"  rows={rows:,}")
    print(f"  buildings={buildings:,}")
    print(f"  anomalies={anomalies:,} (rate={anomaly_rate:.4f})")
    print(f"  timestamp range: {t_min}  ->  {t_max}")

    monthly = per_month_summary(df, time_col, target_col)
    if not monthly.empty:
        print("  per-month anomaly rate:")
        for _, row in monthly.iterrows():
            print(
                f"    {row['month']}: rows={int(row['rows']):>8,} "
                f"anomalies={int(row['anomalies']):>6,} "
                f"rate={row['anomaly_rate']:.4f}"
            )


def main() -> None:
    args = parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        print("--train-ratio must be in (0, 1)", file=sys.stderr)
        raise SystemExit(1)
    if args.gap_hours < 0:
        print("--gap-hours must be >= 0", file=sys.stderr)
        raise SystemExit(1)

    input_path = args.input_file.resolve()
    train_output = args.train_output.resolve()
    test_output = args.test_output.resolve()

    print(f"Reading input file: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded shape: {df.shape}")

    for col in (args.time_col, args.node_col, args.target_col):
        if col not in df.columns:
            print(f"Error: required column '{col}' not found in input.", file=sys.stderr)
            raise SystemExit(1)

    df[args.time_col] = pd.to_datetime(df[args.time_col])

    cutoff = compute_cutoff(df, args.time_col, args.train_ratio)
    gap_end = cutoff + pd.Timedelta(hours=args.gap_hours)
    print(f"Cutoff timestamp:      {cutoff}")
    print(f"Gap end timestamp:     {gap_end} (gap_hours={args.gap_hours})")

    train_df, test_df = split_by_building(
        df,
        time_col=args.time_col,
        node_col=args.node_col,
        cutoff=cutoff,
        gap_hours=args.gap_hours,
    )

    dropped = len(df) - len(train_df) - len(test_df)
    input_buildings = set(df[args.node_col].unique())
    train_buildings = set(train_df[args.node_col].unique())
    test_buildings = set(test_df[args.node_col].unique())
    missing_train = input_buildings - train_buildings
    missing_test = input_buildings - test_buildings

    print("\nSplit summary:")
    print(f"  input rows:    {len(df):,}")
    print(f"  train rows:    {len(train_df):,}")
    print(f"  test rows:     {len(test_df):,}")
    print(f"  dropped (gap): {dropped:,}")
    if missing_train:
        print(
            f"  WARNING: {len(missing_train)} building(s) have no rows in train_val "
            f"(all their data fell after the cutoff)."
        )
    if missing_test:
        print(
            f"  WARNING: {len(missing_test)} building(s) have no rows in test "
            f"(all their data fell before cutoff + gap)."
        )

    print_file_summary(
        "train_val",
        train_df,
        time_col=args.time_col,
        node_col=args.node_col,
        target_col=args.target_col,
    )
    print_file_summary(
        "test",
        test_df,
        time_col=args.time_col,
        node_col=args.node_col,
        target_col=args.target_col,
    )

    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting train_val -> {train_output}")
    train_df.to_csv(train_output, index=False)
    print(f"Writing test      -> {test_output}")
    test_df.to_csv(test_output, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
