#!/usr/bin/env python3
"""
Split the Power Consumption Anomaly training CSV files into local client CSVs.

The power dataset is accepted by fl-backend/data/power_consumption_anomaly.py
with columns such as:
- ctime / timestamp
- activePower / active_power
- label / anomaly

By default this script reads every CSV under:

    fl-backend/datasets/PowerConsumptionAnomaly/
    Power-Consumption-Anomaly-Dataset-main/data

That is the training data folder. It does not use overall_eval_set.csv unless
you explicitly pass it with --input-file.

If the input contains an appliance/device column, whole series are kept together
in one client file. If not, the script tries to infer a device_id from timestamp
resets in combined files such as overall_eval_set.csv. As a last fallback, it
creates contiguous temporal chunks.

Example:
    python scripts/powerCsv_splitter.py --num-parts 5

Then configure local mode with:
    data_dir=Path("datasets/PowerConsumptionAnomaly/partitions")
    file_pattern="data{client_index}.csv"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILENAME = Path(
    "fl-backend/datasets/PowerConsumptionAnomaly/"
    "Power-Consumption-Anomaly-Dataset-main/overall_eval_set.csv"
)
INPUT_DIR = Path(
    "fl-backend/datasets/PowerConsumptionAnomaly/"
    "Power-Consumption-Anomaly-Dataset-main/data"
)
OUTPUT_DIR = Path("fl-backend/datasets/PowerConsumptionAnomaly/partitions")

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "time",
    "datetime",
    "date_time",
    "ctime",
    "created_at",
]
TARGET_CANDIDATES = [
    "label",
    "anomaly",
    "is_anomaly",
    "target",
    "class",
]
NODE_CANDIDATES = [
    "series_id",
    "device_id",
    "appliance",
    "appliance_type",
    "device",
    "building_id",
    "household",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the power consumption anomaly dataset for local FL."
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        required=True,
        help="Number of output CSV files to create.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help=(
            "Optional single CSV to split. If omitted, all CSVs under "
            "--input-dir are used."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory of training CSVs to read recursively.",
    )
    parser.add_argument(
        "--file-glob",
        type=str,
        default="*.csv",
        help="CSV glob used under --input-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where data1.csv ... dataN.csv will be saved.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Target column used to count anomalies. Auto-detected by default.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=None,
        help="Timestamp column used for temporal splits. Auto-detected by default.",
    )
    parser.add_argument(
        "--node-col",
        type=str,
        default=None,
        help="Optional series/appliance column. Auto-detected by default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used only to break ties between otherwise similar series.",
    )
    return parser.parse_args()


def read_single_csv(path: Path, root: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    has_timestamp = first_existing_column(df, TIMESTAMP_CANDIDATES)
    if has_timestamp is None:
        print(f"Skipping CSV without timestamp column: {path}")
        return pd.DataFrame()

    if first_existing_column(df, TARGET_CANDIDATES) is None:
        df["label"] = 0

    relative_parts = path.relative_to(root).parts if root is not None else path.parts
    appliance = relative_parts[0] if len(relative_parts) >= 1 else ""
    device = relative_parts[1] if len(relative_parts) >= 2 else path.stem
    scenario = relative_parts[2] if len(relative_parts) >= 3 else ""

    df["appliance"] = appliance
    df["scenario"] = scenario
    df["source_file"] = path.stem
    df["source_path"] = str(path)
    df["device_id"] = "__".join(
        part
        for part in [appliance, device, scenario, path.stem]
        if part
    )

    return df


def load_input_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_file is not None:
        input_path = args.input_file.resolve()
        print(f"Reading input file: {input_path}")
        df = read_single_csv(input_path)
        if df.empty:
            print(f"Error: no usable rows found in {input_path}", file=sys.stderr)
            raise SystemExit(1)
        return df

    input_dir = args.input_dir.resolve()
    print(f"Reading input directory: {input_dir}")

    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        raise SystemExit(1)

    paths = sorted(input_dir.rglob(args.file_glob))
    if not paths:
        print(
            f"Error: no files matching '{args.file_glob}' found under {input_dir}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    frames = [read_single_csv(path, root=input_dir) for path in paths]
    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        print(f"Error: no usable CSV files found under {input_dir}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Loaded {len(frames)} CSV files")
    return pd.concat(frames, ignore_index=True)


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    explicit: str | None = None,
) -> str | None:
    if explicit:
        return explicit if explicit in df.columns else None

    by_lower = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        column = by_lower.get(candidate.lower())
        if column is not None:
            return str(column)
    return None


def encoded_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    series = df[target_col]

    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype(np.int64)

    normalized = series.astype(str).str.strip().str.lower()
    positive_values = {
        "1",
        "true",
        "yes",
        "y",
        "anomaly",
        "abnormal",
        "fault",
        "malfunction",
        "error",
    }
    return normalized.isin(positive_values).astype(np.int64)


def infer_node_column_from_time_resets(
    df: pd.DataFrame,
    time_col: str,
) -> tuple[pd.DataFrame, str | None]:
    """
    Add a device_id when a combined CSV contains repeated time-series runs.

    overall_eval_set.csv is concatenated from several appliance/scenario files:
    ctime moves forward within each run and then jumps back to the start. Those
    jumps are reliable boundaries for creating separate local series.
    """
    times = pd.to_datetime(df[time_col], errors="coerce")
    if times.isna().all():
        return df, None

    resets = times.diff() < pd.Timedelta(0)
    num_resets = int(resets.sum())
    if num_resets == 0:
        return df, None

    inferred = df.copy()
    inferred["device_id"] = "inferred_series_" + (resets.cumsum() + 1).astype(str)
    print(
        f"Inferred device_id from {num_resets} timestamp resets "
        f"({inferred['device_id'].nunique()} series)."
    )
    return inferred, "device_id"


def assign_series_to_partitions(
    df: pd.DataFrame,
    num_parts: int,
    node_col: str,
    target_col: str,
    seed: int,
) -> list[pd.DataFrame]:
    """
    Assign each whole appliance/device series to exactly one partition.

    This mirrors the shared-mode PCAD handler: series with many anomalies are
    placed first into the currently lightest client by anomaly and row count.
    """
    rng = np.random.default_rng(seed)
    y = encoded_target(df, target_col)

    stats = (
        df.assign(_encoded_target=y)
        .groupby(node_col)
        .agg(
            num_rows=("_encoded_target", "size"),
            num_anomalies=("_encoded_target", "sum"),
        )
        .reset_index()
    )

    if num_parts > len(stats):
        print(
            f"Error: --num-parts={num_parts} is greater than the number of "
            f"available series={len(stats)}. This would create empty files.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    stats["_tie_break"] = rng.random(len(stats))
    stats = stats.sort_values(
        ["num_anomalies", "num_rows", "_tie_break"],
        ascending=[False, False, True],
    )

    partitions = [
        {"series": [], "rows": 0, "anomalies": 0}
        for _ in range(num_parts)
    ]

    for _, row in stats.iterrows():
        best_idx = min(
            range(num_parts),
            key=lambda idx: (
                partitions[idx]["anomalies"],
                partitions[idx]["rows"],
            ),
        )
        partitions[best_idx]["series"].append(row[node_col])
        partitions[best_idx]["rows"] += int(row["num_rows"])
        partitions[best_idx]["anomalies"] += int(row["num_anomalies"])

    parts: list[pd.DataFrame] = []
    print("Partition summary:")
    print(
        f"total_series={len(stats)}, total_rows={len(df)}, "
        f"total_anomalies={int(y.sum())}"
    )

    for idx, partition in enumerate(partitions, start=1):
        part_df = df[df[node_col].isin(partition["series"])].copy()
        parts.append(part_df)
        anomaly_rate = (
            partition["anomalies"] / partition["rows"]
            if partition["rows"] > 0
            else 0.0
        )
        print(
            f"Partition {idx}: {partition['rows']} rows, "
            f"{len(partition['series'])} series, "
            f"{partition['anomalies']} anomalies, "
            f"anomaly_rate={anomaly_rate:.4f}"
        )

    return parts


def assign_temporal_chunks_to_partitions(
    df: pd.DataFrame,
    num_parts: int,
    time_col: str,
    target_col: str,
) -> list[pd.DataFrame]:
    """
    Split a single-series CSV into contiguous temporal chunks.

    overall_eval_set.csv has no device identifier, so keeping chunks contiguous
    preserves the time-series ordering better than a random split.
    """
    if num_parts > len(df):
        print(
            f"Error: --num-parts={num_parts} is greater than the number of "
            f"rows={len(df)}. This would create empty files.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    work = df.copy()
    split_time = pd.to_datetime(work[time_col], errors="coerce")
    if split_time.isna().all():
        print(
            f"Error: could not parse any timestamps from column '{time_col}'.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    work["_split_time"] = split_time
    work = work.sort_values("_split_time").drop(columns=["_split_time"])
    index_chunks = np.array_split(np.arange(len(work)), num_parts)
    parts = [work.iloc[index_chunk].copy() for index_chunk in index_chunks]

    print("Partition summary:")
    print(f"total_rows={len(work)}, total_anomalies={int(encoded_target(work, target_col).sum())}")

    for idx, part in enumerate(parts, start=1):
        anomalies = int(encoded_target(part, target_col).sum())
        rows = len(part)
        anomaly_rate = anomalies / rows if rows else 0.0
        start_time = part[time_col].iloc[0] if rows else "n/a"
        end_time = part[time_col].iloc[-1] if rows else "n/a"
        print(
            f"Partition {idx}: {rows} rows, {anomalies} anomalies, "
            f"anomaly_rate={anomaly_rate:.4f}, "
            f"time_range=[{start_time}, {end_time}]"
        )

    return parts


def write_parts(parts: list[pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_file in output_dir.glob("data*.csv"):
        old_file.unlink()

    for idx, part in enumerate(parts, start=1):
        output_file = output_dir / f"data{idx}.csv"
        part.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")


def verify_parts(parts: list[pd.DataFrame], expected_rows: int) -> None:
    actual_rows = sum(len(part) for part in parts)
    if actual_rows != expected_rows:
        print(
            f"Error: split row count mismatch. Expected {expected_rows}, "
            f"got {actual_rows}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    empty_parts = [idx for idx, part in enumerate(parts, start=1) if part.empty]
    if empty_parts:
        print(
            f"Error: empty output partitions created: {empty_parts}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("Verification passed: all rows are assigned to non-empty files.")


def main() -> None:
    args = parse_args()

    if args.num_parts < 1:
        print("--num-parts must be at least 1", file=sys.stderr)
        raise SystemExit(1)

    output_dir = args.output_dir.resolve()

    df = load_input_dataframe(args)

    time_col = first_existing_column(df, TIMESTAMP_CANDIDATES, args.time_col)
    target_col = first_existing_column(df, TARGET_CANDIDATES, args.target_col)
    node_col = first_existing_column(df, NODE_CANDIDATES, args.node_col)

    if time_col is None:
        print(
            "Error: could not find a timestamp column. Tried: "
            f"{TIMESTAMP_CANDIDATES}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if target_col is None:
        print(
            "Error: could not find a target/anomaly column. Tried: "
            f"{TARGET_CANDIDATES}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Loaded shape: {df.shape}")
    print(f"Using time_col='{time_col}', target_col='{target_col}', node_col='{node_col}'")
    print(f"Saving split files to: {output_dir}")

    if node_col is None:
        df, node_col = infer_node_column_from_time_resets(df, time_col)

    if node_col is not None:
        parts = assign_series_to_partitions(
            df,
            args.num_parts,
            node_col=node_col,
            target_col=target_col,
            seed=args.seed,
        )
    else:
        print("No series/appliance column found; using contiguous temporal chunks.")
        parts = assign_temporal_chunks_to_partitions(
            df,
            args.num_parts,
            time_col=time_col,
            target_col=target_col,
        )

    verify_parts(parts, expected_rows=len(df))
    write_parts(parts, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
