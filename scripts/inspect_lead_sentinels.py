"""
Inspect LEAD feature CSVs for sentinel values.

Sentinels in LEAD raw data come from upstream uint8/uint16 dtype downcasts
where NaN cannot be represented and gets coerced to the dtype's max value
(255, 65535) or to NOAA ISD missing markers (-1, -2).

Usage:
    python scripts/inspect_lead_sentinels.py --input datasets/LEAD/train_features.csv

Reports:
    - Per-column sentinel occurrence count and percentage.
    - Min/max/mean for each numeric column.
    - NaN counts.
    - Pass/fail flag if zero sentinels remain (useful for cleaned-file verification).

This script is read-only and never modifies the input file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# (column_name, predicate(series) -> mask, human description)
SENTINEL_CHECKS = [
    ("cloud_coverage",            lambda s: s == 255,    "== 255 (uint8 sentinel)"),
    ("year_built",                lambda s: s == 255,    "== 255 (uint8 sentinel)"),
    ("year_built",                lambda s: s == 0,      "== 0 (suspicious zero)"),
    ("floor_count",               lambda s: s == 0,      "== 0 (likely missing)"),
    ("wind_direction",            lambda s: s == 65535,  "== 65535 (uint16 sentinel)"),
    ("wind_speed",                lambda s: s == -1,     "== -1 (negative wind speed impossible)"),
    ("precip_depth_1_hr",         lambda s: s == -1,     "== -1 (NOAA trace marker)"),
    ("precip_depth_1_hr",         lambda s: s == -2,     "== -2 (NOAA missing marker)"),
    ("air_temperature_std_lag7",  lambda s: s < 0,       "< 0 (std cannot be negative)"),
    ("air_temperature_std_lag73", lambda s: s < 0,       "< 0 (std cannot be negative)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect LEAD feature CSV for sentinel values.")
    parser.add_argument("--input", type=Path, required=True, help="Path to features CSV")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any sentinel rows are found (useful for CI/verification of cleaned files)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input.resolve()
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    n = len(df)
    print(f"Shape: {df.shape}")
    print()

    print("=== Sentinel value report ===")
    total_sentinels = 0
    rows_unset = []
    for col, pred, desc in SENTINEL_CHECKS:
        if col not in df.columns:
            rows_unset.append(f"  [skipped] {col}: column not present")
            continue
        try:
            mask = pred(df[col])
            cnt = int(mask.sum())
        except Exception as exc:
            rows_unset.append(f"  [error] {col} ({desc}): {exc}")
            continue
        total_sentinels += cnt
        pct = 100.0 * cnt / n if n else 0.0
        marker = "  " if cnt == 0 else "!!"
        print(f"{marker} {col:30s} {desc:45s} {cnt:>10,} ({pct:6.2f}%)")
    for row in rows_unset:
        print(row)
    print()

    print("=== NaN counts (only columns with NaN) ===")
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if nan_counts.empty:
        print("  (none)")
    else:
        for col, c in nan_counts.items():
            pct = 100.0 * c / n if n else 0.0
            print(f"  {col:30s} {c:>10,} ({pct:6.2f}%)")
    print()

    print("=== Numeric column summary ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df[numeric_cols].describe().T[["min", "max", "mean"]]
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary.to_string())
    print()

    print("=== Result ===")
    if total_sentinels == 0:
        print("PASS: zero sentinel rows detected.")
    else:
        print(f"FAIL: {total_sentinels:,} sentinel cell occurrences detected across known columns.")

    if args.strict and total_sentinels > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
