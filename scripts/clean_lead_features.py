"""
Clean LEAD feature CSVs by removing sentinel values and producing derived files.

The LEAD raw CSVs encode missing values as dtype-max sentinels (255 for uint8
columns like cloud_coverage and year_built, 65535 for uint16 wind_direction)
and as NOAA ISD numeric markers (-1 = trace, -2 = missing) in
precip_depth_1_hr. These corrupt InstanceNorm1d statistics and any downstream
feature scaling.

This script:
  1. Reads the original train/test feature CSVs (UNCHANGED on disk).
  2. Replaces sentinel values per the table below, fitting imputation
     statistics on the TRAIN set ONLY and applying them identically to TEST.
  3. Adds 8 binary `*_was_missing` indicator columns.
  4. Replaces the raw `wind_direction` column with cyclic encoding
     `wind_dir_x` / `wind_dir_y` plus a `wind_dir_missing` flag.
  5. Writes derived files:
        datasets/LEAD/train_features_clean.csv
        datasets/LEAD/test_features_clean.csv
        datasets/LEAD/imputation_stats.json   (for reproducibility)

Cleaning rules:
    cloud_coverage           255          -> median per site_id           (+ flag)
    year_built               255, 0       -> median per primary_use       (+ flag)
    floor_count              0            -> median per primary_use       (+ flag)
    wind_direction           65535        -> cyclic (drop raw col)        (+ flag)
    wind_speed               -1           -> median per site_id           (+ flag)
    precip_depth_1_hr        -1 (trace)   -> 0.1
    precip_depth_1_hr        -2 (missing) -> 0.0                          (+ flag)
    air_temperature_std_lag7  < 0         -> 0.0                          (+ flag)
    air_temperature_std_lag73 < 0         -> 0.0                          (+ flag)

Usage:
    python scripts/clean_lead_features.py \
        --train fl-backend/datasets/LEAD/train_features.csv \
        --test  fl-backend/datasets/LEAD/test_features.csv \
        --out-dir fl-backend/datasets/LEAD

Windows users:
    python scripts/clean_lead_features.py --train fl-backend/datasets/LEAD/train_features.csv --test fl-backend/datasets/LEAD/test_features.csv --out-dir fl-backend/datasets/LEAD
        
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


GLOBAL_MEDIAN_KEY = "__global__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean LEAD feature CSVs (produces derived clean files).")
    parser.add_argument("--train", type=Path, required=True, help="Path to raw train_features.csv")
    parser.add_argument("--test",  type=Path, required=True, help="Path to raw test_features.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for clean files")
    parser.add_argument(
        "--train-out-name",
        type=str,
        default="train_features_clean.csv",
        help="Filename for cleaned training CSV",
    )
    parser.add_argument(
        "--test-out-name",
        type=str,
        default="test_features_clean.csv",
        help="Filename for cleaned test CSV",
    )
    parser.add_argument(
        "--stats-name",
        type=str,
        default="imputation_stats.json",
        help="Filename for imputation stats JSON",
    )
    return parser.parse_args()


def fit_group_median(df: pd.DataFrame, value_col: str, group_col: str, valid_mask: pd.Series) -> Dict[str, float]:
    """Fit median of value_col per group (only over rows where valid_mask is True).

    Returns a dict mapping group key (as str) -> median, plus a special
    GLOBAL_MEDIAN_KEY entry used as fallback for unseen groups or empty groups.
    """
    valid = df.loc[valid_mask, [group_col, value_col]]
    medians = valid.groupby(group_col)[value_col].median()
    # JSON keys must be strings.
    out: Dict[str, float] = {str(k): float(v) for k, v in medians.items() if pd.notna(v)}
    global_med = float(valid[value_col].median()) if not valid[value_col].empty else 0.0
    if not np.isfinite(global_med):
        global_med = 0.0
    out[GLOBAL_MEDIAN_KEY] = global_med
    return out


def apply_group_median(df: pd.DataFrame, value_col: str, group_col: str, sentinel_mask: pd.Series, stats: Dict[str, float]) -> None:
    """In place: replace sentinel rows in value_col with the per-group median, fallback to global."""
    fallback = stats.get(GLOBAL_MEDIAN_KEY, 0.0)
    # Build a per-row replacement series via map; missing groups get NaN -> fallback.
    group_keys = df[group_col].astype(str)
    replacement = group_keys.map(stats).astype(float)
    replacement = replacement.fillna(fallback)
    df.loc[sentinel_mask, value_col] = replacement.loc[sentinel_mask].values


def add_flag(df: pd.DataFrame, flag_name: str, mask: pd.Series) -> None:
    df[flag_name] = mask.astype("float32").values


def clean_dataframe(df: pd.DataFrame, stats: dict, fit: bool) -> pd.DataFrame:
    """Apply all cleaning operations. If fit=True, populate `stats` from this df.
    If fit=False, use values already in `stats`.
    """
    df = df.copy()

    # Promote integer columns we will impute with floats to float dtype upfront,
    # so that median replacements (which are non-integer) don't trigger
    # pandas' incompatible-dtype warnings.
    for col in ("cloud_coverage", "year_built", "floor_count", "wind_speed", "precip_depth_1_hr"):
        if col in df.columns:
            df[col] = df[col].astype("float64")

    # ---------- cloud_coverage: 255 -> median per site_id ----------
    mask = df["cloud_coverage"] == 255
    if fit:
        stats["cloud_coverage_by_site_id"] = fit_group_median(df, "cloud_coverage", "site_id", ~mask)
    apply_group_median(df, "cloud_coverage", "site_id", mask, stats["cloud_coverage_by_site_id"])
    add_flag(df, "cloud_coverage_was_missing", mask)

    # ---------- year_built: {255, 0} -> median per primary_use ----------
    mask = (df["year_built"] == 255) | (df["year_built"] == 0)
    if fit:
        stats["year_built_by_primary_use"] = fit_group_median(df, "year_built", "primary_use", ~mask)
    apply_group_median(df, "year_built", "primary_use", mask, stats["year_built_by_primary_use"])
    add_flag(df, "year_built_was_missing", mask)

    # ---------- floor_count: 0 -> median per primary_use ----------
    mask = df["floor_count"] == 0
    if fit:
        stats["floor_count_by_primary_use"] = fit_group_median(df, "floor_count", "primary_use", ~mask)
    apply_group_median(df, "floor_count", "primary_use", mask, stats["floor_count_by_primary_use"])
    add_flag(df, "floor_count_was_missing", mask)

    # ---------- wind_direction: cyclic encoding + missing flag ----------
    mask = df["wind_direction"] == 65535
    radians = np.deg2rad(df["wind_direction"].astype(np.float64))
    wind_dir_x = np.cos(radians).astype(np.float32)
    wind_dir_y = np.sin(radians).astype(np.float32)
    # Sentinels collapse to (0, 0) so they sit at the origin, distinguishable from any real direction.
    wind_dir_x[mask.values] = 0.0
    wind_dir_y[mask.values] = 0.0
    df["wind_dir_x"] = wind_dir_x
    df["wind_dir_y"] = wind_dir_y
    add_flag(df, "wind_dir_missing", mask)
    df = df.drop(columns=["wind_direction"])

    # ---------- wind_speed: -1 -> median per site_id ----------
    mask = df["wind_speed"] == -1
    if fit:
        stats["wind_speed_by_site_id"] = fit_group_median(df, "wind_speed", "site_id", ~mask)
    apply_group_median(df, "wind_speed", "site_id", mask, stats["wind_speed_by_site_id"])
    add_flag(df, "wind_speed_was_missing", mask)

    # ---------- precip_depth_1_hr: -1 (trace) -> 0.1; -2 (missing) -> 0.0 ----------
    trace_mask = df["precip_depth_1_hr"] == -1
    missing_mask = df["precip_depth_1_hr"] == -2
    df.loc[trace_mask, "precip_depth_1_hr"] = 0.1
    df.loc[missing_mask, "precip_depth_1_hr"] = 0.0
    add_flag(df, "precip_depth_was_missing", missing_mask)

    # ---------- air_temperature_std_lag7: < 0 -> 0 ----------
    mask = df["air_temperature_std_lag7"] < 0
    df.loc[mask, "air_temperature_std_lag7"] = 0.0
    add_flag(df, "air_temp_std_lag7_was_missing", mask)

    # ---------- air_temperature_std_lag73: < 0 -> 0 ----------
    mask = df["air_temperature_std_lag73"] < 0
    df.loc[mask, "air_temperature_std_lag73"] = 0.0
    add_flag(df, "air_temp_std_lag73_was_missing", mask)

    return df


def report_residual_sentinels(label: str, df: pd.DataFrame) -> None:
    """Sanity check after cleaning. wind_direction has been dropped on purpose."""
    checks = [
        ("cloud_coverage",            (df["cloud_coverage"] == 255).sum()),
        ("year_built",                ((df["year_built"] == 255) | (df["year_built"] == 0)).sum()),
        ("floor_count",               (df["floor_count"] == 0).sum()),
        ("wind_speed",                (df["wind_speed"] == -1).sum()),
        ("precip_depth_1_hr <0",      (df["precip_depth_1_hr"] < 0).sum()),
        ("air_temperature_std_lag7",  (df["air_temperature_std_lag7"] < 0).sum()),
        ("air_temperature_std_lag73", (df["air_temperature_std_lag73"] < 0).sum()),
    ]
    bad = [(c, int(n)) for c, n in checks if n > 0]
    if bad:
        print(f"  [{label}] residual sentinels detected:")
        for c, n in bad:
            print(f"      {c}: {n:,}")
    else:
        print(f"  [{label}] zero sentinels remaining.")


def main() -> None:
    args = parse_args()

    train_path: Path = args.train.resolve()
    test_path:  Path = args.test.resolve()
    out_dir:    Path = args.out_dir.resolve()

    if not train_path.exists():
        print(f"ERROR: train file not found: {train_path}", file=sys.stderr)
        raise SystemExit(1)
    if not test_path.exists():
        print(f"ERROR: test file not found: {test_path}", file=sys.stderr)
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / args.train_out_name
    test_out  = out_dir / args.test_out_name
    stats_out = out_dir / args.stats_name

    print(f"Reading train: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"  shape: {df_train.shape}")

    print(f"Reading test:  {test_path}")
    df_test = pd.read_csv(test_path)
    print(f"  shape: {df_test.shape}")

    stats: dict = {}

    print("Cleaning train (fitting imputation stats) ...")
    df_train_clean = clean_dataframe(df_train, stats, fit=True)
    report_residual_sentinels("train", df_train_clean)

    print("Cleaning test  (applying fitted stats) ...")
    df_test_clean = clean_dataframe(df_test, stats, fit=False)
    report_residual_sentinels("test", df_test_clean)

    print(f"Writing cleaned train to: {train_out}")
    df_train_clean.to_csv(train_out, index=False)
    print(f"  shape: {df_train_clean.shape}")

    print(f"Writing cleaned test to:  {test_out}")
    df_test_clean.to_csv(test_out, index=False)
    print(f"  shape: {df_test_clean.shape}")

    print(f"Writing imputation stats to: {stats_out}")
    with open(stats_out, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print("Done.")


if __name__ == "__main__":
    main()
