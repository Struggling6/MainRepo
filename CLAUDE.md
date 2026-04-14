# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.12 with a `.venv` virtual environment — activate with `source .venv/bin/activate`
- Primary work happens in **Jupyter notebooks** — launch with `jupyter notebook` or `jupyter lab`
- No formal test suite, build system, or linter is configured

## Project Overview

Research project exploring **anomaly and cyberattack detection in time-series sensor data** using a hybrid CNN-Transformer architecture. Two main datasets are targeted:

- **LEAD** — building energy meter readings across multiple buildings (`datasets/LEAD/`)
- **Smart Grid Security** — electrical grid sensor data with attack labels (`datasets/Smart Grid Security Data/`)
- **Tommy Morris / FDIA** — additional power grid attack datasets in `datasets/`

## Main Notebook: `CNN_transformer.ipynb`

All model development lives here. The notebook is structured as a pipeline:

1. **Data loading & type optimization** — float64→float32, int64→int32, strings→categories to cut memory usage
2. **Feature engineering** — lag features (lag1, lag24), rolling mean/std (window=24), diff features, z-score normalization, one-hot encoding
3. **Windowing** — `create_windowed_data()` converts flat time-series into sliding windows per `house_id` (prevents cross-building leakage)
4. **Temporal grouped train/test split** — `temporal_grouped_split()` uses a single global cutoff timestamp with a gap (73h for LEAD) to prevent leakage from lag features
5. **Scaling** — `StandardScaler` fitted on training windows only
6. **Model** — `CNNTransformer`: two Conv1d blocks (64→128→d_model channels) with BatchNorm + MaxPool, then sinusoidal positional encoding, then a Transformer encoder, then mean pooling + linear classification head
7. **Training** — Adam optimizer, `BCEWithLogitsLoss` (binary) or `CrossEntropyLoss` (multiclass), GPU-aware (CUDA/ROCm with CPU fallback)

## Key Hyperparameters

| Parameter | LEAD | Smart Grid |
|-----------|------|------------|
| `window_size` | 168 (1 week) | 8 |
| `stride` | 24 (1 day) | 4 |
| `train_ratio` | 0.8 | 0.6 |
| `gap_hours` | 73 | 0 |
| `d_model` | 128 | — |
| `nhead` | 4 | — |
| `num_layers` | 2 | — |
| `batch_size` | 64 | — |
| `lr` | 1e-3 | — |

## Known Issues

- The model currently **overfits** — see recent commits. Regularization work is in progress.
- The LEAD dataset is ~1.3 GB and git-ignored; it must be sourced separately.
- The GitHub Actions workflow (`.github/workflows/format.yml`) is a placeholder and not functional.
