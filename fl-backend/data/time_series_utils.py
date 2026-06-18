import pandas as pd
import numpy as np

#Creates sliding windows for a time series dataset.

def create_windowed_data(
        df           : pd.DataFrame, 
        feature_cols : list, 
        window_size  : int, 
        stride       : int, 
        target       : str
    ):
    """
    Slide a window over a single node's DataFrame and return
    feature windows and their corresponding labels.

    The label for each window is taken from the timestep immediately
    after the window ends — this is the "predict the next step" framing
    used throughout the pipeline.

    Returns
    -------
    X_train, y_train, X_val, y_val : np.ndarray
    """
    #X = past observations in a window, y = the target value just after the window
    X_windows, y_windows = [], []
    # Force float32 to prevent any stray float64 column from upcasting the
    # whole windowed array (which doubles peak RAM usage).
    data   = df[feature_cols].values.astype(np.float32, copy=False)   # input features for all rows, shape: (n_rows, n_features)



    labels = df[target].values         # target labels for each row, shape: (n_rows,)

    # range(start=0, stop=last valid start, step=stride)
    # last valid start ensures the window i..i+window_size never
    # falls off the end of the array
    for i in range(0, len(data) - window_size, stride):
        X_windows.append(data[i : i + window_size])
        y_windows.append(labels[i + window_size])

    return np.array(X_windows, dtype=np.float32), np.array(y_windows)


def _prepare_groups(df, node_col, time_col):
    """
    Shared preprocessing step for both temporal_grouped_split and
    create_test_windows. Ensures the DataFrame is a fresh copy with
    proper datetime types and sorted correctly before any windowing.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([node_col, time_col])
    return df


def _window_groups(df, feature_cols, window_size, stride, node_col, time_col, target):
    """
    Window all nodes in a DataFrame without any splitting.
    Assumes the DataFrame has already been prepared by _prepare_groups.
    """
    X_list, y_list = [], []

    for node, group in df.groupby(node_col):
        group = group.sort_values(time_col)
        # Skip nodes that do not have enough rows for even one window
        if len(group) > window_size:
            X, y = create_windowed_data(group, feature_cols, window_size, stride, target)
            X_list.append(X)
            y_list.append(y)

    return (
        np.concatenate(X_list, axis=0),
        np.concatenate(y_list, axis=0),
    )


def temporal_grouped_train_val_eval_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    stride: int,
    node_col: str,
    time_col="timestamp",
    train_ratio=0.6,
    val_ratio=0.2,
    eval_ratio=0.2,
    gap_hours=0,
    target="anomaly",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a time series DataFrame into train, validation, and final-evaluation
    windows per node, using global temporal cutoffs.

    Intended split:
      - train: used for model fitting and may be over/undersampled
      - val: used during training/early stopping/hyperparameter selection and may
             optionally be over/undersampled if your config says so
      - eval: final held-out evaluation set; should never be sampled or used during
              training

    The same gap is applied between train->val and val->eval to reduce leakage from
    lagged features across split boundaries.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_eval, y_eval : np.ndarray
    """
    total = train_ratio + val_ratio + eval_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + eval_ratio must equal 1.0, got {total:.4f}"
        )

    df = _prepare_groups(df, node_col, time_col)

    all_times = df[time_col].sort_values().reset_index(drop=True)
    if all_times.empty:
        raise ValueError("Cannot split an empty DataFrame")

    train_cutoff_idx = int(len(all_times) * train_ratio)
    val_cutoff_idx = int(len(all_times) * (train_ratio + val_ratio))

    # Keep indices inside bounds for very small datasets.
    train_cutoff_idx = min(max(train_cutoff_idx, 0), len(all_times) - 1)
    val_cutoff_idx = min(max(val_cutoff_idx, 0), len(all_times) - 1)

    train_cutoff = pd.Timestamp(all_times.iloc[train_cutoff_idx])
    val_cutoff = pd.Timestamp(all_times.iloc[val_cutoff_idx])

    train_end = train_cutoff
    val_start = train_cutoff + pd.Timedelta(hours=gap_hours)
    val_end = val_cutoff
    eval_start = val_cutoff + pd.Timedelta(hours=gap_hours)

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_eval_list, y_eval_list = [], []

    for node, group in df.groupby(node_col):
        group = group.sort_values(time_col)

        train_df = group[group[time_col] <= train_end]
        val_df = group[(group[time_col] > val_start) & (group[time_col] <= val_end)]
        eval_df = group[group[time_col] > eval_start]

        if len(train_df) > window_size:
            Xtr, ytr = create_windowed_data(train_df, feature_cols, window_size, stride, target)
            X_train_list.append(Xtr)
            y_train_list.append(ytr)

        if len(val_df) > window_size:
            Xva, yva = create_windowed_data(val_df, feature_cols, window_size, stride, target)
            X_val_list.append(Xva)
            y_val_list.append(yva)

        if eval_ratio > 0 and len(eval_df) > window_size:
            Xev, yev = create_windowed_data(eval_df, feature_cols, window_size, stride, target)
            X_eval_list.append(Xev)
            y_eval_list.append(yev)

    def _concat_or_raise(items, split_name: str):
        if not items:
            raise ValueError(
                f"No {split_name} windows were created. "
                "Try reducing window_size/stride/gap_hours or changing split ratios."
            )
        return np.concatenate(items, axis=0)

    return (
        _concat_or_raise(X_train_list, "train"),
        _concat_or_raise(y_train_list, "train labels"),
        _concat_or_raise(X_val_list, "validation"),
        _concat_or_raise(y_val_list, "validation labels"),
        _concat_or_raise(X_eval_list, "evaluation") if eval_ratio > 0 else np.empty((0,), dtype=np.float32),
        _concat_or_raise(y_eval_list, "evaluation labels") if eval_ratio > 0 else np.empty((0,), dtype=np.int64),
    )


def temporal_grouped_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    stride: int,
    node_col: str,
    time_col="timestamp",
    train_ratio=0.8,
    gap_hours=0,
    target="anomaly",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Backwards-compatible two-way split.
    New code should prefer temporal_grouped_train_val_eval_split.
    """
    X_train, y_train, X_val, y_val, _, _ = temporal_grouped_train_val_eval_split(
        df=df,
        feature_cols=feature_cols,
        window_size=window_size,
        stride=stride,
        node_col=node_col,
        time_col=time_col,
        train_ratio=train_ratio,
        val_ratio=1.0 - train_ratio,
        eval_ratio=0.0,
        gap_hours=gap_hours,
        target=target,
    )
    return X_train, y_train, X_val, y_val

def create_test_windows(
    df: pd.DataFrame,
    feature_cols: list,
    window_size: int,
    stride: int,
    node_col: str,
    time_col="timestamp",
    target="anomaly",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Window a test DataFrame without any train/val splitting.
    All rows from all nodes are windowed and returned as a single set.

    Use this only for the dedicated test set that was never touched
    during training or hyperparameter search.

    Returns
    -------
    X_test : np.ndarray — shape (n_windows, window_size, n_features)
    y_test : np.ndarray — shape (n_windows,)
    """
    df = _prepare_groups(df, node_col, time_col)
    return _window_groups(df, feature_cols, window_size, stride, node_col, time_col, target)


## Rough memory estimate for windowed feature arrays
def estimate_split_mem_usage(df, feature_cols):
    """
    Rough estimate of memory required for the training split.
    Useful for checking whether your machine can hold the windowed
    arrays in RAM before running the full pipeline.
    """
    total_rows     = len(df)
    window_size    = 168
    approx_windows = total_rows / window_size   # assumes stride == window_size
    n_features     = len(feature_cols)

    memory_gb = (approx_windows * window_size * n_features * 4) / 1e9  # float32 = 4 bytes
    print(f"Approximate number of windows: {approx_windows:,.0f}")
    print(f"Estimated memory for X_train alone: {memory_gb:.1f} GB")