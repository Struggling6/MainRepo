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


def temporal_grouped_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    stride: int,
    node_col: str,
    time_col="timestamp",
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    gap_hours=0,
    target="anomaly",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a time series DataFrame into train, validation, and test windows
    per node, using global temporal cutoffs computed from all rows across all nodes.

    Ratios: train_ratio (default 0.6), val_ratio (default 0.2), test_ratio (default 0.2)
    These must sum to 1.0.

    The gap between train end and val start prevents lag features from
    leaking across the boundary. For example, if your longest lag is
    73 hours, set gap_hours=73 so the validation set never contains
    rows whose lag features were computed from training data.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
    """
    # Validate ratios
    if not (abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6):
        raise ValueError(
            f"Ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    df = _prepare_groups(df, node_col, time_col)
    df = df.sort_values([node_col, time_col])

    # Compute global cutoffs from all rows sorted by time
    all_times = df[time_col].sort_values()
    train_cutoff = pd.Timestamp(all_times.iloc[int(len(all_times) * train_ratio)])
    val_cutoff = pd.Timestamp(
        all_times.iloc[int(len(all_times) * (train_ratio + val_ratio))]
    )

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    # Process each node separately so different time series are not mixed together
    for node, group in df.groupby(node_col):
        group = group.sort_values(time_col)

        # Train: everything up to and including train_cutoff
        train_df = group[group[time_col] <= train_cutoff]

        # Val: everything after train_cutoff + gap, up to val_cutoff
        val_df = group[
            (group[time_col] > train_cutoff + pd.Timedelta(hours=gap_hours))
            & (group[time_col] <= val_cutoff)
        ]

        # Test: everything after val_cutoff + gap
        test_df = group[group[time_col] > val_cutoff + pd.Timedelta(hours=gap_hours)]

        # Window train data
        if len(train_df) > window_size:
            Xtr, ytr = create_windowed_data(
                train_df, feature_cols, window_size, stride, target
            )
            X_train_list.append(Xtr)
            y_train_list.append(ytr)

        # Window validation data
        if len(val_df) > window_size:
            Xva, yva = create_windowed_data(
                val_df, feature_cols, window_size, stride, target
            )
            X_val_list.append(Xva)
            y_val_list.append(yva)

        # Window test data
        if len(test_df) > window_size:
            Xte, yte = create_windowed_data(
                test_df, feature_cols, window_size, stride, target
            )
            X_test_list.append(Xte)
            y_test_list.append(yte)

    # Concatenate all windows from all nodes
    X_train = (
        np.concatenate(X_train_list, axis=0)
        if X_train_list
        else np.array([], dtype=np.float32).reshape(0, window_size, len(feature_cols))
    )
    y_train = (
        np.concatenate(y_train_list, axis=0)
        if y_train_list
        else np.array([], dtype=np.int64)
    )

    X_val = (
        np.concatenate(X_val_list, axis=0)
        if X_val_list
        else np.array([], dtype=np.float32).reshape(0, window_size, len(feature_cols))
    )
    y_val = (
        np.concatenate(y_val_list, axis=0)
        if y_val_list
        else np.array([], dtype=np.int64)
    )

    X_test = (
        np.concatenate(X_test_list, axis=0)
        if X_test_list
        else np.array([], dtype=np.float32).reshape(0, window_size, len(feature_cols))
    )
    y_test = (
        np.concatenate(y_test_list, axis=0)
        if y_test_list
        else np.array([], dtype=np.int64)
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

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
    approx_windows = total_rows / 168   # stride=24
    window_size    = 168
    n_features     = len(feature_cols)

    memory_gb = (approx_windows * window_size * n_features * 4) / 1e9  # float32 = 4 bytes
    print(f"Approximate number of windows: {approx_windows:,.0f}")
    print(f"Estimated memory for X_train alone: {memory_gb:.1f} GB")