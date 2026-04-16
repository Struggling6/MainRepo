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
    data   = df[feature_cols].values   # input features for all rows, shape: (n_rows, n_features)
    labels = df[target].values         # target labels for each row, shape: (n_rows,)

    # range(start=0, stop=last valid start, step=stride)
    # last valid start ensures the window i..i+window_size never
    # falls off the end of the array
    for i in range(0, len(data) - window_size, stride):
        X_windows.append(data[i : i + window_size])
        y_windows.append(labels[i + window_size])

    return np.array(X_windows), np.array(y_windows)


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
    train_ratio=0.8,
    gap_hours=73,
    target="anomaly",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a time series DataFrame into train and validation windows
    per node, using a single global temporal cutoff computed from all
    rows across all nodes combined.

    The gap between train end and val start prevents lag features from
    leaking across the boundary. For example, if your longest lag is
    73 hours, set gap_hours=73 so the validation set never contains
    rows whose lag features were computed from training data.

    Returns
    -------
    X_train, y_train, X_val, y_val : np.ndarray
    """
    df = _prepare_groups(df, node_col, time_col)
    #Ensures rows are ordered correctly within each node over time
    #ER IKKE SIKKER PÅ OM DET HER BARE GØR DET SAMME???? TJEK LIGE
    df = df.sort_values([node_col, time_col])

    # Compute the global cutoff from all rows sorted by time
    all_times = df[time_col].sort_values()
    cutoff    = pd.Timestamp(all_times.iloc[int(len(all_times) * train_ratio)])

    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []
    nid_train, nid_test        = [], []

    #Process each node seperately so different time series are not mixed together
    for node, group in df.groupby(node_col):
        group    = group.sort_values(time_col)
        
        print(f"\n--- NODE {node} ---")
        print(f"total rows: {len(group)}")

        train_df = group[group[time_col] <= cutoff]
        val_df   = group[group[time_col] > cutoff + pd.Timedelta(hours=gap_hours)]

        print(f"train rows: {len(train_df)}")
        print(f"val rows: {len(val_df)}")
        print(f"window_size: {window_size}")

        if len(train_df) > window_size:
            Xtr, ytr = create_windowed_data(train_df, feature_cols, window_size, stride, target)
            X_train_list.append(Xtr)
            y_train_list.append(ytr)
            nid_train.extend([node] * len(Xtr)) #Store which node each window came from

        if len(val_df) > window_size:
            Xva, yva = create_windowed_data(val_df, feature_cols, window_size, stride, target)
            X_val_list.append(Xva)
            y_val_list.append(yva)

    return (
        np.concatenate(X_train_list, axis=0),
        np.concatenate(y_train_list, axis=0),
        np.concatenate(X_val_list,   axis=0),
        np.concatenate(y_val_list,   axis=0),
    )


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
    approx_windows = total_rows / 24   # stride=24
    window_size    = 24
    n_features     = len(feature_cols)

    memory_gb = (approx_windows * window_size * n_features * 4) / 1e9  # float32 = 4 bytes
    print(f"Approximate number of windows: {approx_windows:,.0f}")
    print(f"Estimated memory for X_train alone: {memory_gb:.1f} GB")