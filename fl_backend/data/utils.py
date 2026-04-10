import pandas as pd

import numpy as np

def create_windowed_data(df, feature_cols, window_size, stride, target):

    X_windows, y_windows = [], []
    data   = df[feature_cols].values   # shape: (n_rows, n_features)
    labels = df[target].values         # shape: (n_rows,)

    # range(start, stop, step):P
    #   start = 0               → begin at first row
    #   stop  = len-window_size → last valid start so window doesn't fall off the end
    #   step  = stride          → how far to shift each iteration
    for i in range(0, len(data) - window_size, stride):
        X_windows.append(data[i : i + window_size])   # rows i..i+window_size-1
        y_windows.append(labels[i + window_size])     # label just after the window

    return np.array(X_windows), np.array(y_windows)

def temporal_grouped_split(
    df: pd.DataFrame,
    feature_cols: list,
    window_size: int,
    stride: int,
    node_col: str,
    time_col="timestamp",
    train_ratio=0.8,
    gap_hours = 0, # In case of no lag features, set gap_hours=0.
    target="anomaly",
):
    """
    Parameters
    ----------
    df           : pre-processed DataFrame
    feature_cols : list of feature column names
    node_col     : column name for node identifier
    time_col     : column name for timestamp
    train_ratio  : fraction of time to use for training (e.g. 0.8 = 80%)
    gap_hours    : hours to skip between train end and test start
                   (For LEAD, use at least max lag = 73 to be safe)
    window_size  : look-back window length in timesteps 
    stride       : window shift per iteration (24 = one window per day)

    Returns
    -------
    X_train, y_train : training windows and labels
    X_test,  y_test  : test windows and labels
    nid_train        : node_id for each training window (useful for analysis)
    nid_test         : node_id for each test window
    """

    df = df.sort_values([node_col, time_col])

    # Single global cutoff — the timestamp at the train_ratio position
    # across ALL rows (all buildings combined), sorted by time.
    # iloc[] is used because we need positional indexing, not label indexing.
    all_times = df[time_col].sort_values()
    cutoff    = all_times.iloc[int(len(all_times) * train_ratio)]

    X_train_list, y_train_list = [], []
    X_val_list,  y_val_list  = [], []
    nid_train, nid_val        = [], []

    for node, group in df.groupby(node_col):
        group = group.sort_values(time_col)

        # Split at cutoff, with a gap after cutoff to avoid lag leakage
        train_df = group[group[time_col] <= cutoff]
        test_df  = group[group[time_col] >  pd.Timestamp(cutoff) + pd.Timedelta(hours=gap_hours)] # use pandas' Timedelta class to add hours to a timestamp

        # Only proceed if the split has enough rows for at least one full window
        if len(train_df) > window_size:
            Xtr, ytr = create_windowed_data(train_df, feature_cols, window_size, stride, target)
            X_train_list.append(Xtr)
            y_train_list.append(ytr)
            nid_train.extend([node] * len(Xtr))

        if len(test_df) > window_size:
            Xte, yte = create_windowed_data(test_df, feature_cols, window_size, stride, target)
            X_val_list.append(Xte)
            y_val_list.append(yte)
            nid_val.extend([node] * len(Xte))
    
    # Concatenate all windows from all buildings into single arrays for train and test sets.
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val  = np.concatenate(X_val_list,  axis=0)
    y_val  = np.concatenate(y_val_list,  axis=0)

    return X_train, y_train, X_val, y_val, np.array(nid_train), np.array(nid_val)

def estimate_split_mem_usage(df, feature_cols):
    total_rows = len(df)
    approx_windows = total_rows / 24  # stride=24
    window_size = 168
    n_features = len(feature_cols)

    memory_gb = (approx_windows * window_size * n_features * 4) / 1e9  # float32 = 4 bytes
    print(f"Approximate number of windows: {approx_windows:,.0f}")
    print(f"Estimated memory for X_train alone: {memory_gb:.1f} GB")