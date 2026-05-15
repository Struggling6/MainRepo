from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .BaseDataHandler import BaseDatasetHandler
from .time_series_utils import temporal_grouped_split
from .ts_augment_utils import jitter, magnitude_warp, scaling


class PowerConsumptionAnomalyHandler(BaseDatasetHandler):
    """
    Dataset handler for the Papaioannou et al. power consumption anomaly data.

    The public dataset is distributed as several appliance CSV files. This handler
    accepts either one combined file or a directory of CSV files and turns them
    into temporal windows compatible with the existing supervised models.

    Partition modes:
    - shared: load all files and partition internally by series_id
    - local:  load this client's individual CSV file from disk
    - optuna: alias for shared mode with num_clients=1 (uses the full dataset)
    """

    TIMESTAMP_CANDIDATES = [
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "ctime",
        "created_at",
    ]
    POWER_CANDIDATES = [
        "active_power",
        "activePower",
        "power",
        "power_consumption",
        "consumption",
        "value",
    ]
    TARGET_CANDIDATES = [
        "anomaly",
        "is_anomaly",
        "label",
        "target",
        "class",
    ]
    NODE_CANDIDATES = [
        "appliance",
        "appliance_type",
        "device",
        "device_id",
        "building_id",
        "household",
    ]

    def __init__(self, config):
        super().__init__(config)

        data_cfg = config.data
        self.file_path = Path(data_cfg.file_path)
        self.data_dir = Path(data_cfg.data_dir)
        self.file_pattern = data_cfg.file_pattern
        self.target = data_cfg.target
        self.batch_size = config.model.batch_size
        self.test_split = data_cfg.test_split
        self.seed = data_cfg.seed

        self.partition_mode = config.federation.partition_mode
        self.num_clients = config.federation.num_clients

        if self.partition_mode == "optuna":
            print("[PCAD] Optuna mode: forcing num_clients=1 for full-dataset trial")
            self.num_clients = 1
            self.partition_mode = "shared"

        self.window_size = data_cfg.window_size
        self.stride = data_cfg.stride
        self.gap_hours = data_cfg.gap_hours
        self.normalize = data_cfg.normalize

        self.oversampling_method = data_cfg.oversampling_method
        self.oversampling_ratio = data_cfg.oversampling_ratio
        self.oversample_val = data_cfg.oversample_val
        self.smote_k_neighbors = data_cfg.smote_k_neighbors

        self.tsaug_jitter_sigma = data_cfg.tsaug_jitter_sigma
        self.tsaug_scaling_sigma = data_cfg.tsaug_scaling_sigma
        self.tsaug_magwarp_sigma = data_cfg.tsaug_magwarp_sigma
        self.tsaug_magwarp_knots = data_cfg.tsaug_magwarp_knots
        self.tsaug_use_jitter = data_cfg.tsaug_use_jitter
        self.tsaug_use_scaling = data_cfg.tsaug_use_scaling
        self.tsaug_use_magwarp = data_cfg.tsaug_use_magwarp

        self._node_col = "series_id"
        self._time_col = "timestamp"
        self.feature_cols = self._build_feature_columns()
        self.df = None

        if self.partition_mode == "shared":
            print("[PCAD] Loading shared data")
            self.df = self._load_shared_dataframe()
            self._prepare_data(self.df)
            self._prepare_partitions()

        elif self.partition_mode == "local":
            self.client_indices = list(range(self.num_clients))

        else:
            raise ValueError(
                f"Unsupported partition_mode: {self.partition_mode}. "
                "Expected 'shared', 'local', or 'optuna'."
            )

    # ── Public API ─────────────────────────────────────────────────────── #

    def get_dataloaders(self, partition_id: int):
        print(
            f"[PCAD] get_dataloaders partition_mode={self.partition_mode} "
            f"partition_id={partition_id}"
        )

        if self.partition_mode == "local":
            path = self._resolve_local_file_path(partition_id)
            self._prepare_data(self._read_csv_with_source(path))
            client_df = self.df
        else:
            if partition_id < 0 or partition_id >= len(self.client_indices):
                raise ValueError(f"Invalid partition_id: {partition_id}")
            client_df = self.df[
                self.df[self._node_col].isin(self.client_indices[partition_id])
            ]

        X_train, y_train, X_val, y_val = temporal_grouped_split(
            client_df,
            feature_cols=self.feature_cols,
            node_col=self._node_col,
            time_col=self._time_col,
            train_ratio=1.0 - self.test_split,
            gap_hours=self.gap_hours,
            window_size=self.window_size,
            stride=self.stride,
            target=self.target,
        )

        X_train, X_val = self._scale_temporal_arrays(X_train, X_val)

        X_train, y_train = self._oversample_anomalies(
            X_train,
            y_train,
            method=self.oversampling_method,
            target_ratio=self.oversampling_ratio,
            smote_k_neighbors=self.smote_k_neighbors,
            seed=self.seed + 20_000 + partition_id,
            split_name="train",
        )

        if self.oversample_val:
            X_val, y_val = self._oversample_anomalies(
                X_val,
                y_val,
                method=self.oversampling_method,
                target_ratio=self.oversampling_ratio,
                smote_k_neighbors=self.smote_k_neighbors,
                seed=self.seed + 30_000 + partition_id,
                split_name="val",
            )
        else:
            print("[PCAD] Keeping validation set unchanged after oversampling")

        print(f"[PCAD] Split done: X_train={X_train.shape}, X_val={X_val.shape}")

        return self._build_dataloaders_from_arrays(X_train, y_train, X_val, y_val)

    def get_metadata(self):
        return {
            "input_dim": len(self.feature_cols),
            "num_classes": self.config.model.num_classes,
            "num_samples": 0 if self.df is None else self.df.shape[0],
            "task_type": self.config.task.name,
            "data_format": "tabular",
        }

    def get_num_partitions(self) -> int:
        return len(self.client_indices)

    # ── Data preparation helpers ───────────────────────────────────────── #

    def _build_feature_columns(self):
        return [
            "active_power",         # power usage
            "power_lag1",           # power usage 1 minute ago
            "power_lag5",           # power usage 5 minutes ago
            "power_roll_mean_5",    # rolling mean over last 5 minutes
            "power_roll_std_5",     # rolling std over last 5 minutes
            "power_roll_mean_30",   # rolling mean over last 30 minutes
            "power_roll_std_30",    # rolling std over last 30 minutes
            "power_diff_1",         # delta vs 1 minute ago
            "power_zscore_30",      # current value normalized by last-30-min stats
            "hour_x",
            "hour_y",
            "weekday_x",
            "weekday_y",
        ]

    def _load_shared_dataframe(self) -> pd.DataFrame:
        if self.file_path.exists() and self.file_path.is_file():
            return self._read_csv_with_source(self.file_path)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Power consumption anomaly data not found. Expected either "
                f"file_path={self.file_path} or data_dir={self.data_dir}."
            )

        paths = sorted(self.data_dir.rglob(self.file_pattern))
        if not paths:
            raise FileNotFoundError(
                f"No files matching '{self.file_pattern}' found under {self.data_dir}."
            )

        frames = [self._read_csv_with_source(path) for path in paths]
        return pd.concat(frames, ignore_index=True)

    def _read_csv_with_source(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)

        has_timestamp = self._first_existing_column(df, self.TIMESTAMP_CANDIDATES)
        has_power = self._first_existing_column(df, self.POWER_CANDIDATES)
        if has_timestamp is None or has_power is None:
            return pd.DataFrame()

        if "label" not in df.columns:
            df["label"] = 0

        df["_source_file"] = path.stem
        return df

    def _first_existing_column(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        by_lower = {column.lower(): column for column in df.columns}
        for candidate in candidates:
            column = by_lower.get(candidate.lower())
            if column is not None:
                return column
        return None

    def _encode_target(self, series: pd.Series) -> pd.Series:
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
        negative_values = {
            "0",
            "false",
            "no",
            "n",
            "normal",
            "healthy",
            "regular",
            "none",
        }

        unknown_values = set(normalized.unique()) - positive_values - negative_values
        if unknown_values:
            raise ValueError(
                "Unknown target labels found in power consumption anomaly data: "
                f"{sorted(unknown_values)}"
            )

        return normalized.isin(positive_values).astype(np.int64)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(column).strip() for column in df.columns]

        timestamp_col = self._first_existing_column(df, self.TIMESTAMP_CANDIDATES)
        power_col = self._first_existing_column(df, self.POWER_CANDIDATES)
        target_col = self.target if self.target in df.columns else None
        if target_col is None:
            target_col = self._first_existing_column(df, self.TARGET_CANDIDATES)
        node_col = self._first_existing_column(df, self.NODE_CANDIDATES)

        if timestamp_col is None:
            raise ValueError(
                "Could not find a timestamp column. Tried: "
                f"{self.TIMESTAMP_CANDIDATES}"
            )
        if power_col is None:
            raise ValueError(
                "Could not find an active power column. Tried: "
                f"{self.POWER_CANDIDATES}"
            )
        if target_col is None:
            raise ValueError(
                "Could not find a target/anomaly label column. Tried configured "
                f"target='{self.target}' and {self.TARGET_CANDIDATES}"
            )

        try:
            df[self._time_col] = pd.to_datetime(
                df[timestamp_col],
                errors="coerce",
                format="mixed",
            )
        except ValueError:
            df[self._time_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df["active_power"] = pd.to_numeric(df[power_col], errors="coerce")
        df[self.target] = self._encode_target(df[target_col])

        if node_col is not None:
            df[self._node_col] = df[node_col].astype(str)
        else:
            df[self._node_col] = df["_source_file"].astype(str)

        df = df.dropna(subset=[self._time_col, "active_power", self.target])
        df = df.sort_values([self._node_col, self._time_col])

        groups = df.groupby(self._node_col, sort=False)
        df["power_lag1"] = groups["active_power"].shift(1)
        df["power_lag5"] = groups["active_power"].shift(5)
        df["power_roll_mean_5"] = groups["active_power"].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df["power_roll_std_5"] = groups["active_power"].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        df["power_roll_mean_30"] = groups["active_power"].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        df["power_roll_std_30"] = groups["active_power"].transform(
            lambda x: x.rolling(window=30, min_periods=1).std()
        )
        df["power_diff_1"] = df["active_power"] - df["power_lag1"]
        df["power_zscore_30"] = (
            (df["active_power"] - df["power_roll_mean_30"])
            / (df["power_roll_std_30"] + 1e-6)
        )

        hour = df[self._time_col].dt.hour
        weekday = df[self._time_col].dt.weekday
        df["hour_x"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_y"] = np.cos(2 * np.pi * hour / 24.0)
        df["weekday_x"] = np.sin(2 * np.pi * weekday / 7.0)
        df["weekday_y"] = np.cos(2 * np.pi * weekday / 7.0)

        df[self.feature_cols] = df[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=self.feature_cols)

        print(
            f"[PCAD] Processed shape={df.shape}, "
            f"series={df[self._node_col].nunique()}, "
            f"anomaly_rate={df[self.target].mean():.4f}"
        )
        return df

    def _prepare_data(self, raw_df: pd.DataFrame):
        df = self._preprocess(raw_df)
        self.df = df
        X = df[self.feature_cols].values.astype(np.float32)
        y = df[self.target].values.astype(np.int64)
        return X, y

    def _scale_temporal_arrays(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.normalize:
            return X_train.astype(np.float32), X_val.astype(np.float32)

        scaler = StandardScaler()
        train_shape = X_train.shape
        val_shape = X_val.shape

        # StandardScaler returns float64; floor back to float32 to avoid carrying
        # a 2x-size array through the rest of the pipeline.
        X_train_scaled = scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(train_shape).astype(np.float32)
        X_val_scaled = scaler.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(val_shape).astype(np.float32)

        print("[PCAD] Applied StandardScaler using training windows only")
        return X_train_scaled, X_val_scaled

    def _build_dataloaders_from_arrays(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )

        return (
            torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            ),
            torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            ),
        )

    # ── Path helpers ───────────────────────────────────────────────────── #

    def _resolve_local_file_path(self, partition_id: int) -> Path:
        if partition_id < 0:
            raise ValueError(f"Invalid partition_id: {partition_id}")

        client_index = partition_id + 1
        path = self.data_dir / self.file_pattern.format(client_index=client_index)

        if not path.exists():
            raise FileNotFoundError(
                f"Local client file not found for partition {partition_id}: {path}"
            )

        return path

    # ── Partitioning ───────────────────────────────────────────────────── #

    def _prepare_partitions(self):
        series_stats = (
            self.df.groupby(self._node_col)
            .agg(num_rows=(self.target, "size"), num_anomalies=(self.target, "sum"))
            .reset_index()
        )

        if self.num_clients > len(series_stats):
            raise ValueError(
                f"num_clients={self.num_clients} is greater than available "
                f"series/appliances={len(series_stats)}."
            )

        rng = np.random.default_rng(self.seed)
        series_stats["_tie_break"] = rng.random(len(series_stats))
        series_stats = series_stats.sort_values(
            ["num_anomalies", "num_rows", "_tie_break"],
            ascending=[False, False, True],
        )

        clients = [
            {"series": [], "rows": 0, "anomalies": 0}
            for _ in range(self.num_clients)
        ]

        for _, row in series_stats.iterrows():
            best_idx = min(
                range(self.num_clients),
                key=lambda idx: (
                    clients[idx]["anomalies"],
                    clients[idx]["rows"],
                ),
            )
            clients[best_idx]["series"].append(row[self._node_col])
            clients[best_idx]["rows"] += int(row["num_rows"])
            clients[best_idx]["anomalies"] += int(row["num_anomalies"])

        self.client_indices = [
            np.array(client["series"])
            for client in clients
        ]

        print("[PCAD] Shared partition summary:")
        for idx, client in enumerate(clients):
            anomaly_rate = client["anomalies"] / client["rows"] if client["rows"] else 0.0
            print(
                f"[PCAD] client={idx} series={len(client['series'])} "
                f"rows={client['rows']} anomalies={client['anomalies']} "
                f"anomaly_rate={anomaly_rate:.4f}"
            )

    # ── Resampling ─────────────────────────────────────────────────────── #

    def _oversample_anomalies(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        method: str,
        target_ratio: float,
        smote_k_neighbors: int,
        seed: int,
        split_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if method in [None, "none"]:
            return X, y

        positives = int(y.sum())
        negatives = int(len(y) - positives)

        if positives == 0 or negatives == 0:
            print(
                f"[PCAD] WARNING: Oversampling skipped for {split_name}; "
                "requires both classes."
            )
            return X, y

        current_ratio = positives / negatives

        if target_ratio <= current_ratio:
            print(
                f"[PCAD] Oversampling skipped for {split_name}: "
                f"current positive:negative ratio={current_ratio:.4f} "
                f"is already >= target={target_ratio:.4f}"
            )
            return X, y

        rng = np.random.default_rng(seed)

        if method in ("ts_augment", "time_series_augment"):
            target_pos_count = int(target_ratio * negatives)
            n_needed = target_pos_count - positives
            if n_needed <= 0:
                return X, y

            anomaly_idx = np.where(y == 1)[0]
            base_idx = rng.choice(anomaly_idx, size=n_needed, replace=True)
            X_synth = X[base_idx].copy()

            if self.tsaug_use_jitter:
                X_synth = jitter(X_synth, self.tsaug_jitter_sigma, rng)
            if self.tsaug_use_scaling:
                X_synth = scaling(X_synth, self.tsaug_scaling_sigma, rng)
            if self.tsaug_use_magwarp:
                X_synth = magnitude_warp(
                    X_synth,
                    self.tsaug_magwarp_sigma,
                    self.tsaug_magwarp_knots,
                    rng,
                )

            X_out = np.concatenate([X, X_synth], axis=0)
            y_out = np.concatenate(
                [y, np.ones(n_needed, dtype=y.dtype)], axis=0
            )

            perm = rng.permutation(len(y_out))
            X_out = X_out[perm]
            y_out = y_out[perm]

            print(
                f"[PCAD] Applied {split_name} oversampling: "
                f"method={method}, "
                f"target_pos_neg={target_ratio}:1, "
                f"before={len(y)}, pos_before={positives}, "
                f"after={len(y_out)}, pos_after={int(y_out.sum())}, "
                f"anomaly_rate={float(y_out.mean()):.4f}"
            )

            return X_out, y_out

        original_shape = X.shape
        X_flat = X.reshape(original_shape[0], -1)

        if method == "random_over":
            from imblearn.over_sampling import RandomOverSampler

            sampler = RandomOverSampler(
                sampling_strategy=target_ratio,
                random_state=seed,
            )

        elif method == "smote":
            if positives < 2:
                print(
                    f"[PCAD] WARNING: SMOTE skipped for {split_name}; "
                    "requires at least two positive samples."
                )
                return X, y

            from imblearn.over_sampling import SMOTE

            effective_k = min(smote_k_neighbors, positives - 1)
            sampler = SMOTE(
                sampling_strategy=target_ratio,
                random_state=seed,
                k_neighbors=effective_k,
            )

        else:
            raise ValueError(
                f"Unknown oversampling_method='{method}'. "
                "Expected 'none', 'random_over', 'smote', or 'ts_augment'."
            )

        X_resampled, y_resampled = sampler.fit_resample(X_flat, y)

        # imblearn (especially SMOTE) returns float64 from its interpolation;
        # floor back to float32 to keep memory consistent with the rest of the
        # pipeline.
        X_resampled = X_resampled.reshape((-1, *original_shape[1:])).astype(
            np.float32,
            copy=False,
        )

        print(
            f"[PCAD] Applied {split_name} oversampling: "
            f"method={method}, "
            f"target_pos_neg={target_ratio}:1, "
            f"before={len(y)}, pos_before={positives}, "
            f"after={len(y_resampled)}, pos_after={int(y_resampled.sum())}, "
            f"anomaly_rate={float(y_resampled.mean()):.4f}"
        )

        return X_resampled, y_resampled
