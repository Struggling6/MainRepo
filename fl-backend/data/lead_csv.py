import torch
import numpy as np
import pandas as pd

import hashlib
import json
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .time_series_utils import temporal_grouped_split
from .BaseDataHandler import BaseDatasetHandler


class LeadCSVHandler(BaseDatasetHandler):
    """
    Dataset handler for the LEAD building energy dataset.

    Supports two modes:
    - shared: load one large CSV and partition internally by building_id
    - local: load one already client-specific CSV file

    Also enforces a fixed one-hot schema for `primary_use` so all clients
    produce the same input dimensionality.
    """

    _gap_hours = 73
    _preprocess_cache_version = 1
    _window_cache_version = 1

    PRIMARY_USE_CATEGORIES = [
        "Education",
        "Entertainment/public assembly",
        "Food sales and service",
        "Healthcare",
        "Lodging/residential",
        "Manufacturing/industrial",
        "Office",
        "Other",
        "Parking",
        "Public services",
        "Religious worship",
        "Services",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.file_path = config.data.file_path
        self.target = config.data.target
        self.batch_size = config.data.batch_size
        self.test_split = config.data.test_split
        self.num_clients = config.federation.num_clients
        self.seed = config.data.seed
        self.partition_mode = getattr(config.federation, "partition_mode")

        self.data_dir = getattr(config.data, "data_dir", None)
        self.file_pattern = getattr(config.data, "file_pattern", None)
        self.cache_preprocessed = getattr(config.data, "cache_preprocessed", True)
        self.cache_windowed = getattr(config.data, "cache_windowed", True)
        self.cache_dir = getattr(config.data, "cache_dir", None)

        self._node_col = "building_id"
        self._time_col = "timestamp"

        self.df = None
        self.features = None
        self.labels = None
        self.num_samples = None
        self.feature_cols = self._build_feature_columns()

        if self.partition_mode == "shared":
            self.features, self.labels = self._prepare_file(self.file_path)
            self._prepare_partitions()

        elif self.partition_mode == "local":
            self.client_indices = list(range(self.num_clients))

        else:
            raise ValueError(
                f"Unsupported partition_mode: {self.partition_mode}. "
                f"Expected 'shared' or 'local'."
            )

    def _resolve_local_file_path(self, partition_id: int) -> Path:
        if partition_id < 0:
            raise ValueError(f"Invalid partition_id: {partition_id}")

        if self.data_dir and self.file_pattern:
            client_index = partition_id + 1
            path = Path(self.data_dir) / self.file_pattern.format(
                client_index=client_index
            )
        else:
            path = Path(self.file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Local client file not found for partition {partition_id}: {path}"
            )

        return path

    def _build_feature_columns(self):
        base_features = [
            "meter_reading",
            "site_id",
            "square_feet",
            "year_built",
            "floor_count",
            "air_temperature",
            "cloud_coverage",
            "dew_temperature",
            "precip_depth_1_hr",
            "sea_level_pressure",
            "wind_direction",
            "wind_speed",
            "air_temperature_mean_lag7",
            "air_temperature_max_lag7",
            "air_temperature_min_lag7",
            "air_temperature_std_lag7",
            "air_temperature_mean_lag73",
            "air_temperature_max_lag73",
            "air_temperature_min_lag73",
            "air_temperature_std_lag73",
            "hour_x",
            "hour_y",
            "month_x",
            "month_y",
            "weekday_x",
            "weekday_y",
            "is_holiday",
            "meter_lag1",
            "meter_lag24",
            "meter_roll_mean_24",
            "meter_roll_std_24",
            "meter_diff_1",
            "meter_diff_24",
            "meter_zscore_24",
        ]

        primary_use_cols = [
            f"primary_use_{cat}" for cat in self.PRIMARY_USE_CATEGORIES
        ]

        return base_features + primary_use_cols

    def _cache_path_for(self, source_path: Path) -> Path:
        source_path = Path(source_path)

        if self.cache_dir is None:
            return source_path.with_name(
                f"{source_path.name}.lead_v{self._preprocess_cache_version}.pkl"
            )

        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:12]
        return cache_dir / (
            f"{source_path.stem}.{digest}.lead_v{self._preprocess_cache_version}.pkl"
        )

    def _source_signature(self, source_path: Path) -> dict:
        stat = Path(source_path).stat()
        return {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "preprocess_version": self._preprocess_cache_version,
        }

    def _window_cache_signature(
        self,
        source_path: Path,
        partition_id: int,
        partition_values=None,
    ) -> dict:
        partition_hash = None

        if partition_values is not None:
            values = np.asarray(partition_values).astype(str)
            partition_hash = hashlib.sha1(
                "\n".join(values.tolist()).encode("utf-8")
            ).hexdigest()

        return {
            "source_path": str(Path(source_path).resolve()),
            "source_signature": self._source_signature(source_path),
            "partition_mode": self.partition_mode,
            "partition_id": partition_id,
            "partition_hash": partition_hash,
            "target": self.target,
            "feature_cols": self.feature_cols,
            "test_split": self.test_split,
            "gap_hours": self._gap_hours,
            "window_size": self._window_size,
            "stride": self._stride,
            "window_cache_version": self._window_cache_version,
        }

    def _window_cache_path_for(
        self,
        source_path: Path,
        partition_id: int,
        partition_values=None,
    ) -> Path:
        signature_json = json.dumps(
            self._window_cache_signature(
                source_path=source_path,
                partition_id=partition_id,
                partition_values=partition_values,
            ),
            sort_keys=True,
        )
        digest = hashlib.sha1(signature_json.encode("utf-8")).hexdigest()[:12]

        source_path = Path(source_path)
        if self.cache_dir is None:
            cache_dir = source_path.parent
        else:
            cache_dir = Path(self.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir / (
            f"{source_path.stem}.partition{partition_id}."
            f"{digest}.lead_windows_v{self._window_cache_version}.npz"
        )

    def _load_preprocessed_cache(self, source_path: Path) -> pd.DataFrame | None:
        if not self.cache_preprocessed:
            return None

        cache_path = self._cache_path_for(source_path)
        if not cache_path.exists():
            return None

        try:
            payload = pd.read_pickle(cache_path)
        except Exception as exc:
            print(f"[LEAD] Ignoring unreadable cache {cache_path}: {exc}")
            return None

        if not isinstance(payload, dict):
            print(f"[LEAD] Ignoring invalid cache format: {cache_path}")
            return None

        if payload.get("source_signature") != self._source_signature(source_path):
            print(f"[LEAD] Cache is stale: {cache_path}")
            return None

        df = payload.get("df")
        if not isinstance(df, pd.DataFrame):
            print(f"[LEAD] Cache did not contain a DataFrame: {cache_path}")
            return None

        print(f"[LEAD] Loaded preprocessed cache: {cache_path}")
        return df

    def _write_preprocessed_cache(self, source_path: Path, df: pd.DataFrame) -> None:
        if not self.cache_preprocessed:
            return

        cache_path = self._cache_path_for(source_path)
        tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.{os.getpid()}.tmp")
        payload = {
            "source_signature": self._source_signature(source_path),
            "df": df,
        }

        try:
            pd.to_pickle(payload, tmp_path)
            tmp_path.replace(cache_path)
            print(f"[LEAD] Wrote preprocessed cache: {cache_path}")
        except Exception as exc:
            print(f"[LEAD] Could not write preprocessed cache {cache_path}: {exc}")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _load_window_cache(
        self,
        source_path: Path,
        partition_id: int,
        partition_values=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
        if not self.cache_windowed:
            return None

        cache_path = self._window_cache_path_for(
            source_path=source_path,
            partition_id=partition_id,
            partition_values=partition_values,
        )

        if not cache_path.exists():
            return None

        expected_signature_json = json.dumps(
            self._window_cache_signature(
                source_path=source_path,
                partition_id=partition_id,
                partition_values=partition_values,
            ),
            sort_keys=True,
        )

        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                if payload["signature_json"].item() != expected_signature_json:
                    print(f"[LEAD] Window cache is stale: {cache_path}")
                    return None

                print(f"[LEAD] Loaded window cache: {cache_path}")
                return (
                    payload["X_train"],
                    payload["y_train"],
                    payload["X_val"],
                    payload["y_val"],
                    int(payload["num_samples"].item()),
                )
        except Exception as exc:
            print(f"[LEAD] Ignoring unreadable window cache {cache_path}: {exc}")
            return None

    def _write_window_cache(
        self,
        source_path: Path,
        partition_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_samples: int,
        partition_values=None,
    ) -> None:
        if not self.cache_windowed:
            return

        cache_path = self._window_cache_path_for(
            source_path=source_path,
            partition_id=partition_id,
            partition_values=partition_values,
        )
        tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.{os.getpid()}.tmp")
        signature_json = json.dumps(
            self._window_cache_signature(
                source_path=source_path,
                partition_id=partition_id,
                partition_values=partition_values,
            ),
            sort_keys=True,
        )

        try:
            with tmp_path.open("wb") as tmp_file:
                np.savez(
                    tmp_file,
                    signature_json=signature_json,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    num_samples=np.array(num_samples, dtype=np.int64),
                )
            tmp_path.replace(cache_path)
            print(f"[LEAD] Wrote window cache: {cache_path}")
        except Exception as exc:
            print(f"[LEAD] Could not write window cache {cache_path}: {exc}")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _load_or_preprocess_file(self, file_path: Path) -> pd.DataFrame:
        file_path = Path(file_path)
        cached_df = self._load_preprocessed_cache(file_path)
        if cached_df is not None:
            return cached_df

        print(f"[LEAD] Loading CSV file: {file_path}")
        raw_df = pd.read_csv(file_path)
        print(f"[LEAD] Raw shape: {raw_df.shape}")

        if self.target not in raw_df.columns:
            raise ValueError(
                f"CSV file must contain target column '{self.target}', "
                f"but columns were: {list(raw_df.columns)}"
            )

        df = self._preprocess(raw_df)
        self._write_preprocessed_cache(file_path, df)
        return df

    def _prepare_file(self, file_path: Path):
        return self._prepare_data(
            self._load_or_preprocess_file(file_path),
            already_preprocessed=True,
        )

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[LEAD] Starting _prepare_data")
        df = df.copy()

        df[self._time_col] = pd.to_datetime(df[self._time_col])
        df = df.sort_values(by=[self._node_col, self._time_col])

        float_cols = df.select_dtypes(include="float64").columns
        df[float_cols] = df[float_cols].astype("float32")

        int_cols = df.select_dtypes(include="int64").columns
        df[int_cols] = df[int_cols].astype("int32")

        df["primary_use"] = pd.Categorical(
            df["primary_use"],
            categories=self.PRIMARY_USE_CATEGORIES,
        )

        groups = df.groupby(self._node_col)

        df["meter_lag1"] = groups["meter_reading"].shift(1)
        df["meter_lag24"] = groups["meter_reading"].shift(24)

        df["meter_roll_mean_24"] = groups["meter_reading"].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )
        df["meter_roll_std_24"] = groups["meter_reading"].transform(
            lambda x: x.rolling(window=24, min_periods=1).std()
        )

        df["meter_diff_1"] = df["meter_reading"] - df["meter_lag1"]
        df["meter_diff_24"] = df["meter_reading"] - df["meter_lag24"]

        df["meter_zscore_24"] = (
            (df["meter_reading"] - df["meter_roll_mean_24"])
            / (df["meter_roll_std_24"] + 1e-6)
        )

        df = pd.get_dummies(df, columns=["primary_use"], drop_first=False)

        expected_primary_use_cols = [
            f"primary_use_{cat}" for cat in self.PRIMARY_USE_CATEGORIES
        ]

        for col in expected_primary_use_cols:
            if col not in df.columns:
                df[col] = 0.0

        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype("float32")

        df = df.dropna()

        print(f"[LEAD] Processed shape: {df.shape}")
        return df

    def _prepare_data(
        self,
        raw_df: pd.DataFrame,
        already_preprocessed: bool = False,
    ):
        df = raw_df if already_preprocessed else self._preprocess(raw_df)

        print(f"[LEAD] Feature count: {len(self.feature_cols)}")

        missing_features = [
            col for col in self.feature_cols if col not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing expected feature columns: {missing_features}"
            )

        self.df = df
        self.num_samples = df.shape[0]

        X = df[self.feature_cols].values.astype(np.float32)
        y = df[self.target].values.astype(np.int64)

        return X, y

    def _scale_temporal_arrays(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Standardize features after the temporal train/validation split.

        The scaler is fitted only on X_train to avoid validation leakage.
        Input shape is expected to be:
        (num_windows, window_size, num_features)
        """

        scaler = StandardScaler()

        original_train_shape = X_train.shape
        original_val_shape = X_val.shape

        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        X_val_2d = X_val.reshape(-1, X_val.shape[-1])

        X_train_scaled = scaler.fit_transform(X_train_2d).reshape(
            original_train_shape
        )
        X_val_scaled = scaler.transform(X_val_2d).reshape(original_val_shape)

        print("[LEAD] Applied StandardScaler using training data only")

        return (
            X_train_scaled.astype(np.float32),
            X_val_scaled.astype(np.float32),
        )

    def _prepare_partitions(self):
        """
        Shared mode: partition buildings across clients.
        Each building_id belongs to exactly one client partition.
        """

        rng = np.random.default_rng(self.seed)
        buildings = self.df[self._node_col].unique()
        rng.shuffle(buildings)
        self.client_indices = np.array_split(buildings, self.num_clients)

    def get_dataloaders(self, partition_id: int):
        print(
            f"[LEAD] get_dataloaders partition_mode={self.partition_mode} "
            f"partition_id={partition_id}"
        )

        if self.partition_mode == "local":
            file_path = self._resolve_local_file_path(partition_id)
            print(f"[LEAD] Preparing local client file: {file_path}")
            source_path = file_path
            partition_values = None

            cached_windows = self._load_window_cache(
                source_path=source_path,
                partition_id=partition_id,
                partition_values=partition_values,
            )

            if cached_windows is not None:
                X_train, y_train, X_val, y_val, num_samples = cached_windows
                self.num_samples = num_samples
            else:
                self.features, self.labels = self._prepare_file(file_path)
                client_df = self.df
                self.num_samples = client_df.shape[0]

                print(f"[LEAD] Client df shape: {client_df.shape}")

                X_train, y_train, X_val, y_val = temporal_grouped_split(
                    client_df,
                    feature_cols=self.feature_cols,
                    node_col=self._node_col,
                    time_col=self._time_col,
                    train_ratio=1.0 - self.test_split,
                    gap_hours=self._gap_hours,
                    window_size=self._window_size,
                    stride=self._stride,
                    target=self.target,
                )

                print(
                    f"[LEAD] Split done: "
                    f"X_train={X_train.shape}, X_val={X_val.shape}"
                )

                X_train, X_val = self._scale_temporal_arrays(X_train, X_val)
                self._write_window_cache(
                    source_path=source_path,
                    partition_id=partition_id,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    num_samples=self.num_samples,
                    partition_values=partition_values,
                )

        else:
            if partition_id < 0 or partition_id >= len(self.client_indices):
                raise ValueError(f"Invalid partition_id: {partition_id}")

            source_path = self.file_path
            partition_values = self.client_indices[partition_id]

            cached_windows = self._load_window_cache(
                source_path=source_path,
                partition_id=partition_id,
                partition_values=partition_values,
            )

            if cached_windows is not None:
                X_train, y_train, X_val, y_val, num_samples = cached_windows
                self.num_samples = num_samples
            else:
                client_df = self.df[
                    self.df[self._node_col].isin(partition_values)
                ]
                self.num_samples = client_df.shape[0]

                print(f"[LEAD] Client df shape: {client_df.shape}")

                X_train, y_train, X_val, y_val = temporal_grouped_split(
                    client_df,
                    feature_cols=self.feature_cols,
                    node_col=self._node_col,
                    time_col=self._time_col,
                    train_ratio=1.0 - self.test_split,
                    gap_hours=self._gap_hours,
                    window_size=self._window_size,
                    stride=self._stride,
                    target=self.target,
                )

                print(
                    f"[LEAD] Split done: "
                    f"X_train={X_train.shape}, X_val={X_val.shape}"
                )

                X_train, X_val = self._scale_temporal_arrays(X_train, X_val)
                self._write_window_cache(
                    source_path=source_path,
                    partition_id=partition_id,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    num_samples=self.num_samples,
                    partition_values=partition_values,
                )

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )

        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )

        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        valloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        return trainloader, valloader

    def get_metadata(self):
        if self.num_samples is None:
            if self.partition_mode == "local":
                sample_path = self._resolve_local_file_path(0)
                print(
                    f"[LEAD] Loading representative local file for metadata: "
                    f"{sample_path}"
                )
                self.features, self.labels = self._prepare_file(sample_path)
            else:
                raise RuntimeError("Metadata requested before dataset was prepared")

        return {
            "input_dim": len(self.feature_cols),
            "num_classes": self.config.model.num_classes,
            "num_samples": self.num_samples,
            "task_type": self.config.task.name,
            "data_format": "tabular",
        }

    def get_num_partitions(self) -> int:
        return len(self.client_indices)
