import torch
import numpy as np
import pandas as pd

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
        self.batch_size = config.model.batch_size
        self.test_split = config.data.test_split
        self.num_clients = config.federation.num_clients
        self.seed = config.data.seed
        self.partition_mode = getattr(config.federation, "partition_mode")

        # Optional local-mode settings
        self.data_dir = getattr(config.data, "data_dir", None)
        self.file_pattern = getattr(config.data, "file_pattern", None)

        self._node_col = "building_id"
        self._time_col = "timestamp"

        self.df = None
        self.features = None
        self.labels = None
        self.feature_cols = self._build_feature_columns()
        self._gap_hours = config.data.gap_hours
        self._stride = config.data.stride
        self._window_size = config.data.window_size

        if self.partition_mode == "shared":
            print(f"[LEAD] Loading shared file: {self.file_path}")
            self.df = pd.read_csv(self.file_path)
            print(f"[LEAD] Raw shape: {self.df.shape}")

            if self.target not in self.df.columns:
                raise ValueError(
                    f"CSV file must contain target column '{self.target}', "
                    f"but columns were: {list(self.df.columns)}"
                )

            self.features, self.labels = self._prepare_data(self.df)
            self._prepare_partitions()

        # In local mode each client gets its own file later in get_dataloaders(...)
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
            path = Path(self.file_path) # Fallback to single-file behavior if no local pattern is configured

        if not path.exists():
            raise FileNotFoundError(
                f"Local client file not found for partition {partition_id}: {path}"
            )

        return path

    def _build_feature_columns(self):
        # Static per-building identity features (site_id, square_feet, year_built,
        # floor_count, primary_use_*) are excluded — they let the model memorize
        # building identity and overfit when train/val share buildings.
        base_features = [
            "meter_reading",
            "air_temperature",
            "cloud_coverage",
            "dew_temperature",
            "precip_depth_1_hr",
            "sea_level_pressure",
            # wind_direction (raw degrees) is replaced by cyclic encoding below.
            "wind_dir_x",
            "wind_dir_y",
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
            # Missingness indicators emitted by scripts/clean_lead_features.py.
            # Each flag is 1.0 when the corresponding raw feature was a sentinel
            # in the source CSV and was replaced by an imputed value.
            "cloud_coverage_was_missing",
            "wind_dir_missing",
            "wind_speed_was_missing",
            "precip_depth_was_missing",
            "air_temp_std_lag7_was_missing",
            "air_temp_std_lag73_was_missing",
        ]

        return base_features

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

        # Re-cast all float columns to float32 after feature engineering
        # (lag/rolling/diff/zscore ops above silently upcast to float64).
        float_cols = df.select_dtypes(include=["float64", "float32"]).columns
        df[float_cols] = df[float_cols].astype("float32")

        df = df.dropna()

        print(f"[LEAD] Processed shape: {df.shape}")
        return df

    def _prepare_data(self, raw_df: pd.DataFrame):
        df = self._preprocess(raw_df)

        print(f"[LEAD] Feature count: {len(self.feature_cols)}")

        missing_features = [
            col for col in self.feature_cols if col not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing expected feature columns: {missing_features}"
            )

        self.df = df

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
        X_train_scaled = (
            scaler.fit_transform(X_train_2d)
            .astype(np.float32, copy=False)
            .reshape(original_train_shape)
        )
        del X_train_2d

        X_val_2d = X_val.reshape(-1, X_val.shape[-1])
        X_val_scaled = (
            scaler.transform(X_val_2d)
            .astype(np.float32, copy=False)
            .reshape(original_val_shape)
        )
        del X_val_2d

        print("[LEAD] Applied StandardScaler using training data only")

        return X_train_scaled, X_val_scaled

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
            print(f"[LEAD] Loading local client file: {file_path}")

            raw_df = pd.read_csv(file_path)
            print(f"[LEAD] Raw local shape: {raw_df.shape}")

            if self.target not in raw_df.columns:
                raise ValueError(
                    f"CSV file must contain target column '{self.target}', "
                    f"but columns were: {list(raw_df.columns)}"
                )

            _, _ = self._prepare_data(raw_df)
            client_df = self.df

        else:
            if partition_id < 0 or partition_id >= len(self.client_indices):
                raise ValueError(f"Invalid partition_id: {partition_id}")

            client_df = self.df[
                self.df[self._node_col].isin(self.client_indices[partition_id])
            ]

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

        print(f"[LEAD] Split done: X_train={X_train.shape}, X_val={X_val.shape}")

        X_train, X_val = self._scale_temporal_arrays(X_train, X_val)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )

        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
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
        if self.df is None:
            if self.partition_mode == "local":
                sample_path = self._resolve_local_file_path(0)
                print(
                    f"[LEAD] Loading representative local file for metadata: "
                    f"{sample_path}"
                )
                raw_df = pd.read_csv(sample_path)

                if self.target not in raw_df.columns:
                    raise ValueError(
                        f"CSV file must contain target column '{self.target}', "
                        f"but columns were: {list(raw_df.columns)}"
                    )

                self._prepare_data(raw_df)
            else:
                raise RuntimeError("Metadata requested before dataset was prepared")

        return {
            "input_dim": len(self.feature_cols),
            "num_classes": self.config.model.num_classes,
            "num_samples": self.df.shape[0],
            "task_type": self.config.task.name,
            "data_format": "tabular",
        }

    def get_num_partitions(self) -> int:
        return len(self.client_indices)