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

    Supports three modes:
    - shared: load one large CSV and partition internally by building_id
    - local: load one already client-specific CSV file
    - optuna: load one large CSV and use the entire dataset as one partition

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
        self.use_undersampling = getattr(config.data, "use_undersampling", False)
        self.undersampling_ratio = getattr(config.data, "undersampling_ratio", 1.0)
        self.undersample_val = getattr(config.data, "undersample_val", False)

        self.use_oversampling = getattr(config.data, "use_oversampling", False)
        self.oversampling_method = getattr(config.data, "oversampling_method", "none")
        self.oversampling_ratio = getattr(config.data, "oversampling_ratio", 1.0)
        self.smote_k_neighbors = getattr(config.data, "smote_k_neighbors", 5)
        self.oversample_val = getattr(config.data, "oversample_val", False)

        self.data_dir = getattr(config.data, "data_dir", None)
        self.file_pattern = getattr(config.data, "file_pattern", None)
        self.precomputed_dir = getattr(config.data, "precomputed_dir", None)
        self.precomputed_pattern = getattr(config.data, "precomputed_pattern", None)
        self.use_precomputed_windows = getattr(
            config.data,
            "use_precomputed_windows",
            False,
        )

        self._node_col = "building_id"
        self._time_col = "timestamp"

        self.df = None
        self.features = None
        self.labels = None
        self.feature_cols = self._build_feature_columns()
        self.aggregation_weight = 0
        self.num_train_windows_before_undersampling = 0
        self.num_train_anomalies_before_undersampling = 0
        self.num_train_windows_after_undersampling = 0
        self._gap_hours = config.data.gap_hours
        self._stride = config.data.stride
        self._window_size = config.data.window_size
        self.global_pos_weight = None

        if self.partition_mode in ["shared", "optuna"]:
            print(f"[LEAD] Loading {self.partition_mode} file: {self.file_path}")
            self.df = pd.read_csv(self.file_path)
            print(f"[LEAD] Raw shape: {self.df.shape}")

            if self.target not in self.df.columns:
                raise ValueError(
                    f"CSV file must contain target column '{self.target}', "
                    f"but columns were: {list(self.df.columns)}"
                )

            self.features, self.labels = self._prepare_data(self.df)

            n_pos = int(self.labels.sum())
            n_neg = len(self.labels) - n_pos
            if n_pos > 0:
                self.global_pos_weight = min(
                    n_neg / n_pos,
                    getattr(self.config.model, "pos_weight_cap", 10.0),
                )
            else:
                self.global_pos_weight = getattr(
                    self.config.model, "pos_weight_cap", 10.0
                )

            if self.partition_mode == "shared":
                self._prepare_partitions()
            else:
                self.client_indices = [np.array(self.df[self._node_col].unique())]
                print(
                    "[LEAD] Optuna mode enabled: using the full dataset "
                    "as one partition."
                )

        elif self.partition_mode == "local":
            self.client_indices = list(range(self.num_clients))

        else:
            raise ValueError(
                f"Unsupported partition_mode: {self.partition_mode}. "
                f"Expected 'shared', 'local', or 'optuna'."
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

    def _resolve_precomputed_file_path(self, partition_id: int) -> Path:
        if partition_id < 0:
            raise ValueError(f"Invalid partition_id: {partition_id}")

        if not self.precomputed_dir or not self.precomputed_pattern:
            raise ValueError(
                "Precomputed LEAD windows require data.precomputed_dir and "
                "data.precomputed_pattern to be configured."
            )

        client_index = partition_id + 1
        path = Path(self.precomputed_dir) / self.precomputed_pattern.format(
            client_index=client_index
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Precomputed windows not found for partition {partition_id}: {path}. "
                "Run scripts/precompute_lead_windows.py before starting Flower."
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
            "wind_dir_x",
            "wind_dir_y",
            "wind_dir_missing",
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

    def _load_precomputed_dataloaders(self, partition_id: int):
        path = self._resolve_precomputed_file_path(partition_id)
        print(f"[LEAD] Loading precomputed windows: {path}")

        with np.load(path) as artifact:
            required = ["X_train", "y_train", "X_val", "y_val"]
            missing = [key for key in required if key not in artifact]

            if missing:
                raise ValueError(
                    f"Precomputed artifact {path} is missing arrays: {missing}"
                )

            X_train = artifact["X_train"].astype(np.float32, copy=False)
            y_train = artifact["y_train"].astype(np.int64, copy=False)
            X_val = artifact["X_val"].astype(np.float32, copy=False)
            y_val = artifact["y_val"].astype(np.int64, copy=False)

            self.num_train_windows_before_undersampling = int(
                artifact["num_train_windows_before_undersampling"][0]
            ) if "num_train_windows_before_undersampling" in artifact else len(y_train)
            self.num_train_anomalies_before_undersampling = int(
                artifact["num_train_anomalies_before_undersampling"][0]
            ) if "num_train_anomalies_before_undersampling" in artifact else int(y_train.sum())
            self.num_train_windows_after_undersampling = int(
                artifact["num_train_windows_after_undersampling"][0]
            ) if "num_train_windows_after_undersampling" in artifact else len(y_train)
            self.aggregation_weight = int(
                artifact["aggregation_weight"][0]
            ) if "aggregation_weight" in artifact else self.num_train_windows_before_undersampling
            oversampling_method = (
                str(artifact["oversampling_method"][0])
                if "oversampling_method" in artifact
                else "unknown"
            )
            num_train_windows_after_oversampling = int(
                artifact["num_train_windows_after_oversampling"][0]
            ) if "num_train_windows_after_oversampling" in artifact else len(y_train)
            num_train_anomalies_after_oversampling = int(
                artifact["num_train_anomalies_after_oversampling"][0]
            ) if "num_train_anomalies_after_oversampling" in artifact else int(y_train.sum())
            oversample_val = bool(
                artifact["oversample_val"][0]
            ) if "oversample_val" in artifact else False
            num_val_windows_after_oversampling = int(
                artifact["num_val_windows_after_oversampling"][0]
            ) if "num_val_windows_after_oversampling" in artifact else len(y_val)
            num_val_anomalies_after_oversampling = int(
                artifact["num_val_anomalies_after_oversampling"][0]
            ) if "num_val_anomalies_after_oversampling" in artifact else int(y_val.sum())

        self.df = None
        self.features = X_train
        self.labels = y_train

        print(
            f"[LEAD] Precomputed shapes: "
            f"X_train={X_train.shape}, X_val={X_val.shape}, "
            f"aggregation_weight={self.aggregation_weight}, "
            f"oversampling={oversampling_method}, "
            f"train_after_oversampling={num_train_windows_after_oversampling}, "
            f"train_anomalies_after_oversampling={num_train_anomalies_after_oversampling}, "
            f"oversample_val={oversample_val}, "
            f"val_after_oversampling={num_val_windows_after_oversampling}, "
            f"val_anomalies_after_oversampling={num_val_anomalies_after_oversampling}"
        )

        return self._build_dataloaders_from_arrays(
            X_train,
            y_train,
            X_val,
            y_val,
        )

    def _prepare_partitions(self):
        if self.df is None:
            raise RuntimeError("Cannot prepare partitions before dataframe is loaded.")

        if self._node_col not in self.df.columns:
            raise ValueError(f"Missing node column: {self._node_col}")

        if self.target not in self.df.columns:
            raise ValueError(f"Missing target column: {self.target}")

        rng = np.random.default_rng(self.seed)

        building_stats = (
            self.df
            .groupby(self._node_col)
            .agg(
                num_rows=(self.target, "size"),
                num_anomalies=(self.target, "sum"),
            )
            .reset_index()
        )

        building_stats["num_rows"] = building_stats["num_rows"].astype(int)
        building_stats["num_anomalies"] = building_stats["num_anomalies"].astype(int)
        building_stats["anomaly_rate"] = (
            building_stats["num_anomalies"] / building_stats["num_rows"]
        ).fillna(0.0)
        building_stats["_tie_break"] = rng.random(len(building_stats))

        building_stats = building_stats.sort_values(
            by=["anomaly_rate", "num_rows", "_tie_break"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        total_rows = int(building_stats["num_rows"].sum())
        total_anomalies = int(building_stats["num_anomalies"].sum())
        total_buildings = int(len(building_stats))

        if self.num_clients > total_buildings:
            raise ValueError(
                f"num_clients={self.num_clients} is greater than number of "
                f"available buildings={total_buildings}. This would create empty clients."
            )

        clients = [
            {
                "buildings": [],
                "num_rows": 0,
                "num_anomalies": 0,
                "num_buildings": 0,
            }
            for _ in range(self.num_clients)
        ]

        # Round-robin by anomaly rate: walk buildings from highest anomaly rate
        # to lowest, cycling through clients (0,1,...,K-1,0,1,...). Each client
        # gets every K-th building in the sorted order, guaranteeing balanced
        # building counts and a representative mix of rates.
        for i, (_, row) in enumerate(building_stats.iterrows()):
            client_idx = i % self.num_clients
            clients[client_idx]["buildings"].append(row[self._node_col])
            clients[client_idx]["num_rows"] += int(row["num_rows"])
            clients[client_idx]["num_anomalies"] += int(row["num_anomalies"])
            clients[client_idx]["num_buildings"] += 1

        self.client_indices = [
            np.array(client["buildings"])
            for client in clients
        ]

        print("[LEAD] Shared partition summary:")
        print(
            f"[LEAD] total_buildings={total_buildings}, "
            f"total_rows={total_rows}, total_anomalies={total_anomalies}"
        )

        for idx, client in enumerate(clients):
            rows = client["num_rows"]
            anomalies = client["num_anomalies"]
            anomaly_rate = anomalies / rows if rows > 0 else 0.0

            print(
                f"[LEAD] client={idx} "
                f"buildings={client['num_buildings']} "
                f"rows={rows} "
                f"anomalies={anomalies} "
                f"anomaly_rate={anomaly_rate:.4f}"
            )

        zero_anomaly_clients = [
            idx for idx, client in enumerate(clients)
            if client["num_anomalies"] == 0
        ]

        if zero_anomaly_clients:
            print(
                "[LEAD] WARNING: Some clients received zero anomaly rows: "
                f"{zero_anomaly_clients}. "
                "This may hurt federated anomaly detection."
            )

    def get_dataloaders(self, partition_id: int):
        print(
            f"[LEAD] get_dataloaders partition_mode={self.partition_mode} "
            f"partition_id={partition_id}"
        )

        if self.partition_mode == "local" and self.use_precomputed_windows:
            return self._load_precomputed_dataloaders(partition_id)

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

        elif self.partition_mode == "optuna":
            if partition_id != 0:
                print(
                    f"[LEAD] Optuna mode ignores partition_id={partition_id}; "
                    "using the full dataset."
                )

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

        self.num_train_windows_before_undersampling = len(y_train)
        self.num_train_anomalies_before_undersampling = int(y_train.sum())
        self.aggregation_weight = self.num_train_windows_before_undersampling

        X_train, X_val = self._scale_temporal_arrays(X_train, X_val)
        if self.use_undersampling:
            X_train, y_train = self._undersample_normals(
                X_train,
                y_train,
                normal_to_anomaly_ratio=self.undersampling_ratio,
                seed=self.seed + partition_id,
            )

            if self.undersample_val:
                X_val, y_val = self._undersample_normals(
                    X_val,
                    y_val,
                    normal_to_anomaly_ratio=self.undersampling_ratio,
                    seed=self.seed + 10_000 + partition_id,
                )
            else:
                print("[LEAD] Keeping validation set unchanged after undersampling step")

        if self.use_oversampling:
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
                print("[LEAD] Keeping validation set unchanged after oversampling step")

        self.num_train_windows_after_undersampling = len(y_train)
                
        return self._build_dataloaders_from_arrays(X_train, y_train, X_val, y_val)

    def get_metadata(self):
        if self.df is None:
            if self.partition_mode == "local":
                return {
                    "input_dim": len(self.feature_cols),
                    "num_classes": self.config.model.num_classes,
                    "num_samples": 0,
                    "task_type": self.config.task.name,
                    "data_format": "tabular",
                }
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
    
    def _undersample_normals(
            
        self,
        X: np.ndarray,
        y: np.ndarray,
        normal_to_anomaly_ratio: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        y = y.astype(np.int64)

        anomaly_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]

        n_anomalies = len(anomaly_idx)

        if n_anomalies == 0:
            print("[LEAD] WARNING: No anomalies found; skipping undersampling")
            return X, y

        n_normals_to_keep = int(n_anomalies * normal_to_anomaly_ratio)

        if len(normal_idx) <= n_normals_to_keep:
            print(
                "[LEAD] WARNING: Not enough normal samples for undersampling; "
                "keeping original data"
            )
            return X, y

        sampled_normal_idx = rng.choice(
            normal_idx,
            size=n_normals_to_keep,
            replace=False,
        )

        selected_idx = np.concatenate([anomaly_idx, sampled_normal_idx])
        rng.shuffle(selected_idx)

        X_under = X[selected_idx]
        y_under = y[selected_idx]

        print(
            f"[LEAD] Applied undersampling: "
            f"ratio={normal_to_anomaly_ratio}:1, "
            f"before={len(y)}, after={len(y_under)}, "
            f"anomaly_rate={y_under.mean():.4f}"
        )

        return X_under, y_under
    
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
        y = y.astype(np.int64, copy=False)

        if method in [None, "none"]:
            return X, y

        positives = int(y.sum())
        negatives = int(len(y) - positives)

        if positives == 0 or negatives == 0:
            print(
                f"[LEAD] WARNING: Oversampling skipped for {split_name}; "
                "requires both classes."
            )
            return X, y

        current_ratio = positives / negatives

        if target_ratio <= current_ratio:
            print(
                f"[LEAD] Oversampling skipped for {split_name}: "
                f"current positive:negative ratio={current_ratio:.4f} "
                f"is already >= target={target_ratio:.4f}"
            )
            return X, y

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
                    f"[LEAD] WARNING: SMOTE skipped for {split_name}; "
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

        elif method in ["borderline_smote", "borderlinesmote"]:
            if positives < 2:
                print(
                    f"[LEAD] WARNING: BorderlineSMOTE skipped for {split_name}; "
                    "requires at least two positive samples."
                )
                return X, y

            from imblearn.over_sampling import BorderlineSMOTE

            effective_k = min(smote_k_neighbors, positives - 1)

            sampler = BorderlineSMOTE(
                sampling_strategy=target_ratio,
                random_state=seed,
                k_neighbors=effective_k,
                m_neighbors=min(10, max(1, len(y) - 1)),
                kind="borderline-1",
            )
        elif method in ["time_series_augment", "ts_augment"]:
            rng = np.random.default_rng(seed)

            anomaly_idx = np.where(y == 1)[0]
            normal_idx = np.where(y == 0)[0]

            target_positives = int(target_ratio * len(normal_idx))
            n_to_generate = max(0, target_positives - len(anomaly_idx))

            if n_to_generate <= 0:
                print(
                    f"[LEAD] Time-series augmentation skipped for {split_name}; "
                    "target ratio already reached."
                )
                return X, y

            source_idx = rng.choice(
                anomaly_idx,
                size=n_to_generate,
                replace=True,
            )

            X_new = X[source_idx].copy()

            # 1. Jittering: small Gaussian noise
            noise_std = 0.02
            X_new = X_new + rng.normal(
                loc=0.0,
                scale=noise_std,
                size=X_new.shape,
            ).astype(np.float32)

            # 2. Magnitude scaling: slightly scale each window
            scale = rng.normal(
                loc=1.0,
                scale=0.05,
                size=(n_to_generate, 1, 1),
            ).astype(np.float32)
            X_new = X_new * scale

            # 3. Time shift: roll the sequence slightly forward/backward
            max_shift = 3
            for i in range(n_to_generate):
                shift = rng.integers(-max_shift, max_shift + 1)
                X_new[i] = np.roll(X_new[i], shift=shift, axis=0)

            y_new = np.ones(n_to_generate, dtype=np.int64)

            X_resampled = np.concatenate([X, X_new], axis=0)
            y_resampled = np.concatenate([y, y_new], axis=0)

            shuffle_idx = rng.permutation(len(y_resampled))
            X_resampled = X_resampled[shuffle_idx].astype(np.float32, copy=False)
            y_resampled = y_resampled[shuffle_idx].astype(np.int64, copy=False)

            print(
                f"[LEAD] Applied {split_name} time-series augmentation: "
                f"target_pos_neg={target_ratio}:1, "
                f"before={len(y)}, pos_before={positives}, "
                f"generated={n_to_generate}, "
                f"after={len(y_resampled)}, pos_after={int(y_resampled.sum())}, "
                f"anomaly_rate={float(y_resampled.mean()):.4f}"
            )

            return X_resampled, y_resampled
        else:
            raise ValueError(
                f"Unknown oversampling_method='{method}'. "
                "Expected 'none', 'random_over', 'smote', 'borderline_smote', or 'time_series_augment'."
            )

        X_resampled, y_resampled = sampler.fit_resample(X_flat, y)

        X_resampled = X_resampled.reshape((-1, *original_shape[1:])).astype(
            np.float32,
            copy=False,
        )
        y_resampled = y_resampled.astype(np.int64, copy=False)

        print(
            f"[LEAD] Applied {split_name} oversampling: "
            f"method={method}, "
            f"target_pos_neg={target_ratio}:1, "
            f"before={len(y)}, pos_before={positives}, "
            f"after={len(y_resampled)}, pos_after={int(y_resampled.sum())}, "
            f"anomaly_rate={float(y_resampled.mean()):.4f}"
        )

        return X_resampled, y_resampled