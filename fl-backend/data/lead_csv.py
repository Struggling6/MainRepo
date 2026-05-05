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

    _gap_hours = 73

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

        self.data_dir = getattr(config.data, "data_dir", None)
        self.file_pattern = getattr(config.data, "file_pattern", None)

        self._node_col = "building_id"
        self._time_col = "timestamp"

        self.df = None
        self.features = None
        self.labels = None
        self.feature_cols = self._build_feature_columns()

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
        ]

        primary_use_cols = [
            f"primary_use_{cat}" for cat in self.PRIMARY_USE_CATEGORIES
        ]

        return base_features + primary_use_cols

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
        building_stats["_tie_break"] = rng.random(len(building_stats))

        building_stats = building_stats.sort_values(
            by=["num_anomalies", "num_rows", "_tie_break"],
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

        ideal_rows = total_rows / self.num_clients
        ideal_anomalies = (
            total_anomalies / self.num_clients if total_anomalies > 0 else 0.0
        )
        ideal_buildings = total_buildings / self.num_clients

        clients = [
            {
                "buildings": [],
                "num_rows": 0,
                "num_anomalies": 0,
                "num_buildings": 0,
            }
            for _ in range(self.num_clients)
        ]

        def score_client_after_assignment(client, rows_to_add, anomalies_to_add):
            new_rows = client["num_rows"] + rows_to_add
            new_anomalies = client["num_anomalies"] + anomalies_to_add
            new_buildings = client["num_buildings"] + 1

            row_score = ((new_rows - ideal_rows) / max(ideal_rows, 1.0)) ** 2

            if total_anomalies > 0:
                anomaly_score = (
                    (new_anomalies - ideal_anomalies)
                    / max(ideal_anomalies, 1.0)
                ) ** 2
            else:
                anomaly_score = 0.0

            building_score = (
                (new_buildings - ideal_buildings)
                / max(ideal_buildings, 1.0)
            ) ** 2

            return (
                1.0 * row_score
                + 3.0 * anomaly_score
                + 0.2 * building_score
            )

        remaining_buildings = building_stats.copy()

        for client_idx in range(self.num_clients):
            row = remaining_buildings.iloc[0]
            remaining_buildings = remaining_buildings.iloc[1:].reset_index(drop=True)

            building_id = row[self._node_col]
            rows = int(row["num_rows"])
            anomalies = int(row["num_anomalies"])

            clients[client_idx]["buildings"].append(building_id)
            clients[client_idx]["num_rows"] += rows
            clients[client_idx]["num_anomalies"] += anomalies
            clients[client_idx]["num_buildings"] += 1

        for _, row in remaining_buildings.iterrows():
            building_id = row[self._node_col]
            rows = int(row["num_rows"])
            anomalies = int(row["num_anomalies"])

            best_client_idx = min(
                range(self.num_clients),
                key=lambda idx: score_client_after_assignment(
                    clients[idx],
                    rows,
                    anomalies,
                ),
            )

            clients[best_client_idx]["buildings"].append(building_id)
            clients[best_client_idx]["num_rows"] += rows
            clients[best_client_idx]["num_anomalies"] += anomalies
            clients[best_client_idx]["num_buildings"] += 1

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
                print("[LEAD] Keeping validation set unchanged")
                
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