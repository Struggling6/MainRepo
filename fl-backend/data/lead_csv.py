import torch
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .BaseDataHandler import BaseDatasetHandler
from .time_series_utils import temporal_grouped_split
from .ts_augment_utils import (
    jitter,
    magnitude_warp,
    mixup,
    scaling,
    time_warp,
    window_slicing,
)


class LeadCSVHandler(BaseDatasetHandler):
    """
    Dataset handler for the LEAD building energy dataset.

    Partition modes:
    - shared: load one large CSV and partition internally by building_id
    - local:  load this client's precomputed window artifact from disk
    - optuna: alias for shared mode with num_clients=1 (uses the full dataset)

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

        data_cfg = config.data
        self.file_path = data_cfg.file_path
        self.target = data_cfg.target
        self.batch_size = config.model.batch_size
        self.test_split = data_cfg.test_split
        self.seed = data_cfg.seed

        self.partition_mode = config.federation.partition_mode
        self.num_clients = config.federation.num_clients

        if self.partition_mode == "optuna":
            print("[LEAD] Optuna mode: forcing num_clients=1 for full-dataset trial")
            self.num_clients = 1
            self.partition_mode = "shared"

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

        self.precomputed_dir = data_cfg.precomputed_dir
        self.precomputed_pattern = data_cfg.precomputed_pattern

        self.use_precomputed_windows = getattr(data_cfg, 'use_precomputed_windows', True)
        self.data_dir = getattr(data_cfg, 'data_dir', None)

        self._node_col = "building_id"
        self._time_col = "timestamp"
        self._gap_hours = data_cfg.gap_hours
        self._stride = data_cfg.stride
        self._window_size = data_cfg.window_size

        self.df = None
        self.feature_cols = self._build_feature_columns()
        self.aggregation_weight = 0
        self.global_pos_weight = None

        if self.partition_mode == "shared":
            print(f"[LEAD] Loading shared file: {self.file_path}")
            self.df = pd.read_csv(self.file_path)
            print(f"[LEAD] Raw shape: {self.df.shape}")

            if self.target not in self.df.columns:
                raise ValueError(
                    f"CSV file must contain target column '{self.target}', "
                    f"but columns were: {list(self.df.columns)}"
                )

            _, labels = self._prepare_data(self.df)

            n_pos = int(labels.sum())
            n_neg = len(labels) - n_pos
            pos_weight_cap = getattr(config.model, "pos_weight_cap", 10.0)
            if n_pos > 0:
                self.global_pos_weight = min(n_neg / n_pos, pos_weight_cap)
            else:
                self.global_pos_weight = pos_weight_cap

            self._prepare_partitions()

        elif self.partition_mode == "local":
            self.client_indices = list(range(self.num_clients))

        else:
            raise ValueError(
                f"Unsupported partition_mode: {self.partition_mode}. "
                f"Expected 'shared', 'local', or 'optuna'."
            )

    # ── Public API ─────────────────────────────────────────────────────── #

    def get_dataloaders(self, partition_id: int, round_seed_salt: int = 0):
        print(
            f"[LEAD] get_dataloaders partition_mode={self.partition_mode} "
            f"partition_id={partition_id} round_seed_salt={round_seed_salt}"
        )

        if self.partition_mode == "local":
            if self.use_precomputed_windows:
                return self._load_precomputed_dataloaders(partition_id)
            return self._generate_local_dataloaders(partition_id)

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

        self.aggregation_weight = len(y_train)

        X_train, X_val = self._scale_temporal_arrays(X_train, X_val)

        X_train, y_train = self._oversample_anomalies(
            X_train,
            y_train,
            method=self.oversampling_method,
            target_ratio=self.oversampling_ratio,
            smote_k_neighbors=self.smote_k_neighbors,
            seed=self.seed + 20_000 + partition_id + round_seed_salt * 1_000_000,
            split_name="train",
        )
        # ← ADD THIS
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        print(
            f"[LEAD] Post-oversample train: total={len(y_train)}, "
            f"pos={n_pos}, neg={n_neg}, "
            f"ratio={n_pos/n_neg:.4f}, "
            f"global_pos_weight={self.global_pos_weight:.2f}"
        )

        # Recompute pos_weight to reflect the actual training distribution
        # (after oversampling, data may be balanced — the pre-computed
        #  global_pos_weight from the raw full dataset does not apply).

        if n_pos > 0:
            self.global_pos_weight = min(
                n_neg / n_pos, self.config.model.pos_weight_cap
            )
        else:
            self.global_pos_weight = self.config.model.pos_weight_cap

        if self.oversample_val:
            X_val, y_val = self._oversample_anomalies(
                X_val,
                y_val,
                method=self.oversampling_method,
                target_ratio=self.oversampling_ratio,
                smote_k_neighbors=self.smote_k_neighbors,
                seed=self.seed + 30_000 + partition_id + round_seed_salt * 1_000_000,
                split_name="val",
            )
        else:
            print("[LEAD] Keeping validation set unchanged after oversampling step")

        return self._build_dataloaders_from_arrays(X_train, y_train, X_val, y_val)

    def get_metadata(self):
        if self.df is None and self.partition_mode != "local":
            raise RuntimeError("Metadata requested before dataset was prepared")

        return {
            "input_dim": len(self.feature_cols),
            "num_classes": self.config.model.num_classes,
            "num_samples": self.df.shape[0] if self.df is not None else 0,
            "task_type": self.config.task.name,
            "data_format": "tabular",
        }

    def get_num_partitions(self) -> int:
        return len(self.client_indices)

    # ── Data preparation helpers ───────────────────────────────────────── #

    def _build_feature_columns(self):
        # Static per-building identity features (site_id, square_feet, year_built,
        # floor_count, primary_use_*) are excluded — they let the model memorize
        # building identity and overfit when train/val share buildings.
        return [
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
            "wind_dir_missing",
            "wind_speed_was_missing",
            "precip_depth_was_missing",
            "air_temp_std_lag7_was_missing",
            "air_temp_std_lag73_was_missing",
        ]

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[LEAD] Starting _preprocess")
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

        # StandardScaler returns float64; floor back to float32 to avoid carrying
        # a 2x-size array through the rest of the pipeline.
        X_train_scaled = scaler.fit_transform(X_train_2d).reshape(
            original_train_shape
        ).astype(np.float32)
        X_val_scaled = scaler.transform(X_val_2d).reshape(
            original_val_shape
        ).astype(np.float32)

        print("[LEAD] Applied StandardScaler")

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

    # ── Path helpers ───────────────────────────────────────────────────── #

    def _resolve_precomputed_file_path(self, partition_id: int) -> Path:
        if partition_id < 0:
            raise ValueError(f"Invalid partition_id: {partition_id}")

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

    # ── Partitioning ───────────────────────────────────────────────────── #

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
            X_synth = time_warp(X_synth, rng=rng)
            X_synth = window_slicing(X_synth, rng=rng)
            X_synth = mixup(X_synth, rng=rng)

            X_out = np.concatenate([X, X_synth], axis=0)
            y_out = np.concatenate(
                [y, np.ones(n_needed, dtype=y.dtype)], axis=0
            )

            perm = rng.permutation(len(y_out))
            X_out = X_out[perm]
            y_out = y_out[perm]

            print(
                f"[LEAD] Applied {split_name} oversampling: "
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
            f"[LEAD] Applied {split_name} oversampling: "
            f"method={method}, "
            f"target_pos_neg={target_ratio}:1, "
            f"before={len(y)}, pos_before={positives}, "
            f"after={len(y_resampled)}, pos_after={int(y_resampled.sum())}, "
            f"anomaly_rate={float(y_resampled.mean()):.4f}"
        )

        return X_resampled, y_resampled

    # ── Local (non-precomputed) generation ─────────────────────────────── #

    def _generate_local_dataloaders(self, partition_id: int):
        client_index = partition_id + 1
        data_path = Path(self.data_dir) / f"data{client_index}.csv"
        print(f"[LEAD] Loading local client file: {data_path}")

        df = pd.read_csv(data_path)

        if self.target not in df.columns:
            raise ValueError(
                f"CSV {data_path} missing target column '{self.target}'. "
                f"Columns: {list(df.columns)}"
            )

        df = self._preprocess(df)

        missing_features = [
            col for col in self.feature_cols if col not in df.columns
        ]
        if missing_features:
            raise ValueError(
                f"Missing expected feature columns in {data_path}: {missing_features}"
            )

        X = df[self.feature_cols].values.astype(np.float32)
        y = df[self.target].values.astype(np.int64)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight_cap = getattr(self.config.model, "pos_weight_cap", 10.0)
        if n_pos > 0:
            self.global_pos_weight = min(n_neg / n_pos, pos_weight_cap)
        else:
            self.global_pos_weight = pos_weight_cap

        X_train, y_train, X_val, y_val = temporal_grouped_split(
            df,
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

        self.aggregation_weight = len(y_train)

        self.num_train_windows_before_undersampling = len(y_train)
        self.num_train_anomalies_before_undersampling = int(y_train.sum())

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

        self.num_train_windows_after_undersampling = len(y_train)

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        print(
            f"[LEAD] Post-oversample train: total={len(y_train)}, "
            f"pos={n_pos}, neg={n_neg}, "
            f"ratio={n_pos/n_neg:.4f}, "
            f"global_pos_weight={self.global_pos_weight:.2f}"
        )

        if n_pos > 0:
            self.global_pos_weight = min(
                n_neg / n_pos, self.config.model.pos_weight_cap
            )
        else:
            self.global_pos_weight = self.config.model.pos_weight_cap

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

        return self._build_dataloaders_from_arrays(X_train, y_train, X_val, y_val)

    # ── Precomputed loader ─────────────────────────────────────────────── #

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

            X_train = artifact["X_train"]
            y_train = artifact["y_train"]
            X_val = artifact["X_val"]
            y_val = artifact["y_val"]

            self.aggregation_weight = (
                int(artifact["aggregation_weight"][0])
                if "aggregation_weight" in artifact
                else len(y_train)
            )
            oversampling_method = (
                str(artifact["oversampling_method"][0])
                if "oversampling_method" in artifact
                else "unknown"
            )

        print(
            f"[LEAD] Precomputed shapes: X_train={X_train.shape}, X_val={X_val.shape}, "
            f"train_anomalies={int(y_train.sum())}, val_anomalies={int(y_val.sum())}, "
            f"aggregation_weight={self.aggregation_weight}, "
            f"oversampling={oversampling_method}"
        )

        return self._build_dataloaders_from_arrays(X_train, y_train, X_val, y_val)
