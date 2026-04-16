import torch
import numpy as np
import pandas as pd
from .time_series_utils import temporal_grouped_split
from config import CONFIG
from .BaseDataHandler import BaseDatasetHandler


class LeadCSVHandler(BaseDatasetHandler):
    """
    Dataset handler for the LEAD building energy dataset.

    The gap between train and validation is set to 73 hours to match
    the longest lag feature (air_temperature_*_lag73), preventing those
    features from leaking across the train/val boundary.
    """
    # Override base class default — matches longest lag feature
    _gap_hours = 73

    def __init__(self, config):    # config: LeadCSVConfig

        super().__init__(config)
        self.file_path    = config.file_path
        self.target       = config.target
        self.batch_size   = config.batch_size
        self.test_split   = config.test_split
        self.num_clients  = config.num_clients
        self.seed         = config.seed
        self._node_col    = "building_id"
        self._time_col    = "timestamp"


        # Load raw CSV into self.df so _prepare_data can use it
        self.df = pd.read_csv(self.file_path)

        # _prepare_data must run before _prepare_partitions because
        # _prepare_partitions needs self.df to have building IDs
        self.features, self.labels = self._prepare_data()
        self._prepare_partitions()

    # ------------------------------------------------------------------ #
    #  Template Method implementations                                     #
    # ------------------------------------------------------------------ #

    def _preprocess(self, df):
        """
        Apply LEAD-specific feature engineering to any raw DataFrame.
        Called by both _prepare_data (training) and load_test_set (testing)
        so the exact same cleaning steps are applied to both.
        """

        df[self._time_col] = pd.to_datetime(df[self._time_col])
        df = df.sort_values(by=[self._node_col, self._time_col])

        # Memory optimisation — halves RAM usage for large datasets
        float_cols = df.select_dtypes(include="float64").columns
        df[float_cols] = df[float_cols].astype("float32")
        int_cols = df.select_dtypes(include="int64").columns
        df[int_cols] = df[int_cols].astype("int32")
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].astype("category")

        # Lag features — computed per building so no cross-building leakage
        groups = df.groupby(self._node_col)
        df["meter_lag1"]  = groups["meter_reading"].shift(1)   # 1 hour ago
        df["meter_lag24"] = groups["meter_reading"].shift(24)  # 24 hours ago

        # Rolling statistics over the last 24 hours
        # min_periods=1 ensures values are produced near the start of the series
        df["meter_roll_mean_24"] = groups["meter_reading"].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )
        df["meter_roll_std_24"] = groups["meter_reading"].transform(
            lambda x: x.rolling(window=24, min_periods=1).std()
        )

        # Difference features — how much has consumption changed?
        df["meter_diff_1"]  = df["meter_reading"] - df["meter_lag1"]
        df["meter_diff_24"] = df["meter_reading"] - df["meter_lag24"]

        # Z-score — how many std devs is current reading from 24hr mean?
        # +1e-6 prevents division by zero when std is 0 (flat signal)
        df["meter_zscore_24"] = (
            (df["meter_reading"] - df["meter_roll_mean_24"])
            / (df["meter_roll_std_24"] + 1e-6)
        )

        # One-hot encode primary_use — drop_first avoids dummy variable trap
        df = pd.get_dummies(df, columns=["primary_use"], drop_first=True)

        # pandas 2.x returns bool columns from get_dummies — cast to float32
        # so the feature matrix stays a single numeric dtype
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype("float32")

        return df.dropna()

    def _prepare_data(self):
        """
        Preprocess the training DataFrame and populate self.df and
        self.feature_cols. Returns X, y as numpy arrays for use by
        get_metadata and _prepare_partitions.
        """
        df = self._preprocess(self.df)

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

        # Collect any one-hot columns created from primary_use
        primary_use_cols  = [col for col in df.columns if col.startswith("primary_use_")]
        self.feature_cols = base_features + primary_use_cols

        # Store the fully processed DataFrame so run_split and
        # get_dataloaders can use it with correct timestamps and features
        self.df = df

        X = df[self.feature_cols].values.astype(np.float32)
        y = df[self.config.target].values.astype(np.float32)
        return X, y
    
    # ------------------------------------------------------------------ #
    #  Partitioning                                                        #
    # ------------------------------------------------------------------ #

    def _prepare_partitions(self):
        """Split sample indices into num_clients equal partitions, shuffled."""
        rng     = np.random.default_rng(self.seed)
        buildings = self.df["building_id"].unique()
        rng.shuffle(buildings)
        self.client_indices = np.array_split(buildings, self.num_clients)
        
    # ------------------------------------------------------------------ #
    #  BaseDatasetHandler interface                                        #
    # ------------------------------------------------------------------ #
    
    def get_dataloaders(self, partition_id: int):
        if partition_id < 0 or partition_id >= self.num_clients:
            raise ValueError(f"Invalid partition_id: {partition_id}")

        # Filter to only the buildings assigned to this client
        client_df = self.df[
            self.df[self._node_col].isin(self.client_indices[partition_id])
        ]

        # Temporal split on this client's buildings only — each client
        # trains on its own time-ordered slice of the data
        X_train, y_train, X_val, y_val = temporal_grouped_split(
            client_df,
            feature_cols=self.feature_cols,
            node_col=self._node_col,
            time_col=self._time_col,
            train_ratio=1.0 - self.test_split,
            gap_hours=self._gap_hours,
            window_size=self._window_size,
            stride=self._stride,
            target=self.config.target,
        )

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        valloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        return trainloader, valloader
        
    def get_metadata(self):
        return {
            "input_dim"  : len(self.feature_cols),
            "num_classes": CONFIG.model.num_classes,
            "num_samples": self.df.shape[0], 
            "task_type"  : CONFIG.task.name,
        "data_format": "tabular",
        }


    def get_num_partitions(self) -> int:
        return self.num_clients
