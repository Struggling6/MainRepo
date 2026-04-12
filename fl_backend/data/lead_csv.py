# Data manipulation and visualization libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from data.base import BaseDatasetHandler
from fl_backend.data.time_series_utils import temporal_grouped_split

# Machine learning libraries
import torch
import torch.nn as nn #neural network module
import torch.nn.functional as F #functional module contains functions that don't have parameters, like activation functions and loss functions
from torch.utils.data import TensorDataset, DataLoader #Dataset is an abstract class representing a dataset, and DataLoader is a utility that provides an iterable over the given dataset.
from transformers import AutoConfig, AutoModel # AutoConfig is used to load the configuration of a pre-trained model, and AutoModel is used to load the pre-trained model itself.
from sklearn.model_selection import train_test_split #train_test_split is a function from scikit-learn that splits arrays or matrices into random train and test subsets.
from sklearn.preprocessing import StandardScaler # StandsardScaler is a class from scikit-learn that standardizes features by removing the mean and scaling to unit variance.
from sklearn.model_selection import TimeSeriesSplit # TimeSeriesSplit is a class from scikit-learn that provides train/test indices to split time series data samples that are observed at fixed time intervals.
from sklearn.preprocessing import LabelEncoder # LabelEncoder is a class from scikit-learn that encodes target labels with value between 0 and n_classes-1, where n is the number of distinct labels.
from sklearn.metrics import f1_score, classification_report

class LeadCSVHandler(BaseDatasetHandler):
    def __init__(self, config: dict):
        super().__init__(config)

        self.file_path = config["file_path"]
        self.batch_size = config.get("batch_size")
        self.num_clients = config.get("num_clients")
        self.test_split = config.get("test_split")
        self.normalize = config.get("normalize")
        self.seed = config.get("seed", 42)
        self.label_column = config.get("label_column", "marker")

        self.df = pd.read_csv(self.file_path)

        if self.label_column not in self.df.columns:
            raise ValueError(
                f"CSV file must contain '{self.label_column}' as the target label column."
            )
        self._prepare_data()
        self._prepare_partitions()
        self.features, self.labels = self._prepare_data()
        self.client_indices = np.array_split(indices, self.num_clients)

    def _prepare_data(self):

    # ------------------------------------------------------------------ #
    #  Load & Sort                                                         #
    # ------------------------------------------------------------------ #


        NODE_ID = "building_id"
        TIMESTAMP = "timestamp"

        df = pd.read_csv(self.file_path)

        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
        df = df.sort_values(by=[NODE_ID, TIMESTAMP])

        #OPTIMIZATION: Reduce memory usage by downcasting data types
        # Float64 → float32 (half the memory)
        float_cols = df.select_dtypes(include="float64").columns
        df[float_cols] = df[float_cols].astype("float32")

        # Int64 → int32 (half the memory)
        int_cols = df.select_dtypes(include="int64").columns
        df[int_cols] = df[int_cols].astype("int32")

        # Object (string) → category (much less memory if there are many repeated values)
        str_cols = df.select_dtypes(include="str").columns
        df[str_cols] = df[str_cols].astype("category")

        # Check memory usage after optimization
        print(df.info(memory_usage="deep"))

        for col in df.select_dtypes(include="str").columns:
            print(f"{col}: {df[col].memory_usage(deep=True) / 1e6:.1f} MB")


    # ------------------------------------------------------------------ #
    #  Features                                                            #
    # ------------------------------------------------------------------ #

        features = [
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

    # ------------------------------------------------------------------ #
    #  Preprocessing                                                       #
    # ------------------------------------------------------------------ #

        # These must be computed per building (via groupby) and BEFORE the train/test
        # split — they are feature engineering, not data leakage, because each value
        # only looks backwards in time within its own building.

        groups = df.groupby(NODE_ID)  # group the data by node so we can compute features separately for each node

        # copy a past value into the current row so the model can see history.
        df["meter_lag1"]  = groups["meter_reading"].shift(1)  #what was the meter reading 1 hour ago?
        df["meter_lag24"] = groups["meter_reading"].shift(24) #what was the meter reading 24 hours ago?


        # Rolling mean over the last 24 hours — captures the building's "normal" baseline.
        # min_periods=1 means it still produces a value even near the start of the series.
        df["meter_roll_mean_24"] = groups["meter_reading"].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )

        # Rolling std over the last 24 hours — captures how volatile the recent period was.
        # A low std means stable consumption; a high std means erratic behaviour.
        df["meter_roll_std_24"] = groups["meter_reading"].transform(
            lambda x: x.rolling(window=24, min_periods=1).std()
        )

        # Differences — how much has consumption changed since N steps ago?
        # We add new columns for the change since 1 hour ago and since 24 hours ago.
        df["meter_diff_1"]  = df["meter_reading"] - df["meter_lag1"]   # change in last hour
        df["meter_diff_24"] = df["meter_reading"] - df["meter_lag24"]  # change since yesterday

        # Z-score — how many standard deviations the current reading is from the 24-hour mean.
        # e.g. zscore=0.3 → normal, zscore=7.0 → very likely anomalous.
        # +1e-6 avoids division by zero when std is 0 (flat signal with no variation).
        df["meter_zscore_24"] = (
            (df["meter_reading"] - df["meter_roll_mean_24"])
            / (df["meter_roll_std_24"] + 1e-6)
        )

        # ONE-HOT ENCODING: Convert the categorical "primary_use" column into multiple binary columns.
        df = pd.get_dummies(df, columns=["primary_use"], drop_first=True) # one-hot encoding
        df = df.dropna() # drop rows with NaN values

        primary_use_cols = []
        for col in df.columns:
            if col.startswith("primary_use_"):
                primary_use_cols.append(col)

        feature_cols = features + primary_use_cols

        X = df[feature_cols] # the features (input variables) for the model
        y = df["anomaly"] # the target variable (what we want to predict)

        return X, y
        
    # ------------------------------------------------------------------ #
    #  Run The Split                                                       #
    # ------------------------------------------------------------------ #

    def run_split(self, feature_cols):
        return temporal_grouped_split(
            self.df,
            feature_cols=feature_cols,
            node_col="building_id",
            time_col="timestamp",       
            train_ratio=0.8,
            gap_hours=73,               # matches longest lag feature (lag73)
            window_size=168,            # 1 week of hourly data
            stride=24,                  # one window per day
            target="anomaly",
        )        
    
    def _prepare_partitions(self):
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(self.features))
        rng.shuffle(indices)

        
    def get_metadata(self):
        return {
            "input_dim": self.features.shape[1],
            "num_classes": 1,
            "num_samples": len(self.features),
            "task_type": "binary_classification",
            "data_format": "tabular",
        }

    def get_dataloaders(self, partition_id):
        if partition_id < 0 or partition_id >= self.num_clients:
            raise ValueError(f"Invalid partition_id: {partition_id}")

        idx = self.client_indices[partition_id]

        features, labels = self.run_split(feature_cols=self.features.columns.tolist())

        x_client = features[idx]
        y_client = labels[idx]

        x_tensor = torch.tensor(x_client, dtype=torch.float32)
        y_tensor = torch.tensor(y_client, dtype=torch.long)

        dataset = TensorDataset(x_tensor, y_tensor)

        test_size = 0.2
        train_size = len(dataset) - test_size

        generator = torch.Generator().manual_seed(self.seed)

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=generator,
        )

        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return trainloader, testloader


    def get_num_partitions(self) -> int:
        return self.num_clients