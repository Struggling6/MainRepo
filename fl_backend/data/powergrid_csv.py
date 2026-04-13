import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from .BaseDataHandler import BaseDatasetHandler


class PowerGridCSVHandler(BaseDatasetHandler):
    def __init__(self, config):     # config: PowerGridCSVConfig
        super().__init__(config)

        self.file_path = config.file_path
        self.batch_size = config.batch_size
        self.num_clients = config.num_clients
        self.test_split = config.test_split
        self.normalize = config.normalize
        self.seed = config.seed
        self.label_column = config.target
     

        self.df = pd.read_csv(self.file_path)

        if self.label_column not in self.df.columns:
            raise ValueError(
                f"CSV file must contain '{self.label_column}' as the target label column."
            )

        self._prepare_data()
        self._prepare_partitions()

    def _prepare_data(self):
        feature_df = self.df.drop(columns=[self.label_column])
        labels_series = self.df[self.label_column].astype(str).str.strip()

        self.label_to_idx = {
            "Natural": 0,
            "Attack": 1,
        }

        unknown_labels = set(labels_series.unique()) - set(self.label_to_idx.keys())
        if unknown_labels:
            raise ValueError(f"Unknown labels found: {unknown_labels}")

        labels_encoded = labels_series.map(self.label_to_idx)

        self.features = feature_df.values.astype("float32")
        self.labels = labels_encoded.values.astype("int64")

        self.features = np.nan_to_num(
            self.features,
            nan = 0.0,
            posinf = 0.0,
            neginf = 0.0,
        )
        if self.normalize:
            mean = self.features.mean(axis=0)
            std = self.features.std(axis=0)
            std[std == 0] = 1.0
            self.features = (self.features - mean) / std

    def _prepare_partitions(self):
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(self.features))
        rng.shuffle(indices)

        self.client_indices = np.array_split(indices, self.num_clients)

    def get_metadata(self):
        return {
            "input_dim": self.features.shape[1],
            "num_classes": 2,
            "num_samples": len(self.features),
            "label_mapping": self.label_to_idx,
            "task_type": "binary_classification",
            "data_format": "tabular",
        }

    def get_dataloaders(self, partition_id: int):
        if partition_id < 0 or partition_id >= self.num_clients:
            raise ValueError(f"Invalid partition_id: {partition_id}")

        idx = self.client_indices[partition_id]

        x_client = self.features[idx]
        y_client = self.labels[idx]

        x_tensor = torch.tensor(x_client, dtype=torch.float32)
        y_tensor = torch.tensor(y_client, dtype=torch.long)

        dataset = TensorDataset(x_tensor, y_tensor)

        test_size = int(len(dataset) * self.test_split)
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