import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from .base import BaseDatasetHandler

class PowerGridCSVHandler(BaseDatasetHandler):

    def __init__(self, config:dict):
        super().__init__(config)

        self.file_path = config['file_path']
        self.batch_size = config.get('batch_size', 32)
        self.num_clients = config.get('num_clients', 1)
        self.test_split = config.get('test_split', 0.2)
        self.normalize = config.get('normalize', True)

        self.df = pd.read_csv(self.file_path)

        if "marker" not in self.df.columns:
            raise ValueError("CSV file must contain a 'marker' column for client identification.")
        
        self._prepare_data()
    
    def _prepare_data(self):

        # Separate features and labels
        feature_df = self.df.drop(columns=['marker'])
        labels_series = self.df['marker']

        # Convert labels to integers
        self.label_to_idx = {
            label: idx for idx, label in enumerate(labels_series.unique())
        }

        labels_encoded = labels_series.map(self.label_to_idx)

        # Convert to numpy arrays
        self.features = feature_df.values.astype("float32")
        self.labels = labels_encoded.values.astype("int64")

        # Normalize features if enabled
        if self.normalize:
            mean = self.features.mean(axis=0)
            std = self.features.std(axis=0) 
            std[std == 0] = 1.0
            self.features = (self.features - mean) / std
        
        def get_metadata(self):
            return {
                "input_dim": self.features.shape[1],
                "num_classes": len(self.label_to_idx),
                "num_samples": len(self.features)
                "label_mapping": self.label_to_idx
            }

        def get_dataloaders(self, config:dict):

            if partition_id < 0 or partition_id >= self.num_clients:
                raise ValueError(f"Invalid partition_id")
            
            total_samples = len(self.features)
            samples_per_client = total_samples // self.num_clients

            start = partition_id * samples_per_client
            end = (
                total_samples
                if partition_id == self.num_clients - 1
                else start + samples_per_client
            )


            
