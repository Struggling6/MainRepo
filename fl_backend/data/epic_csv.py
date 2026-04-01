import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from .base import BaseDatasetHandler
from .dataInjection import inject_noise
from sklearn.model_selection import train_test_split



class EpicCSVHandler(BaseDatasetHandler):

    def __init__(self, config: dict):
        super().__init__(config) #Add din egen config

        self.file_path = config["file_path"]
        self.clean_path = config["clean_path"]
        self.batch_size = config.get("batch_size", 32)
        self.num_clients = config.get("num_clients", 1)
        self.test_split = config.get("test_split", 0.2)
        self.noise_level = config.get("noise_level", 0.7)
        self.normalize = config.get("normalize", True)
        self.seed = config.get("seed", 42)
        self.label_column = config.get("label_column", "marker")

        self._prepare_data()
        self._prepare_partitions()


    def _prepare_data(self):    
        inject_noise(self.clean_path, self.file_path, noise_level=self.noise_level)

        self.df = pd.read_csv(self.file_path)
        self.df = self.df.drop(columns=["Timestamp"])

        self.X = self.preprocess_dataframe(self.df)


    def _prepare_partitions(self):
        total_samples = len(self.X)
        device_data_amount = total_samples // self.num_clients
        
        self.partitions = []
        
        for partition_id in range(self.num_clients):
            # Beregn start og slut indeks
            start_idx = partition_id * device_data_amount
            end_idx = start_idx + device_data_amount
            
            # Håndter sidste partition (får resten af data)
            if partition_id == self.num_clients - 1:
                end_idx = total_samples
            
            # Hent data for denne partition
            X_partition = self.X[start_idx:end_idx]
            
            # Gem partition i liste
            self.partitions.append({
                "partition_id": partition_id,
                "X": X_partition,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "size": len(X_partition),
            })
            
            print(f"Partition {partition_id}: rows {start_idx}-{end_idx} ({len(X_partition)} samples)")


    def preprocess_dataframe(self, df: pd.DataFrame) -> torch.Tensor:
        df_numeric = df.apply(pd.to_numeric, errors="coerce")
        df_numeric = df_numeric.astype(float)
        df_numeric = df_numeric.fillna(0.0)

        mean = df_numeric.mean()
        std = df_numeric.std().replace(0, 1)

        df_numeric = (df_numeric - mean) / std
        df_numeric = df_numeric.fillna(0.0)

        X = torch.tensor(df_numeric.values, dtype=torch.float32)
        X = X.unsqueeze(1)   # [num_samples, 1, num_features]
        return X


    def make_autoencoder_dataloader(self,
        X: torch.Tensor,
        batch_size: int = 32,
    ) -> DataLoader:
        dataset = TensorDataset(X)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0) 

    
    def get_dataloaders(self, partition_id: int):

        partition = self.partitions[partition_id]
        X_partition = partition["X"]
    
        X_train, X_test = train_test_split(X_partition, test_size=0.2, random_state=self.seed + partition_id, shuffle=False)

        # Dataloaders
        train_loader = self.make_autoencoder_dataloader(X_train, batch_size=32)
        test_loader = self.make_autoencoder_dataloader(X_test, batch_size=32)
        
        return train_loader, test_loader

    def get_num_partitions(self) -> int:
        return self.num_clients
    

    def get_metadata(self):
        return {
            "input_dim": self.X.shape[2],  # X har shape [num_samples, 1, num_features]
            "num_features": self.X.shape[2],
            "num_classes": 2, #SKal slettes fordi vi ikke har klasser men predictions
            "num_samples": len(self.X),
            "label_mapping": {}, 
            "task_type": "binary_classification",
            "data_format": "tabular",
        }