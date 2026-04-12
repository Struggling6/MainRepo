import pandas as pd #Library to read csv files and work with tables
import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset, random_split 
from .base import BaseDatasetHandler #Import out base class


class PowerGridCSVHandler(BaseDatasetHandler):
    def __init__(self, config: dict):
        super().__init__(config)

        #Path to the CSV file.
        self.file_path = config["file_path"]

        #Use from config if provided, otherwise default to 32
        self.batch_size = config.get("batch_size", 32) 

        #How many clients to split data into. Default is 1
        self.num_clients = config.get("num_clients", 1)

        #Percentage of data to use for testing (rest is for training).
        self.test_split = config.get("test_split", 0.2)

        #Whether to normalize features.
        self.normalize = config.get("normalize", True)

        #Ensure reproducibility by using a fixed random seed for shuffling and splitting data.
        self.seed = config.get("seed", 42)
        
        #Name of the column in the CSV that reveals if its an attack or natural
        self.label_column = config.get("label_column", "marker")
     
        #Load the CSV file into a pandas DataFrame (table).
        self.df = pd.read_csv(self.file_path)

        #Check that the label column exists in the CSV
        if self.label_column not in self.df.columns:
            raise ValueError(
                f"CSV file must contain '{self.label_column}' as the target label column."
            )
        #Prepare the data and create partitions for clients.
        self._prepare_data()
        self._prepare_partitions()

    def _prepare_data(self):
        feature_df = self.df.drop(columns=[self.label_column])
        #Get the label coloumn (target values) and convert to string, remove whitespace
        labels_series = self.df[self.label_column].astype(str).str.strip()

        #Define how labels are converted to numbers
        self.label_to_idx = {
            "Natural": 0,
            "Attack": 1,
        }

        #Check if there are labels not in our mapping and raise an error if so
        unknown_labels = set(labels_series.unique()) - set(self.label_to_idx.keys())
        if unknown_labels:
            raise ValueError(f"Unknown labels found: {unknown_labels}")

        #Convert string labels to numbers using the mapping
        labels_encoded = labels_series.map(self.label_to_idx)

        #Convert features and labels to NumPy arrays of type float32 and int64
        self.features = feature_df.values.astype("float32")
        self.labels = labels_encoded.values.astype("int64")

        #Replace any NaN or infinite values in features with 0.0
        self.features = np.nan_to_num(
            self.features,
            nan = 0.0,
            posinf = 0.0,
            neginf = 0.0,
        )
        #Normalize features if enabled. Ensures no single feature dominates due to scale.
        if self.normalize:
            mean = self.features.mean(axis=0) #Calculate mean of each feature column
            std = self.features.std(axis=0) #Calculate std of each feature column
            std[std == 0] = 1.0 # If std is 0, replace with 1 to avoid division errors
            self.features = (self.features - mean) / std #Standardization/Z-score normalization. Mean = 0, Std = 1.

    def _prepare_partitions(self):
        #Create random number generator with fixed seed.
        rng = np.random.default_rng(self.seed)
        
        #Create array of indices
        indices = np.arange(len(self.features))

        #Shuffle the indices randomly to create random partitions for clients.
        rng.shuffle(indices)

        #Split the shuffled indices into num_clients equal parts (or as close as possible)
        self.client_indices = np.array_split(indices, self.num_clients)

    #Return infomation about the dataset
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
        #Check that the requested partition_id is valid
        if partition_id < 0 or partition_id >= self.num_clients:
            raise ValueError(f"Invalid partition_id: {partition_id}")

        #Get indices for this client
        idx = self.client_indices[partition_id]

        #Select features and labels for this client using the indices
        x_client = self.features[idx]
        y_client = self.labels[idx]

        #Convert NumPy arrays --> PyTorch tensors
        x_tensor = torch.tensor(x_client, dtype=torch.float32)
        y_tensor = torch.tensor(y_client, dtype=torch.long)

        #Create a TensorDataset from the features and labels tensors
        dataset = TensorDataset(x_tensor, y_tensor)

        #Compute number of test samples
        test_size = int(len(dataset) * self.test_split)

        #Remain samples will be used for training
        train_size = len(dataset) - test_size

        generator = torch.Generator().manual_seed(self.seed)

        #Split dataset into train and test
        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=generator,
        )
        #Create dataloader for training
        trainloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True)
        
        #Create dataloader for testing (no shuffling)
        testloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False)

        return trainloader, testloader

    def get_num_partitions(self) -> int:
        return self.num_clients