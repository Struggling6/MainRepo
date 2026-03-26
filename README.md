## Installed dependencies
See requirements.txt
## Virtual environment
Made requirements.txt to install all the packages required and the correct versions.

**IMPORTANT**
Python 3.11 is required to make this project work

To use first make a virtual environment using ``py -3.11 -m venv .venv`` This should take 30 seconds ish and make a new folder


To access venv use: ``.\.venv\Scripts\Activate.ps1`` in terminal. it should now look something like :

```(.venv) PS C:\Users\phili\Documents\GitHub\MainRepo> ```


After ensuring terminal line looks like the one above, update pip using the command ``python -m pip install --upgrade pip`` and afterwards install the dependencies using: ```pip install -r requirements.txt```  all the dependencies should now be downloaded with the correct versions.

if you install new packages be sure to update ``requirements.txt`` using the command ``pip freeze > requirements.txt``

This was done to be sure to have the correct version of the libraries and is apparently standard practice


# Expanding the backend

## Making new Dataset handlers
A dataset handler is responsible for loading a dataset, preparing it for training, splitting it into client partiotions, and returning PyTorch dataloaders for a given client.

The goal is the the rest of the system should not need to know anything about the raw dataset format. Whether the dataset comes from a csv file, parquet file, database export or a custom preporcessing pipeline, the handler should expose the same interface to the rest of the backend

### Responsibilities of a Dataset handler
Each dataset handler should:
- load the raw dataset 
- validate that required columns or fields exist
- separate features and labels
- apply dataset-specific preprocessing
- optionally normalize or transform the input data
- partition the dataset across federated clients
- return train/test dataloaders for a given client
- expose metadata needed by models and tasks

### Required Interface
Every dataset handler should inherit from `BaseDatasetHandler` and implement the methods expected by the framework.

a Handler should at minimum provide:
- get_metadata()
    - Returns information about the dataset, such as input size, number of classes, and data format.
- get_dataloaders(partition_id: int)
    - Returns the train and test dataloaders for a specific client partition.
- get_num_partitions()
    - Returns the number of available client partitions.

### General structure
A typical dataset handler follow this pattern:

1. Read configuration values
2. Load the dataset
3. Validate required inputs
4. Prepare features and labels
5. Apply preprocessing
6. Create federated client partitions
7. Expose dataloaders and metadata

A simplified structure looks like this: 
```Python
class MyDatasetHandler(BaseDatasetHandler):
    def __init__(self, config: dict):
        super().__init__(config)

        self.file_path = config["file_path"]
        self.batch_size = config.get("batch_size", 32)
        self.num_clients = config.get("num_clients", 1)
        self.test_split = config.get("test_split", 0.2)
        self.seed = config.get("seed", 42)

        self.data = self._load_data()
        self._prepare_data()
        self._prepare_partitions()

    def _load_data(self):
        # Load raw data from file or another source
        pass

    def _prepare_data(self):
        # Extract features/labels and apply preprocessing
        pass

    def _prepare_partitions(self):
        # Split dataset into client subsets
        pass

    def get_metadata(self):
        return {...}

    def get_dataloaders(self, partition_id: int):
        return trainloader, testloader

    def get_num_partitions(self) -> int:
        return self.num_clients

```