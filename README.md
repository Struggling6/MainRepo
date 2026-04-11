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



## Docker compose:
rebuild + start Docker Compose commands
Run:
docker compose down
docker compose up --build -d
What this does:
- down = stops/removes old containers
- up --build -d = rebuilds using our pyproject.toml and starts everything again

docker compose ps
- Shows running containers




## New Docker COmmands:

You only need to rebuild the image if you changed:

Python code
pyproject.toml
the Dockerfile

Then run:

docker build -t fl-backend-app:latest ./fl_backend
docker compose up -d

If you only change:

number of clients
ports
resource limits

then just regenerate compose and restart:

python generate_compose.py --num-clients 15
docker compose up -d

No rebuild needed.

## Flower CLI commands:
flwr config list
What it does:
- Shows your Flower config file location
- Shows available connections

add this to your flwr config.toml
```
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

flwr run . local-deployment --stream
 What it does:
- Runs your Flower app (`.` = current folder)
- Connects to Docker backend (`local-deployment`)
- `-stream` = shows logs live
- remember to cd fl_backend
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

### Federated Learning with Flower

This project uses Flower to implement a federated learning system with a custom client and server.

## Overview

The system follows the standard federated learning workflow:

1. The server sends a global model to clients  
2. Clients train the model on their local data partitions  
3. Clients return updated model parameters  
4. The server aggregates updates using FedAvg  
5. The process repeats for multiple rounds  

---

## Client

The client is implemented by extending `NumPyClient`.

Each client:
- Loads its assigned data partition  
- Builds the model and task  
- Trains locally on its data  
- Evaluates the global model  

### Methods

- `get_parameters`: Returns current model parameters  
- `fit`: Trains the model locally and returns updated parameters and metrics  
- `evaluate`: Evaluates the model on local test data  

The `client_fn` function creates a client using a partition ID provided by Flower.

---

## Server

The server defines the training strategy and number of rounds.

### Strategy

FedAvg is used to aggregate client updates:

- Model parameters from clients are averaged  
- Each client’s contribution is weighted by its number of training examples  

### Configuration

- `fraction_fit`: Fraction of clients used for training each round  
- `fraction_evaluate`: Fraction of clients used for evaluation  
- `min_*`: Minimum number of clients required  

---

## Data

- The dataset is split into partitions  
- Each client trains only on its own partition  
- No raw data is shared between clients or server  

---

## What is Context?

`context` is a Flower-provided object that contains runtime information about the current client or server.

- In the client, it is used to determine which data partition to load  
- In the server, it can be used for configuration if needed  