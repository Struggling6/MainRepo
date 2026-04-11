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
## `training.optimize` — Optuna Hyperparameter Optimization

This script runs [Optuna](https://optuna.org/) hyperparameter search over a model, then updates the global `CONFIG` object with the best trial's parameters so downstream training uses them automatically.

### How it works

1. **Parses CLI flags** for epochs, trials, and patience (see below).
2. **Loads the dataset** via `LeadCSVHandler(CONFIG.data)` and splits it into train/validation tensors using `run_split()`. The split is temporal and grouped by `node_id`, so no node leaks between train and val. (node refers to building, sensor, device, etc.)
3. **Instantiates `OptunaOptimizer`** with the training/validation tensors and the chosen trial budget.
4. **Runs the study.** Each Optuna trial samples a hyperparameter combination (model width, number of heads, layers, dropout, learning rate, weight decay, batch size, pos-weight cap) and trains a fresh model for up to `--epochs` number of epochs with early stopping controlled by `--patience`.
5. **Writes the best trial's parameters back into `CONFIG`** — specifically `CONFIG.model`, `CONFIG.training`, and `CONFIG.data.batch_size` — and prints the updated config to stdout.

### Prerequisites

- You must run the script **from the `fl_backend/` directory**. The imports (`data.lead_csv`, `models.supervised_cnn_transformer`, `config`) resolve relative to that folder.
- The virtual environment must be activated and all dependencies installed (PyTorch, Optuna, pandas, numpy, etc.).
- `CONFIG.data.file_path` must as of now point to LEAD CSV file.

### Running

Because `training` is a Python package (a module with an `__init__.py`), run the script with `-m` so relative imports inside `training/` resolve correctly:

```bash
cd fl_backend
python -m training.optimize
```

### Command-line arguments

All three flags are optional. If omitted, the defaults shown below are used.

| Flag              | Short | Type | Default | Description                                                                 |
|-------------------|-------|------|---------|-----------------------------------------------------------------------------|
| `--epochs`        | `-e`  | int  | `10`    | Maximum number of training epochs per Optuna trial.                          |
| `--trials`        | `-t`  | int  | `50`    | Number of Optuna trials to run in the study.                                 |
| `--patience`      | `-p`  | int  | `10`    | Early-stopping patience (epochs without val improvement before a trial stops). |

Run `python -m training.optimize --help` to see this same information at the command line.

## Hyperparameters searched

The `OptunaOptimizer` samples and returns values for the following keys in `study.best_trial.params`:

- `d_model` — transformer model dimension
- `nhead` — number of attention heads
- `num_layers` — number of transformer encoder layers
- `dropout` — dropout rate
- `pos_weight_cap` — cap on the positive-class weight in the loss
- `lr` — learning rate
- `weight_decay` — AdamW weight decay
- `batch_size` — training batch size

See `training/training_utils/OptunaOptimizer.py` for the exact search spaces.