# FL Backend

Federated anomaly detection backend using [Flower](https://flower.ai/), PyTorch, and multiple model architectures (CNNTransformer, LSTM, MLP, Transformer, PatchTST) for the LEAD building energy dataset.

---

## Setup

Python 3.12 is required.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
.\.venv\Scripts\Activate.ps1       # Windows

# Install dependencies
pip install -e ./fl_backend
```

If you install new packages, update `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

## Running Locally (Single Client)

For local development and testing the training pipeline without federation:

```bash
cd fl_backend
python main.py
```

This runs a single client end-to-end: data loading → training → evaluation. All logs print directly to the terminal.

To test the full federated simulation locally:

```bash
python main.py --simulate
```

---

## Docker Deployment

Docker is used to run the full federated system locally with multiple isolated containers. After code changes, rebuild and restart:

```bash
docker compose down
docker compose up --build -d
```

The `--build` flag rebuilds the images automatically — no separate `docker build` step needed.

By default images use CPU PyTorch. To build with GPU support pass a build arg:

```bash
# ROCm (AMD)
docker compose build --build-arg TORCH_VARIANT=rocm

# CUDA (Nvidia)
docker compose build --build-arg TORCH_VARIANT=cuda
```

If you only changed the number of clients, ports, or resource limits — no rebuild needed:

```bash
python generate_compose.py --num-clients 4
docker compose up -d
```

### Useful Commands

```bash
docker compose ps                                # show running containers
docker compose down                              # stop and remove containers
docker compose logs -f                           # follow logs from all containers
docker logs fl-backend-superexec-serverapp-1     # serverapp logs
docker logs fl-backend-superexec-clientapp-1-1   # clientapp logs
```

---

## Running with Flower CLI

Make sure the Docker containers are running first, then from the `fl_backend/` directory:

```bash
cd fl_backend
flwr run . local-deployment --stream
```

The `--stream` flag shows live logs. The `local-deployment` federation connects to the Docker SuperLink at `127.0.0.1:9093`.

Your `~/.flwr/config.toml` should contain:

```toml
[superlink]
default = "local-deployment"

[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

Run `flwr config list` to see the config file location and available connections.

---

## AI-LAB (HPC)

Federated training with real clients runs on AAU AI-LAB using SLURM and Singularity. From the AI-LAB frontend:

```bash
cd /ceph/project/sw6P6/fl_backend
bash aiLab_scripts/launch_all.sh
```

For Optuna hyperparameter search:

```bash
bash aiLab_scripts/run_optuna.sh --trials 50 --epochs 10 --patience 10
```

Monitor jobs:

```bash
squeue --me
tail -f /ceph/project/sw6P6/fl_backend/logs/optuna_<jobid>_0.out
```

AI-LAB limits: max 8 concurrent jobs, max 8 GPUs per user, max 12h per job.

---

## `config.py` — Configuration Guide

All experiment settings live in `config.py`. The single `CONFIG` object at the bottom of the file is the only thing you ever need to change to switch models, datasets, or training parameters.

---

### How It Works

The configuration system is built around Python dataclasses composed together into a top-level `ExperimentConfig`. Every field has a sensible default, so you only specify what differs from those defaults.

```python
# Uses all defaults
CONFIG = ExperimentConfig()

# Override only what you need
CONFIG = ExperimentConfig(
    model=LSTMConfig(hidden_size=256),
    training=TrainingConfig(learning_rate=5e-4),
)
```

`ExperimentConfig` has six sub-configs: `task`, `model`, `data`, `training`, `federation`, and `evaluation`. Swapping any of them is a one-line change at the bottom of the file.

| Field | Controls |
|---|---|
| `model` | Architecture and hyperparameters |
| `data` | Dataset, file paths, batch size |
| `training` | Learning rate, epochs per round, early stopping |
| `federation` | Number of rounds, clients, FedProx regularisation |
| `evaluation` | Saved model path, classification threshold |
| `task` | Problem type (binary classification, anomaly detection, etc.) |

One important note on `local_epochs` in `TrainingConfig` — keep it low (1–2) in a federated setting. More local epochs cause each client's model to drift further from the global model before aggregation, which destabilises training across rounds.

---

### Adding a New Model

The model class lives in `models/` and should extend `BaseModel`. The config class lives in `config.py` and must be a `@dataclass` with a `build(input_dim)` method — this is the only contract the pipeline requires. The pipeline always calls `CONFIG.model.build(input_dim=metadata["input_dim"])`, so as long as your config implements `build()`, everything else works automatically.

```python
@dataclass
class MyModelConfig:
    name:           str   = "my_model"
    hidden_size:    int   = 128
    num_classes:    int   = 1
    pos_weight_cap: float = 10.0
    loss_fn:        type  = nn.BCEWithLogitsLoss

    def build(self, input_dim: int, context_length: int = None) -> nn.Module:
        from models.my_model import MyModel
        return MyModel(
            in_channels=input_dim,
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
        )
```

The `context_length` argument is optional — only include it if your model needs to know the sequence length (e.g. for positional embeddings).

> **Note on PatchTST:** `PatchTSTConfig` is a plain `@dataclass` that does not inherit from HuggingFace's `PatchTSTConfig`. Instead, `build()` creates a fresh `HF_PatchTSTConfig` internally and passes it to the model. This avoids conflicts with HuggingFace's complex `PretrainedConfig` initialisation chain.

---

### Adding a New Dataset Handler

A dataset handler is responsible for loading a dataset, preparing it for training, splitting it into client partitions, and returning PyTorch DataLoaders for a given client. The goal is that the rest of the system should not need to know anything about the raw dataset format — whether the data comes from a CSV, Parquet, database export, or a custom pipeline, the handler exposes the same interface to the rest of the backend.

#### Responsibilities

Each dataset handler should:
- Load the raw dataset and validate that required columns exist
- Separate features and labels and apply dataset-specific preprocessing
- Partition the dataset across federated clients
- Return train/test DataLoaders for a given client partition
- Expose metadata needed by models and tasks

#### Required Interface

Every handler inherits from `BaseDatasetHandler` and must implement:

| Method | Description |
|---|---|
| `get_metadata()` | Returns input size, number of classes, data format, etc. |
| `get_dataloaders(partition_id)` | Returns `(trainloader, testloader)` for a given client |
| `get_num_partitions()` | Returns the number of available client partitions |

#### General Structure

```python
class MyDatasetHandler(BaseDatasetHandler):
    def __init__(self, config):
        super().__init__(config)
        self.file_path  = config.file_path
        self.batch_size = config.batch_size
        self.test_split = config.test_split
        self.seed       = config.seed

        self.df = self._load_data()
        self._prepare_data()
        self._prepare_partitions()

    def _load_data(self):
        # Load raw data from file or another source
        pass

    def _prepare_data(self):
        # Extract features/labels and apply preprocessing
        pass

    def _prepare_partitions(self):
        # Split dataset into per-client subsets
        pass

    def get_metadata(self) -> dict:
        return {
            "input_dim":   self.X.shape[-1],
            "num_classes": 1,
            "task_type":   "binary_classification",
        }

    def get_dataloaders(self, partition_id: int):
        return trainloader, testloader

    def get_num_partitions(self) -> int:
        return self.num_clients
```

#### Adding the Config

There is no registry — each data config has its own `build_handler()` method that instantiates the correct handler directly, following the same pattern as model configs. Add the config to `config.py`:

```python
@dataclass
class MyDatasetConfig:
    name:       str   = "my_dataset"
    file_path:  Path  = Path(__file__).parent / "datasets" / "MyData" / "data.csv"
    target:     str   = "label"
    batch_size: int   = 64
    test_split: float = 0.2
    seed:       int   = 42

    def build_handler(self, config=None):
        from data.my_dataset import MyDatasetHandler
        return MyDatasetHandler(config or CONFIG)
```

Use `Path(__file__).parent` so paths resolve correctly on any machine regardless of working directory.

Then activate it:

```python
CONFIG = ExperimentConfig(
    data=MyDatasetConfig(),
)
```

To use it in code:

```python
dataset_handler = config.data.build_handler()
```

---

## Optuna Hyperparameter Search

Runs an Optuna study over model hyperparameters and writes the best params back into `CONFIG`.

```bash
cd fl_backend
python -m training.optimize --trials 50 --epochs 10 --patience 10
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--trials` | `-t` | `50` | Number of Optuna trials |
| `--epochs` | `-e` | `10` | Max epochs per trial |
| `--patience` | `-p` | `10` | Early stopping patience |
| `--storage` | `-s` | None | SQLite URL for shared study (AI-LAB) |
| `--study-name` | `-n` | `lead_anomaly_detection` | Optuna study name |

On AI-LAB, `run_optuna.sh` submits one SLURM array task per trial so all trials run in parallel across GPUs, writing results to a shared SQLite database.

---