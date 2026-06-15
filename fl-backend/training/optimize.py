import argparse, os, signal, sys

from pathlib import Path
from models.utils import get_device
from optuna_config import OPTUNA_CONFIG
from local_experiment import CONFIG
from .training_utils.OptunaOptimizer import OptunaOptimizer

# Compute default storage path
_db_path = Path(__file__).resolve().parents[1] / "optuna_study.db"
_db_path.parent.mkdir(parents=True, exist_ok=True)

_storage_default = f"sqlite:///{_db_path.as_posix()}"

parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization")
parser.add_argument("--epochs", "-e", type=int, default=OPTUNA_CONFIG.general.epochs, help="Epochs per trial")
parser.add_argument("--trials", "-t", type=int, default=OPTUNA_CONFIG.general.n_trials, help="Number of Optuna trials")
parser.add_argument("--storage", "-s", type=str, default=os.environ.get("STORAGE", _storage_default), help="Optuna storage URL (default: fl-backend/optuna_study.db)")
parser.add_argument("--study-name", "-n", type=str, default=OPTUNA_CONFIG.general.study_name, help="Optuna study name")
parser.add_argument(
    "--dashboard",
    "-d",
    action="store_true",
    help="Launch Optuna Dashboard after optimization finishes",
)
parser.add_argument("--jobs", "-j", type=int, default= OPTUNA_CONFIG.general.n_jobs, help="Number of parallel jobs (default: 1)")
args = parser.parse_args()

print("OPTUNA ARGS")
print("  storage    =", args.storage)
print("  study      =", args.study_name)
print("  epochs     =", args.epochs)
print("  trials     =", args.trials)
print("CONFIG")
print("  model      =", CONFIG.model.name)
print("  training   =", CONFIG.training)
print("  federation =", CONFIG.federation)
print("  dataset    =", CONFIG.data.name)
print("  device     =", get_device())

datahandler = CONFIG.data.build_handler(CONFIG) #Optuna will choose the dataset that is in local_experiment.py

# Use partition 0 just to get a dataset (Optuna is centralized anyway)
train_loader, val_loader = datahandler.get_dataloaders(partition_id=0)

# Convert loaders → numpy (since your optimizer expects arrays)
X_train = train_loader.dataset.tensors[0].numpy()
y_train = train_loader.dataset.tensors[1].numpy()

X_val = val_loader.dataset.tensors[0].numpy()
y_val = val_loader.dataset.tensors[1].numpy()

optimizer = OptunaOptimizer(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    config=CONFIG,
    optuna_config=OPTUNA_CONFIG,
    n_trials=args.trials,
    epochs=args.epochs,
    storage=args.storage,
    study_name=args.study_name,
    n_jobs=args.jobs,
)

study = optimizer.run()

best_params = study.best_trial.params

if args.dashboard:
    try:
        import subprocess
        dashboard_proc = subprocess.Popen(
            ["optuna-dashboard", args.storage, "--host", "127.0.0.1", "--port", "8080"],
        )
        print("\nLaunching Optuna Dashboard on http://127.0.0.1:8080")
        print("Press Ctrl+C to stop.")
        dashboard_proc.wait()
    except ImportError:
        print("\noptuna-dashboard is not installed. Run: pip install optuna-dashboard")
    except KeyboardInterrupt:
        dashboard_proc.terminate()
        dashboard_proc.wait()
        print("\nDashboard stopped.")
