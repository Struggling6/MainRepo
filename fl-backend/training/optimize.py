import argparse, os, signal, sys

from pathlib import Path
from models.utils import get_device
from .training_utils.OptunaOptimizer import OptunaOptimizer
from data.lead_csv import LeadCSVHandler
from local_experiment import CONFIG

# Compute default storage path
_db_dir = Path(__file__).resolve().parent.parent.parent / "fl-backend"
_storage_default = f"sqlite:////{str(_db_dir).lstrip('/')}/optuna_study.db"

parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization")
parser.add_argument("--epochs", "-e", type=int, default=10, help="Epochs per trial (default: 10)")
parser.add_argument("--trials", "-t", type=int, default=50, help="Number of Optuna trials (default: 50)")
parser.add_argument("--storage", "-s", type=str, default=os.environ.get("STORAGE", _storage_default), help="Optuna storage URL (default: fl-backend/optuna_study.db)")
parser.add_argument("--study-name", "-n", type=str, help="Optuna study name")
parser.add_argument(
    "--dashboard",
    "-d",
    action="store_true",
    help="Launch Optuna Dashboard after optimization finishes",
)
parser.add_argument("--jobs", "-j", type=int, default=1, help="Number of parallel jobs (default: 1)")
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

datahandler = LeadCSVHandler(CONFIG)

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
