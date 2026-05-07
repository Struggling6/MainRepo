import argparse
import os
import signal
import sys
from pathlib import Path

# Compute default storage path
_db_dir = Path(__file__).resolve().parent.parent.parent / "fl-backend"
_db_path = _db_dir / "optuna_study.db"
_storage_default = f"sqlite:///{_db_path.as_posix()}"

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
args = parser.parse_args()

from models.utils import get_device
from .training_utils.OptunaOptimizer import OptunaOptimizer
from data.lead_csv import LeadCSVHandler
from local_experiment import CONFIG

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
)

study = optimizer.run()

best_params = study.best_trial.params

if args.dashboard:
    try:
        from optuna_dashboard import run_server
    except ImportError:
        print("\nDashboard requested, but optuna-dashboard is not installed.")
        print("Install it with: pip install optuna-dashboard")
    else:
        def signal_handler(sig, frame):
            print("\n\nShutting down dashboard server...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        
        print("\nLaunching Optuna Dashboard on http://127.0.0.1:8080")
        print("Press Ctrl+C to stop the dashboard server.")
        try:
            run_server(args.storage, host="127.0.0.1", port=8080)
        except KeyboardInterrupt:
            print("\n\nShutting down dashboard server...")
            sys.exit(0)
