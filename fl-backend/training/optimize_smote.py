import argparse
import os
from pathlib import Path

import numpy as np

from models.utils import get_device
from training.training_utils.OptunaOptimizer import OptunaOptimizer
from local_experiment import CONFIG


_db_path = Path(__file__).resolve().parents[1] / "optuna_smote_study.db"
_db_path.parent.mkdir(parents=True, exist_ok=True)
_storage_default = f"sqlite:///{_db_path.as_posix()}"


parser = argparse.ArgumentParser()
parser.add_argument("--data", default="datasets/LEAD/lead_smote_windows.npz")
parser.add_argument("--epochs", "-e", type=int, default=10)
parser.add_argument("--trials", "-t", type=int, default=50)
parser.add_argument("--storage", "-s", default=os.environ.get("STORAGE", _storage_default))
parser.add_argument("--study-name", "-n", type=str, default="lead_smote")
args = parser.parse_args()

print("OPTUNA SMOTE ARGS")
print("  data    =", args.data)
print("  storage =", args.storage)
print("  study   =", args.study_name)
print("  epochs  =", args.epochs)
print("  trials  =", args.trials)
print("  device  =", get_device())

data = np.load(args.data)

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]

print("DATA")
print("  X_train:", X_train.shape)
print("  y_train:", y_train.shape, "anomaly_rate=", y_train.mean())
print("  X_val:  ", X_val.shape)
print("  y_val:  ", y_val.shape, "anomaly_rate=", y_val.mean())

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