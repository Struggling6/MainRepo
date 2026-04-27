import argparse

parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization")
parser.add_argument("--epochs", "-e", type=int, default=10, help="Epochs per trial (default: 10)")
parser.add_argument("--trials", "-t", type=int, default=50, help="Number of Optuna trials (default: 50)")
parser.add_argument("--patience", "-p", type=int, default=10, help="Early-stopping patience (default: 10)")
parser.add_argument("--storage", type=str, help="Optuna storage URL")
parser.add_argument("--study-name", type=str, help="Optuna study name")
args = parser.parse_args()

from .training_utils.OptunaOptimizer import OptunaOptimizer
from data.lead_csv import LeadCSVHandler
from config import CONFIG

print("OPTUNA ARGS")
print("  storage   =", args.storage)
print("  study     =", args.study_name)
print("  epochs    =", args.epochs)
print("  trials    =", args.trials)
print("  patience  =", args.patience)

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
    patience=args.patience,
    storage=args.storage,
    study_name=args.study_name,
)

study = optimizer.run()

best = study.best_trial.params

CONFIG.model.d_model = best["d_model"]
CONFIG.model.nhead = best["nhead"]
CONFIG.model.num_layers = best["num_layers"]
CONFIG.model.dropout = best["dropout"]
CONFIG.model.pos_weight_cap = best["pos_weight_cap"]

CONFIG.training.learning_rate = best["lr"]
CONFIG.training.weight_decay = best["weight_decay"]
CONFIG.data.batch_size = best["batch_size"]

print("CONFIG updated with best Optuna params:")
print(f"  model:    {CONFIG.model}")
print(f"  training: {CONFIG.training}")
print(f"  data:     {CONFIG.data}")
