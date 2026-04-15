# This script runs Optuna hyperparameter optimization for the SupervisedTransformerCNN model.
# It uses the AnomalyTrainerBase class to share training and validation logic across trials.
# After running, it updates the global CONFIG with the best hyperparameters found.

# argsparse is used at the top level before imports for better CLI performance (avoid importing heavy libraries if just asking for --help)
import argparse
parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization")
parser.add_argument("--epochs",   "-e", type=int, default=10, help="Epochs per trial (default: 10)")
parser.add_argument("--trials",   "-t", type=int, default=50, help="Number of Optuna trials (default: 50)")
parser.add_argument("--patience", "-p", type=int, default=10, help="Early-stopping patience (default: 10)")
args = parser.parse_args()

from .training_utils.OptunaOptimizer import OptunaOptimizer
from data.lead_csv import LeadCSVHandler
from config import CONFIG


datahandler = LeadCSVHandler(CONFIG.data)
X_train, y_train, X_val, y_val = datahandler.run_split()

optimizer = OptunaOptimizer(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    config=CONFIG,
    n_trials=args.trials,
    epochs=args.epochs,
    patience=args.patience,
)

study = optimizer.run()

# --- Update CONFIG with best Optuna params ---
best = study.best_trial.params

CONFIG.model.d_model        = best["d_model"]
CONFIG.model.nhead          = best["nhead"]
CONFIG.model.num_layers     = best["num_layers"]
CONFIG.model.dropout        = best["dropout"]
CONFIG.model.pos_weight_cap = best["pos_weight_cap"]

CONFIG.training.learning_rate = best["lr"]
CONFIG.training.weight_decay  = best["weight_decay"]

CONFIG.data.batch_size = best["batch_size"]

print("CONFIG updated with best Optuna params:")
print(f"  model:    {CONFIG.model}")
print(f"  training: {CONFIG.training}")
print(f"  data:     {CONFIG.data}")