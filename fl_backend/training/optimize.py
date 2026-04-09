from .training_utils.OptunaOptimizer import OptunaOptimizer
from data.lead_csv import LeadCSVHandler
from models.supervised_cnn_transformer import SupervisedTansformerCNN
from config import CONFIG

datahandler = LeadCSVHandler(CONFIG)
X_train, y_train, X_val, y_val = datahandler.run_split


optimizer = OptunaOptimizer(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    in_channels=X_train.shape[2],
    n_trials=50,
    epochs=10,
    patience=10,
)

study = optimizer.run()

# --- Update CONFIG with best Optuna params ---
best = study.best_trial.params

CONFIG["model"].update({
    "hidden_dim":  best["d_model"],
    "embed_dim":   best["d_model"],
    "num_heads":   best["nhead"],
    "num_layers":  best["num_layers"],
    "dropout":     best["dropout"],
})

CONFIG["training"].update({
    "learning_rate": best["lr"],
    "weight_decay":  best["weight_decay"],
    "pos_weight_cap": best["pos_weight_cap"],
})

CONFIG["data"].update({
    "batch_size": best["batch_size"],
})

print("CONFIG updated with best Optuna params:")
for section in ["model", "training", "data"]:
    print(f"  {section}: {CONFIG[section]}")

# Train final model using updated CONFIG
final_model = SupervisedTansformerCNN(
    in_channels=X_train.shape[2],
    d_model=CONFIG["model"]["embed_dim"],
    nhead=CONFIG["model"]["num_heads"],
    num_layers=CONFIG["model"]["num_layers"],
    dropout=CONFIG["model"]["dropout"],
)