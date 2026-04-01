from OptunaOptimizer import OptunaOptimizer
from models.supervised_cnn_transformer import SupervisedTansformerCNN
from Trainer import Trainer
from utils import get_device
import torch
import torch.nn as nn
from data.utils import temporal_grouped_split

# LEAD DATASET
X_train, y_train, X_test, y_test, nid_train, nid_test = temporal_grouped_split(
    df,
    feature_cols,
    node_col="building_id",
    time_col="timestamp",       
    train_ratio=0.8,
    gap_hours=73,               # matches longest lag feature (lag73)
    window_size=168,            # 1 week of hourly data
    stride=24,                  # one window per day
    target=LEAD_target,
)        

# --- Hyperparameter search ---
optuna_optimizer = OptunaOptimizer(
    X_train=X_train, y_train=y_train,
    X_val=X_test,    y_val=y_test,
    in_channels=X_train.shape[2],
    n_trials=50,
    epochs=30,
    patience=7,
)
study = optuna_optimizer.run()

# --- Train final model with best params ---
best = study.best_trial.params

pos_weight = optuna_optimizer._compute_pos_weight(y_train, cap=best["pos_weight_cap"])
loss_fn    = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight], device=get_device())
)

model = SupervisedTansformerCNN(
    in_channels=X_train.shape[2],
    d_model=best["d_model"],
    nhead=best["nhead"],
    num_layers=best["num_layers"],
    dropout=best["dropout"],
)

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    lr=best["lr"],
    batch_size=best["batch_size"],
    weight_decay=best["weight_decay"],
    epochs=30,
    patience=7,
    num_classes=1,
)
trained_model = trainer.train(X_train, y_train, X_test, y_test)