# diagnose_patience.py
import optuna
import numpy as np
import matplotlib.pyplot as plt
from training.training_utils.OptunaOptimizer import OptunaOptimizer
from data.lead_csv import LeadCSVHandler
from config import CONFIG

N_TRIALS   = 5    # enough to see variance
N_EPOCHS   = 40   # generous — you want to see the full curve

datahandler = LeadCSVHandler(CONFIG)
train_loader, val_loader = datahandler.get_dataloaders(partition_id=0)

X_train = train_loader.dataset.tensors[0].numpy()
y_train = train_loader.dataset.tensors[1].numpy()
X_val   = val_loader.dataset.tensors[0].numpy()
y_val   = val_loader.dataset.tensors[1].numpy()

optimizer = OptunaOptimizer(
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    config=CONFIG,
    n_trials=N_TRIALS,
    epochs=N_EPOCHS,
    patience=N_EPOCHS,   # disable early stopping
)

study = optimizer.run()