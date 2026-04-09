import optuna
import torch
import torch.nn as nn
from models.supervised_cnn_transformer import SupervisedTansformerCNN
from training.training_utils.AnomalyTrainerBase import AnomalyTrainerBase

class OptunaOptimizer(AnomalyTrainerBase):
    """
    Runs an Optuna hyperparameter search using the shared train/val
    epoch logic from AnomalyTrainerBase. Builds a fresh model and
    Trainer for each trial.
    """

    def __init__(
        self,
        X_train, y_train,
        X_val,   y_val,
        in_channels,
        model = SupervisedTansformerCNN,
        loss_fn=None,
        n_trials=50,
        epochs=10,
        patience=10,
    ):
        super().__init__(epochs, patience, num_classes=1)
        self.X_train     = X_train
        self.y_train     = y_train
        self.X_val       = X_val
        self.y_val       = y_val
        self.in_channels = in_channels
        self.n_trials    = n_trials
        self.loss_fn     = loss_fn
        self.model       = model
        self.raw_pw      = self._compute_pos_weight(y_train)  # computed once, reused every trial

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self):
        """Create the Optuna study and run all trials."""
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        study.optimize(self._objective, n_trials=self.n_trials)
        self._print_results(study)
        return study

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _objective(self, trial):
        # Sample hyperparameters for this trial
        d_model        = trial.suggest_categorical("d_model",        [32, 64, 128])
        nhead          = trial.suggest_categorical("nhead",          [2, 4])
        num_layers     = trial.suggest_int(        "num_layers",     1, 3)
        dropout        = trial.suggest_float(      "dropout",        0.2, 0.5)
        lr             = trial.suggest_float(      "lr",             1e-4, 1e-2, log=True)
        weight_decay   = trial.suggest_float(      "weight_decay",   1e-4, 1e-1, log=True)
        batch_size     = trial.suggest_categorical("batch_size",     [32, 64, 128])
        pos_weight_cap = trial.suggest_float(      "pos_weight_cap", 5.0, 20.0)

        loss_fn   = self._build_loss(pos_weight_cap)
        model     = self._build_model(d_model, nhead, num_layers, dropout)
        train_dl  = self._build_dataloader(self.X_train, self.y_train, batch_size, shuffle=True)
        val_dl    = self._build_dataloader(self.X_val,   self.y_val,   batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_f1, no_improve = 0.0, 0

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(model, train_dl, optimizer, loss_fn)
            _, val_f1, _, _ = self._val_epoch(model, val_dl, loss_fn)

            # Report to Optuna so it can prune unpromising trials early
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_f1 > best_f1:
                best_f1    = val_f1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        return float(best_f1)  # explicit cast to Python float for Optuna

    def _build_loss(self, pos_weight_cap):
        if (self.loss_fn is None):
            pw = min(self.raw_pw, pos_weight_cap)
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pw], device=self.device)
            )
        
        return self.loss_fn

    def _build_model(self, d_model, num_heads, num_layers, dropout):
        return self.model(
            in_channels=self.in_channels,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=1,
            dropout=dropout,
        ).to(self.device)

    def _print_results(self, study):
        print("\n=== Best Trial ===")
        print(f"  F1:     {study.best_trial.value:.4f}")
        print("  Params:")
        for k, v in study.best_trial.params.items():
            print(f"    {k}: {v}")