import optuna
import os
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0" # Silence ROCm warning
import torch
import torch.nn as nn
from models.supervised_cnn_transformer import SupervisedTransformerCNN
from training.training_utils.TrainEvalBase import TrainEvalBase
from training.training_utils.utils import compute_pos_weight
from models.utils import get_device
from config import CONFIG
from models.registry import MODEL_REGISTRY

class OptunaOptimizer(TrainEvalBase):
    """
    Runs an Optuna hyperparameter search using the shared train/val
    epoch logic from AnomalyTrainerBase. Builds a fresh model and
    Trainer for each trial.
    """
        
    # Silence Optuna's default one-line trial logger
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def __init__(
        self,
        X_train, y_train,
        X_val,   y_val,
        config=CONFIG,
        n_trials=50,
        epochs=10,
        patience=10,
    ):
        super().__init__(epochs, patience, num_classes=1)
        self.X_train     = X_train
        self.y_train     = y_train
        self.X_val       = X_val
        self.y_val       = y_val
        self.in_channels = config.model.in_channels
        self.n_trials    = n_trials
        self.loss_fn     = config.model.loss_fn
        self.model       = self._resolve_model(config.model.name)
        self.raw_pw      = compute_pos_weight(y_train)  # computed once, reused every trial

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self):
        """Create the Optuna study and run all trials."""
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        study.optimize(
            self._objective, 
            n_trials=self.n_trials, 
            callbacks=[self._pretty_trial_callback],
            show_progress_bar=True
        )
        self._print_results(study)
        return study

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_model(self, model_name):
        """Resolve a configured model identifier to a callable model class."""
        if callable(model_name):
            return model_name

        if model_name in MODEL_REGISTRY:
            return MODEL_REGISTRY[model_name]

        raise ValueError(f"Unsupported model name: {model_name}")

    def _objective(self, trial):
        print(f"\n▶ Trial {trial.number + 1}/{self.n_trials} starting...")
        # Sample hyperparameters for this trial
        d_model        = trial.suggest_categorical("d_model",        [32, 64, 128])
        valid_nheads   = [n for n in [2, 4, 8] if d_model % n == 0] # nhead must divide d_model
        nhead          = trial.suggest_categorical("nhead",          valid_nheads)
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
       
        best_pr_auc, no_improve = 0.0, 0
        
        for epoch in range(1, self.epochs + 1):
            self._train_epoch(model, train_dl, optimizer, loss_fn)
            _, val_f1, _, pr_auc = self._val_epoch(model, val_dl, loss_fn)

            # Report to Optuna so it can prune unpromising trials early
            trial.report(pr_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        return float(best_pr_auc)  # explicit cast to Python float for Optuna

    def _build_loss(self, pos_weight_cap):
        pw = min(self.raw_pw, pos_weight_cap)
        return self.loss_fn(
            pos_weight=torch.tensor([pw], device=get_device())
        )

    def _build_model(self, d_model, nhead, num_layers, dropout):
        return self.model(
            in_channels=self.in_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=1,
            dropout=dropout,
        ).to(get_device())

    def _print_results(self, study):
        print("\n=== Best Trial ===")
        print(f"  PR-AUC:     {study.best_trial.value:.4f}")
        print("  Params:")
        for k, v in study.best_trial.params.items():
            print(f"    {k}: {v}")



    def _pretty_trial_callback(self,study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Print each finished trial in a readable multi-line block."""
        is_best = study.best_trial.number == trial.number
        marker  = "★ NEW BEST" if is_best else ""
        duration = trial.duration.total_seconds() if trial.duration else 0.0

        print(f"\n── Trial {trial.number:>3}  {marker}")
        print(f"   value    : {trial.value:.6f}")
        print(f"   duration : {duration:6.1f}s")
        print(f"   params   :")
        for k, v in trial.params.items():
            if isinstance(v, float):
                print(f"     {k:<16} = {v:.6g}")
            else:
                print(f"     {k:<16} = {v}")
        print("─" * 40 + "\n")