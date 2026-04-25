import optuna
import os
from copy import deepcopy
import torch.nn as nn

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"  # Silence ROCm warning

import torch
from training.training_utils.TrainEvalBase import TrainEvalBase
from training.training_utils.utils import compute_pos_weight
from models.utils import get_device
from config import CONFIG, resolve_loss_fn

class OptunaOptimizer(TrainEvalBase):
    """
    Runs an Optuna hyperparameter search using the shared train/val
    epoch logic from TrainEvalBase. Builds a fresh model for each trial.
    """

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def __init__(
        self,
        X_train, y_train,
        X_val, y_val,
        config=CONFIG,
        n_trials=50,
        epochs=10,
        patience=10,
        storage=None,
        study_name=None,
    ):
        super().__init__(epochs, patience, num_classes=1)
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.in_channels = X_train.shape[-1]
        self.n_trials = n_trials
        self.loss_fn = config.model.loss_fn
        self.raw_pw = compute_pos_weight(y_train)

        self.storage = storage
        self.study_name = study_name

    def run(self):
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            load_if_exists=True,
        )
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            callbacks=[self._pretty_trial_callback],
            show_progress_bar=True,
        )
        self._print_results(study)
        return study

    def _objective(self, trial):
        print(f"\n▶ Trial {trial.number + 1}/{self.n_trials} starting...")

        # Sample shared hyperparameters
        lr           = trial.suggest_float("lr",           1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128])
        pos_weight_cap = trial.suggest_float("pos_weight_cap", 5.0, 20.0)
        dropout      = trial.suggest_float("dropout", 0.1, 0.5)

        # Sample model-specific hyperparameters
        model = self._build_model(trial, dropout)

        loss_fn  = self._build_loss(pos_weight_cap)
        train_dl = self._build_dataloader(self.X_train, self.y_train, batch_size, shuffle=True)
        val_dl   = self._build_dataloader(self.X_val,   self.y_val,   batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_pr_auc, no_improve = 0.0, 0

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(model, train_dl, optimizer, loss_fn)
            _, val_f1, best_thresh, pr_auc = self._val_epoch(model, val_dl, loss_fn)

            trial.report(pr_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if pr_auc > best_pr_auc:
                best_pr_auc    = pr_auc
                best_threshold = best_thresh
                no_improve     = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        trial.set_user_attr("best_threshold", float(best_threshold))
        return float(best_pr_auc)


    def _build_model(self, trial, dropout) -> nn.Module:
        """Sample model-specific hyperparameters and build a fresh model for this trial."""
        from config import CNNTransformerConfig, TransformerConfig, LSTMConfig, MLPConfig, PatchTSTConfig

        model_config = deepcopy(self.config.model)

        if isinstance(model_config, (CNNTransformerConfig, TransformerConfig)):
            model_config.d_model    = trial.suggest_categorical("d_model", [32, 64, 128])
            valid_nheads            = [n for n in [2, 4, 8] if model_config.d_model % n == 0]
            model_config.nhead      = trial.suggest_categorical("nhead", valid_nheads)
            model_config.num_layers = trial.suggest_int("num_layers", 1, 4)
            model_config.dropout    = dropout

        elif isinstance(model_config, LSTMConfig):
            model_config.hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
            model_config.num_layers  = trial.suggest_int("num_layers", 1, 4)
            model_config.dropout     = dropout

        elif isinstance(model_config, MLPConfig):
            model_config.hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
            model_config.num_layers  = trial.suggest_int("num_layers", 1, 5)
            model_config.dropout     = dropout

        elif isinstance(model_config, PatchTSTConfig):
            model_config.d_model             = trial.suggest_categorical("d_model", [32, 64, 128])
            valid_nheads                     = [n for n in [2, 4, 8] if model_config.d_model % n == 0]
            model_config.nhead               = trial.suggest_categorical("nhead", valid_nheads)
            model_config.num_layers          = trial.suggest_int("num_layers", 1, 4)
            model_config.dropout             = dropout
            model_config.ffn_dim             = trial.suggest_categorical("ffn_dim", [128, 256, 512])

        else:
            raise TypeError(
                f"No search space defined for {type(model_config).__name__}. "
                f"Add a branch to _build_model() in OptunaOptimizer."
            )

        return model_config.build(
            input_dim=self.config.evaluation.input_dim
        ).to(get_device())
    
    def _build_loss(self, pos_weight_cap):
        pw = min(self.raw_pw, pos_weight_cap)
        loss_cls = resolve_loss_fn(self.loss_fn)
        return loss_cls(
            pos_weight=torch.tensor([pw], device=get_device())
        ) 

    def _print_results(self, study):
        print("\n=== Best Trial ===")
        print(f"  PR-AUC:     {study.best_trial.value:.4f}")
        print("  Params:")
        for k, v in study.best_trial.params.items():
            print(f"    {k}: {v}")

    def _pretty_trial_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        is_best = study.best_trial.number == trial.number
        marker = "★ NEW BEST" if is_best else ""
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
