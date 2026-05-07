import os
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

import torch, optuna
import torch.nn as nn

from copy import deepcopy
from training.training_utils.TrainEvalBase import TrainEvalBase
from training.training_utils.utils import compute_pos_weight
from models.utils import get_device
from local_experiment import CONFIG, resolve_loss_fn


class OptunaOptimizer(TrainEvalBase):
    """
    Runs an Optuna hyperparameter search using the shared train/val
    epoch logic from TrainEvalBase. Builds a fresh model for each trial.
    Pruning is handled by HyperbandPruner — no manual patience needed.
    """

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _logger(self, msg: str):
        print(f"[Optuna] {msg}", flush=True)

    def __init__(
        self,
        X_train, y_train,
        X_val, y_val,
        config=CONFIG,
        n_trials=50,
        epochs=20,
        storage=None,
        study_name=None,
    ):
        super().__init__(num_classes=1, epochs=epochs)
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val
        self.in_channels = int(X_train.shape[-1])
        self.config.evaluation.input_dim = self.in_channels

        self.n_trials = n_trials
        self.loss_fn = config.model.loss_fn
        self.raw_pw = compute_pos_weight(y_train)

        self.storage = storage
        self.study_name = study_name

    def run(self):
        min_resource = max(3, self.epochs // 4)
        max_resource = self.epochs

        self._logger(
            f"Starting study name={self.study_name}, trials={self.n_trials}, epochs={self.epochs}, "
            f"train_shape={self.X_train.shape}, val_shape={self.X_val.shape}"
            f"device={get_device()}"
            
        )
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=min_resource,
                max_resource=max_resource,
                reduction_factor=3,
            ),
            load_if_exists=True,
        )
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            callbacks=[self._pretty_trial_callback],
            show_progress_bar=True,
            
        )
        self._print_results(study)
        self._logger("Study finished")
        return study

    def _objective(self, trial):
        print(f"\n▶ Trial {trial.number + 1}/{self.n_trials} starting...")

        lr                = trial.suggest_float("lr",                   3e-5, 3e-3, log=True)
        weight_decay      = trial.suggest_float("weight_decay",         1e-4, 1e-2, log=True)
        num_layers        = trial.suggest_int("num_layers",             1, 5)
        batch_size        = trial.suggest_categorical("batch_size",     [8, 16, 32, 64, 128, 256])
        pos_weight_cap    = trial.suggest_float("pos_weight_cap",       20.0, 60.0, log=True)
        dropout           = trial.suggest_float("dropout",              0.1, 0.4)
        self._logger(
            f"Trial {trial.number + 1}: sampled lr={lr:.3e}, wd={weight_decay:.3e}, "
            f"layers={num_layers}, batch_size={batch_size}, dropout={dropout:.3f}"
            f"pos_weight_cap={pos_weight_cap}"
        )
        model             = self._build_model(trial, dropout, num_layers, batch_size)
        loss_fn           = self._build_loss(pos_weight_cap)
        train_dl          = self._build_dataloader(self.X_train, self.y_train, batch_size, shuffle=True)
        val_dl            = self._build_dataloader(self.X_val,   self.y_val,   batch_size, shuffle=False)
        self._logger(
            f"Trial {trial.number + 1}: dataloaders built train_batches={len(train_dl)}, val_batches={len(val_dl)}"
        )
        optimizer         = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_pr_auc, best_threshold = 0.0, 0.5

        for epoch in range(self.epochs):  # 0-indexed for Hyperband
            train_loss = self._train_epoch(model, train_dl, optimizer, loss_fn)
            val_loss, val_f1, best_thresh, pr_auc = self._val_epoch(model, val_dl, loss_fn)

            # Track best-so-far PR-AUC and the threshold that achieved it.
            # We update this BEFORE reporting to the pruner so the value
            # passed to Hyperband is a non-decreasing curve.
            if pr_auc > best_pr_auc:
                best_pr_auc    = pr_auc
                best_threshold = best_thresh

            self._logger(
                f"Trial {trial.number + 1} epoch {epoch + 1}/{self.epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_f1={val_f1:.4f}, "
                f"pr_auc={pr_auc:.4f}, best_pr_auc={best_pr_auc:.4f}, best_thresh={best_thresh:.2f}"
            )

            # Hyperband (and Optuna pruners in general) compares the value
            # reported at step `epoch` across trials. If we feed the raw
            # per-epoch PR-AUC, a single noisy dip can mask a trial that is
            # actually trending upward and lead to premature pruning.
            #
            # Reporting the running maximum (`best_pr_auc`) yields a
            # monotonically non-decreasing curve, which is the contract
            # Hyperband implicitly assumes ("more resource = at least as
            # good"). This stabilises pruning decisions without changing
            # which trial wins overall (the final return value is unchanged).
            trial.report(best_pr_auc, epoch)
            if trial.should_prune():
                self._logger(
                    f"Trial {trial.number + 1} pruned at epoch {epoch + 1} "
                    f"with best_pr_auc={best_pr_auc:.4f} (last_epoch_pr_auc={pr_auc:.4f})"
                )
                raise optuna.TrialPruned()

        trial.set_user_attr("best_threshold", float(best_threshold))
        self._logger(
            f"Trial {trial.number + 1} complete: best_pr_auc={best_pr_auc:.4f}, "
            f"best_threshold={best_threshold:.2f}"
        )
        return float(best_pr_auc)

    def _build_model(self, trial, dropout, num_layers, batch_size) -> nn.Module:
        from config import CNNTransformerConfig, TransformerConfig, LSTMConfig, PatchTSTConfig
        
        model_config = deepcopy(self.config.model)

        model_config.batch_size = batch_size
        model_config.num_layers = num_layers

        def valid_nheads():
            return [n for n in [2, 4, 8, 16] if model_config.d_model % n == 0]

        # For transformer-based models, ensure nhead divides d_model
        if hasattr(model_config, "nhead") and valid_nheads():
            model_config.nhead = trial.suggest_categorical("nhead", valid_nheads())
            self._logger(f"Trial {trial.number + 1}: valid_nheads={valid_nheads()}, selected_nhead={model_config.nhead}")

        def valid_patch_lengths(context_length):
            return [p for p in [4, 8, 16, 32] if context_length % p == 0]

        if isinstance(model_config, (CNNTransformerConfig, TransformerConfig)):
            model_config.d_model     = trial.suggest_categorical("d_model", [32, 64, 128])
            model_config.dropout     = dropout

            self._logger(
                f"Trial {trial.number + 1}: building {type(model_config).__name__} "
                f"d_model={model_config.d_model}, num_layers={model_config.num_layers}, dropout={model_config.dropout:.3f}"
            )

        elif isinstance(model_config, LSTMConfig):
            model_config.hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256])
            model_config.dropout     = dropout
            self._logger(
                f"Trial {trial.number + 1}: building LSTMConfig hidden_size={model_config.hidden_size}, "
                f"num_layers={model_config.num_layers}, dropout={model_config.dropout:.3f}"
            )

        elif isinstance(model_config, PatchTSTConfig):
            patch_candidates = valid_patch_lengths(model_config.context_length)
            model_config.d_model     = trial.suggest_categorical("d_model", [32, 64, 128])
            model_config.patch_length = trial.suggest_categorical("patch_length", patch_candidates)
            model_config.patch_stride = trial.suggest_int("patch_stride", 1, model_config.patch_length)
            model_config.ffn_dim    = trial.suggest_categorical("ffn_dim", [32, 64, 128, 256, 512])
            model_config.channel_attention = trial.suggest_categorical("channel_attention", [True, False])
            model_config.attention_dropout = trial.suggest_float("attention_dropout", 0.1, 0.4)
            model_config.positional_dropout = trial.suggest_float("positional_dropout", 0.1, 0.4)
            model_config.head_dropout = trial.suggest_float("head_dropout", 0.1, 0.4)
            self._logger(
                f"Trial {trial.number + 1}: building PatchTSTConfig context_length={model_config.context_length}, "
                f"patch_candidates={patch_candidates}, selected_patch_length={model_config.patch_length}, "
                f"patch_stride={model_config.patch_stride}, d_model={model_config.d_model}, ffn_dim={model_config.ffn_dim}"
            )
        else:
            raise TypeError(
                f"No search space defined for {type(model_config).__name__}. "
                f"Add a branch to _build_model() in OptunaOptimizer."
            )
        self._logger(f"Trial {trial.number + 1}: model moved to device={get_device()}")

        return model_config.build(input_dim=self.in_channels).to(get_device())

    def _build_loss(self, pos_weight_cap):
        pw = min(self.raw_pw, pos_weight_cap)
        loss_cls = resolve_loss_fn(self.loss_fn)
        self._logger(
            f"Loss setup: loss_fn={self.loss_fn}, raw_pos_weight={self.raw_pw:.4f}, "
            f"cap={pos_weight_cap}, effective_pos_weight={pw:.4f}"
        )
        return loss_cls(pos_weight=torch.tensor([pw], device=get_device()))

    def _print_results(self, study):
        print("\n=== Best Trial ===")
        print(f"  PR-AUC:     {study.best_trial.value:.4f}")
        print("  Params:")
        for k, v in study.best_trial.params.items():
            print(f"    {k}: {v}")

    def _pretty_trial_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        is_best = (study.best_trial.number + 1) == (trial.number + 1)
        marker   = "★ NEW BEST" if is_best else ""
        duration = trial.duration.total_seconds() if trial.duration else 0.0

        print(f"\n── Trial {trial.number + 1:>3}  {marker}")
        print(f"   value    : {trial.value:.6f}")
        print(f"   duration : {duration:6.1f}s")
        print(f"   params   :")
        for k, v in trial.params.items():
            if isinstance(v, float):
                print(f"     {k:<16} = {v:.6g}")
            else:
                print(f"     {k:<16} = {v}")
        print("─" * 40 + "\n")
