import threading
import optuna
import os
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"  # Has to happen before torch import

import torch
import torch.nn as nn
import gc

from copy import deepcopy
from training.training_utils.TrainEvalBase import TrainEvalBase
from training.training_utils.utils import compute_pos_weight
from models.utils import get_device
from local_experiment import CONFIG
from config import resolve_loss_fn, FloatRange, IntRange
from optuna_config import OPTUNA_CONFIG


def _suggest(trial, name, spec):
    """Translate a FloatRange / IntRange / list into a trial.suggest_* call."""
    if isinstance(spec, FloatRange):
        return trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    if isinstance(spec, IntRange):
        return trial.suggest_int(name, spec.low, spec.high, log=spec.log)
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)

    raise TypeError(
        f"Unsupported search-space spec for {name!r}: {type(spec).__name__}"
    )


class OptunaOptimizer(TrainEvalBase):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _logger(self, msg: str):
        _print_lock = threading.Lock()
        with _print_lock:
            print(f"[Optuna] {msg}", flush=True)

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        config=CONFIG,
        optuna_config=OPTUNA_CONFIG,
        n_trials=50,
        epochs=20,
        storage=None,
        study_name=None,
        n_jobs=1,
    ):
        super().__init__(num_classes=1, epochs=epochs)
        self.config = config
        self.optuna_config = optuna_config
        self.search = self._select_search_space(config.model, optuna_config)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.in_channels = int(X_train.shape[-1])
        self.config.evaluation.input_dim = self.in_channels

        self.n_trials = n_trials
        self.loss_fn = config.model.loss_fn
        self.raw_pw = compute_pos_weight(y_train)

        self.storage = storage
        self.study_name = study_name
        self.device = get_device()
        self.n_jobs = n_jobs

    @staticmethod
    def _select_search_space(model_config, optuna_config):
        from config import (
            CNNTransformerConfig,
            TransformerConfig,
            LSTMConfig,
            PatchTSTConfig,
            TimesNetConfig,
            SupervisedCNNConfig,
        )

        if isinstance(model_config, LSTMConfig):
            return optuna_config.lstm
        if isinstance(model_config, TransformerConfig):
            return optuna_config.transformer
        if isinstance(model_config, CNNTransformerConfig):
            return optuna_config.cnn_transformer
        if isinstance(model_config, PatchTSTConfig):
            return optuna_config.patchtst
        if isinstance(model_config, TimesNetConfig):
            return optuna_config.timesnet
        if isinstance(model_config, SupervisedCNNConfig):
            return optuna_config.cnn

        raise TypeError(
            f"No Optuna search space configured for {type(model_config).__name__}"
        )

    def run(self):
        self._logger(
            f"Starting study name={self.study_name}, trials={self.n_trials}, "
            f"epochs={self.epochs}, train_shape={self.X_train.shape}, "
            f"val_shape={self.X_val.shape}, device={self.device}"
        )

        pruner_cfg = self.optuna_config.pruner
        min_resource = max(1, int(self.epochs * pruner_cfg.min_resource_fraction))

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=min_resource,
                max_resource=self.epochs,
                reduction_factor=pruner_cfg.reduction_factor,
            ),
            load_if_exists=True,
        )

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            callbacks=[self._pretty_trial_callback],
            gc_after_trial=True,
            show_progress_bar=True,
        )

        self._print_results(study)
        self._logger("Study finished")
        return study

    def _objective_score(self, pr_auc: float, f1: float) -> float:
        """Weighted scalar Optuna optimizes. Set pr_auc_weight=1, f1_weight=0
        for pure PR-AUC; flip for pure F1; mix for a blended objective."""
        s = self.optuna_config.scoring
        return s.pr_auc_weight * float(pr_auc) + s.f1_weight * float(f1)

    def _objective(self, trial):
        model = None
        optimizer = None
        train_dl = None
        val_dl = None

        try:
            s = self.search

            lr = _suggest(trial, "lr", s.lr)
            weight_decay = _suggest(trial, "weight_decay", s.weight_decay)
            num_layers = _suggest(trial, "num_layers", s.num_layers)
            batch_size = _suggest(trial, "batch_size", s.batch_size)
            pos_weight_cap = _suggest(trial, "pos_weight_cap", s.pos_weight_cap)

            self._logger(
                f"Trial {trial.number + 1}: sampled lr={lr:.3e}, "
                f"wd={weight_decay:.3e}, layers={num_layers}, "
                f"batch_size={batch_size}, pos_weight_cap={pos_weight_cap}"
            )

            model = self._build_model(trial, num_layers, batch_size)
            loss_fn = self._build_loss(pos_weight_cap)

            train_dl = self._build_dataloader(
                self.X_train,
                self.y_train,
                batch_size,
                shuffle=True,
            )

            val_dl = self._build_dataloader(
                self.X_val,
                self.y_val,
                batch_size,
                shuffle=False,
            )

            self._logger(
                f"Trial {trial.number + 1}: dataloaders built "
                f"train_batches={len(train_dl)}, val_batches={len(val_dl)}"
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

            best_pr_auc = 0.0
            best_f1 = 0.0
            best_score = 0.0
            best_threshold = 0.5

            for epoch in range(self.epochs):
                train_loss = self._train_epoch(model, train_dl, optimizer, loss_fn)
                val_loss, val_f1, best_thresh, pr_auc, roc_auc  = self._val_epoch(
                    model,
                    val_dl,
                    loss_fn,
                )

                score = self._objective_score(pr_auc, val_f1)

                # Track best epoch by the weighted score so PR-AUC, F1, or any
                # mix can drive the search just by changing the weights.
                if score > best_score:
                    best_pr_auc = pr_auc
                    best_f1 = val_f1
                    best_score = score
                    best_threshold = best_thresh

                self._logger(
                    f"Trial {trial.number + 1} epoch {epoch + 1}/{self.epochs}: "
                    f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                    f"val_f1={val_f1:.4f}, pr_auc={pr_auc:.4f}, "
                    f"score={score:.4f}, best_score={best_score:.4f}, "
                    f"best_pr_auc={best_pr_auc:.4f}, best_f1={best_f1:.4f}, "
                    f"best_thresh={best_threshold:.2f}"
                )

                trial.report(best_score, epoch)
                trial.set_user_attr("score", float(best_score))
                trial.set_user_attr("pr_auc", float(best_pr_auc))
                trial.set_user_attr("f1", float(best_f1))
                trial.set_user_attr("threshold", float(best_threshold))
                trial.set_user_attr("best_threshold", float(best_threshold))
                
                if trial.should_prune():
                    self._logger(
                        f"Trial {trial.number + 1} pruned at epoch {epoch + 1} "
                        f"with best_score={best_score:.4f} "
                        f"(last_epoch_score={score:.4f})"
                    )
                    raise optuna.TrialPruned()



            self._logger(
                f"Trial {trial.number + 1} complete: "
                f"best_pr_auc={best_pr_auc:.4f}, "
                f"best_f1={best_f1:.4f}, "
                f"best_score={best_score:.4f}, "
                f"best_threshold={best_threshold:.2f}"
            )

            return float(best_score)

        except torch.cuda.OutOfMemoryError as e:
            self._logger(
                f"Trial {trial.number + 1} hit CUDA OOM "
                f"({e.__class__.__name__}); marking pruned and continuing study."
            )
            raise optuna.TrialPruned() from e

        finally:
            del model, optimizer, train_dl, val_dl
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _valid_nheads(d_model, candidates):
        return [n for n in candidates if d_model % n == 0]

    @staticmethod
    def _valid_patch_lengths(context_length, candidates):
        valid = [p for p in candidates if context_length % p == 0]

        if not valid:
            raise ValueError(
                f"No patch_length in {candidates} divides "
                f"context_length={context_length}. "
                f"Add a divisor of {context_length} to patch_length_candidates."
            )

        return valid

    def _build_model(self, trial, num_layers, batch_size) -> nn.Module:
        from config import (
            CNNTransformerConfig,
            TransformerConfig,
            LSTMConfig,
            PatchTSTConfig,
            TimesNetConfig,
            SupervisedCNNConfig,
        )

        model_config = deepcopy(self.config.model)
        s = self.search

        model_config.batch_size = batch_size
        if not isinstance(model_config, SupervisedCNNConfig):
            model_config.num_layers = num_layers

        if isinstance(model_config, (CNNTransformerConfig, TransformerConfig)):
            model_config.d_model = _suggest(trial, "d_model", s.d_model)
            model_config.nhead = trial.suggest_categorical(
                "nhead",
                self._valid_nheads(model_config.d_model, s.nhead_candidates),
            )
            model_config.dropout = _suggest(trial, "dropout", s.dropout)
        elif isinstance(model_config, SupervisedCNNConfig):
            model_config.d_model = _suggest(trial, "d_model", s.d_model)
            model_config.dropout = _suggest(trial, "dropout", s.dropout)
            model_config.pos_weight_cap = _suggest(trial, "pos_weight_cap", s.pos_weight_cap)

        elif isinstance(model_config, LSTMConfig):
            model_config.hidden_size = _suggest(trial, "hidden_size", s.hidden_size)
            model_config.dropout = _suggest(trial, "dropout", s.dropout)

        elif isinstance(model_config, PatchTSTConfig):
            model_config.d_model = _suggest(trial, "d_model", s.d_model)
            model_config.nhead = trial.suggest_categorical(
                "nhead",
                self._valid_nheads(model_config.d_model, s.nhead_candidates),
            )
            model_config.patch_length = trial.suggest_categorical(
                "patch_length",
                self._valid_patch_lengths(
                    model_config.context_length,
                    s.patch_length_candidates,
                ),
            )
            model_config.patch_stride = max(1, model_config.patch_length // 2)
            model_config.ffn_dim = _suggest(trial, "ffn_dim", s.ffn_dim)
            model_config.channel_attention = _suggest(
                trial,
                "channel_attention",
                s.channel_attention,
            )
            model_config.attention_dropout = _suggest(
                trial,
                "attention_dropout",
                s.attention_dropout,
            )
            model_config.positional_dropout = _suggest(
                trial,
                "positional_dropout",
                s.positional_dropout,
            )
            model_config.head_dropout = _suggest(
                trial,
                "head_dropout",
                s.head_dropout,
            )

        elif isinstance(model_config, TimesNetConfig):
            model_config.d_model = _suggest(trial, "d_model", s.d_model)
            model_config.top_k = _suggest(trial, "top_k", s.top_k)
            model_config.d_ffn = _suggest(trial, "d_ffn", s.d_ffn)
            model_config.n_kernels = _suggest(trial, "n_kernels", s.n_kernels)
            model_config.dropout = _suggest(trial, "dropout", s.dropout)

        else:
            raise TypeError(
                f"No search space defined for {type(model_config).__name__}. "
                f"Add a branch to _build_model() in OptunaOptimizer."
            )

        self._logger(
            f"Trial {trial.number + 1}: built {type(model_config).__name__} "
            f"params={dict(trial.params)}"
        )

        return model_config.build(input_dim=self.in_channels).to(self.device)

    def _build_loss(self, pos_weight_cap):
        pw = min(self.raw_pw, pos_weight_cap)
        loss_cls = resolve_loss_fn(self.loss_fn)

        self._logger(
            f"Loss setup: loss_fn={self.loss_fn}, "
            f"raw_pos_weight={self.raw_pw:.4f}, "
            f"cap={pos_weight_cap:.4f}, "
            f"effective_pos_weight={pw:.4f}"
        )

        return loss_cls(pos_weight=torch.tensor([pw], device=self.device))

    def _print_results(self, study):
        print("\n=== Best Trial ===")
        print(f"  Score:      {study.best_trial.value:.4f}")
        print(f"  PR-AUC:     {study.best_trial.user_attrs.get('pr_auc')}")
        print(f"  F1:         {study.best_trial.user_attrs.get('f1')}")
        print(f"  Threshold:  {study.best_trial.user_attrs.get('threshold')}")
        print("  Params:")

        for k, v in study.best_trial.params.items():
            print(f"    {k}: {v}")

    def _pretty_trial_callback(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ):
        try:
            is_best = study.best_trial.number == trial.number
        except ValueError:
            is_best = False

        marker = "★ NEW BEST" if is_best else ""
        duration = trial.duration.total_seconds() if trial.duration else 0.0

        print(f"\n── Trial {trial.number + 1:>3}  {marker}")

        if trial.value is None:
            print("   value    : None")
        else:
            print(f"   value    : {trial.value:.6f}")

        print(f"   duration : {duration:6.1f}s")
        print("   metrics  :")
        print(f"     score           = {trial.user_attrs.get('score')}")
        print(f"     pr_auc          = {trial.user_attrs.get('pr_auc')}")
        print(f"     f1              = {trial.user_attrs.get('f1')}")
        print(f"     threshold       = {trial.user_attrs.get('threshold')}")
        print("   params   :")

        for k, v in trial.params.items():
            if isinstance(v, float):
                print(f"     {k:<16} = {v:.6g}")
            else:
                print(f"     {k:<16} = {v}")

        print("─" * 40 + "\n")