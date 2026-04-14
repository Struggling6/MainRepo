import torch
import torch.nn as nn
import numpy as np
from training.training_utils.TrainEvalBase import TrainEvalBase


class Trainer(TrainEvalBase):
    """
    Trains a single model with fixed hyperparameters.
    Inherits dataloader, train/val epoch logic from AnomalyTrainerBase.
    """

    def __init__(
        self,
        model:        nn.Module,
        loss_fn:      nn.Module,
        lr:           float,
        batch_size:   int,
        epochs:       int,
        patience:     int,
        num_classes:  int,
        weight_decay: float,
    ):
        super().__init__(epochs, patience, num_classes)  # initialise base class
        self.model       = model.to(self.device)
        self.loss_fn     = loss_fn
        self.batch_size  = batch_size
        self.optimizer   = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> nn.Module:
        """Train the model and restore the best checkpoint."""
        train_loader = self._build_dataloader(X_train, y_train, self.batch_size, shuffle=True)
        val_loader   = self._build_dataloader(X_val,   y_val,   self.batch_size, shuffle=False)

        best_f1, best_state, epochs_without_improvement = -1.0, None, 0
        self.history = []

        for epoch in range(1, self.epochs + 1):
            train_loss                            = self._train_epoch(self.model, train_loader, self.optimizer, self.loss_fn)
            val_loss, val_f1, best_thresh, pr_auc = self._val_epoch(self.model, val_loader, self.loss_fn)

            metrics = {
                "epoch":       epoch,
                "train_loss":  train_loss,
                "val_loss":    val_loss,
                "val_f1":      val_f1,
                "best_thresh": best_thresh,
                "pr_auc":      pr_auc,
            }
            self.history.append(metrics)
            self._print_epoch(metrics)

            # Early stopping
            if val_f1 > best_f1:
                best_f1    = val_f1
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch} (best val F1: {best_f1:.4f})")
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best model (val F1: {best_f1:.4f})")

        return self.model

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _print_epoch(self, m):
        print(
            f"Epoch {m['epoch']:>3}/{self.epochs} | "
            f"Train loss: {m['train_loss']:.4f} | "
            f"Val loss: {m['val_loss']:.4f}  "
            f"F1: {m['val_f1']:.4f}  "
            f"PR-AUC: {m['pr_auc']:.4f}  "
            f"thresh: {m['best_thresh']:.2f}"
        )