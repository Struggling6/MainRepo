import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from fl_backend.training.training_utils.TrainEvalBase import TrainEvalBase
from models.utils import load_model


class Evaluator(TrainEvalBase):
    """
    Evaluates a trained model on a given dataset.
    Inherits _val_epoch and _build_dataloader from AnomalyTrainerBase.

    Two modes of operation:
      - Flower per-round evaluation: call evaluate_round(model, testloader)
      - Full final evaluation:       call evaluate_final(config)
    """

    def __init__(
        self,
        model_config,
        num_classes: int = 1,
    ):
        # epochs and patience are irrelevant for evaluation
        # but required by AnomalyTrainerBase.__init__
        super().__init__(num_classes=num_classes)
        self.model_config = model_config

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate_round(self, model, testloader):
        """
        Lightweight evaluation for Flower's per-round client evaluation.
        Called by FlowerClient.evaluate() after each federation round to
        report how the global aggregated model performs on local data.

        Returns
        -------
        dict with loss, f1, pr_auc, best_threshold, num_examples
        """
        loss_fn = self._build_loss(testloader)

        loss, f1, thresh, pr_auc = self._val_epoch(model, testloader, loss_fn)

        return {
            "loss":           loss,
            "val_f1":         f1,
            "best_threshold": thresh,
            "pr_auc":         pr_auc,
            "num_examples":   int(len(testloader.dataset)),
        }

    def evaluate_final(
            self, 
            model: nn.Module, 
            testloader: torch.utils.data.DataLoader, 
            threshold: float = None
        ):
        """
        Full evaluation on the held-out test set.
        Produces a detailed diagnostic report including classification
        report, confusion matrix, ROC-AUC and PR-AUC.
        Run once after federation completes for final honest assessment.

        Parameters
        ----------
        model      : nn.Module  — trained model
        testloader : DataLoader — held-out test set, never used during training
        threshold  : float      — override decision threshold (uses best found
                                  during _val_epoch if None)
        """
        loss_fn = self._build_loss(testloader)

        loss, f1, best_thresh, pr_auc = self._val_epoch(model, testloader, loss_fn)

        # Use provided threshold override if given
        threshold = threshold if threshold is not None else best_thresh

        # Collect raw probabilities for detailed metrics
        all_probs, all_labels = self._collect_probs(model, testloader)
        all_preds = (all_probs >= threshold).astype(float)

        roc_auc = roc_auc_score(all_labels, all_probs)

        print("\n=== Final Evaluation Results ===")
        print(f"  Threshold : {threshold:.2f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  PR-AUC    : {pr_auc:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")
        print("\n--- Classification Report ---")
        print(classification_report(
            all_labels, all_preds,
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        ))
        print("--- Confusion Matrix ---")
        print(confusion_matrix(all_labels, all_preds))

        return {
            "loss":      loss,
            "f1":        f1,
            "pr_auc":    pr_auc,
            "roc_auc":   roc_auc,
            "threshold": threshold,
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_loss(self, testloader):
        """Build loss function using pos_weight computed from test labels."""
        y      = self._extract_labels(testloader)
        pw     = self._compute_pos_weight(y, cap=self.model_config.pos_weight_cap)
        return self.model_config.loss_fn(
            pos_weight=torch.tensor([pw], device=self.device)
        )

    def _collect_probs(self, model, testloader):
        """Collect raw probabilities and labels for detailed metric computation."""
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for features, labels in testloader:
                features = features.to(self.device)
                logits   = model(features)
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels)

        return (
            torch.cat(all_probs).numpy(),
            torch.cat(all_labels).numpy(),
        )

    @staticmethod
    def _extract_labels(loader):
        """Extract all labels from a DataLoader into a numpy array."""
        all_y = []
        for _, y in loader:
            all_y.append(y.numpy())
        return np.concatenate(all_y)