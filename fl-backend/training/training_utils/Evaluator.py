import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from training.training_utils.TrainEvalBase import TrainEvalBase
from training.training_utils.utils import compute_pos_weight
from local_experiment import CONFIG, resolve_loss_fn

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
        metadata
    ):
        # epochs are irrelevant for evaluation
        # but required by AnomalyTrainerBase.__init__
        super().__init__(num_classes=model_config.num_classes)
        self.model_config = model_config
        self.metadata = metadata
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
        model_path: Path,
        testloader: torch.utils.data.DataLoader,
        threshold:  float = None,
    ):
        """
        Full evaluation on the held-out test set.
        Loads the model from disk, runs inference, and produces a detailed
        diagnostic report. Run once after federation completes.

        Parameters
        ----------
        model_path : Path      — path to the saved checkpoint (.pt file)
        testloader : DataLoader — held-out test set, never used during training
        threshold  : float     — override decision threshold. If None, uses
                                    the threshold found during _val_epoch.
        """
        model            = self._load_model(model_path)
        loss_fn          = self._build_loss(testloader)
        loss, f1, best_thresh, pr_auc = self._val_epoch(model, testloader, loss_fn)

        threshold = threshold if threshold is not None else best_thresh

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

    def _load_model(self, path: Path) -> nn.Module:
        """
        Load model weights from a checkpoint file into a fresh model
        built from self.model_config.
        """

        model = CONFIG.model.build(input_dim=self.metadata["input_dim"])

        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self._loaded_threshold = checkpoint.get("threshold", 0.5)
        self._loaded_metrics   = checkpoint.get("metrics",   {})

        print(f"Model loaded from {path}  (threshold={self._loaded_threshold:.2f})")
        return model

    def _build_loss(self, testloader):
        """Build loss function using pos_weight computed from test labels."""
        y      = self._extract_labels(testloader)
        pw     = compute_pos_weight(y, cap=self.model_config.pos_weight_cap)
        loss_cls = resolve_loss_fn(self.model_config.loss_fn)
        return loss_cls(
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