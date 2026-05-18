import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from training.training_utils.TrainEvalBase import TrainEvalBase
from training.training_utils.utils import compute_pos_weight
from local_experiment import CONFIG, resolve_loss_fn


class Evaluator(TrainEvalBase):
    """
    Runs final evaluation of the saved global model on the held-out test
    set. Loads weights from a checkpoint, runs inference, and produces a
    full diagnostic report (classification report + confusion matrix).
    """

    def __init__(self):
        super().__init__(num_classes=CONFIG.model.num_classes)
        self.model_config = CONFIG.model
        self._data_handler = None
        self._metadata = None
        self._checkpoint_threshold = None

    @property
    def data_handler(self):
        if self._data_handler is None:
            self._data_handler = CONFIG.data.build_handler(CONFIG)
        return self._data_handler

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self.data_handler.get_metadata()
        return self._metadata

    @classmethod
    def evaluate(
        cls,
        model_path: Path = None,
        testloader: DataLoader = None,
        threshold: float = None,
    ):
        self = cls()
        model_path = model_path or CONFIG.evaluation.model_path
        testloader = testloader if testloader is not None else self._build_testloader()

        model = self._load_model(model_path)

        all_probs, all_labels = self._collect_probs(model, testloader)

        # Threshold precedence: explicit arg > checkpoint > PR-curve best > 0.5
        if threshold is None:
            threshold = self._checkpoint_threshold

        if threshold is None:
            precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
            precision = precision[:-1]
            recall = recall[:-1]
            if thresholds.size > 0:
                denom = precision + recall
                f1_scores = np.zeros_like(denom, dtype=float)
                np.divide(2 * precision * recall, denom, out=f1_scores, where=denom > 0)
                threshold = float(thresholds[np.argmax(f1_scores)])
            else:
                threshold = 0.5

        all_preds = (all_probs >= threshold).astype(float)

        f1_val = f1_score(all_labels, all_preds, zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)
        precision_val = precision_score(all_labels, all_preds, zero_division=0)
        recall_val = recall_score(all_labels, all_preds, zero_division=0)
        pr_auc = average_precision_score(all_labels, all_probs)
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            roc_auc = 0.0

        loss = self._compute_loss(model, testloader)

        print("\n=== Final Evaluation Results ===")
        print(f"  Threshold : {threshold:.2f}")
        print(f"  F1        : {f1_val:.4f}")
        print(f"  PR-AUC    : {pr_auc:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")
        print(f"  Accuracy  : {accuracy:.4f}")
        print(f"  Precision : {precision_val:.4f}")
        print(f"  Recall    : {recall_val:.4f}")
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
            "f1":        f1_val,
            "pr_auc":    pr_auc,
            "roc_auc":   roc_auc,
            "threshold": threshold,
            "accuracy":  accuracy,
            "precision": precision_val,
            "recall":    recall_val,
        }

    def _build_testloader(self) -> DataLoader:
        test_path = Path(CONFIG.evaluation.test_path)

        if not test_path.exists():
            app_root = Path(__file__).resolve().parents[2]
            candidate = app_root / test_path

            if candidate.exists():
                test_path = candidate
                
        X_test, y_test = self.data_handler.load_test_set(CONFIG.evaluation.test_path)
        return DataLoader(
            TensorDataset(
                torch.tensor(X_test.astype(np.float32)),
                torch.tensor(y_test.astype(np.float32)),
            ),
            batch_size=CONFIG.evaluation.batch_size,
            shuffle=False,
        )

    def _load_model(self, path: Path) -> nn.Module:
        model = CONFIG.model.build(input_dim=self.metadata["input_dim"])

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self._checkpoint_threshold = checkpoint.get("threshold")
        loaded_info = self._checkpoint_threshold
        if loaded_info is not None:
            print(f"Model loaded from {path}  (threshold={loaded_info:.2f})")
        else:
            print(f"Model loaded from {path}  (no threshold in checkpoint)")
        return model

    def _build_loss(self, testloader: DataLoader):
        y = np.concatenate([y.numpy() for _, y in testloader])
        pw = compute_pos_weight(y, cap=self.model_config.pos_weight_cap)
        loss_cls = resolve_loss_fn(self.model_config.loss_fn)
        return loss_cls(
            pos_weight=torch.tensor([pw], device=self.device)
        )

    def _collect_probs(self, model: nn.Module, testloader: DataLoader):
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for features, labels in testloader:
                features = features.to(self.device)
                logits = model(features)
                all_probs.append(torch.sigmoid(logits).cpu().view(-1))
                all_labels.append(labels.cpu().view(-1))

        return (
            torch.cat(all_probs).numpy(),
            torch.cat(all_labels).numpy(),
        )

    def _compute_loss(self, model: nn.Module, testloader: DataLoader) -> float:
        loss_fn = self._build_loss(testloader)
        model.eval()
        total_loss, total_samples = 0.0, 0

        with torch.no_grad():
            for features, labels in testloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                logits = model(features)
                loss_labels = labels.float().view_as(logits)
                total_loss += loss_fn(logits, loss_labels).item() * features.size(0)
                total_samples += features.size(0)

        return total_loss / total_samples if total_samples > 0 else 0.0
