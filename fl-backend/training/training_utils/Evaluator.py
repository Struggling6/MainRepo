import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
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
        loss_fn = self._build_loss(testloader)
        loss, f1, best_thresh, pr_auc, roc_auc = self._val_epoch(model, testloader, loss_fn)

        threshold = threshold if threshold is not None else best_thresh

        all_probs, all_labels = self._collect_probs(model, testloader)
        all_preds = (all_probs >= threshold).astype(float)

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

        loaded_threshold = checkpoint.get("threshold", 0.5)
        print(f"Model loaded from {path}  (threshold={loaded_threshold:.2f})")
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
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels)

        return (
            torch.cat(all_probs).numpy(),
            torch.cat(all_labels).numpy(),
        )
