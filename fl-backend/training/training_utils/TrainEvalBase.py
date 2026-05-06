import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
from models.utils import get_device


class TrainEvalBase:
    """
    Shared base class for Trainer and OptunaOptimizer.
    Holds common state and utility methods so neither subclass
    has to reimplement them.
    """

    def __init__(
        self,
        num_classes: int = 1,
        epochs: int = 1,
    ):
        self.device = get_device()
        self.epochs = epochs
        self.num_classes = num_classes

    # ------------------------------------------------------------------ #
    # Shared utilities                                                    #
    # ------------------------------------------------------------------ #

    def _build_dataloader(self, features, labels, batch_size, shuffle):
        feature_tensor = torch.tensor(features.astype(np.float32))
        label_tensor = torch.tensor(
            labels,
            dtype=torch.long if self.num_classes > 1 else torch.float32,
        )
        dataset = TensorDataset(feature_tensor, label_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _train_epoch(self, model, loader, optimizer, loss_fn):
        model.train()
        total_loss, total_samples = 0.0, 0

        for features, labels in loader:
            features = features.to(self.device)
            labels   = labels.to(self.device).float()   # ← cast to float32

            optimizer.zero_grad()
            logits = model(features)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * features.size(0)
            total_samples += features.size(0)

        return total_loss / total_samples

    def _val_epoch(self, model, loader, loss_fn):
        model.eval()
        total_loss, total_samples = 0.0, 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits = model(features)

                if self.num_classes == 1:
                    loss_labels = labels.float().view_as(logits)
                else:
                    loss_labels = labels

                total_loss += loss_fn(logits, loss_labels).item() * features.size(0)
                total_samples += features.size(0)

                all_probs.append(torch.sigmoid(logits).cpu().view(-1))
                all_labels.append(labels.cpu().view(-1))

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # ------------------------------------------------------------------
        # Decision-threshold search via the precision-recall curve.
        #
        # `precision_recall_curve` returns precision/recall at every unique
        # score that occurs in `all_probs`, which is the finest threshold
        # set that can ever change the confusion matrix (no information is
        # lost vs. a fine grid, and no unreachable thresholds are tested).
        #
        # F1 is computed per-threshold from those precision/recall arrays,
        # then we pick the argmax. This sweeps the full [0.0, 1.0) range,
        # fixing the previous bug where np.arange(0.01, 0.5, 0.01) could
        # never select an optimum at or above 0.5.
        # ------------------------------------------------------------------
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

        # precision_recall_curve appends a final (precision=1, recall=0)
        # sentinel point that has no associated threshold, so trim it.
        precision = precision[:-1]
        recall = recall[:-1]

        if thresholds.size == 0:
            # Degenerate batch (all labels identical or empty). Fall back.
            best_f1, best_thresh = 0.0, 0.5
        else:
            denom = precision + recall
            f1_scores = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
            best_idx = int(np.argmax(f1_scores))
            best_f1 = float(f1_scores[best_idx])
            best_thresh = float(thresholds[best_idx])

        # ------------------------------------------------------------------
        # PR-AUC (average precision).
        #
        # Note on interpretation: PR-AUC is class-imbalance sensitive. The
        # baseline expected score for a random classifier equals the
        # positive class prevalence (e.g., for LEAD `anomaly` ~= 0.0213,
        # so a random model scores PR-AUC ~= 0.0213, not 0.5). Always
        # compare PR-AUC to the prevalence baseline, not to 0.5.
        # ------------------------------------------------------------------
        pr_auc = average_precision_score(all_labels, all_probs)

        return total_loss / total_samples, best_f1, best_thresh, pr_auc