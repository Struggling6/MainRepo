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
            labels   = labels.to(self.device)

            optimizer.zero_grad()
            logits = model(features)
            loss   = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        # -----------------------------
        # PR-AUC (threshold-independent)
        # -----------------------------
        pr_auc = average_precision_score(all_labels, all_probs)

        # -----------------------------
        # Find best threshold via PR curve
        # -----------------------------
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

        # Remove last sentinel point
        precision = precision[:-1]
        recall = recall[:-1]

        if thresholds.size == 0:
            best_f1 = 0.0
            best_thresh = 0.5
 
        else:
            denom = precision + recall
            f1_scores = np.zeros_like(denom, dtype=float)

            np.divide(
                2 * precision * recall,
                denom,
                out=f1_scores,
                where=denom > 0,
            )

            best_idx = int(np.argmax(f1_scores))

            best_f1 = float(f1_scores[best_idx])
            best_thresh = float(thresholds[best_idx])


        return (
            total_loss / total_samples,
            best_f1,
            best_thresh,
            pr_auc,
        )