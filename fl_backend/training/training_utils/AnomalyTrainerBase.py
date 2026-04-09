import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score
from models.utils import get_device

class AnomalyTrainerBase:
    """
    Shared base class for Trainer and OptunaOptimizer.
    Holds common state and utility methods so neither subclass
    has to reimplement them.
    """

    def __init__(
            self,
            epochs:      int, 
            patience:    int, 
            num_classes: int = 1
            ):
        self.device      = get_device()
        self.epochs      = epochs
        self.patience    = patience # how many epochs to wait for improvement before stopping
        self.num_classes = num_classes

    # ------------------------------------------------------------------ #
    #  Shared utilities                                                    #
    # ------------------------------------------------------------------ #

    def _build_dataloader(self, features, labels, batch_size, shuffle):
        feature_tensor = torch.tensor(features.astype(np.float32))  # cast first to avoid object dtype error
        label_tensor   = torch.tensor(
            labels,
            dtype=torch.long if self.num_classes > 1 else torch.float32
        )
        dataset = TensorDataset(feature_tensor, label_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _compute_pos_weight(self, y_train, cap=None):
        raw_pw = float((y_train == 0).sum() / (y_train == 1).sum())
        return min(raw_pw, cap) if cap else raw_pw

    def _train_epoch(self, model, loader, optimizer, loss_fn):
        model.train()
        total_loss, total_samples = 0.0, 0

        for features, labels in loader:
            features, labels = features.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels)
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
                features, labels = features.to(self.device), labels.to(self.device)
                logits = model(features)
                total_loss    += loss_fn(logits, labels).item() * features.size(0)
                total_samples += features.size(0)
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels.cpu())

        all_probs  = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Search for the decision threshold that maximises F1
        best_f1, best_thresh = 0.0, 0.5
        for thresh in np.arange(0.01, 0.5, 0.01):
            preds = (all_probs >= thresh).astype(float)
            score = f1_score(all_labels, preds, pos_label=1, zero_division=0)
            if score > best_f1:
                best_f1, best_thresh = score, thresh

        pr_auc = average_precision_score(all_labels, all_probs)

        return total_loss / total_samples, best_f1, best_thresh, pr_auc