import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score
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

        for batch_idx, (features, labels) in enumerate(loader):            
            features = features.to(self.device)
            labels   = labels.to(self.device).float()

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
                    loss_labels = labels.long()

                total_loss += loss_fn(logits, loss_labels).item() * features.size(0)
                total_samples += features.size(0)

                all_probs.append(torch.sigmoid(logits).cpu().view(-1))
                all_labels.append(labels.cpu().view(-1))

        all_probs = torch.cat(all_probs).numpy()
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