import torch
import torch.nn as nn
from training.training_utils.TrainEvalBase import TrainEvalBase


class Trainer(TrainEvalBase):
    def __init__(
        self,
        model:         nn.Module,
        loss_fn:       nn.Module | None, # Optionally pass None for HuggingFace models that compute loss internally
        lr:            float,
        epochs:        int,
        num_classes:   int,
        weight_decay:  float,
        patience:      int,
        proximal_mu:   float = 0.0,
        global_params: list[torch.Tensor] | None = None,
    ):
        super().__init__(num_classes=num_classes, epochs=epochs)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs), eta_min=lr * 0.01
        )
        self.patience = patience
        self.proximal_mu = proximal_mu

        # FedProx saves the global model before the client starts local training.
        # The copy is kept on the training device so we do not move it every batch.
        if global_params is None:
            self.global_params = [
                p.detach().clone().to(self.device)
                for p in self.model.parameters()
            ]
        else:
            self.global_params = [
                p.detach().clone().to(self.device)
                for p in global_params
            ]

    def train(self, train_loader, val_loader) -> nn.Module:
        best_pr_auc, epochs_without_improvement = 0.0, 0
        self.history = []

        for epoch in range(1, self.epochs + 1):
            train_loss                            = self._train_epoch(self.model, train_loader, self.optimizer, self.loss_fn)
            val_loss, val_f1, best_thresh, pr_auc, roc_auc = self._val_epoch(self.model, val_loader, self.loss_fn)
            self.scheduler.step()

            metrics = {
                "epoch":       epoch,
                "train_loss":  train_loss,
                "val_loss":    val_loss,
                "val_f1":      val_f1,
                "best_thresh": best_thresh,
                "pr_auc":      pr_auc,
                "roc_auc":     roc_auc,
            }
            self.history.append(metrics)
            self._print_epoch(metrics)

            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch} (best PR-AUC: {best_pr_auc:.4f})")
                    break

        return self.model

    def _train_epoch(self, model, loader, optimizer, loss_fn):
        model.train()
        total_loss, total_samples = 0.0, 0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device).float()

            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)

            # FedProx penalty: keep the local client model close to the global model.
            if self.proximal_mu > 0:
                prox_term = 0.0

                # Sum the squared distance between current local weights and saved global weights.
                for local_param, global_param in zip(model.parameters(), self.global_params):
                    prox_term += torch.sum((local_param - global_param) ** 2)

                # Add the FedProx term to the normal loss: loss + (mu / 2) * distance.
                loss = loss + (self.proximal_mu / 2.0) * prox_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

        return total_loss / total_samples

    def _print_epoch(self, m):
        print(
            f"Epoch {m['epoch']:>3}/{self.epochs} | "
            f"Train loss: {m['train_loss']:.4f} | "
            f"Val loss: {m['val_loss']:.4f}  "
            f"F1: {m['val_f1']:.4f}  "
            f"PR-AUC: {m['pr_auc']:.4f}  "
            f"ROC-AUC: {m['roc_auc']:.4f}  "
            f"thresh: {m['best_thresh']:.2f}"
        )
