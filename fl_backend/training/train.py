import torch
import torch.nn as nn
from training.training_utils.Trainer import Trainer


def train_model(model, trainloader, task, training_config, device):
    """
    Entry point for Flower's client training loop.
    Delegates to Trainer for the actual training logic.
    """
    pos_weight = Trainer._compute_pos_weight(
        None,  # static-style call — no instance needed
        y_train=_extract_labels(trainloader),
        cap=getattr(training_config, "pos_weight_cap", 10.0),
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        batch_size=trainloader.batch_size,
        epochs=training_config.local_epochs,
        patience=training_config.patience,
        num_classes=1,
    )

    # Flower passes pre-built dataloaders — extract arrays for Trainer
    X_train, y_train = _dataloader_to_arrays(trainloader)
    X_val,   y_val   = _dataloader_to_arrays(trainloader)  # use same for now — Flower handles eval separately

    trained_model = trainer.train(X_train, y_train, X_val, y_val)

    total_examples = len(trainloader.dataset)

    return {
        "loss":         trainer.history[-1]["train_loss"],
        "val_f1":       trainer.history[-1]["val_f1"],
        "num_examples": total_examples,
    }


def _dataloader_to_arrays(loader):
    """Extract all features and labels from a DataLoader into numpy arrays."""
    import numpy as np
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x.numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_x), np.concatenate(all_y)


def _extract_labels(loader):
    """Extract only labels from a DataLoader for pos_weight computation."""
    import numpy as np
    all_y = []
    for _, y in loader:
        all_y.append(y.numpy())
    return np.concatenate(all_y)