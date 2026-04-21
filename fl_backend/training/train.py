import torch
import torch.nn as nn
import numpy as np
from config import TrainingConfig, CNNTransformerConfig
from training.training_utils.utils import compute_pos_weight

def train_model(
        model: nn.Module, 
        trainloader: torch.utils.data.DataLoader, 
        training_config: TrainingConfig, 
        model_config: CNNTransformerConfig, 
        device: torch.device
    ):
    """
    Entry point for Flower's client training loop.
    Trains the model using config-specified loss function and hyperparameters.
    """

    # Compute pos_weight from training labels
    y_train    = _extract_labels(trainloader)
    pos_weight = compute_pos_weight(y_train, cap=model_config.pos_weight_cap)

    # Loss function comes from model config — not hardcoded
    loss_fn = model_config.loss_fn(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    # Extract arrays from dataloader for Trainer
    X_train, y_train = _dataloader_to_arrays(trainloader)

    from training.training_utils.Trainer import Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        batch_size=trainloader.batch_size,
        epochs=training_config.local_epochs,
        patience=training_config.patience,
        num_classes=model_config.num_classes,
    )

    # Use same data for val — Flower handles proper evaluation separately
    trainer.train(X_train, y_train, X_train, y_train)

    return {
        "train_loss":   trainer.history[-1]["train_loss"],
        "val_f1":       trainer.history[-1]["val_f1"],
        "num_examples": len(trainloader.dataset),
    }


def _dataloader_to_arrays(loader):
    """Extract all features and labels from a DataLoader into numpy arrays."""
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x.numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_x), np.concatenate(all_y)


def _extract_labels(loader):
    """Extract only labels from a DataLoader for pos_weight computation."""
    all_y = []
    for _, y in loader:
        all_y.append(y.numpy())
    return np.concatenate(all_y)
