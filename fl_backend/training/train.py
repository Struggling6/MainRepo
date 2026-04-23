from logging import INFO
import torch
import torch.nn as nn
import numpy as np
from config import TrainingConfig, CNNTransformerConfig
from training.training_utils.Trainer import Trainer
from training.training_utils.utils import compute_pos_weight


def train_model(
        model:           nn.Module,
        trainloader:     torch.utils.data.DataLoader,
        valloader:       torch.utils.data.DataLoader,
        training_config: TrainingConfig,
        model_config:    CNNTransformerConfig,
        device:          torch.device,
        proximal_mu: float = 0.0,
    ):
    """
    Entry point for Flower's client training loop.
    Trains the model using config-specified loss function and hyperparameters.
    """
    if hasattr(model_config, "loss_fn"):
        print("[TRAIN] entering train_model", flush=True)

        # --------------------------------------------------
        # Extract labels (for pos_weight)
        # --------------------------------------------------
        print("[TRAIN] extracting labels for pos_weight...", flush=True)
        y_train = _extract_labels(trainloader)
        print(f"[TRAIN] labels extracted: shape={y_train.shape}", flush=True)

        pos_weight = compute_pos_weight(y_train, cap=model_config.pos_weight_cap)
        print(f"[TRAIN] pos_weight computed: {pos_weight}", flush=True)




        # --------------------------------------------------
        # Loss function
        # --------------------------------------------------
        print("[TRAIN] creating loss function...", flush=True)
        loss_fn = model_config.loss_fn(
            pos_weight=torch.tensor([pos_weight], device=device)
        ) if model_config.loss_fn == nn.BCEWithLogitsLoss else model_config.loss_fn()
    else:
        # HuggingFace models — loss is handled internally, pass None
        loss_fn = None
        
    # --------------------------------------------------
    # Convert DataLoader -> numpy arrays (THIS IS HEAVY)
    # --------------------------------------------------
    print("[TRAIN] converting dataloader to arrays...", flush=True)
    X_train, y_train = _dataloader_to_arrays(trainloader)
    print(f"[TRAIN] arrays created: X={X_train.shape}, y={y_train.shape}", flush=True)

    # --------------------------------------------------
    # Trainer setup
    # --------------------------------------------------
    print("[TRAIN] initializing Trainer...", flush=True)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        batch_size=trainloader.batch_size,
        epochs=training_config.local_epochs,
        patience=training_config.patience,
        num_classes=model_config.num_classes,
        proximal_mu=proximal_mu,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    print("[TRAIN] starting training loop...", flush=True)
    try:
        trainer.train(trainloader, valloader)
    except Exception as e:
        torch.log(INFO, "Training failed: %s", e)
        raise
    print("[TRAIN] training finished", flush=True)

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    result = {
        "train_loss":   trainer.history[-1]["train_loss"],
        "val_f1":       trainer.history[-1]["val_f1"],
        "num_examples": len(trainloader.dataset),
    }

    print(f"[TRAIN] returning results: {result}", flush=True)

    return result


def _dataloader_to_arrays(loader):
    """Extract all features and labels from a DataLoader into numpy arrays."""
    print("[TRAIN] _dataloader_to_arrays start", flush=True)

    all_x, all_y = [], []

    for i, (x, y) in enumerate(loader):
        if i == 0:
            print("[TRAIN] first batch loaded", flush=True)

        if i % 50 == 0:
            print(f"[TRAIN] loading batch {i}", flush=True)

        all_x.append(x.numpy())
        all_y.append(y.numpy())

    print("[TRAIN] concatenating arrays...", flush=True)
    X = np.concatenate(all_x)
    y = np.concatenate(all_y)

    print("[TRAIN] _dataloader_to_arrays done", flush=True)
    return X, y


def _extract_labels(loader):
    """Extract only labels from a DataLoader for pos_weight computation."""
    print("[TRAIN] _extract_labels start", flush=True)

    all_y = []

    for i, (_, y) in enumerate(loader):
        if i == 0:
            print("[TRAIN] first label batch loaded", flush=True)

        if i % 50 == 0:
            print(f"[TRAIN] extracting labels batch {i}", flush=True)

        all_y.append(y.numpy())

    print("[TRAIN] concatenating labels...", flush=True)
    y = np.concatenate(all_y)

    print("[TRAIN] _extract_labels done", flush=True)
    return y
