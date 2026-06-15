import torch
import torch.nn as nn
import numpy as np
import os

from flwr.common.logger import log
from logging import INFO
from config import TrainingConfig, ExperimentConfig, resolve_loss_fn
from training.training_utils.Trainer import Trainer
from training.training_utils.utils import compute_pos_weight

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"  # Silence ROCm warning

def train_model(
        model:           nn.Module,
        trainloader:     torch.utils.data.DataLoader,
        valloader:       torch.utils.data.DataLoader,
        training_config: TrainingConfig,
        model_config:    ExperimentConfig,
        device:          torch.device,
        proximal_mu:     float = 0.0,
        global_pos_weight: float | None = None,
    ):
    """
    Entry point for Flower's client training loop.
    Trains the model using config-specified loss function and hyperparameters.
    """

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    print(f"[TRAIN] device={device}" + (f" ({gpu_name})" if gpu_name else ""), flush=True)


    if hasattr(model_config, "loss_fn"):
        print("[TRAIN] entering train_model", flush=True)

        # --------------------------------------------------
        # Extract labels (for pos_weight)
        # --------------------------------------------------
        if global_pos_weight is not None:
            pos_weight = global_pos_weight
            print(f"[TRAIN] using global pos_weight: {pos_weight}", flush=True)
        else:
            print("[TRAIN] extracting labels for pos_weight...", flush=True)
            y_train = _extract_labels(trainloader)
            print(f"[TRAIN] labels extracted: shape={y_train.shape}", flush=True)

            pos_weight = compute_pos_weight(y_train, cap=model_config.pos_weight_cap)
            print(f"[TRAIN] pos_weight computed: {pos_weight}", flush=True)


        # --------------------------------------------------
        # Loss function
        # --------------------------------------------------
        print("[TRAIN] creating loss function...", flush=True)
        loss_cls = resolve_loss_fn(model_config.loss_fn)
        loss_fn = loss_cls(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
        ) if loss_cls == nn.BCEWithLogitsLoss else loss_cls()
    else:
        # HuggingFace models — loss is handled internally, pass None
        loss_fn = None
        
    # --------------------------------------------------
    # Trainer setup
    # --------------------------------------------------
    print("[TRAIN] initializing Trainer...", flush=True)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        patience=training_config.patience,
        epochs=training_config.local_epochs,
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
        log(INFO, "Training failed: %s", e)
        raise

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    result = {
        "train_loss":   trainer.history[-1]["train_loss"],
        "val_loss":     trainer.history[-1]["val_loss"],
        "f1":           trainer.history[-1]["val_f1"],
        "pr_auc":       trainer.history[-1]["pr_auc"],
        "roc_auc":      trainer.history[-1]["roc_auc"],
        "threshold":    trainer.history[-1]["best_thresh"],
        "num_examples": len(trainloader.dataset),
    }

    print(f"[TRAIN] returning results: {result}", flush=True)

    return result


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
