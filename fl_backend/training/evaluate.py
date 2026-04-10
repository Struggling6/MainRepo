import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from config import CONFIG, ExperimentConfig
from data.lead_csv import LeadCSVHandler
from models.registry import create_model
from models.utils import load_model, get_device


def evaluate(config: ExperimentConfig): 
    device = get_device()

    # ── Load test data ───────────────────────────────────────────────── #
    print(f"Loading test data from {config.evaluation.test_path}")
    test_handler = LeadCSVHandler(config.data)
    _, _, X_test, y_test, _, _ = test_handler.run_split()

    X_tensor = torch.tensor(X_test.astype(np.float32))
    y_tensor = torch.tensor(y_test.astype(np.float32))

    dataset    = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    testloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
    )

    # ── Load model ───────────────────────────────────────────────────── #
    data_metadata = test_handler.get_metadata()
    model         = create_model(config.model, data_metadata)
    model, threshold, saved_metrics = load_model(
        model,
        path=config.evaluation.model_path,
        device=device,
    )

    # Override threshold from config if explicitly set
    threshold = config.evaluation.threshold if config.evaluation.threshold != 0.5 else threshold

    # ── Run inference ────────────────────────────────────────────────── #
    all_probs, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for features, labels in testloader:
            features = features.to(device)
            logits   = model(features)
            probs    = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds  = (all_probs >= threshold).astype(float)

    # ── Compute metrics ──────────────────────────────────────────────── #
    f1     = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    pr_auc = average_precision_score(all_labels, all_probs)

    print("\n=== Evaluation Results ===")
    print(f"  Threshold : {threshold:.2f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Anomaly"]))
    print("--- Confusion Matrix ---")
    print(confusion_matrix(all_labels, all_preds))

    return {
        "f1":        f1,
        "pr_auc":    pr_auc,
        "threshold": threshold,
    }


if __name__ == "__main__":
    evaluate(CONFIG)