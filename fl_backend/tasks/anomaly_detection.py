import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

class AnomalyDetectionTask:
    """Task class for anomaly detection using reconstruction error."""
    
    def __init__(self):
        self.name = "anomaly_detection"
        self.max_norm = 1.0  # Gradient clipping threshold
        self.model_ref = None  # Vil blive sat af train_model

    def compute_loss(self, model, batch, device):
        """Compute reconstruction error loss for a batch."""
        (x_batch,) = batch
        x_batch = x_batch.to(device)
        
        # Forward pass
        x_hat = model(x_batch)
        
        # Reconstruction error
        loss = ((x_hat - x_batch) ** 2).mean()
        
        # ✅ CHECK: Fanger NaN/Inf før det bliver værre
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Loss became NaN or Inf during training.")
        
        # Gem model reference for gradient clipping efter backward
        self.model_ref = model
        
        return loss, x_hat, x_batch

    def apply_gradient_clipping(self, model=None):
        """Apply gradient clipping to prevent exploding gradients."""
        if model is None:
            model = self.model_ref
        
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_norm)

    
    def get_anomaly_scores(self, model, dataloader, device):
        """Get anomaly scores for samples."""
        model.eval()
        scores = []

        with torch.no_grad():
            for (x_batch,) in dataloader:
                x_batch = x_batch.to(device)
                x_hat = model(x_batch)
                batch_scores = ((x_hat - x_batch) ** 2).mean(dim=(1, 2))
                scores.append(batch_scores.cpu())

        return torch.cat(scores, dim=0)

    def detect_anomalies(self, model, eval_loader, df, train_size, device, threshold_std=3.0):
        """Detect anomalies using reconstruction error threshold."""
        scores = self.get_anomaly_scores(model, eval_loader, device)

        plt.plot(scores.numpy())
        plt.title("Anomaly Scores")
        plt.xlabel("Sample Index")
        plt.ylabel("Score")
        plt.show()

        threshold = scores.mean() + threshold_std * scores.std()
        anomalies = scores > threshold
        anomaly_indices = torch.nonzero(anomalies, as_tuple=True)[0].tolist()

        print(f"Number of anomalies detected: {anomalies.sum()}")
        print(f"Anomaly threshold: {threshold:.6f}")
        print(f"Anomaly scores shape: {scores.shape}")
        print(f"\nAnomalies at indices: {anomaly_indices}")

        for i, idx in enumerate(anomaly_indices):
            print(f"\nAnomaly {i+1}:")
            print(f"  Test index: {idx}")
            print(f"  Original index: {train_size + idx}")
            print(f"  Anomaly score: {scores[idx]:.6f}")
        
        return anomalies, scores, threshold
    
    def compute_metrics(self, outputs, targets):
        """Compute metrics for anomaly detection."""
        reconstruction_errors = ((outputs - targets) ** 2).mean(dim=(1, 2))
        
        threshold = reconstruction_errors.mean() + 3.0 * reconstruction_errors.std()
        anomalies = reconstruction_errors > threshold
        
        num_anomalies = anomalies.sum().item()
        total_samples = len(reconstruction_errors)
        
        # ✅ Apply gradient clipping efter metrics beregning
        # Så sker det lige før optimizer.step() i train.py
        if self.model_ref is not None:
            self.apply_gradient_clipping(self.model_ref)
        
        return {
            "correct": num_anomalies,
            "total": total_samples,
        }