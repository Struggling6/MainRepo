import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import pandas as pd
from training.evaluate import evaluate_model
from training.train import train_one_epoch


class AnomalyDetectionTask:
    #Skal også inkluerer compute loss og måske rename til anomalyPrediction task og have alt loss og alt muligt her inde??
    """Task class for anomaly detection using reconstruction error."""
    
    def __init__(self):
        self.name = "anomaly_detection"
    
    #def compute_loss(self, model, train_loader, eval_loader, optimizer, device):
        #train_loss = train_one_epoch(model, train_loader, optimizer, device)
        #eval_loss = evaluate(model, eval_loader, device)
        #return train_loss, eval_loss

    def compute_loss(self, model, batch, device):
        """Compute reconstruction error loss for a batch."""
        (x_batch,) = batch
        x_batch = x_batch.to(device)
        
        # Forward pass
        x_hat = model(x_batch)
        
        # Reconstruction error
        loss = ((x_hat - x_batch) ** 2).mean()
        
        return loss, x_hat, x_batch

    def compute_metrics(self, outputs, targets):
        """Compute metrics for anomaly detection."""
        # For autoencoder, we don't have accuracy in traditional sense
        # Return empty metrics for now
        return {"correct": 0}
    
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
        original_anomaly_indices = [train_size + idx for idx in anomaly_indices]
        anomalous_rows = df.iloc[original_anomaly_indices]

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