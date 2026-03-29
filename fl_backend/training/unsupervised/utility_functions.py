import matplotlib.pyplot as plt
import torch.nn as nn
import torch


def get_anomaly_scores(
    model: nn.Module,
    dataloader,
    device: torch.device,
):
    model.eval()
    scores = []

    with torch.no_grad():
        for (x_batch,) in dataloader:
            x_batch = x_batch.to(device)

            x_hat = model(x_batch)
            batch_scores = ((x_hat - x_batch) ** 2).mean(dim=(1, 2))
            scores.append(batch_scores.cpu())

    return torch.cat(scores, dim=0)


def detect_anomalies(
    model: nn.Module,
    eval_loader,
    df: pd.DataFrame,
    train_size: int,
    device: torch.device,
    threshold_std: float = 3.0,
):
    scores = get_anomaly_scores(model, eval_loader, device)

    #Plotting resultatet
    plt.plot(scores.numpy())
    plt.title("Anomaly Scores")
    plt.xlabel("Sample Index")
    plt.ylabel("Score")
    plt.show()

    # Calculate threshold
    threshold = scores.mean() + threshold_std * scores.std()
    anomalies = scores > threshold

    anomaly_indices = torch.nonzero(anomalies, as_tuple=True)[0].tolist()
    original_anomaly_indices = [train_size + idx for idx in anomaly_indices]
    anomalous_rows = df.iloc[original_anomaly_indices]

    print(f"Number of anomalies detected: {anomalies.sum()}")
    print(f"Anomaly threshold: {threshold:.6f}")
    print(f"Anomaly scores shape: {scores.shape}")
    print(f"\nAnomalies at indices: {anomaly_indices}")

    # Vis anomaly scores for disse rækker
    print("\n" + "="*60)
    for i, idx in enumerate(anomaly_indices):
        print(f"\nAnomaly {i+1}:")
        print(f"  Test index: {idx}")
        print(f"  Original index: {train_size + idx}")
        print(f"  Anomaly score: {scores[idx]:.6f}")
    
