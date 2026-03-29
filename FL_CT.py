import torch
import torch.nn as nn
import torch.nn.functional as F

#Preprocessing data
import pandas as pd
import torch
from fl_backend.data.EPIC import preprocess_dataframe, make_autoencoder_dataloader
from torch.utils.data import TensorDataset, DataLoader

#Training
import torch
import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataInjection import inject_noise
import matplotlib.pyplot as plt

class CNNTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, embed_dim, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(32, in_channels, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = z.transpose(1, 2)      # [batch, seq_len, embed_dim]
        z = self.transformer(z)
        z = z.transpose(1, 2)      # [batch, embed_dim, seq_len]

        x_hat = self.decoder(z)
        x_hat = F.interpolate(
            x_hat,
            size=x.shape[-1],
            mode="linear",
            align_corners=False,
        )
        return x_hat

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    device: torch.device,
    max_norm: float = 1.0,
):
    model.train()
    criterion = nn.MSELoss()

    running_loss = 0.0
    num_batches = 0

    for (x_batch,) in dataloader:
        x_batch = x_batch.to(device)

        x_hat = model(x_batch)
        loss = criterion(x_hat, x_batch)

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Loss became NaN or Inf during training.")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


#Evaluate funktion
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
):
    model.eval()
    criterion = nn.MSELoss()

    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for (x_batch,) in dataloader:
            x_batch = x_batch.to(device)

            x_hat = model(x_batch)
            loss = criterion(x_hat, x_batch)

            running_loss += loss.item()
            num_batches += 1

    return running_loss / max(num_batches, 1)


#Anomoly score funktion
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


#Utility functyion 
def get_model_parameters(model: nn.Module):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model: nn.Module, parameters):
    state_dict = model.state_dict()
    new_state_dict = {}

    for key, value in zip(state_dict.keys(), parameters):
        new_state_dict[key] = torch.tensor(value)

    model.load_state_dict(new_state_dict, strict=True)



#"Lokal klientkode", så baisically main koden  

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
    

def run_test_func():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    clean_path = Path("datasets/EPIC/Scenario_1/EpicLog_Scenario 1_19_Oct_2018_14_44.csv")
    path = Path("datasets/EPIC/Scenario_1/EpicLog_noisy.csv")

    inject_noise(clean_path, path, noise_level=0.7)

    df = pd.read_csv(path)
    df = df.drop(columns=["Timestamp"])

    # Preprocess
    X = preprocess_dataframe(df)

    # Split the data into training and test sets
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42, shuffle=False)

    # Dataloaders
    train_loader = make_autoencoder_dataloader(X_train, batch_size=32, shuffle=False)
    eval_loader = make_autoencoder_dataloader(X_test, batch_size=32, shuffle=False)

    # Model
    model = CNNTransformer(
        in_channels=1,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    epochs = 10

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        eval_loss = evaluate(model, eval_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_loss:.6f}, "
            f"Eval Loss: {eval_loss:.6f}"
        )
    
    detect_anomalies(
        model=model,
        eval_loader=eval_loader,
        df=df,
        train_size=len(X_train),
        device=device,
        threshold_std=3.0)