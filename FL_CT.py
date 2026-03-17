import torch
import torch.nn as nn
import torch.nn.functional as F

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
    


#Preprocessing data
import pandas as pd
import torch

def preprocess_dataframe(df: pd.DataFrame) -> torch.Tensor:
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    df_numeric = df_numeric.astype(float)
    df_numeric = df_numeric.fillna(0.0)

    mean = df_numeric.mean()
    std = df_numeric.std().replace(0, 1)

    df_numeric = (df_numeric - mean) / std
    df_numeric = df_numeric.fillna(0.0)

    X = torch.tensor(df_numeric.values, dtype=torch.float32)
    X = X.unsqueeze(1)   # [num_samples, 1, num_features]
    return X



#Dataloader funktion
from torch.utils.data import TensorDataset, DataLoader

def make_autoencoder_dataloader(
    X: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


#Train funktion
import torch
import torch.nn as nn

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


#Lokal klientkode 
import torch
import torch.optim as optim
import pandas as pd
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
path = Path("datasets/EPIC/Scenario_1/EpicLog_Scenario 1_19_Oct_2018_14_44.csv")
df = pd.read_csv(path)
df = df.drop(columns=["Timestamp"])

# Preprocess
X = preprocess_dataframe(df)

# Dataloaders
train_loader = make_autoencoder_dataloader(X, batch_size=32, shuffle=True)
eval_loader = make_autoencoder_dataloader(X, batch_size=32, shuffle=False)

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

scores = get_anomaly_scores(model, eval_loader, device)
print(scores[:10])




#Example på output
#Epoch 1/20, Train Loss: 0.198603, Eval Loss: 0.166171
#Epoch 2/20, Train Loss: 0.155465, Eval Loss: 0.144920
#Epoch 3/20, Train Loss: 0.141268, Eval Loss: 0.136764
#Epoch 4/20, Train Loss: 0.134611, Eval Loss: 0.131075
#Epoch 5/20, Train Loss: 0.130228, Eval Loss: 0.128099
#Epoch 6/20, Train Loss: 0.128087, Eval Loss: 0.126918
#Epoch 7/20, Train Loss: 0.127096, Eval Loss: 0.126269
#Epoch 8/20, Train Loss: 0.126491, Eval Loss: 0.125835
#Epoch 9/20, Train Loss: 0.126150, Eval Loss: 0.125592
#Epoch 10/20, Train Loss: 0.125878, Eval Loss: 0.125500
#Epoch 11/20, Train Loss: 0.125664, Eval Loss: 0.125209
#Epoch 12/20, Train Loss: 0.125519, Eval Loss: 0.125073
#Epoch 13/20, Train Loss: 0.125385, Eval Loss: 0.125093
#Epoch 14/20, Train Loss: 0.125333, Eval Loss: 0.124945
#Epoch 15/20, Train Loss: 0.125196, Eval Loss: 0.124854
#Epoch 16/20, Train Loss: 0.125114, Eval Loss: 0.124818
#Epoch 17/20, Train Loss: 0.125053, Eval Loss: 0.124786
#Epoch 18/20, Train Loss: 0.125006, Eval Loss: 0.124740
#Epoch 19/20, Train Loss: 0.124981, Eval Loss: 0.124724
#Epoch 20/20, Train Loss: 0.124936, Eval Loss: 0.124684
#tensor([0.5257, 0.5269, 0.5313, 0.5480, 0.5480, 0.5295, 0.5298, 0.5298, 0.5599,0.5270])

#Train loss : gennemsnittet af fejl mellem input og output fra modellen under træning. En lavere værdi betyder at modellen bliver bedre til at genskabe input data. 
#Eval loss : Dette er rekonstruktionsfejlen på evaluerings dats (Så data som modellen ikke træner på), Lav eval loss vetyder at modellen generaliserer godt på unseen data. 
#Så siden vores loss værdier stabiliserer ved 0.124... betyder det at modellen har lært at genkende dataen godt uden at over tilpasse. 
#Anomoly score : Representerer "rekustruktionsfejlen" for hver datapunkt i eval dataet. En høj score betyder modellen havde det svært ved at genkende datapunktet, hvilket betyder det er en anomaly. 

# Så pointen er at modellen skal træne på datapunkter som er normale, så den kan genkende hvilke mønstrer er normale
# Ideen er så at hvis den kan konsturerer et resultat som minder som de patterns vi har fundet fra train data, så burde anomoly score være lav, fordi de er ens
# men hvis den ikke kan konstruerer et resultat som minder om vores patterns, så betyder det at det er en anomaly.


#For at kører : python FL_CT.py


import matplotlib.pyplot as plt

plt.plot(scores.numpy())
plt.title("Anomaly Scores")
plt.xlabel("Sample Index")
plt.ylabel("Score")
plt.show()

threshold = scores.mean() + 3 * scores.std()  # Eksempel: 3 standardafvigelser over gennemsnittet
anomalies = scores > threshold
print(f"Number of anomalies: {anomalies.sum()}")