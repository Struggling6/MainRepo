import torch

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

def run_test_func():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Model
def train(model, optimizer, device):
    model = CNNTransformer(model_config=CONFIG["model"], data_metadata={"num_features": 45})

    epochs = training_config.get("local_epochs", 10)
    lr = training_config.get("learning_rate", 0.001)

    optimizer = optim.Adam(model.parameters(),lr , weight_decay=1e-5)

    for _ in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        eval_loss = evaluate(model, eval_loader, device)

    detect_anomalies(
        model=model,
        eval_loader=eval_loader,
        df=df,
        train_size=len(X_train),
        device=device,
        threshold_std=3.0)