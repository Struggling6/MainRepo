import torch
import torch.nn as nn

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
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


def train_model(model, trainloader, task, training_config: dict ,  device):
    model.to(device)
    model.train()

    epochs = training_config.get("local_epochs", 10)
    lr = training_config.get("learning_rate", 0.001)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = train_one_epoch(
            model=model,
            dataloader=trainloader,
            optimizer=optimizer,
            device=device,
            max_norm=1.0
        )
        total_loss += epoch_loss
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.6f}")
    
    avg_loss = total_loss / max(epochs, 1)
    num_examples = len(trainloader.dataset) if hasattr(trainloader, 'dataset') else len(trainloader)
    
    return {
        "loss": avg_loss,
        "accuracy": 0.0,
        "num_examples": num_examples,
    }


'''
    for _ in range (epochs):
        for batch in trainloader:
            optimizer.zero_grad()

            loss, outputs, targets = task.compute_loss(model, batch, device)
            loss.backward()
            optimizer.step()

            metrics = task.compute_metrics(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += metrics["correct"]
            total_examples += batch_size


    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "num_examples": total_examples,
    }
'''


