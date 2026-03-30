import torch
import torch.nn as nn

def evaluate(
    model,
    dataloader,
    device,
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
