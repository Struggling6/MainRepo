import torch

def train_model(model, trainloader, epochs, lr,  device):
    model.to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    total_examples = 0
    correct = 0

    for _ in range (epochs):
        for x_batch, y_batch in trainloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()

    avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
    accuracy = correct / total_examples if total_examples > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "num_examples": total_examples,
    }



