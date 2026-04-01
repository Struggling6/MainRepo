import torch

def train_model(model, trainloader, task, training_config: dict ,  device):
    model.to(device)
    model.train()

    epochs = training_config.get("local_epochs", 10)
    lr = training_config.get("learning_rate", 0.001)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    

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