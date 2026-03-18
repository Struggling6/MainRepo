import torch

def evaluate_model(model, testloader, device):
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
    correct = 0

    with torch.no_grad():
        for x_batch, y_batch in testloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

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
    