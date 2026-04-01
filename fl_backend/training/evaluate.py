import torch

def evaluate_model(model, testloader, task, device):
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    

    with torch.no_grad():
        for batch in testloader:
     
            loss, outputs, targets = task.compute_loss(model, batch, device)    
          
            metrics = task.compute_metrics(outputs, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += metrics["correct"]
            total_examples += batch_size

        avg_loss = total_loss / max(total_examples, 1)


        results =  {
            "loss": avg_loss,
            "accuracy": total_correct / total_examples,   
            "num_examples": total_examples,
        }

        return results
    