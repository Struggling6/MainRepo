import torch
import torch.nn.functional as F
from .base import BaseTask


class ClassificationTask(BaseTask):
    def compute_loss(self, model, batch, device):
        x, y = batch

        x = x.to(device)
        y = y.to(device)

        outputs = model(x) 
        loss = F.cross_entropy(outputs, y)
        return loss,outputs,y
    
    def compute_metrics(self, outputs, targets):
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)

        return {
            "correct": correct,
            "total": total,
        }