from abc import ABC, abstractmethod


class BaseTask(ABC):
    
    @abstractmethod
    def compute_loss(self, model, batch, device):
        pass

    @abstractmethod
    def compute_metrics(self, outputs, targets):
        pass