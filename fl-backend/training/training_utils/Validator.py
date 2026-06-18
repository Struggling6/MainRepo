import torch

from training.training_utils.TrainEvalBase import TrainEvalBase
from training.training_utils.utils import compute_pos_weight
from local_experiment import resolve_loss_fn


class Validator(TrainEvalBase):
    """
    Runs validation on the aggregated global model after each federation
    round. Reports loss, F1, PR-AUC, ROC-AUC and the best threshold found
    on the client's local validation split.
    """

    def __init__(self, model_config, metadata):
        super().__init__(num_classes=model_config.num_classes)
        self.model_config = model_config
        self.metadata = metadata

    def validate(self, model, valloader):
        loss_fn = self._build_loss(valloader)
        loss, f1, thresh, pr_auc, roc_auc, accuracy, precision, recall = self._val_epoch(model, valloader, loss_fn)

        return {
            "loss":         loss,
            "f1":           f1,
            "threshold":    thresh,
            "pr_auc":       pr_auc,
            "roc_auc":      roc_auc,
            "accuracy":     accuracy,
            "precision":    precision,
            "recall":       recall,
            "num_examples": int(len(valloader.dataset)),
        }

    def _build_loss(self, valloader):
        y = _extract_labels(valloader)
        pw = compute_pos_weight(y, cap=self.model_config.pos_weight_cap)
        loss_cls = resolve_loss_fn(self.model_config.loss_fn)
        return loss_cls(
            pos_weight=torch.tensor([pw], device=self.device)
        )


def _extract_labels(loader):
    import numpy as np
    return np.concatenate([y.numpy() for _, y in loader])
