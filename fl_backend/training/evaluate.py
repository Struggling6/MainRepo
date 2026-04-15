import torch
import numpy as np

from config import ExperimentConfig
from training.training_utils.Evaluator import Evaluator
from models.utils import get_device
from data.registry import create_dataset_handler

def evaluate(config: ExperimentConfig):
    device       = get_device()
    # Use registry to create the correct handler based on config.data.name
    # This works for LeadCSVHandler, PowerGridCSVHandler, or any future handler
    test_handler = create_dataset_handler(config.evaluation)
    data_metadata = test_handler.get_metadata()


    _, _, X_test, y_test = test_handler.run_split()

    testloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test.astype(np.float32)),
            torch.tensor(y_test.astype(np.float32)),
        ),
        batch_size=config.evaluation.batch_size,
        shuffle=False,
    )

    evaluator = Evaluator(model_config=config.model)

    return evaluator.evaluate_final(
        model_path=config.evaluation.model_path,
        testloader=testloader,
        threshold=config.evaluation.threshold,
    )
