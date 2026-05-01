from config import CONFIG

import torch
import numpy as np
import os

from training.training_utils.Evaluator import Evaluator

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"  # Silence ROCm warning

def evaluate():
    test_handler = CONFIG.data.build_handler(CONFIG)
    metadata = test_handler.load_metadata()
    X_test, y_test = test_handler.load_test_set(CONFIG.evaluation.test_path)

    testloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test.astype(np.float32)),
            torch.tensor(y_test.astype(np.float32)),
        ),
        batch_size=CONFIG.evaluation.batch_size,
        shuffle=False,
    )

    evaluator = Evaluator(model_config=CONFIG.model, metadata=metadata)

    return evaluator.evaluate_final(
        model_path=CONFIG.evaluation.model_path,
        testloader=testloader,
        threshold=CONFIG.evaluation.threshold,
    )
    