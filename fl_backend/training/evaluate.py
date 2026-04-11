import torch
import numpy as np
import pandas as pd

from pathlib import Path
from config import ExperimentConfig
from data.lead_csv import LeadCSVHandler
from training.training_utils.Evaluator import Evaluator
from models.registry import create_model
from models.utils import load_model, get_device

def evaluate(config: ExperimentConfig):
    device       = get_device()
    test_handler = LeadCSVHandler(config.data)
    evaluator    = Evaluator(model_config=config.model)
    _, _, X_test, y_test, _, _ = test_handler.run_split()
    
    testloader = evaluator._build_dataloader(X_test, y_test, config.evaluation.batch_size)

    model            = create_model(config.model, test_handler.get_metadata())
    model, thresh, _ = load_model(model, path=config.evaluation.model_path, device=device)

    return evaluator.evaluate_final(model, testloader, threshold=thresh)