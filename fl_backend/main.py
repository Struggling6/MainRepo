#This file is just to test that everything works end to end before implementing flower.

import torch
from config import CONFIG
from data.registry import create_dataset_handler
from models.registry import create_model
from tasks.registry import create_task
from training.train import train_model
from training.evaluate import evaluate_model
from models.utils import get_device
from training.training_utils.Evaluator import Evaluator

def main():
    config = CONFIG
    device = get_device()

    # ── Dataset ──────────────────────────────────────────────────────── #
    dataset_handler = create_dataset_handler(config.data)
    metadata   = dataset_handler.get_metadata()
    CONFIG.evaluation.input_dim = metadata["input_dim"]
    print(f"Dataset metadata: {metadata}")

    # ── Model ─────────────────────────────────────────────────────────── #
    model = create_model(config.model, metadata)
    print(f"Model created: {config.model.name}")

    # ── DataLoaders ───────────────────────────────────────────────────── #
    trainloader, valloader = dataset_handler.get_dataloaders(partition_id=0)
    print("Loaded train and val dataloaders for client 0")

    # ── Training ──────────────────────────────────────────────────────── #
    train_results = train_model(
        model=model,
        trainloader=trainloader,
        training_config=config.training,
        model_config=config.model,         
        device=device,
    )
    print(f"Training results: {train_results}")

    # ── Evaluation ───────────────────────────────────────────────────── #
    evaluator = Evaluator(
        model_config=config.model,
        input_dim=metadata["input_dim"],
    )
    eval_results = evaluator.evaluate_round(model, valloader)
    print(f"Evaluation results: {eval_results}")

    plot_diagrams(eval_results["targets"], eval_results["preds"], train_results["train_losses"], train_results["train_accuracies"])

if __name__ == "__main__":
    main()