#This file is just to test that everything works end to end before implementing flower.

from config import CONFIG
from data.registry import create_dataset_handler
from training.train import train_model
from training.evaluate import evaluate
from models.utils import get_device
from training.training_utils.Evaluator import Evaluator
from plotting.plotting_config import plot_diagrams
from plotting.plotting_config import plot_diagrams

def main():
    config = CONFIG
    device = get_device()

    # ── Dataset ──────────────────────────────────────────────────────── #
    dataset_handler = create_dataset_handler(config.data)
    metadata        = dataset_handler.get_metadata()
    CONFIG.evaluation.input_dim = metadata["input_dim"]
    print(f"Dataset metadata: {metadata}")

    # ── Model ─────────────────────────────────────────────────────────── #
    model = CONFIG.model.build(input_dim=metadata["input_dim"])
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
    )
    eval_results = evaluator.evaluate_round(model, valloader)
    print(f"Evaluation results: {eval_results}")

    #plot_diagrams(eval_results["targets"], eval_results["preds"], train_results["train_losses"], train_results["train_accuracies"])

if __name__ == "__main__":
    main()