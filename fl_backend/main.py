from config import CONFIG
from data.registry import create_dataset_handler
from training.train import train_model
from training.evaluate import evaluate
from models.utils import get_device
from training.training_utils.Evaluator import Evaluator
from plotting.plotting_config import plot_diagrams
from plotting.plotting_config import plot_diagrams

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--simulate", action="store_true", help="Run Flower simulation instead of single-client test")
args = parser.parse_args()


def simulate():
    """Run full federated simulation with all clients using Flower."""
    from flwr.simulation import run_simulation
    from client_app import app as client_app
    from server_app import app as server_app

    num_clients = CONFIG.data.num_clients

    print(f"Starting simulation with {num_clients} clients, {CONFIG.federation.num_rounds} rounds")

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config={
            "client_resources": {
                "num_cpus": 4,
                "num_gpus": 1.0,  # share the GPU across clients
            }
        },
    )


def main():
    config = CONFIG
    device = get_device()

    # ── Dataset ──────────────────────────────────────────────────────── #
    dataset_handler             = create_dataset_handler(config)
    metadata                    = dataset_handler.get_metadata()
    config.evaluation.input_dim = metadata["input_dim"]
    
    print(f"Dataset metadata: {metadata}")

    # ── Model ─────────────────────────────────────────────────────────── #
    model = config.model.build(input_dim=metadata["input_dim"])
    print(f"Model created: {config.model.name}")

    # ── DataLoaders ───────────────────────────────────────────────────── #
    trainloader, valloader = dataset_handler.get_dataloaders(partition_id=0)
    print("Loaded train and val dataloaders for client 0")

    # ── Training ──────────────────────────────────────────────────────── #
    train_results = train_model(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
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
    if args.simulate:
        simulate()
    else:
        main()