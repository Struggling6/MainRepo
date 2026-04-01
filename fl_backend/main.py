#This file is just to test that everything works end to end before implementing flower.

import torch

from config import CONFIG
from data.registry import create_dataset_handler
from models.registry import create_model
from tasks.registry import create_task
from training.train import train_model
from training.evaluate import evaluate_model
from models.utils import get_device

def main():
    config = CONFIG

    device = get_device()
    print(f"Using device: {device}")

    # Create dataset handler
    dataset_handler = create_dataset_handler(config["data"])

    # Read metadata from dataset
    data_metadata = dataset_handler.get_metadata()
    print(f"Dataset metadata: {data_metadata}")

    # Create model
    model = create_model(config["model"], data_metadata)
    print(f"Model created:{config['model']['name']}")

    # Create task
    task = create_task(config["task"])
    print(f"Task created: {config['task']['name']}")

    # get train/test dataloaders for client / partition 0
    trainloader, testloader = dataset_handler.get_dataloaders(partition_id=0)
    print("Loaded train and test dataloaders for client 0")

    train_results = train_model(
        model=model,
        trainloader=trainloader,
        task=task,
        device=device,
        training_config=config["training"],
    )
    print(f"Training results: {train_results}")

    eval_results = evaluate_model(
        model=model,
        testloader=testloader,
        task=task,
        device=device,
    )
    print(f"Evaluation results: {eval_results}")


    # Detect anomalies
    '''
    print("\n--- Anomaly Detection ---")
    anomalies, scores, threshold = task.detect_anomalies(
        model=model,
        eval_loader=testloader,
        df=dataset_handler.df,
        train_size=len(trainloader.dataset),
        device=device,
        threshold_std=1.0
    )
    '''

if __name__ == "__main__":
    main()
