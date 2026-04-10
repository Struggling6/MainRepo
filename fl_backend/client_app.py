import torch
from flwr.client import ClientApp, NumPyClient

from config import CONFIG
from data.registry import create_dataset_handler
from models.registry import create_model
from models.utils import get_model_parameters, set_model_parameters
from training.train import train_model
from training.evaluate import evaluate_model
from models.utils import get_device


class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int):
        self.config          = CONFIG
        self.partition_id    = partition_id
        self.device          = get_device()
        self.dataset_handler = create_dataset_handler(self.config.data)
        self.data_metadata   = self.dataset_handler.get_metadata()

        # Model created from config — architecture and loss both come from config
        self.model = create_model(self.config.model, self.data_metadata)

        self.trainloader, self.testloader = self.dataset_handler.get_dataloaders(
            partition_id=self.partition_id
        )

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)

        results = train_model(
            model=self.model,
            trainloader=self.trainloader,
            training_config=self.config.training,  # TrainingConfig
            model_config=self.config.model,         # CNNTransformerConfig — contains loss_fn
            device=self.device,
        )

        return get_model_parameters(self.model), results["num_examples"], results

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)

        results = evaluate_model(
            model=self.model,
            testloader=self.testloader,
            device=self.device,
        )

        return results["loss"], results["num_examples"], {"f1": results["f1"]}


def client_fn(context):
    partition_id = int(context.node_config["partition-id"])
    return FlowerClient(partition_id=partition_id).to_client()


app = ClientApp(client_fn=client_fn)