import torch
from flwr.client import ClientApp, NumPyClient

from config import CONFIG
from data.registry import create_dataset_handler
from models.registry import create_model
from models.utils import get_model_parameters, set_model_parameters
from tasks.registry import create_task
from training.train import train_model
from training.evaluate import evaluate_model


class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int):
        self.config = CONFIG
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset_handler = create_dataset_handler(self.config.data)
        self.data_metadata = self.dataset_handler.get_metadata()

        self.model = create_model(self.config.model, self.data_metadata)
        self.task = create_task(self.config.task)

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
            task=self.task,
            training_config=self.config.training,
            device=self.device,
        )

        return get_model_parameters(self.model), results["num_examples"], results

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)

        results = evaluate_model(
            model=self.model,
            testloader=self.testloader,
            task=self.task,
            device=self.device,
        )

        return results["loss"], results["num_examples"], {
            "accuracy": results["accuracy"]
        }


def client_fn(context):
    partition_id = int(context.node_config["partition-id"])
    return FlowerClient(partition_id=partition_id).to_client()


app = ClientApp(client_fn=client_fn)