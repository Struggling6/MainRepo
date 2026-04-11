import torch
from copy import deepcopy
from flwr.client import ClientApp, NumPyClient

from config import CONFIG
from data.registry import create_dataset_handler
from models.registry import create_model
from models.utils import get_model_parameters, set_model_parameters
from tasks.registry import create_task
from training.train import train_model
from training.evaluate import evaluate_model


class FlowerClient(NumPyClient):
    def __init__(
        self,
        partition_id: int,
        facility_id: str | None = None,
        data_path: str | None = None,
    ):
        self.config = deepcopy(CONFIG)
        self.partition_id = partition_id
        self.facility_id = facility_id or f"client-{partition_id}"

        if data_path is not None:
            self.config["data"]["file_path"] = data_path
            self.config["data"]["partition_mode"] = "local"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(
            f"[Client Init] facility_id={self.facility_id}, "
            f"partition_id={self.partition_id}, "
            f"data_path={self.config['data']['file_path']}, "
            f"partition_mode={self.config['data'].get('partition_mode', 'shared')}, "
            f"device={self.device}"
        )

        self.dataset_handler = create_dataset_handler(self.config["data"])
        self.data_metadata = self.dataset_handler.get_metadata()

        self.model = create_model(self.config["model"], self.data_metadata)
        self.task = create_task(self.config["task"])

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
            training_config=self.config["training"],
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
    partition_id = int(context.node_config.get("partition-id", 0))
    facility_id = context.node_config.get("facility-id", f"client-{partition_id}")
    data_path = context.node_config.get("data-path")

    return FlowerClient(
        partition_id=partition_id,
        facility_id=facility_id,
        data_path=data_path,
    ).to_client()


app = ClientApp(client_fn=client_fn)