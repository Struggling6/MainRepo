from copy import deepcopy
from pathlib import Path

import torch
from flwr.client import ClientApp, NumPyClient

from config import CONFIG
from data.registry import create_dataset_handler
from models.utils import get_device, get_model_parameters, set_model_parameters
from training.train import train_model
from training.training_utils.Evaluator import Evaluator


class FlowerClient(NumPyClient):
    def __init__(
        self,
        partition_id: int,
        facility_id: str | None = None,
        data_path: str | None = None,
    ):
        # Make a per-client copy so one client does not mutate global CONFIG
        self.config = deepcopy(CONFIG)

        self.partition_id = partition_id
        self.facility_id = facility_id or f"client-{partition_id}"
        self.device = get_device()

        # If a specific dataset file is passed from SuperNode, override config
        if data_path is not None:
            self.config.data.file_path = Path(data_path)
            self.config.data.partition_mode = "local"

        # Build dataset/model from config
        self.dataset_handler = create_dataset_handler(self.config.data)
        self.metadata = self.dataset_handler.get_metadata()
        self.model = CONFIG.model.build(input_dim=self.metadata["input_dim"])

        self.trainloader, self.testloader = self.dataset_handler.get_dataloaders(
            partition_id=self.partition_id
        )

        self.evaluator = Evaluator(model_config=self.config.model)

        print(
            f"[Client Init] facility_id={self.facility_id}, "
            f"partition_id={self.partition_id}, "
            f"data_path={self.config.data.file_path}, "
            f"partition_mode={self.config.data.partition_mode}, "
            f"device={self.device}"
        )

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)

        results = train_model(
            model=self.model,
            trainloader=self.trainloader,
            training_config=self.config.training,
            model_config=self.config.model,
            device=self.device,
        )

        return get_model_parameters(self.model), results["num_examples"], results

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)

        results = self.evaluator.evaluate_round(self.model, self.testloader)

        return results["loss"], results["num_examples"], {
            "f1": results["val_f1"],
            "pr_auc": results["pr_auc"],
            "threshold": results["best_threshold"],
        }


def client_fn(context):
    partition_id = int(context.node_config["partition-id"])
    facility_id = context.node_config.get("facility-id", f"client-{partition_id}")
    data_path = context.node_config.get("data-path")

    return FlowerClient(
        partition_id=partition_id,
        facility_id=facility_id,
        data_path=data_path,
    ).to_client()


app = ClientApp(client_fn=client_fn)