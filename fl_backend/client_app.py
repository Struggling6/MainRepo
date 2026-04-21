from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from flwr.client import ClientApp, NumPyClient

from config import CONFIG
from data.registry import create_dataset_handler
from models.utils import get_device, get_model_parameters, set_model_parameters
from training.train import train_model
from training.training_utils.Evaluator import Evaluator


def _resolve_real_model(model):
    """Return the actual nn.Module containing weights."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"build(...) returned {type(model).__name__}, expected nn.Module")

    # Try common wrapper attributes first
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "net", None),
        getattr(model, "network", None),
        getattr(model, "module", None),
    ]

    for cand in candidates:
        if isinstance(cand, nn.Module):
            num_tensors = len(cand.state_dict())
            num_params = sum(p.numel() for p in cand.parameters())
            if num_tensors > 0 or num_params > 0:
                return cand

    raise RuntimeError(
        f"Model {type(model).__name__} has no registered tensors/parameters. "
        "Your build(...) likely returns a wrapper or stores layers incorrectly."
    )


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
        self.device = get_device()

        if data_path is not None:
            self.config.data.file_path = Path(data_path)
            self.config.data.partition_mode = "local"

        print("[Client Init] creating dataset handler", flush=True)
        self.dataset_handler = create_dataset_handler(self.config.data)

        print("[Client Init] getting metadata", flush=True)
        self.metadata = self.dataset_handler.get_metadata()
        print(f"[Client Init] metadata={self.metadata}", flush=True)

        print("[Client Init] creating model", flush=True)
        self.model = CONFIG.model.build(input_dim=self.metadata["input_dim"])

        print("[Client Init] creating dataloaders", flush=True)
        self.trainloader, self.testloader = self.dataset_handler.get_dataloaders(
            partition_id=self.partition_id
        )

        print(
            f"[Client Init] facility_id={self.facility_id}, "
            f"partition_id={self.partition_id}, "
            f"data_path={self.config.data.file_path}, "
            f"partition_mode={self.config.data.partition_mode}, "
            f"device={self.device}",
            flush=True,
        )

        self.evaluator = Evaluator(model_config=self.config.model)

    def get_parameters(self, config):
        print("=== CLIENT get_parameters ENTERED ===", flush=True)
        params = get_model_parameters(self.model)
        print(f"[DEBUG] returning {len(params)} parameter arrays", flush=True)
        return params

    def fit(self, parameters, config):
        print(f"[FIT] start facility_id={self.facility_id}", flush=True)
        set_model_parameters(self.model, parameters)

        proximal_mu = config.get("proximal-mu", self.config.federation.proximal_mu)

        results = train_model(
            model=self.model,
            trainloader=self.trainloader,
            training_config=self.config.training,
            model_config=self.config.model,
            device=self.device,
            proximal_mu=proximal_mu,
        )

        print(f"[FIT] done facility_id={self.facility_id} results={results}", flush=True)
        return get_model_parameters(self.model), results["num_examples"], results

    def evaluate(self, parameters, config):
        print(f"[EVAL] start facility_id={self.facility_id}", flush=True)
        set_model_parameters(self.model, parameters)
        self.model = self.model.to(self.device)

        results = self.evaluator.evaluate_round(self.model, self.testloader)

        print(f"[EVAL] done facility_id={self.facility_id} results={results}", flush=True)

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