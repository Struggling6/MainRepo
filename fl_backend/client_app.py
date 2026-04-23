from pathlib import Path
from torch import nn
from flwr.client import ClientApp, NumPyClient
from training.training_utils.Evaluator import Evaluator
from flwr.common.logger import log
from logging import INFO
from config import CONFIG
from models.utils import get_device, get_model_parameters, set_model_parameters
from training.train import train_model
from copy import deepcopy
from config import CONFIG


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
        self.config       = deepcopy(CONFIG)
        self.partition_id = partition_id
        self.facility_id  = facility_id or f"client-{partition_id}"
        self.device       = get_device()

        if data_path is not None:
            self.config.data.file_path = Path(data_path)
            self.config.federation.partition_mode = "local"

        print("[Client Init] creating dataset handler", flush=True)
        self.dataset_handler = self.config.data.build_handler(self.config)

        print("[Client Init] getting metadata", flush=True)
        self.metadata = self.dataset_handler.get_metadata()
        print(f"[Client Init] metadata={self.metadata}", flush=True)

        print("[Client Init] creating model", flush=True)
        # In client_app.py __init__
        try:
            self.model = self.config.model.build(input_dim=self.metadata["input_dim"])
        except AttributeError as e:
            log(INFO, "[%s] ERROR: config.model.build() failed — %s", self.facility_id, e)
            log(INFO, "[%s] Make sure your model config has a build() method defined at class level (not inside another method)", self.facility_id)
            raise
        except Exception as e:
            log(INFO, "[%s] ERROR building model: %s", self.facility_id, e)
            raise

        self.trainloader, self.testloader = self.dataset_handler.get_dataloaders(
            partition_id=self.partition_id
        )

        log(INFO, "[%s] initialised | partition=%s | data=%s | mode=%s | device=%s",
            self.facility_id,
            self.partition_id,
            self.config.data.file_path,
            self.config.federation.partition_mode,
            self.device,
        )

        self.evaluator = Evaluator(model_config=self.config.model)

    def get_parameters(self, config):
        print("=== CLIENT get_parameters ENTERED ===", flush=True)
        params = get_model_parameters(self.model)
        print(f"[DEBUG] returning {len(params)} parameter arrays", flush=True)
        return params

    def fit(self, parameters, config):
        round_num = config.get("round", "?")
        log(INFO, "[%s] FIT start | round=%s", self.facility_id, round_num)
        set_model_parameters(self.model, parameters)

        proximal_mu = config.get("proximal-mu", self.config.federation.proximal_mu)

        results = train_model(
            model=self.model,
            trainloader=self.trainloader,
            valloader=self.testloader,
            training_config=self.config.training,
            model_config=self.config.model,
            device=self.device,
        )

        print(f"[FIT] done facility_id={self.facility_id} results={results}", flush=True)
        return get_model_parameters(self.model), results["num_examples"], results

    def evaluate(self, parameters, config):
        round_num = config.get("round", "?")
        log(INFO, "[%s] EVAL start | round=%s", self.facility_id, round_num)

        set_model_parameters(self.model, parameters)
        self.model = self.model.to(self.device)

        results = self.evaluator.evaluate_round(self.model, self.testloader)

        log(INFO, "[%s] EVAL done  | round=%s | loss=%.4f | f1=%.4f | pr_auc=%.4f | thresh=%.2f",
            self.facility_id,
            round_num,
            results["loss"],
            results["val_f1"],
            results["pr_auc"],
            results["best_threshold"],
        )

        return results["loss"], results["num_examples"], {
            "f1":        results["val_f1"],
            "pr_auc":    results["pr_auc"],
            "threshold": results["best_threshold"],
        }


def client_fn(context):
    partition_id = int(context.node_config["partition-id"])
    facility_id  = context.node_config.get("facility-id", f"client-{partition_id}")
    data_path    = context.node_config.get("data-path")

    try:
        return FlowerClient(
            partition_id=partition_id,
            facility_id=facility_id,
            data_path=data_path,
        ).to_client()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Client {partition_id} ({facility_id}): dataset not found — {exc}. "
            "Ensure the datasets directory is mounted and the file exists."
        ) from exc


app = ClientApp(client_fn=client_fn)