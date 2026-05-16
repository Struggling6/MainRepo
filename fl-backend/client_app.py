import torch 
from torch import nn
from contextlib import contextmanager
from pathlib import Path
from flwr.client import ClientApp, NumPyClient
from training.training_utils.Validator import Validator
from flwr.common.logger import log
from logging import INFO
from local_experiment import CONFIG
from models.utils import get_device, get_model_parameters, set_model_parameters
from training.train import train_model
from copy import deepcopy

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@contextmanager
def _gpu_file_lock(enabled: bool, lock_path: Path, label: str):
    if not enabled:
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        lock_file = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115
    except PermissionError:
        lock_path.unlink(missing_ok=True)
        lock_file = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115

    with lock_file:
        try:
            import fcntl

            print(f"[GPU LOCK] waiting for {label}: {lock_path}", flush=True)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            print(f"[GPU LOCK] acquired for {label}", flush=True)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                print(f"[GPU LOCK] released for {label}", flush=True)
        except ModuleNotFoundError:
            print("[GPU LOCK] fcntl unavailable; continuing without file lock", flush=True)
            yield

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

        self.trainloader = None
        self.testloader = None

        log(INFO, "[%s] initialised | partition=%s | data=%s | mode=%s | device=%s",
            self.facility_id,
            self.partition_id,
            self.config.data.file_path,
            self.config.federation.partition_mode,
            self.device,
        )

        self.validator = Validator(model_config=self.config.model, metadata=self.metadata)

    def _use_gpu_lock(self, for_evaluate: bool = False) -> bool:
        if self.device.type != "cuda":
            return False

        fed_config = self.config.federation
        if for_evaluate and not getattr(fed_config, "serialize_gpu_evaluate", True):
            return False

        return getattr(fed_config, "serialize_gpu", False)

    def _gpu_lock_path(self) -> Path:
        return Path(getattr(self.config.federation, "gpu_lock_path", "datasets/.gpu.lock"))

    def _ensure_dataloaders(self):
        if self.trainloader is not None and self.testloader is not None:
            return

        log(INFO, "[%s] loading dataloaders for partition=%s", self.facility_id, self.partition_id)
        self.trainloader, self.testloader = self.dataset_handler.get_dataloaders(
            partition_id=self.partition_id
        )
        self.metadata = self.dataset_handler.get_metadata()
        self.validator.metadata = self.metadata

    def get_parameters(self, config):
        print("=== CLIENT get_parameters ENTERED ===", flush=True)
        params = get_model_parameters(self.model)
        print(f"[DEBUG] returning {len(params)} parameter arrays", flush=True)
        return params

    def fit(self, parameters, config):
        round_num = config.get("round", "?")
        log(INFO, "[%s] FIT start | round=%s", self.facility_id, round_num)
        set_model_parameters(self.model, parameters)
        self._ensure_dataloaders()

        # Flower FedProx sends this value to tell the client how strong the penalty is.
        proximal_mu = config.get("proximal_mu", self.config.federation.proximal_mu)

        lock_label = f"{self.facility_id} fit round={round_num}"
        with _gpu_file_lock(self._use_gpu_lock(), self._gpu_lock_path(), lock_label):
            results = train_model(
                model=self.model,
                trainloader=self.trainloader,
                valloader=self.testloader,
                training_config=self.config.training,
                model_config=self.config.model,
                device=self.device,
                proximal_mu=proximal_mu,
                global_pos_weight=getattr(
                    self.dataset_handler, "global_pos_weight", None
                ),
            )

        results["input_dim"] = self.metadata["input_dim"]
        results["num_examples_trained"] = results["num_examples"]

        aggregation_weight = int(
            getattr(
                self.dataset_handler,
                "aggregation_weight",
                results["num_examples"],
            )
        )
        results["aggregation_weight"] = aggregation_weight

        log(INFO, "[FIT] done facility_id=%s results=%s", self.facility_id, results)

        return get_model_parameters(self.model), aggregation_weight, results

    def evaluate(self, parameters, config):
        round_num = config.get("round", "?")
        log(INFO, "[%s] VAL start | round=%s", self.facility_id, round_num)

        set_model_parameters(self.model, parameters)
        self._ensure_dataloaders()

        lock_label = f"{self.facility_id} validate round={round_num}"
        with _gpu_file_lock(
            self._use_gpu_lock(for_evaluate=True),
            self._gpu_lock_path(),
            lock_label,
        ):
            self.model = self.model.to(self.device)
            results = self.validator.validate(self.model, self.testloader)

        log(INFO, "[%s] VAL done  | round=%s | loss=%.4f | f1=%.4f | pr_auc=%.4f | roc_auc=%.4f | thresh=%.2f| acc=%.4f | prec=%.4f | rec=%.4f",
            self.facility_id,
            round_num,
            results["loss"],
            results["val_f1"],
            results["pr_auc"],
            results["roc_auc"],
            results["best_threshold"],
            results["accuracy"],
            results["precision"],
            results["recall"],
        )

        return results["loss"], results["num_examples"], {
            "f1":        results["val_f1"],
            "pr_auc":    results["pr_auc"],
            "threshold": results["best_threshold"],
            "roc_auc":   results["roc_auc"],
            "accuracy":  results["accuracy"],
            "precision": results["precision"],
            "recall":    results["recall"],
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
