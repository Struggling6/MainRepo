from flwr.server.strategy import FedProx
from flwr.common import parameters_to_ndarrays, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Optional
from config import CONFIG
from models.utils import get_device
from data.registry import create_dataset_handler


class FedProxWithSave(FedProx):
    """
    FedAvg strategy that saves the aggregated model to disk
    after the final federation round completes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict]:
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if (
            aggregated_parameters is not None
            and server_round == CONFIG.federation.num_rounds
        ):
            print(f"Final round {server_round} complete, saving aggregated model...")

            metadata = create_dataset_handler(CONFIG).get_metadata()
            model = CONFIG.model.build(input_dim=metadata["input_dim"])

            # Convert Flower parameters back to numpy arrays, then load into model
            ndarrays    = parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict  = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)

            self._save_model(
                model=model,
                path=CONFIG.evaluation.model_path,
                config=CONFIG.model,
            )

        return aggregated_parameters, metrics

    @staticmethod
    def _save_model(
        model: nn.Module,
        path: Path,
        config: object = None,
        threshold: float = 0.5,
        metrics: dict = None,
    ):
        """
        Save model weights, architecture config, best threshold,
        and training metrics to a single checkpoint file.
        """

        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": config,
                "threshold": threshold,
                "metrics": metrics or {},
            }
            torch.save(checkpoint, path)
            print(f"Model saved to {path}")
        else:
            raise ValueError("Path must be a pathlib.Path object")