from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from config import CONFIG
from data.registry import create_dataset_handler


class FedAvgWithSave(FedAvg):
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

            metadata = create_dataset_handler(CONFIG.data).get_metadata()
            input_dim = metadata["input_dim"]
            model = CONFIG.model.build(input_dim=input_dim)

            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            state_dict_keys = list(model.state_dict().keys())

            if len(state_dict_keys) != len(ndarrays):
                raise RuntimeError(
                    f"Mismatch when saving model: model has {len(state_dict_keys)} "
                    f"state tensors, aggregated result has {len(ndarrays)}"
                )

            state_dict = {
                k: torch.tensor(v, dtype=model.state_dict()[k].dtype)
                for k, v in zip(state_dict_keys, ndarrays)
            }
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