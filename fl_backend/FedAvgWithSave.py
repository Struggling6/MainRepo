from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Optional
from config import CONFIG
from models.registry import create_model
from models.utils import get_device


def _infer_input_dim(ndarrays: list, fallback: int) -> int:
    """Return the model's input feature count from the aggregated parameter arrays.

    Conv1d weights are the only 3-D tensors in the CNN-Transformer architecture
    (shape: out_channels, in_channels, kernel_size); transformer attention weights
    are all 2-D.  We read `in_channels` from the *first* 3-D array found, which
    corresponds to the first convolutional layer.

    Args:
        ndarrays: Flat list of numpy arrays from ``parameters_to_ndarrays``.
        fallback: Value to return when no 3-D array is found (e.g., a non-CNN model).
    """
    return next((int(a.shape[1]) for a in ndarrays if a.ndim == 3), fallback)


class FedAvgWithSave(FedAvg):
    """
    FedAvg strategy that saves the aggregated model to disk
    after the final federation round completes.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict]:

        # Run standard FedAvg aggregation first
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Save only after the final round
        if aggregated_parameters is not None and server_round == CONFIG.federation.num_rounds:
            print(f"Final round {server_round} complete, saving aggregated model...")

            device   = get_device()
            ndarrays = parameters_to_ndarrays(aggregated_parameters)

            # Infer input_dim from the first Conv1d weight in the aggregated parameters.
            # Conv1d weights have shape (out_ch, in_ch, kernel_size) — the only 3-D arrays in
            # this architecture (transformer weights are all 2-D).  This ensures the server
            # reconstructs a model whose first layer matches whatever feature count the clients
            # actually trained with, preventing load_state_dict shape mismatches.
            input_dim = _infer_input_dim(ndarrays, fallback=CONFIG.model.in_channels)
            data_metadata = {"input_dim": input_dim, "num_classes": CONFIG.model.num_classes}
            model         = create_model(CONFIG.model, data_metadata).to(device)

            # Convert Flower parameters back to numpy arrays, then load into model
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
        model:      nn.Module,
        path:       Path,
        config:     object = None,
        threshold:  float  = 0.5,
        metrics:    dict   = None,
    ):
        """
        Save model weights, architecture config, best threshold,
        and training metrics to a single checkpoint file.
        """
    
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)  # create checkpoints/ dir if it doesn't exist

            checkpoint = {
                "model_state_dict": model.state_dict(), # PyTorch convention for saving/loading weights
                "model_config":     config,
                "threshold":        threshold,
                "metrics":          metrics or {},
            }
            torch.save(checkpoint, path)
            print(f"Model saved to {path}")

        else:
            raise ValueError("Path must be a pathlib.Path object")
