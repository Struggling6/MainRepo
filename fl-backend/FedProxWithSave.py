import torch
import torch.nn as nn
import os
from flwr.common.logger import log
from logging import INFO, ERROR
from flwr.server.strategy import FedProx
from flwr.common import parameters_to_ndarrays, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from pathlib import Path
from typing import Union, Optional
from config import CONFIG

class FedProxWithSave(FedProx):
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
        
        print(f"[SAVE] aggregate_fit called, round={server_round}, num_rounds={CONFIG.federation.num_rounds}", flush=True)

        # Run standard FedAvg aggregation first
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None and server_round == CONFIG.federation.num_rounds:
            print(f"[SAVE] starting model save...", flush=True)
            try:
                # Get input_dim from first client result — already computed during data loading
                input_dim = int(results[0][1].metrics["input_dim"])
                print(f"[SAVE] input_dim={input_dim}", flush=True)
                CONFIG.evaluation.input_dim = input_dim  # store it for future use
                model = CONFIG.model.build(input_dim=input_dim)
                print(f"[SAVE] model built", flush=True)

                ndarrays    = parameters_to_ndarrays(aggregated_parameters)
                params_dict = zip(model.state_dict().keys(), ndarrays)
                state_dict  = {k: torch.tensor(v) for k, v in params_dict}
                model.load_state_dict(state_dict, strict=True)

                # Aggregate metrics from all clients for the final checkpoint
                aggregated_metrics = {}
                if results:
                    num_examples_total = sum(fit_res.num_examples for _, fit_res in results)
                    for key in results[0][1].metrics:
                        aggregated_metrics[key] = sum(
                            fit_res.metrics[key] * fit_res.num_examples
                            for _, fit_res in results
                        ) / num_examples_total

                # Use threshold from metrics if available, else fall back to config default
                threshold = aggregated_metrics.pop("threshold", CONFIG.evaluation.threshold)

                self._save_model(
                    model=model,
                    path=CONFIG.evaluation.model_path,
                    config=CONFIG.model,
                    threshold=threshold,
                    metrics=aggregated_metrics,
                )
                print(f"[SAVE] model saved successfully", flush=True)

            except Exception as e:
                print(f"[SAVE] ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
        
        return aggregated_parameters, metrics

    @staticmethod
    def _save_model(
        model:      nn.Module,
        path:       Path,
        config:     object,
        threshold:  float,
        metrics:    dict,
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
            log(INFO, "[SAVE] creating directory %s", path.parent)
            path.parent.mkdir(parents=True, exist_ok=True)
            log(INFO, "[SAVE] directory exists: %s", path.parent.exists())
            log(INFO, "[SAVE] directory writable: %s", os.access(path.parent, os.W_OK))
            torch.save(checkpoint, path)
            torch.save(checkpoint, path)
            print(f"Model saved to {path}")

        else:
            raise ValueError("Path must be a pathlib.Path object")

