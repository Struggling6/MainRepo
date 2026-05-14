import os
import torch
import torch.nn as nn

from pathlib import Path
from typing import Optional, Union
from flwr.common import EvaluateRes, FitRes, Parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx

from logging import INFO, ERROR
from local_experiment import CONFIG


class FedProxWithSave(FedProx):
    """
    FedProx strategy that saves the aggregated model to disk after the final
    federation round and saves evaluation plots after the final evaluation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.f1_history = []
        self.pr_auc_history = []
        self.roc_auc_history = []

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict]:

        print(
            f"[SAVE] aggregate_fit called, round={server_round}, "
            f"num_rounds={CONFIG.federation.num_rounds}",
            flush=True,
        )

        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if (
            aggregated_parameters is not None
            and server_round == CONFIG.federation.num_rounds
        ):
            print("[SAVE] starting model save...", flush=True)

            try:
                input_dim = int(results[0][1].metrics["input_dim"])
                print(f"[SAVE] input_dim={input_dim}", flush=True)

                CONFIG.evaluation.input_dim = input_dim

                model = CONFIG.model.build(input_dim=input_dim)
                print("[SAVE] model built", flush=True)

                ndarrays = parameters_to_ndarrays(aggregated_parameters)
                params_dict = zip(model.state_dict().keys(), ndarrays)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}

                model.load_state_dict(state_dict, strict=True)

                aggregated_metrics = {}

                if results:
                    num_examples_total = sum(
                        fit_res.num_examples for _, fit_res in results
                    )

                    metric_keys = set()
                    for _, fit_res in results:
                        metric_keys.update(fit_res.metrics.keys())

                    for key in metric_keys:
                        values = [
                            fit_res.metrics[key] * fit_res.num_examples
                            for _, fit_res in results
                            if key in fit_res.metrics
                        ]

                        if values:
                            aggregated_metrics[key] = (
                                sum(values) / num_examples_total
                            )

                threshold = aggregated_metrics.pop(
                    "threshold",
                    CONFIG.evaluation.threshold,
                )

                base_dir = Path(os.getenv("CHECKPOINT_DIR", Path(__file__).resolve().parent / "checkpoints"))
                path = base_dir / f"{CONFIG.model.name}_{CONFIG.data.name}.pt"
                CONFIG.evaluation.model_path = path

                self._save_model(
                    model=model,
                    path = path,
                    config=CONFIG.model,
                    threshold=threshold,
                    metrics=aggregated_metrics,
                )

                print("[SAVE] model saved successfully", flush=True)

                if CONFIG.data.name == "power_consumption_anomaly":
                    print(
                        "[EVAL] skipping final held-out evaluation for power_consumption_anomaly",
                        flush=True,
                    )
                else:
                    from training.training_utils.Evaluator import Evaluator

                    print("[EVAL] running final evaluation on held-out test set...", flush=True)
                    Evaluator.evaluate(model_path=path, threshold=threshold)
                    print("[EVAL] final evaluation done", flush=True)

            except Exception as e:
                log(ERROR, "[SAVE] ERROR: %s", e)
                print(f"[SAVE] ERROR: {e}", flush=True)

                import traceback

                traceback.print_exc()

        return aggregated_parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict]:

        aggregated_result = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

        if aggregated_result is None:
            return aggregated_result

        loss, metrics = aggregated_result

        if "f1" in metrics:
            self.f1_history.append((server_round, metrics["f1"]))

        if "pr_auc" in metrics:
            self.pr_auc_history.append((server_round, metrics["pr_auc"]))

        if "roc_auc" in metrics:
            self.roc_auc_history.append((server_round, metrics["roc_auc"]))

        if server_round == CONFIG.federation.num_rounds:
            try:
                from plotting.plotting_config import plot_diagrams

                print("[PLOTS] Final round reached. Saving plots...", flush=True)

                plot_diagrams(
                    pr_history=self.pr_auc_history,
                    f1_history=self.f1_history,
                    roc_history=self.roc_auc_history,
                )

                print("[PLOTS] Plots saved successfully.", flush=True)

            except Exception as e:
                log(ERROR, "[PLOTS] ERROR: %s", e)
                print(f"[PLOTS] ERROR: {e}", flush=True)

                import traceback

                traceback.print_exc()

        return loss, metrics

    @staticmethod
    def _save_model(
        model: nn.Module,
        path: Path,
        config: object,
        threshold: float,
        metrics: dict,
    ):
        """
        Save model weights, architecture config, threshold,
        and training metrics to a single checkpoint file.
        """

        if not isinstance(path, Path):
            raise ValueError("Path must be a pathlib.Path object")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "threshold": threshold,
            "metrics": metrics or {},
        }

        log(INFO, "[SAVE] creating directory %s", path.parent)

        path.parent.mkdir(parents=True, exist_ok=True)

        log(INFO, "[SAVE] directory exists: %s", path.parent.exists())
        log(INFO, "[SAVE] directory writable: %s", os.access(path.parent, os.W_OK))

        torch.save(checkpoint, path)

        print(f"Model saved to {path}")
