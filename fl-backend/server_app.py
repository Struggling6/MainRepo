from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common.logger import log
from logging import INFO
from config import CONFIG
from FedProxWithSave import FedProxWithSave
from plotting.plotting_config import plot_diagrams

def weighted_average_fit(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated = {}

    for key in ["train_loss", "val_f1"]:
        values = [
            num_examples * m[key]
            for num_examples, m in metrics
            if key in m
        ]
        if values:
            aggregated[key] = sum(values) / total_examples
    print(f"[SERVER] Aggregated fit metrics: {aggregated}")
    return aggregated


def weighted_average_evaluate(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated = {}

    for key in ["f1", "pr_auc", "threshold"]:
        values = [
            num_examples * m[key]
            for num_examples, m in metrics
            if key in m
        ]
        if values:
            aggregated[key] = sum(values) / total_examples

    print(f"[SERVER] Aggregated evaluate metrics: {aggregated}")
    return aggregated


def server_fn(context):
    fed_config  = CONFIG.federation
    num_clients = fed_config.num_clients
    proximal_mu = fed_config.proximal_mu

    log(INFO, "Starting federation: %s rounds, %s clients", fed_config.num_rounds, num_clients)
    log(INFO, "Model: %s", CONFIG.model.name)
    log(INFO, "Dataset: %s", CONFIG.data.name)

    strategy = FedProxWithSave(
        fraction_fit=fed_config.fraction_fit,
        fraction_evaluate=fed_config.fraction_evaluate,
        min_fit_clients=max(1, int(num_clients * fed_config.fraction_fit)),
        min_evaluate_clients=max(1, int(num_clients * fed_config.fraction_evaluate)),
        min_available_clients=num_clients,
        on_evaluate_config_fn=lambda server_round: {"round": server_round},
        fit_metrics_aggregation_fn=weighted_average_fit,
        evaluate_metrics_aggregation_fn=weighted_average_evaluate,
        proximal_mu=proximal_mu,
    )

    config = ServerConfig(num_rounds=fed_config.num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)