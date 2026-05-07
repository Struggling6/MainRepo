from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common.logger import log
from logging import INFO
from local_experiment import CONFIG
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
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_evaluate_config_fn=lambda server_round: {"round": server_round},
        fit_metrics_aggregation_fn=weighted_average_fit,
        evaluate_metrics_aggregation_fn=weighted_average_evaluate,
        proximal_mu=proximal_mu,
    )

    config = ServerConfig(num_rounds=fed_config.num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)



def on_train_end(context):
    #Called after all rounds complete.
    strategy = context.strategy
    
    # Extract metrics from eval_history
    f1_values = []
    pr_values = []
    
    for eval_round in strategy.eval_history:
        if "f1" in eval_round:
            f1_values.append(eval_round["f1"])
        if "pr_auc" in eval_round:
            pr_values.append(eval_round["pr_auc"])
    
    print(f"F1 values: {f1_values}")
    print(f"PR-AUC values: {pr_values}")
    
    plot_diagrams(f1_values, pr_values)


app = ServerApp(server_fn=server_fn)
