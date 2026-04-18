from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import ndarrays_to_parameters

from config import CONFIG
from FedAvgWithSave import FedAvgWithSave
from data.registry import create_dataset_handler


def server_fn(context):
    fed_config = CONFIG.federation
    num_clients = CONFIG.data.num_clients

    # Build initial global model on the server
    metadata = create_dataset_handler(CONFIG.data).get_metadata()
    input_dim = metadata["input_dim"]
    global_model = CONFIG.model.build(input_dim=input_dim)

    initial_ndarrays = [
        v.detach().cpu().numpy() for _, v in global_model.state_dict().items()
    ]
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    strategy = FedAvgWithSave(
        fraction_fit=fed_config.fraction_fit,
        fraction_evaluate=fed_config.fraction_evaluate,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_evaluate_config_fn=lambda server_round: {"round": server_round},
        initial_parameters=initial_parameters,
    )

    config = ServerConfig(num_rounds=fed_config.num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)