#Flower tools for building the server.
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from config import CONFIG
from fl_backend import FedAvgWithSave
from models.registry import create_model
from models.utils import save_model, get_device

def server_fn(context):
    fed_config  = CONFIG.federation
    num_clients = CONFIG.data.num_clients

    strategy = FedAvgWithSave(
        fraction_fit=fed_config.fraction_fit,
        fraction_evaluate=fed_config.fraction_evaluate,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_evaluate_config_fn=lambda server_round: {"round": server_round},
    )

    config = ServerConfig(num_rounds=fed_config.num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)