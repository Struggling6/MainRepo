#Flower tools for building the server.
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
import torch

from config import CONFIG
from models.registry import create_model
from models.utils import save_model, get_device

def server_fn(context):
    fed_config = CONFIG.federation
    num_clients = CONFIG.data.num_clients

    #Creates the FL strategy. 
    strategy = FedAvg(
        fraction_fit=fed_config.fraction_fit,
        fraction_evaluate=fed_config.fraction_evaluate,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        # Called by Flower after every aggregation round
        on_evaluate_config_fn=lambda server_round: {"round": server_round},
        # Save model after the final round
        on_fit_config_fn=lambda server_round: {"round": server_round},
    )
    
    config = ServerConfig(num_rounds=fed_config.num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

def save_aggregated_model(server_round, parameters, config):
    """
    Called after the final federation round completes.
    Reconstructs the model from aggregated parameters and saves it.
    """
    if server_round == CONFIG.federation.num_rounds:
        device        = get_device()
        data_metadata = {"input_dim": CONFIG.model.in_channels, "num_classes": 1}
        model         = create_model(CONFIG.model, data_metadata).to(device)

        # Convert Flower parameters back to numpy arrays, then load into model
        ndarrays = parameters_to_ndarrays(parameters)
        params_dict = zip(model.state_dict().keys(), ndarrays)
        state_dict  = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        save_model(
            model=model,
            path=CONFIG.evaluation.model_path,
            config=CONFIG.model,
        )
        print(f"Federated model saved after round {server_round}")


app = ServerApp(server_fn=server_fn)