from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from config import CONFIG

#import numpy as np
#import torch
#from client_app import LocalClient
#from data.registry import create_dataset_handler
#from models.registry import create_model
#from models.utils import get_model_parameters, set_model_parameters
#from tasks.registry import create_task
#from training.evaluate import evaluate_model

def server_fn(context):
    fed_config = CONFIG["federation"]
    num_clients = CONFIG["data"]["num_clients"]

    strategy = FedAvg(
        fraction_fit=fed_config["fraction_fit"],
        fraction_evaluate=fed_config["fraction_evaluate"],
        min_fit_clients=CONFIG["data"]["num_clients"],
        min_evaluate_clients=CONFIG["data"]["num_clients"],
        min_available_clients=CONFIG["data"]["num_clients"],
    )
    
    config = ServerConfig(num_rounds=fed_config["num_rounds"])

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)