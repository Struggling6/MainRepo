#Flower tools for building the server.
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from config import CONFIG

def server_fn(context):
    #Gets the FL settings from config (how many rounds etc.)
    fed_config = CONFIG["federation"]
    num_clients = CONFIG["data"]["num_clients"]

    #Creates the FL strategy. 
    strategy = FedAvg(
        #Both fractions* are used to specify what fraction of clients should be used for training and evaluation in each round. 
        fraction_fit=fed_config["fraction_fit"],
        fraction_evaluate=fed_config["fraction_evaluate"],
        #min* values set to num_clients which means every round waits for all clients.
        min_fit_clients=CONFIG["data"]["num_clients"],
        min_evaluate_clients=CONFIG["data"]["num_clients"],
        min_available_clients=CONFIG["data"]["num_clients"],
    )
    
    #Sets how many FL rounds to run and which strategy.
    config = ServerConfig(num_rounds=fed_config["num_rounds"])
    
    #Tells Flower which strategy and config to use for the server.
    return ServerAppComponents(strategy=strategy, config=config)

#Tells Flower how to create the server
app = ServerApp(server_fn=server_fn)