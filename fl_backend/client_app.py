import torch
from flwr.client import ClientApp, NumPyClient


from config import CONFIG
from data.registry import create_dataset_handler
from models.registry import create_model
from models.utils import get_model_parameters, set_model_parameters
from training.train import train_model
from models.utils import get_device
from training.training_utils.Evaluator import Evaluator
#Creating a class FlowerCLient that is based on NumpyClient which is a built-in class from Flower that defines the interface for clients in FL.
class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int):
        self.config          = CONFIG #Load the global project configuration
        self.partition_id    = partition_id #Store which data partition belongs to the client
        self.device          = get_device()
        self.dataset_handler = create_dataset_handler(self.config.data) #Create the dataset handler from the data configuration
        self.data_metadata   = self.dataset_handler.get_metadata()

        # Model created from config — architecture and loss both come from config
        self.model = create_model(self.config.model, self.data_metadata)

        #Loads the training and testing data for the client’s partition.
        self.trainloader, self.testloader = self.dataset_handler.get_dataloaders(
            partition_id=self.partition_id
        )
        self.evaluator = Evaluator(model_config=self.config.model)

    #Method called by the server to get the current model parameters from the client. 
    def get_parameters(self, config):
        return get_model_parameters(self.model)
    
    #Updates the client's model parameters with the new parameters sent by the server.
    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
    
        #Trains the model locally on the client's data using the training loader etc.
        results = train_model(
            model=self.model,
            trainloader=self.trainloader,
            training_config=self.config.training,  # TrainingConfig
            model_config=self.config.model,         # CNNTransformerConfig — contains loss_fn
            device=self.device,
        )

        #Returns the updated model parameters, number of training examples used, full training results (like loss, accuracy)
        return get_model_parameters(self.model), results["num_examples"], results
    
    #Takes the updated model parameters from server, evaluates the model on the client's test data.
    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        results = self.evaluator.evaluate_round(self.model, self.testloader)
        return results["loss"], results["num_examples"], {
            "f1":        results["val_f1"],
            "pr_auc":    results["pr_auc"],
            "threshold": results["best_threshold"],
        }

#This creates a client
def client_fn(context):
    #Get which data this client should use
    partition_id = int(context.node_config["partition-id"])

    #Create the client with that data
    return FlowerClient(partition_id=partition_id).to_client()

#Tells Flower how to create clients
app = ClientApp(client_fn=client_fn)