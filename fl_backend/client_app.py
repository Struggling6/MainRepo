import torch
from flwr.client import ClientApp, NumPyClient

from config import CONFIG
from data.registry import create_dataset_handler
from models.registry import create_model
from models.utils import get_model_parameters, set_model_parameters
from tasks.registry import create_task
from training.train import train_model
from training.evaluate import evaluate_model

#Creating a class FlowerCLient that is based on NumpyClient which is a built-in class from Flower that defines the interface for clients in FL.
class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int):
        #Load the global project configuration
        self.config = CONFIG

        #Store which data partition belongs to the client
        self.partition_id = partition_id

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #Create the dataset handler from the data configuration
        self.dataset_handler = create_dataset_handler(self.config["data"])

        #Get dataset metadata
        self.data_metadata = self.dataset_handler.get_metadata()

        #Build the model using config and dataset metadata
        self.model = create_model(self.config["model"], self.data_metadata)

        #Create the task definition based on the config (for example anomaly detection)
        self.task = create_task(self.config["task"])

        #Loads the training and testing data for the client’s partition.
        self.trainloader, self.testloader = self.dataset_handler.get_dataloaders(
            partition_id=self.partition_id
        )

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
            task=self.task,
            training_config=self.config["training"],
            device=self.device,
        )

        #Returns the updated model parameters, number of training examples used, full training results (like loss, accuracy)
        return get_model_parameters(self.model), results["num_examples"], results
    
    #Takes the updated model parameters from server, evaluates the model on the client's test data.
    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)

        results = evaluate_model(
            model=self.model, #the updated model
            testloader=self.testloader,
            task=self.task,
            device=self.device,
        )
        #Sends loss, num_examples, and accuracy back to the server.
        return results["loss"], results["num_examples"], {
            "accuracy": results["accuracy"]
        }

#This creates a client
def client_fn(context):
    #Get which data this client should use
    partition_id = int(context.node_config["partition-id"])

    #Create the client with that data
    return FlowerClient(partition_id=partition_id).to_client()

#Tells Flower how to create clients
app = ClientApp(client_fn=client_fn)