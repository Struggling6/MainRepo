from .lead_csv import LeadCSVHandler
from .powergrid_csv import PowerGridCSVHandler 
from config import LeadCSVConfig, PowerGridCSVConfig

#Registry mapping dataset names to their handler classes
DATASET_REGISTRY = {
    "lead_csv":      LeadCSVHandler,
    "powergrid_csv": PowerGridCSVHandler,
}

#Creates dataset handler based on config.
def create_dataset_handler(data_config: LeadCSVConfig | PowerGridCSVConfig):
    handler_class = DATASET_REGISTRY.get(data_config.name)

    dataset_name = data_config.get("name")
    
    #Ensure name is provided
    if dataset_name is None:
        raise ValueError("Dataset 'name' must be specified in config.")
    
    #Check if dataset is registered
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' is not registered.")
    
    if handler_class is None:
        raise ValueError(f"Dataset '{data_config.name}' is not registered. "
                         f"Available datasets: {list(DATASET_REGISTRY.keys())}")
    #Create and return the dataset handler instance
    return handler_class(data_config)
