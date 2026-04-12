from .powergrid_csv import PowerGridCSVHandler 

#Registry mapping dataset names to their handler classes
DATASET_REGISTRY = {
    "powergrid_csv": PowerGridCSVHandler,
}

#Creates dataset handler based on config.
def create_dataset_handler(data_config):
    dataset_name = data_config.get("name")
    
    #Ensure name is provided
    if dataset_name is None:
        raise ValueError("Dataset 'name' must be specified in config.")
    
    #Check if dataset is registered
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' is not registered.")
    
    #Create and return the dataset handler instance
    return DATASET_REGISTRY[dataset_name](data_config)
