from .powergrid_csv import PowerGridCSVHandler

DATASET_REGISTRY = {
    "powergrid_csv": PowerGridCSVHandler
}

def create_dataset_handler(data_config):
    dataset_name = data_config.get("name")
    
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' is not registered.")
    
    return DATASET_REGISTRY[dataset_name](data_config)
