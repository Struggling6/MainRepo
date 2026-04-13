from .lead_csv import LeadCSVHandler
from .powergrid_csv import PowerGridCSVHandler
from config import LeadCSVConfig, PowerGridCSVConfig

DATASET_REGISTRY = {
    "lead_csv":      LeadCSVHandler,
    "powergrid_csv": PowerGridCSVHandler,
}

def create_dataset_handler(data_config: LeadCSVConfig | PowerGridCSVConfig): # type hint for union of config types
    handler_class = DATASET_REGISTRY.get(data_config.name)
    if handler_class is None:
        raise ValueError(f"Dataset '{data_config.name}' is not registered. "
                         f"Available datasets: {list(DATASET_REGISTRY.keys())}")
    return handler_class(data_config)