from .lead_csv import LeadCSVHandler
from .powergrid_csv import PowerGridCSVHandler
from config import LeadCSVConfig, PowerGridCSVConfig

DATASET_REGISTRY = {
    "lead_csv":      LeadCSVHandler,
    "powergrid_csv": PowerGridCSVHandler,
}


def create_dataset_handler(data_config: LeadCSVConfig | PowerGridCSVConfig):
    """
    Create and return the appropriate dataset handler based on
    data_config.name. The handler class is looked up from the registry
    so adding a new dataset only requires registering it here and
    creating the corresponding config dataclass.
    """
    handler_class = DATASET_REGISTRY.get(data_config.name)

    if handler_class is None:
        raise ValueError(
            f"Dataset '{data_config.name}' is not registered. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    return handler_class(data_config)