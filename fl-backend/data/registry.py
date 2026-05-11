from .lead_csv import LeadCSVHandler
from .power_consumption_anomaly import PowerConsumptionAnomalyHandler
from .powergrid_csv import PowerGridCSVHandler
from config import ExperimentConfig
DATASET_REGISTRY = {
    "lead_csv":                  LeadCSVHandler,
    "powergrid_csv":             PowerGridCSVHandler,
    "power_consumption_anomaly": PowerConsumptionAnomalyHandler,
}


def create_dataset_handler(config: ExperimentConfig) -> ExperimentConfig:
    """
    Create and return the appropriate dataset handler based on
    data_config.name. The handler class is looked up from the registry
    so adding a new dataset only requires registering it here and
    creating the corresponding config dataclass.
    """
    handler_class = DATASET_REGISTRY.get(config.data.name)

    if handler_class is None:
        raise ValueError(
            f"Dataset '{config.data.name}' is not registered. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    return handler_class(config)
