from pathlib import Path

from .lead_csv import LeadCSVHandler
from .powergrid_csv import PowerGridCSVHandler 
from config import LeadCSVConfig, PowerGridCSVConfig

#Registry mapping dataset names to their handler classes
DATASET_REGISTRY = {
    "lead_csv":      LeadCSVHandler,
    "powergrid_csv": PowerGridCSVHandler,
}


def _infer_dataset_name_from_path(file_path: Path) -> str | None:
    try:
        import pandas as pd

        df = pd.read_csv(file_path, nrows=1)
    except Exception:
        return None

    cols = set(df.columns)
    if "marker" in cols:
        return "powergrid_csv"
    if {"building_id", "timestamp"}.issubset(cols):
        return "lead_csv"
    return None


#Creates dataset handler based on config.
def create_dataset_handler(data_config: LeadCSVConfig | PowerGridCSVConfig):
    dataset_name = getattr(data_config, "name", None)

    #Ensure name is provided
    if dataset_name is None:
        raise ValueError("Dataset 'name' must be specified in config.")

    handler_class = DATASET_REGISTRY.get(dataset_name)

    file_path = getattr(data_config, "file_path", None)
    if file_path is not None:
        file_path = Path(file_path)
        if file_path.exists():
            inferred_name = _infer_dataset_name_from_path(file_path)
            if inferred_name is not None and inferred_name != dataset_name:
                print(
                    f"Warning: config.name='{dataset_name}' does not match "
                    f"detected dataset type '{inferred_name}' from {file_path}. "
                    "Using the inferred dataset handler."
                )
                dataset_name = inferred_name
                handler_class = DATASET_REGISTRY.get(dataset_name)
                setattr(data_config, "name", dataset_name)

                if inferred_name == "powergrid_csv":
                    if not hasattr(data_config, "normalize"):
                        setattr(data_config, "normalize", True)
                    if not hasattr(data_config, "partition_mode"):
                        setattr(data_config, "partition_mode", "shared")

                if inferred_name == "lead_csv" and not hasattr(data_config, "partition_mode"):
                    setattr(data_config, "partition_mode", "shared")

    #Check if dataset is registered
    if dataset_name not in DATASET_REGISTRY or handler_class is None:
        raise ValueError(
            f"Dataset '{dataset_name}' is not registered. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    #Create and return the dataset handler instance
    return handler_class(data_config)
