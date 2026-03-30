from .classification import ClassificationTask
from .anomaly_detection import AnomalyDetectionTask

TASK_REGISTRY = {
    "classification": ClassificationTask,
    "anomaly_detection": AnomalyDetectionTask,
    # Add more tasks here as needed
}

def create_task(task_config):
    return TASK_REGISTRY[task_config["name"]]()