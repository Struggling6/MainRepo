from .classification import ClassificationTask

TASK_REGISTRY = {
    "classification": ClassificationTask,
    # Add more tasks here as needed
}

def create_task(task_config):
    return TASK_REGISTRY[task_config.name]