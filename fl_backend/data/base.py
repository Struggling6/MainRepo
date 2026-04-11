from abc import ABC, abstractmethod


class BaseDatasetHandler(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def get_dataloaders(self, partition_id: int):
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        pass

    @abstractmethod
    def get_num_partitions(self) -> int:
        pass