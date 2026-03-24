from abc import ABC, abstractmethod

#this is supposed to be an interface for all the data handlers
# so each Data handler has these methods implemented, 
# and the rest of the code can call these methods without worrying about the implementation details of each data handler
class BaseDatasetHandler(ABC):
    def __init__(self, config:dict):
        self.config = config

    @abstractmethod
    def get_dataloaders(self, partion_id:int):
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        pass

    @abstractmethod
    def get_num_partitions(self) -> int:
        pass