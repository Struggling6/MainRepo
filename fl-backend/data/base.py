#Python module used to create interfaces
from abc import ABC, abstractmethod 

#Base interface for all dataset handlers. 
#@abstractmethod is a decorator that indicates that the method is abstract and must be implemented by any subclass.
class BaseDatasetHandler(ABC):
    #When the object is created, save the config/settings.
    def __init__(self, config:dict):
        self.config = config

    @abstractmethod
    def get_dataloaders(self, partion_id:int):
        #pass is a placeholder. The subclasses will implement the logic.
        pass 

    @abstractmethod
    #Returns metadata about the dataset, such as input shape, number of classes, etc.
    def get_metadata(self) -> dict: 
        pass

    @abstractmethod
    #Returns how many partitions/client exist.
    def get_num_partitions(self) -> int: 
        pass