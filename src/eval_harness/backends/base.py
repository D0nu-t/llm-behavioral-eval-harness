from abc import ABC, abstractmethod


class ModelBackend(ABC):

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def complete(self, messages):
        pass
