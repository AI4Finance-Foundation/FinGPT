from abc import ABC, abstractmethod

class IHelperMethod(ABC):
    @abstractmethod
    def validate(self, file_path : str) -> tuple[bool, str]:
        pass