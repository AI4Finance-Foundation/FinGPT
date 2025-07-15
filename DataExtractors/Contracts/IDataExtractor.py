from abc import ABC, abstractmethod
from typing import Optional

class IDataExtractor(ABC):
    @abstractmethod
    def extract_text_from_path(self, file_path: str) -> Optional[str]:
        pass