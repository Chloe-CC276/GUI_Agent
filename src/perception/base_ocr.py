from abc import ABC, abstractmethod
import numpy as np

class BaseOCREngine(ABC):

    @abstractmethod
    def detect(self, image: np.ndarray):
        pass