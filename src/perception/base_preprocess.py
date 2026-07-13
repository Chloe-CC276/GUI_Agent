from abc import ABC, abstractmethod
import numpy as np


class BaseImageProcessor(ABC):

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        pass