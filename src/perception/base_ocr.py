from abc import ABC, abstractmethod
import numpy as np
from .gui_element import GUIElement

class BaseOCREngine(ABC):

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
    ) -> list[GUIElement]:
        raise NotImplementedError