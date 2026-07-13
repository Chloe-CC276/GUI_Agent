"""
base_capture abstract class
扩展：跨屏他兼容

"""

from abc import ABC, abstractmethod
import numpy as np


class BaseCapture(ABC):

    @abstractmethod
    def capture_screen(self) -> np.ndarray:
        pass

    @abstractmethod
    def capture_region(
        self,
        left: int,
        top: int,
        width: int,
        height: int
    ) -> np.ndarray:
        pass