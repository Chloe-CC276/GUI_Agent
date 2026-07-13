from dataclasses import dataclass
from typing import Tuple


@dataclass
class GUIElement:

    text: str

    bbox: Tuple[int, int, int, int]

    confidence: float

    element_type: str = "text"

    center: Tuple[int, int] | None = None