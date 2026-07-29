from dataclasses import dataclass
from typing import Tuple


@dataclass
class GUIElement:

    text: str

    bbox: Tuple[int, int, int, int]

    confidence: float

    element_type: str = "text"

    center: Tuple[int, int] | None = None

    # Index within the PerceptionResult that produced this element. It is the
    # handle the Planner asks the model to copy back, so it must stay aligned
    # with the position of this element in merged_elements.
    element_id: int | None = None

    # Which detector produced the element: "ocr", "ui" or "merged".
    source: str | None = None
