from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any


__all__ = [
    "BaseCapture",
    "BaseImageProcessor",
    "BaseOCREngine",
    "GUIElement",
    "ImageProcessor",
    "PaddleOCREngine",
    "PerceptionPipeline",
    "PerceptionResult",
    "ScreenCapture",
    "UIDetector",
]


# Public name -> (module inside src.perception, object name)
_EXPORTS: dict[str, tuple[str, str]] = {
    "BaseCapture": ("base_capture", "BaseCapture"),
    "BaseImageProcessor": ("base_preprocess", "BaseImageProcessor"),
    "BaseOCREngine": ("base_ocr", "BaseOCREngine"),
    "GUIElement": ("gui_element", "GUIElement"),
    "ImageProcessor": ("image_preprocess", "ImageProcessor"),
    "PaddleOCREngine": ("paddle_ocr", "PaddleOCREngine"),
    "PerceptionPipeline": ("perception_pipeline", "PerceptionPipeline"),
    "PerceptionResult": ("perception_pipeline", "PerceptionResult"),
    "ScreenCapture": ("screen_capture", "ScreenCapture"),
    "UIDetector": ("ui_detector", "UIDetector"),
}


def __getattr__(name: str) -> Any:
    """Load a public perception object only when it is first requested."""
    try:
        module_name, object_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc

    module = import_module(f".{module_name}", package=__name__)
    value = getattr(module, object_name)

    # Cache the resolved object so later access has no import overhead.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to IDE completion and introspection."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .base_capture import BaseCapture
    from .base_ocr import BaseOCREngine
    from .base_preprocess import BaseImageProcessor
    from .gui_element import GUIElement
    from .image_preprocess import ImageProcessor
    from .paddle_ocr import PaddleOCREngine
    from .perception_pipeline import PerceptionPipeline, PerceptionResult
    from .screen_capture import ScreenCapture
    from .ui_detector import UIDetector