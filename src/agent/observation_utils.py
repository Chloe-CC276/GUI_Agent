"""
observation_utils
Shared readers for ObservationState.

OCR items and GUI elements arrive as either mappings or objects, so every
detector needs the same defensive access to text, bounding boxes and counts.
These helpers keep that logic in one place.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from src.common.target_validation import normalise_target_text

BoundingBox = tuple[float, float, float, float]

# Marks the observation captured by the post-action stage. Verification only
# reads that capture, so the next planning turn can reuse it instead of paying
# for a second full perception pass of the very same screen.
POST_ACTION_OBSERVATION_KEY: str = "post_action_observation_id"


def bbox_of(item: Any) -> BoundingBox | None:
    """Return a valid (left, top, right, bottom) box, or None."""

    bbox = getattr(item, "bbox", None)
    if bbox is None and isinstance(item, Mapping):
        bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def text_of(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("text") or "").strip()
    return str(getattr(item, "text", "") or "").strip()


def iter_items(observation: Any) -> list[Any]:
    """Return every OCR item and GUI element of an observation."""

    items: list[Any] = []
    if observation is None:
        return items
    for attribute in ("ocr_items", "gui_elements"):
        value = getattr(observation, attribute, None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items.extend(value)
    return items


def iter_labeled_boxes(observation: Any) -> list[tuple[str, BoundingBox]]:
    """Return de-duplicated (text, bbox) pairs for every detected item."""

    result: list[tuple[str, BoundingBox]] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for item in iter_items(observation):
        bbox = bbox_of(item)
        if bbox is None:
            continue
        text = text_of(item)
        key = (text, tuple(int(value) for value in bbox))
        if key in seen:
            continue
        seen.add(key)
        result.append((text, bbox))
    return result


def iter_texts(observation: Any) -> list[str]:
    """Return every non-empty text, including items without a bbox."""

    return [text for text in map(text_of, iter_items(observation)) if text]


def normalised_texts(observation: Any) -> list[str]:
    """Return normalised forms of every labelled text, dropping empties."""

    return [
        normalised
        for text, _bbox in iter_labeled_boxes(observation)
        if (normalised := normalise_target_text(text))
    ]


def screen_size(observation: Any) -> tuple[float, float]:
    """Return the screen size, falling back to the detected box extents."""

    width = getattr(observation, "screen_width", None) if observation else None
    height = getattr(observation, "screen_height", None) if observation else None
    try:
        width_f = float(width) if width else 0.0
    except (TypeError, ValueError):
        width_f = 0.0
    try:
        height_f = float(height) if height else 0.0
    except (TypeError, ValueError):
        height_f = 0.0
    if width_f <= 0 or height_f <= 0:
        boxes = iter_labeled_boxes(observation)
        if boxes:
            width_f = max(width_f, max(box[2] for _text, box in boxes))
            height_f = max(height_f, max(box[3] for _text, box in boxes))
    return max(width_f, 1.0), max(height_f, 1.0)


def observation_counts(observation: Any) -> tuple[int, int]:
    """Return (element_count, ocr_item_count), tolerating stubs and methods."""

    if observation is None:
        return 0, 0

    elements = getattr(observation, "element_count", None)
    if callable(elements):
        try:
            element_count = int(elements())
        except Exception:
            element_count = 0
    elif elements is not None:
        element_count = int(elements)
    else:
        element_count = len(getattr(observation, "gui_elements", None) or [])

    ocr_count = getattr(observation, "ocr_item_count", None)
    if callable(ocr_count):
        try:
            ocr_items = int(ocr_count())
        except Exception:
            ocr_items = 0
    elif ocr_count is not None:
        ocr_items = int(ocr_count)
    else:
        ocr_items = len(getattr(observation, "ocr_items", None) or [])

    return element_count, ocr_items


def mark_post_action_observation(state: Any, observation: Any) -> None:
    """Remember that *observation* was captured right after the last action."""

    metadata = getattr(state, "metadata", None)
    if isinstance(metadata, MutableMapping):
        metadata[POST_ACTION_OBSERVATION_KEY] = getattr(
            observation, "observation_id", None
        )


def consume_post_action_observation(state: Any, observation: Any) -> bool:
    """Report whether *observation* is still the untouched post-action capture.

    The mark is cleared, so one capture is only ever reused once.
    """

    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, MutableMapping):
        return False
    marker = metadata.pop(POST_ACTION_OBSERVATION_KEY, None)
    if marker is None or observation is None:
        return False
    return marker == getattr(observation, "observation_id", None)


__all__ = [
    "BoundingBox",
    "POST_ACTION_OBSERVATION_KEY",
    "bbox_of",
    "consume_post_action_observation",
    "mark_post_action_observation",
    "iter_items",
    "iter_labeled_boxes",
    "iter_texts",
    "normalised_texts",
    "observation_counts",
    "screen_size",
    "text_of",
]
