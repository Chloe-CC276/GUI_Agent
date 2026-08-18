"""
common/target_validation

Single source of truth for click-target evidence checks.

Planner resolves a model-supplied ``target_text`` to a detected element and
records the evidence under ``metadata["target_validation"]``.  The orchestration
layer and the Executor both re-check that evidence before a real click is sent.
Keeping the comparison rules here prevents the three layers from drifting apart
and accepting a target such as ``"Search the web"`` matched against a single
detected character.

This module deliberately depends on the standard library only, so both
``src.agent`` and ``src.executor`` can import it without pulling in PyAutoGUI,
OpenCV or any model SDK.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping


CLICK_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "click",
        "double_click",
        "right_click",
        "middle_click",
        "mouse_down",
        "mouse_up",
    }
)

# A candidate shorter than this cannot carry enough evidence that it is the
# requested target, regardless of how it overlaps.
MIN_MATCH_CHARS: int = 2

# Shortest acceptable length of the shorter text relative to the longer one.
MIN_MATCH_RATIO: float = 0.3

# Keep digits, latin, CJK, and common window-close glyphs (× ✕ ✖ ❌ …).
# Without this, labels like "×" normalise to "" and falsely fail texts_match.
_KEEP_CHARACTERS = re.compile(
    r"[^0-9a-z\u4e00-\u9fff\u00d7\u2715\u2716\u274c\u2573\uff38\uff58]+"
)

# Interchangeable title-bar close marks (not plain latin "x" / word "close").
_CLOSE_GLYPHS = frozenset({"×", "✕", "✖", "❌", "ｘ", "Ｘ", "⨉", "⨯"})


def normalise_target_text(value: Any) -> str:
    """Reduce a label to comparable characters: digits, latin letters and CJK."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return _KEEP_CHARACTERS.sub("", text)


def _loose_tokens(value: Any) -> set[str]:
    return set(str(value or "").casefold().split())


def texts_match(target_text: Any, candidate_text: Any) -> bool:
    """Report whether a detected label is strong enough evidence for a target.

    Accepted as a match when the normalised forms are equal, when one contains
    the other, or when one token set covers the other.  The last two rules only
    apply once the shorter text is long enough both absolutely and relative to
    the longer text.
    """

    raw_a = unicodedata.normalize("NFKC", str(target_text or "")).strip()
    raw_b = unicodedata.normalize("NFKC", str(candidate_text or "")).strip()
    if raw_a and raw_a.casefold() == raw_b.casefold():
        return True
    # Title-bar close control: single glyph labels that look identical in logs.
    if raw_a in _CLOSE_GLYPHS and raw_b in _CLOSE_GLYPHS:
        return True

    expected = normalise_target_text(target_text)
    actual = normalise_target_text(candidate_text)
    if not expected or not actual:
        return False
    if expected == actual:
        return True

    shorter, longer = sorted((expected, actual), key=len)
    if len(shorter) < MIN_MATCH_CHARS:
        return False
    if len(shorter) / len(longer) < MIN_MATCH_RATIO:
        return False
    if shorter in longer:
        return True

    expected_tokens = _loose_tokens(target_text)
    actual_tokens = _loose_tokens(candidate_text)
    if not expected_tokens or not actual_tokens:
        return False
    return expected_tokens <= actual_tokens or actual_tokens <= expected_tokens


def coerce_action_mapping(action: Any) -> dict[str, Any]:
    """Return a plain mapping for an Action object, mapping or dataclass."""

    if isinstance(action, Mapping):
        return dict(action)
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(action, method_name, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                continue
            if isinstance(value, Mapping):
                return dict(value)
    result: dict[str, Any] = {}
    for name in (
        "type",
        "action_type",
        "x",
        "y",
        "metadata",
        "parameters",
    ):
        value = getattr(action, name, None)
        if value is not None:
            result[name] = value
    return result


def _action_type(data: Mapping[str, Any]) -> str:
    raw = data.get("type") or data.get("action_type")
    if hasattr(raw, "value"):
        raw = raw.value
    return str(raw or "").strip().lower()


def _point_in_bbox(x: int, y: int, bbox: Any) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        left, top, right, bottom = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return False
    return left <= x <= right and top <= y <= bottom


def validate_click_target(
    action: Any,
    *,
    require_evidence: bool = True,
) -> str | None:
    """Check one action's click evidence and return an error message or ``None``.

    ``require_evidence`` distinguishes the two call sites.  The orchestration
    layer sets it because every planned click must carry evidence.  The Executor
    leaves it off so that directly constructed actions from scripts and tests
    still run, while any evidence that *is* present is still enforced.
    """

    if action is None:
        return "Action rejected: no action is available for execution."

    data = coerce_action_mapping(action)
    nested = data.get("parameters")
    params = dict(nested) if isinstance(nested, Mapping) else data

    action_type = _action_type(data)
    if action_type not in CLICK_ACTION_TYPES:
        return None

    metadata = params.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = data.get("metadata")
    validation = (
        metadata.get("target_validation") if isinstance(metadata, Mapping) else None
    )
    if not isinstance(validation, Mapping):
        if require_evidence:
            return f"{action_type} rejected: no detected target was matched."
        return None

    target_text = str(validation.get("target_text", "")).strip()
    matched_text = str(
        validation.get("matched_text") or validation.get("detected_text") or ""
    ).strip()
    element_id = validation.get("element_id")
    bbox = validation.get("matched_bbox")

    if not target_text:
        return f"{action_type} rejected: target_text is missing."
    if normalise_target_text(target_text).startswith("element"):
        return (
            f"{action_type} rejected: placeholder target {target_text!r} "
            "is not a semantic GUI label."
        )
    if not matched_text:
        return (
            f"{action_type} rejected: element_id={element_id} has no semantic "
            "label and cannot be verified."
        )
    if not texts_match(target_text, matched_text):
        return (
            f"{action_type} rejected: target_text={target_text!r} is not "
            f"supported by detected_text={matched_text!r}."
        )

    x, y = params.get("x"), params.get("y")
    if x is None or y is None:
        return f"{action_type} rejected: coordinates are missing."
    try:
        x_int, y_int = int(x), int(y)
    except (TypeError, ValueError):
        return f"{action_type} rejected: coordinates must be integers."

    if bbox is None:
        if require_evidence:
            return f"{action_type} rejected: matched_bbox is missing."
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return f"{action_type} rejected: matched_bbox must contain four coordinates."
    if not _point_in_bbox(x_int, y_int, bbox):
        return (
            f"{action_type} rejected: ({x_int}, {y_int}) is outside target "
            f"{target_text!r} bbox={list(bbox)}."
        )
    return None


__all__ = [
    "CLICK_ACTION_TYPES",
    "MIN_MATCH_CHARS",
    "MIN_MATCH_RATIO",
    "coerce_action_mapping",
    "normalise_target_text",
    "texts_match",
    "validate_click_target",
]
