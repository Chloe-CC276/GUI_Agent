"""
prompts/formatters
Safe, compact formatting helpers for GUI Agent prompt contexts.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_WHITESPACE_RE = re.compile(r"\s+")
_MISSING = object()


def compact_whitespace(value: Any, *, preserve_newlines: bool = False) -> str:
    """Normalize whitespace in a value and return a stripped string."""

    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    if not preserve_newlines:
        return _WHITESPACE_RE.sub(" ", text).strip()
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def truncate_text(
    value: Any,
    max_chars: int,
    *,
    suffix: str = "...<truncated>",
    compact: bool = False,
) -> str:
    """Return text no longer than ``max_chars``, including its suffix."""

    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    text = compact_whitespace(value) if compact else ("" if value is None else str(value))
    if len(text) <= max_chars:
        return text
    if max_chars == 0:
        return ""
    if len(suffix) >= max_chars:
        return suffix[:max_chars]
    return text[: max_chars - len(suffix)].rstrip() + suffix


def _json_safe(value: Any, *, max_depth: int, depth: int = 0) -> Any:
    if depth > max_depth:
        return "<max-depth-reached>"
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value, max_depth=max_depth, depth=depth + 1)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, max_depth=max_depth, depth=depth + 1)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item, max_depth=max_depth, depth=depth + 1) for item in value]
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_safe(getattr(value, field.name), max_depth=max_depth, depth=depth + 1)
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe(to_dict(), max_depth=max_depth, depth=depth + 1)
        except Exception:
            pass
    public = getattr(value, "__dict__", None)
    if isinstance(public, Mapping):
        return _json_safe(public, max_depth=max_depth, depth=depth + 1)
    return str(value)


def safe_json_dumps(
    value: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    max_depth: int = 12,
    fallback: str = "<unserializable>",
) -> str:
    """Serialize common project objects to JSON without raising on unknown types."""

    try:
        safe_value = _json_safe(value, max_depth=max_depth)
        return json.dumps(
            safe_value,
            indent=indent,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            allow_nan=False,
        )
    except Exception:
        return json.dumps(fallback, ensure_ascii=ensure_ascii)


def _read(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        item = getattr(value, name, _MISSING)
        if item is not _MISSING:
            return item
    return default


def format_error(error: Any, *, max_chars: int = 2_000, include_traceback: bool = False) -> str:
    """Format an exception, ErrorInfo object, mapping, or error string compactly."""

    if error is None:
        return ""
    error_type = _read(error, "error_type", "type", default=type(error).__name__)
    message = _read(error, "message", default=str(error))
    code = _read(error, "code")
    retryable = _read(error, "retryable")
    parts = [f"{compact_whitespace(error_type)}: {compact_whitespace(message)}"]
    if code is not None:
        parts.append(f"code={code}")
    if retryable is not None:
        parts.append(f"retryable={bool(retryable)}")
    if include_traceback:
        traceback_text = _read(error, "traceback")
        if traceback_text:
            parts.append(f"traceback={compact_whitespace(traceback_text)}")
    return truncate_text("; ".join(parts), max_chars)


def _round_number(value: Any, precision: int) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(float(value), precision)


def _normalize_bbox(value: Any, precision: int) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        candidates = (
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
            ("x", "y", "width", "height"),
        )
        for names in candidates:
            if all(name in value for name in names):
                return [_round_number(value[name], precision) for name in names]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 4:
        return [_round_number(item, precision) for item in value[:4]]
    return None


def format_element(
    element: Any,
    *,
    index: int | None = None,
    max_text_chars: int = 240,
    coordinate_precision: int = 2,
) -> dict[str, Any]:
    """Convert one GUI element into a stable, compact prompt dictionary."""

    element_id = _read(element, "element_id", "id", "uid", "index", default=index)
    text = _read(element, "text", "label", "name", "content", "value", default="")
    role = _read(element, "role", "type", "class_name", "tag", default="unknown")
    bbox = _normalize_bbox(
        _read(element, "bbox", "bounding_box", "bounds", "box", "rect"),
        coordinate_precision,
    )
    center = _read(element, "center", "center_point", "position")
    if center is None and bbox is not None and all(isinstance(v, (int, float)) for v in bbox):
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    if isinstance(center, Sequence) and not isinstance(center, (str, bytes)):
        center = [_round_number(item, coordinate_precision) for item in center[:2]]
    result: dict[str, Any] = {
        "element_id": element_id,
        "text": truncate_text(text, max_text_chars, compact=True),
        "role": compact_whitespace(role) or "unknown",
        "bbox": bbox,
        "center": center,
    }
    for output_name, aliases in {
        "confidence": ("confidence", "score", "probability"),
        "enabled": ("enabled", "is_enabled"),
        "visible": ("visible", "is_visible"),
        "clickable": ("clickable", "is_clickable"),
        "source": ("source",),
    }.items():
        item = _read(element, *aliases)
        if item is not None:
            result[output_name] = _round_number(item, coordinate_precision)
    return {key: value for key, value in result.items() if value not in (None, "")}


def format_elements(
    elements: Iterable[Any] | None,
    *,
    max_items: int = 100,
    max_text_chars: int = 240,
    coordinate_precision: int = 2,
) -> list[dict[str, Any]]:
    """Format and bound a sequence of GUI elements."""

    if max_items < 0:
        raise ValueError("max_items must be non-negative")
    if elements is None:
        return []
    result: list[dict[str, Any]] = []
    for index, element in enumerate(elements):
        if index >= max_items:
            break
        result.append(
            format_element(
                element,
                index=index,
                max_text_chars=max_text_chars,
                coordinate_precision=coordinate_precision,
            )
        )
    return result


def format_list(
    items: Iterable[Any] | None,
    *,
    max_items: int | None = None,
    max_item_chars: int = 500,
    numbered: bool = True,
    empty_text: str = "(none)",
) -> str:
    """Render arbitrary items as a compact numbered or bulleted text list."""

    if items is None:
        return empty_text
    materialized = list(items)
    if max_items is not None:
        materialized = materialized[:max_items]
    lines: list[str] = []
    for index, item in enumerate(materialized, start=1):
        if isinstance(item, str):
            text = compact_whitespace(item)
        else:
            text = compact_whitespace(safe_json_dumps(item, indent=None))
        prefix = f"{index}." if numbered else "-"
        lines.append(f"{prefix} {truncate_text(text, max_item_chars)}")
    return "\n".join(lines) if lines else empty_text


def format_history_item(item: Any, *, max_chars: int = 800) -> str:
    """Format one state-history event or completed Agent step."""

    step = _read(item, "step_index", "step", default="?")
    event = _read(item, "event_type", "type", default=type(item).__name__)
    status = _read(item, "status")
    if isinstance(status, Enum):
        status = status.value
    message = _read(item, "message", "reason", "summary")
    action = _read(item, "action")
    if action is None:
        planner = _read(item, "planner_result")
        action = _read(planner, "action") if planner is not None else None
    parts = [f"step={step}", f"event={compact_whitespace(event)}"]
    if status is not None:
        parts.append(f"status={compact_whitespace(status)}")
    if action is not None:
        parts.append(f"action={compact_whitespace(safe_json_dumps(action, indent=None))}")
    if message:
        parts.append(f"message={compact_whitespace(message)}")
    return truncate_text(" | ".join(parts), max_chars)


def format_history(
    history: Sequence[Any] | Iterable[Any] | None,
    *,
    limit: int = 10,
    max_item_chars: int = 800,
    empty_text: str = "(no history)",
) -> str:
    """Render only the most recent bounded history entries in chronological order."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if history is None or limit == 0:
        return empty_text

    items = list(history)[-limit:]
    lines: list[str] = []

    for index, item in enumerate(items, 1):
        if isinstance(item, dict) and item.get("type") == "reflection":
            content = item.get("content", {})
            if isinstance(content, dict):
                rendered = (
                    "reflection feedback: "
                    f"failure_type={content.get('failure_type', '')}; "
                    f"summary={content.get('summary', '')}; "
                    f"likely_cause={content.get('likely_cause', '')}; "
                    f"avoid={content.get('avoid', [])}; "
                    f"strategy={content.get('strategy', [])}; "
                    f"should_replan={content.get('should_replan', '')}; "
                    f"confidence={content.get('confidence', '')}"
                )
            else:
                rendered = f"reflection feedback: {content}"
        else:
            rendered = format_history_item(item, max_chars=max_item_chars)

        if len(rendered) > max_item_chars:
            rendered = rendered[: max_item_chars - 3] + "..."

        lines.append(f"{index}. {rendered}")

    return "\n".join(lines) if lines else empty_text


__all__ = [
    "compact_whitespace",
    "format_element",
    "format_elements",
    "format_error",
    "format_history",
    "format_history_item",
    "format_list",
    "safe_json_dumps",
    "truncate_text",
]