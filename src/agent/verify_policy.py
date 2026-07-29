"""
Shared rules for skipping post-action verification.

paste_text injects clipboard text and should be followed immediately by Enter.
Ctrl+L / Command+L focuses the browser address bar and should be followed by paste.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.common.target_validation import coerce_action_mapping

_SKIP_VERIFY_ACTION_TYPES: frozenset[str] = frozenset({"paste_text"})

_ADDRESS_BAR_FOCUS_KEY_SETS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"ctrl", "l"}),
        frozenset({"control", "l"}),
        frozenset({"cmd", "l"}),
        frozenset({"command", "l"}),
    }
)


def latest_action_type(action: Any) -> str:
    if action is None:
        return ""
    data = coerce_action_mapping(action)
    nested = data.get("parameters")
    if isinstance(nested, Mapping):
        data = {**data, **dict(nested)}
    raw = data.get("type") or data.get("action_type")
    if hasattr(raw, "value"):
        raw = raw.value
    return str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalized_hotkey_keys(action: Any) -> frozenset[str]:
    if action is None:
        return frozenset()
    data = coerce_action_mapping(action)
    nested = data.get("parameters")
    if isinstance(nested, Mapping):
        data = {**data, **dict(nested)}
    keys = data.get("keys")
    if isinstance(keys, str):
        parts = [part.strip() for part in keys.replace("+", ",").split(",") if part.strip()]
    elif isinstance(keys, Sequence) and not isinstance(keys, (str, bytes)):
        parts = [str(item).strip() for item in keys if str(item).strip()]
    else:
        return frozenset()
    return frozenset(part.lower().replace("_", "") for part in parts)


def should_skip_verification(action: Any) -> bool:
    """Return True when the orchestrator should skip verify and replan immediately."""

    action_type = latest_action_type(action)
    if action_type in _SKIP_VERIFY_ACTION_TYPES:
        return True
    if action_type == "hotkey":
        return _normalized_hotkey_keys(action) in _ADDRESS_BAR_FOCUS_KEY_SETS
    return False


__all__ = [
    "latest_action_type",
    "should_skip_verification",
]
