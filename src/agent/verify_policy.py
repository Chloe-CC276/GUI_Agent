"""
Shared rules for skipping post-action verification.

paste_text injects clipboard text and should be followed immediately by Enter.
Address-bar focus (Ctrl+L) must still observe_after so dropdown evidence can be
checked deterministically before the next paste step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.common.target_validation import coerce_action_mapping

_SKIP_VERIFY_ACTION_TYPES: frozenset[str] = frozenset({"paste_text"})


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


def should_skip_verification(action: Any) -> bool:
    """Return True when the orchestrator should skip verify and replan immediately."""

    return latest_action_type(action) in _SKIP_VERIFY_ACTION_TYPES


__all__ = [
    "latest_action_type",
    "should_skip_verification",
]
