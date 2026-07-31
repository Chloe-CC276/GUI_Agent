"""
Shared rules for skipping post-action verification.

paste_text injects clipboard text and should be followed immediately by Enter
for browser-search flows. Chat-compose pastes must still be OCR-verified inside
the composer, so they do not skip. Chat composer clicks skip verify because a
caret is not OCR-reliable; focus is recorded in chat_progress instead.
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


def should_skip_verification(action: Any, state: Any = None) -> bool:
    """Return True when the orchestrator should skip verify and replan immediately."""

    action_type = latest_action_type(action)
    if state is not None:
        # Local import avoids a circular dependency with chat_send.
        from .chat_send import (
            is_chat_paste_action,
            is_chat_send_task,
            should_skip_chat_focus_verification,
        )
        from .document_tasks import task_instruction

        instruction = task_instruction(state)
        if is_chat_send_task(instruction):
            if should_skip_chat_focus_verification(action, state):
                return True
            if is_chat_paste_action(action) or action_type == "paste_text":
                return False

    return action_type in _SKIP_VERIFY_ACTION_TYPES


__all__ = [
    "latest_action_type",
    "should_skip_verification",
]
