"""
launch_wait
Settling policy for actions that launch an application.

Double-clicking a file or shortcut starts a process, and the window often needs
seconds to appear. Observing too early captures the pre-launch screen, so the
verifier finds no evidence that the file was opened and the planner re-clicks.

Two rules:
1. Launch actions wait longer before the post-action observation.
2. When that observation still looks like the pre-action screen, observe again
   after a short extra wait instead of verifying a stale screenshot.
"""

from __future__ import annotations

from typing import Any

from .observation_utils import iter_texts, observation_counts
from .verify_policy import latest_action_type

LAUNCH_ACTION_TYPES: frozenset[str] = frozenset({"double_click"})

# The launch observation is treated as stale when the screen barely moved:
# element counts within this ratio AND text overlap above this similarity.
_MAX_COUNT_DRIFT_RATIO: float = 0.12
_MIN_TEXT_SIMILARITY: float = 0.90
_MIN_TEXTS_TO_COMPARE: int = 5


def is_app_launch_action(action: Any) -> bool:
    """Return True for actions that can start a new application window."""

    return latest_action_type(action) in LAUNCH_ACTION_TYPES


def resolve_post_action_wait(
    action: Any,
    *,
    base_seconds: float,
    launch_seconds: float,
) -> float:
    """Return the settle time to use before the post-action observation."""

    base = max(0.0, float(base_seconds))
    if not is_app_launch_action(action):
        return base
    return max(base, max(0.0, float(launch_seconds)))


def _similar_counts(before_count: int, after_count: int) -> bool:
    largest = max(before_count, after_count)
    if largest == 0:
        return True
    return abs(after_count - before_count) / largest <= _MAX_COUNT_DRIFT_RATIO


def screen_appears_unchanged(before: Any, after: Any) -> bool:
    """Report whether the after observation still shows the pre-action screen."""

    if before is None or after is None:
        return False

    before_elements, before_ocr = observation_counts(before)
    after_elements, after_ocr = observation_counts(after)
    if not _similar_counts(before_elements, after_elements):
        return False
    if not _similar_counts(before_ocr, after_ocr):
        return False

    before_texts = set(iter_texts(before))
    after_texts = set(iter_texts(after))
    if (
        len(before_texts) < _MIN_TEXTS_TO_COMPARE
        or len(after_texts) < _MIN_TEXTS_TO_COMPARE
    ):
        # Too little text to judge; rely on the counts checked above.
        return True

    union = before_texts | after_texts
    if not union:
        return True
    similarity = len(before_texts & after_texts) / len(union)
    return similarity >= _MIN_TEXT_SIMILARITY


def needs_additional_observation(action: Any, before: Any, after: Any) -> bool:
    """Return True when a launch action still needs another observation."""

    if not is_app_launch_action(action):
        return False
    return screen_appears_unchanged(before, after)


__all__ = [
    "LAUNCH_ACTION_TYPES",
    "is_app_launch_action",
    "needs_additional_observation",
    "resolve_post_action_wait",
    "screen_appears_unchanged",
]
