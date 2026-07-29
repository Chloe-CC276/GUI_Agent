"""Dependency-free helpers shared by the agent, perception and executor layers."""

from .target_validation import (
    CLICK_ACTION_TYPES,
    MIN_MATCH_CHARS,
    MIN_MATCH_RATIO,
    coerce_action_mapping,
    normalise_target_text,
    texts_match,
    validate_click_target,
)

__all__ = [
    "CLICK_ACTION_TYPES",
    "MIN_MATCH_CHARS",
    "MIN_MATCH_RATIO",
    "coerce_action_mapping",
    "normalise_target_text",
    "texts_match",
    "validate_click_target",
]
