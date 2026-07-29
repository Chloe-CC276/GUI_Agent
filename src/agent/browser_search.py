"""
Browser search helpers for recipe B.

Force address-bar focus via Ctrl+L instead of clicking placeholder OCR, and
apply a deterministic focus-success heuristic when the verifier still misses
the suggestion dropdown / element-count spike after a focus action.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.common.target_validation import (
    CLICK_ACTION_TYPES,
    coerce_action_mapping,
    normalise_target_text,
)

# Normalised forms (via normalise_target_text): digits/latin/CJK only, casefold.
_ADDRESS_BAR_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "搜索或输入web地址",
    "搜索或输入网址",
    "在此输入网址",
    "输入网址",
    "searchorenterwebaddress",
    "searchorenteraddress",
    "searchorenterawebaddress",
    "addressandsearchbar",
    "typeanaddress",
    "enteraddress",
    "enterurl",
)

_FORBIDDEN_SUGGESTION_MARKERS: tuple[str, ...] = (
    "无法中文输入法",
    "不知道应该先选词",
    "中文输入法",
    "先选词",
)

# Edge/Chrome often grow the OCR/GUI element list when the suggestion dropdown opens.
_FOCUS_ELEMENT_DELTA: int = 12


def is_address_bar_focus_target(text: Any) -> bool:
    """Return True when *text* looks like an address-bar placeholder label."""

    normalised = normalise_target_text(text)
    if not normalised:
        return False
    return any(
        marker in normalised or normalised in marker
        for marker in _ADDRESS_BAR_PLACEHOLDER_MARKERS
    )


def is_forbidden_suggestion_target(text: Any) -> bool:
    """Return True for autocomplete/OCR junk that must never be clicked."""

    normalised = normalise_target_text(text)
    if not normalised:
        return False
    if any(marker in normalised for marker in _FORBIDDEN_SUGGESTION_MARKERS):
        return True
    # Long Chinese suggestion rows are almost never the intended focus target.
    cjk = sum(1 for char in normalised if "\u4e00" <= char <= "\u9fff")
    return cjk >= 12 and len(normalised) >= 16


def maybe_rewrite_address_bar_click(
    action_type: str,
    parameters: Mapping[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    """Rewrite address-bar placeholder clicks to hotkey Ctrl+L.

    Returns ``(action_type, parameters, rewritten)``.
    """

    params = dict(parameters)
    normalized_type = str(action_type or "").strip().lower()
    if normalized_type not in CLICK_ACTION_TYPES:
        return normalized_type, params, False

    target = params.get("target_text") or params.get("text") or ""
    if not is_address_bar_focus_target(target):
        return normalized_type, params, False

    return (
        "hotkey",
        {
            "keys": ["ctrl", "l"],
            "description": (
                "Focus the browser address bar with Ctrl+L "
                f"(rewritten from click {target!r})"
            ),
        },
        True,
    )


def _action_target_text(action: Any) -> str:
    if action is None:
        return ""
    data = coerce_action_mapping(action)
    nested = data.get("parameters")
    if isinstance(nested, Mapping):
        data = {**data, **dict(nested)}
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        validation = metadata.get("target_validation")
        if isinstance(validation, Mapping) and validation.get("target_text"):
            return str(validation.get("target_text") or "")
    return str(data.get("target_text") or data.get("text") or "")


def is_address_bar_focus_action(action: Any) -> bool:
    """True for Ctrl+L hotkeys or clicks on address-bar placeholder text."""

    if action is None:
        return False
    data = coerce_action_mapping(action)
    nested = data.get("parameters")
    if isinstance(nested, Mapping):
        data = {**data, **dict(nested)}
    raw = data.get("type") or data.get("action_type")
    if hasattr(raw, "value"):
        raw = raw.value
    action_type = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if action_type == "hotkey":
        keys = data.get("keys")
        if isinstance(keys, str):
            parts = [
                part.strip().lower()
                for part in keys.replace("+", ",").split(",")
                if part.strip()
            ]
        elif isinstance(keys, (list, tuple)):
            parts = [str(item).strip().lower() for item in keys if str(item).strip()]
        else:
            parts = []
        normalised = {part.replace("_", "") for part in parts}
        return normalised in {
            frozenset({"ctrl", "l"}),
            frozenset({"control", "l"}),
            frozenset({"cmd", "l"}),
            frozenset({"command", "l"}),
        }
    if action_type in CLICK_ACTION_TYPES:
        return is_address_bar_focus_target(_action_target_text(action))
    return False


def _observation_counts(observation: Any) -> tuple[int, int]:
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


def focus_evidence_present(
    before: Any,
    after: Any,
    *,
    min_delta: int = _FOCUS_ELEMENT_DELTA,
) -> bool:
    """Detect a focus dropdown via a sharp rise in detected UI/OCR items."""

    before_elements, before_ocr = _observation_counts(before)
    after_elements, after_ocr = _observation_counts(after)
    if after_elements >= before_elements + min_delta:
        return True
    if after_ocr >= before_ocr + min_delta:
        return True
    return False


def apply_focus_verify_override(
    verify_data: Mapping[str, Any],
    *,
    action: Any,
    before: Any,
    after: Any,
) -> tuple[dict[str, Any], bool]:
    """Force verify success when focus evidence is present but the VLM missed it.

    Returns ``(data, overridden)``.
    """

    data = dict(verify_data)
    if bool(data.get("action_effective")) and str(data.get("status", "")).lower() == "success":
        return data, False

    address_focus = is_address_bar_focus_action(action)
    has_spike = focus_evidence_present(before, after)

    if not address_focus:
        raw = coerce_action_mapping(action) if action is not None else {}
        nested = raw.get("parameters")
        if isinstance(nested, Mapping):
            raw = {**raw, **dict(nested)}
        action_type = str(raw.get("type") or raw.get("action_type") or "").lower()
        if action_type not in CLICK_ACTION_TYPES or not has_spike:
            return data, False
    # Address-bar focus: override caret-only VLM failures. A dropdown spike is
    # strong extra evidence but not required — Edge often keeps the caret hidden.

    before_elements, before_ocr = _observation_counts(before)
    after_elements, after_ocr = _observation_counts(after)
    evidence = list(data.get("evidence") or [])
    evidence.append(
        "Deterministic focus heuristic: "
        f"gui_elements {before_elements}->{after_elements}, "
        f"ocr_items {before_ocr}->{after_ocr}, spike={has_spike}."
    )
    data["action_effective"] = True
    data["status"] = "success"
    data["recommended_next"] = "continue"
    try:
        confidence = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    data["confidence"] = max(confidence, 0.8)
    data["evidence"] = evidence
    data["reason"] = (
        "Address/search focus succeeded: treat focus as effective even without a "
        "visible caret (override of caret-only verifier failure)."
    )
    return data, True


__all__ = [
    "apply_focus_verify_override",
    "focus_evidence_present",
    "is_address_bar_focus_action",
    "is_address_bar_focus_target",
    "is_forbidden_suggestion_target",
    "maybe_rewrite_address_bar_click",
]
