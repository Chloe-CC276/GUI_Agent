"""
Deterministic browser search-box / address-bar focus detection.

Success when either:
1. A search/address box is present AND a history/suggestion dropdown appears
   directly below it (spatial association), or
2. The box shows caret / border highlight / background change (model-side;
   code uses dropdown + element geometry as the reliable desktop signal).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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

_DROPDOWN_CHROME_MARKERS: tuple[str, ...] = (
    "历史记录",
    "筛选搜索",
    "收藏夹",
    "标签页",
    "history",
    "favorites",
    "favourites",
    "tabs",
    "filtersearch",
)

_FORBIDDEN_SUGGESTION_MARKERS: tuple[str, ...] = (
    "无法中文输入法",
    "不知道应该先选词",
    "中文输入法",
    "先选词",
)

_FOCUS_ELEMENT_DELTA: int = 12
_TOP_CHROME_RATIO: float = 0.28
_DROPDOWN_MAX_BELOW_RATIO: float = 0.55


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
    cjk = sum(1 for char in normalised if "\u4e00" <= char <= "\u9fff")
    return cjk >= 12 and len(normalised) >= 16


def maybe_rewrite_address_bar_click(
    action_type: str,
    parameters: Mapping[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    """Rewrite address-bar placeholder clicks to hotkey Ctrl+L."""

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


def _bbox_of(item: Any) -> tuple[float, float, float, float] | None:
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


def _text_of(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("text") or "").strip()
    return str(getattr(item, "text", "") or "").strip()


def _iter_labeled_boxes(observation: Any) -> list[tuple[str, tuple[float, float, float, float]]]:
    items: list[Any] = []
    for attr in ("ocr_items", "gui_elements"):
        value = getattr(observation, attr, None) if observation is not None else None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items.extend(value)
    result: list[tuple[str, tuple[float, float, float, float]]] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for item in items:
        bbox = _bbox_of(item)
        if bbox is None:
            continue
        text = _text_of(item)
        key = (text, tuple(int(v) for v in bbox))
        if key in seen:
            continue
        seen.add(key)
        result.append((text, bbox))
    return result


def _screen_size(observation: Any) -> tuple[float, float]:
    width = getattr(observation, "screen_width", None) if observation is not None else None
    height = getattr(observation, "screen_height", None) if observation is not None else None
    try:
        width_f = float(width) if width else 0.0
    except (TypeError, ValueError):
        width_f = 0.0
    try:
        height_f = float(height) if height else 0.0
    except (TypeError, ValueError):
        height_f = 0.0
    if width_f <= 0 or height_f <= 0:
        boxes = _iter_labeled_boxes(observation)
        if boxes:
            width_f = max(width_f, max(box[2] for _, box in boxes))
            height_f = max(height_f, max(box[3] for _, box in boxes))
    return max(width_f, 1.0), max(height_f, 1.0)


def _horizontal_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    *,
    pad: float = 48.0,
) -> float:
    return min(a[2] + pad, b[2]) - max(a[0] - pad, b[0])


def _is_dropdown_chrome(text: str) -> bool:
    normalised = normalise_target_text(text)
    if not normalised:
        return False
    return any(
        marker in normalised or normalised in marker
        for marker in (normalise_target_text(item) for item in _DROPDOWN_CHROME_MARKERS)
    )


def detect_search_box_focus(observation: Any) -> tuple[bool, list[str]]:
    """Detect focused search/address bar from a single after-observation.

    Primary rule: search/address box exists + history/suggestion dropdown below
    it with horizontal spatial association.
    """

    if observation is None:
        return False, []

    width, height = _screen_size(observation)
    boxes = _iter_labeled_boxes(observation)
    if not boxes:
        return False, []

    top_band = height * _TOP_CHROME_RATIO
    search_candidates: list[tuple[str, tuple[float, float, float, float]]] = []
    for text, bbox in boxes:
        _left, top, _right, bottom = bbox
        if top > top_band:
            continue
        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
        if is_address_bar_focus_target(text):
            search_candidates.append((text, bbox))
            continue
        # Wide short chrome field near the top (Edge/Chrome address bar).
        if box_w >= width * 0.22 and box_h <= max(90.0, height * 0.08) and top <= height * 0.18:
            search_candidates.append((text or "<address-bar>", bbox))

    if not search_candidates:
        return False, ["No search/address box candidate in the top chrome band."]

    evidence: list[str] = []
    for text, bbox in search_candidates:
        bottom = bbox[3]
        max_below = bottom + height * _DROPDOWN_MAX_BELOW_RATIO
        chrome_hits: list[str] = []
        suggestion_hits: list[str] = []
        for other_text, other_bbox in boxes:
            if other_bbox is bbox:
                continue
            if other_bbox[1] < bottom - 2:
                continue
            if other_bbox[1] > max_below:
                continue
            if _horizontal_overlap(bbox, other_bbox) <= 0:
                continue
            if _is_dropdown_chrome(other_text):
                chrome_hits.append(other_text)
            elif other_text and other_bbox[1] >= bottom + 4:
                suggestion_hits.append(other_text)

        if chrome_hits or len(suggestion_hits) >= 3:
            note = (
                f"search/address box {text!r} focused: "
                f"dropdown_below={len(suggestion_hits)} rows, "
                f"chrome={chrome_hits[:4]}"
            )
            evidence.append(note)
            return True, evidence

    evidence.append(
        "Search/address box found but no spatially associated history dropdown."
    )
    return False, evidence


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


def build_focus_success_verify(
    *,
    evidence: Iterable[str],
    before: Any = None,
    after: Any = None,
) -> dict[str, Any]:
    """Build a verifier payload that marks address/search focus as successful."""

    before_elements, before_ocr = _observation_counts(before)
    after_elements, after_ocr = _observation_counts(after)
    notes = list(evidence)
    notes.append(
        f"gui_elements {before_elements}->{after_elements}, "
        f"ocr_items {before_ocr}->{after_ocr}."
    )
    return {
        "status": "success",
        "action_effective": True,
        "task_complete": False,
        "evidence": notes,
        "reason": (
            "Search/address focus succeeded: search box present with a "
            "history/suggestion dropdown spatially below it "
            "(or equivalent focus highlight)."
        ),
        "confidence": 0.9,
        "recommended_next": "continue",
    }


def apply_focus_verify_override(
    verify_data: Mapping[str, Any],
    *,
    action: Any,
    before: Any,
    after: Any,
) -> tuple[dict[str, Any], bool]:
    """Force verify success when focus evidence is present but the VLM missed it."""

    data = dict(verify_data)
    if bool(data.get("action_effective")) and str(data.get("status", "")).lower() == "success":
        return data, False

    detected, evidence = detect_search_box_focus(after)
    address_focus = is_address_bar_focus_action(action)
    has_spike = focus_evidence_present(before, after)

    if detected or (address_focus and has_spike):
        pass
    else:
        raw = coerce_action_mapping(action) if action is not None else {}
        nested = raw.get("parameters")
        if isinstance(nested, Mapping):
            raw = {**raw, **dict(nested)}
        action_type = str(raw.get("type") or raw.get("action_type") or "").lower()
        if action_type not in CLICK_ACTION_TYPES or not has_spike:
            return data, False
        evidence = list(evidence) + ["Element-count spike after focus click."]

    success = build_focus_success_verify(evidence=evidence, before=before, after=after)
    merged = dict(data)
    merged.update(success)
    prior = list(data.get("evidence") or [])
    if prior:
        merged["evidence"] = prior + list(success["evidence"])
    return merged, True


__all__ = [
    "apply_focus_verify_override",
    "build_focus_success_verify",
    "detect_search_box_focus",
    "focus_evidence_present",
    "is_address_bar_focus_action",
    "is_address_bar_focus_target",
    "is_forbidden_suggestion_target",
    "maybe_rewrite_address_bar_click",
]
