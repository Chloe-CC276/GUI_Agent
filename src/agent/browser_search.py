"""
Deterministic browser search-box / address-bar focus detection.

Success when either:
1. A search/address box is present AND a history/suggestion dropdown appears
   directly below it (spatial association), or
2. The box shows caret / border highlight / background change (model-side;
   code uses dropdown + element geometry as the reliable desktop signal).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.common.target_validation import (
    CLICK_ACTION_TYPES,
    coerce_action_mapping,
    normalise_target_text,
)

from .observation_utils import (
    iter_labeled_boxes,
    normalised_texts,
    observation_counts,
    screen_size,
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

# Persistent search stage. The planner only sees the latest observation, so a
# confirmed focus must survive in state.metadata; otherwise the model keeps
# re-focusing the address bar instead of pasting the query.
#
# Website-open recipe for Google (and similar sites):
#   Ctrl+L → paste google.com → Enter → re-observe → Google logo + search box
#   ⇒ task complete. Bare "google" is forbidden (it becomes a Bing/Edge search).
SEARCH_PROGRESS_KEY: str = "search_progress"

LEG_NAVIGATE: str = "navigate"
LEG_SEARCH: str = "search"

PHASE_INPUT_FOCUSED: str = "input_focused"
PHASE_QUERY_ENTERED: str = "query_entered"
PHASE_NAV_SUBMITTED: str = "nav_submitted"
PHASE_HOMEPAGE_REACHED: str = "homepage_reached"
PHASE_SEARCH_SUBMITTED: str = "search_submitted"
# Backward-compatible alias used by older call sites / logs.
PHASE_QUERY_SUBMITTED: str = PHASE_SEARCH_SUBMITTED

_PHASE_RANK: dict[str, int] = {
    PHASE_INPUT_FOCUSED: 1,
    PHASE_QUERY_ENTERED: 2,
    PHASE_NAV_SUBMITTED: 3,
    PHASE_HOMEPAGE_REACHED: 4,
    PHASE_SEARCH_SUBMITTED: 4,
}

_TERMINAL_PHASES: frozenset[str] = frozenset(
    {PHASE_HOMEPAGE_REACHED, PHASE_SEARCH_SUBMITTED}
)

_NAVIGATION_TEXT: str = "google.com"

_BING_PAGE_MARKERS: tuple[str, ...] = (
    "bing",
    "microsoftbing",
    "必应",
    "wwwbingcom",
    "bingcomsearch",
)

_GOOGLE_BRAND_MARKERS: tuple[str, ...] = (
    "google",
    "谷歌",
)

_PASTE_ACTION_TYPES: frozenset[str] = frozenset(
    {"paste_text", "paste", "type_text", "type", "write"}
)

_FOCUS_SENSITIVE_PHASES: frozenset[str] = frozenset(
    {PHASE_INPUT_FOCUSED, PHASE_QUERY_ENTERED}
)

_FOCUS_BREAKING_ACTION_TYPES: frozenset[str] = frozenset(CLICK_ACTION_TYPES) | {
    "scroll",
    "drag",
    "hotkey",
    "press",
    "press_key",
    "key",
}


def _guidance_for(phase: str, leg: str) -> tuple[str, list[str]]:
    """Return (next_action, must_not_repeat) for a phase/leg pair."""

    if phase == PHASE_INPUT_FOCUSED:
        if leg == LEG_SEARCH:
            return (
                "A Google page search box is focused. "
                "Paste the task query with paste_text now "
                f"(never paste bare 'google' or '{_NAVIGATION_TEXT}' again).",
                [
                    "hotkey ctrl+l",
                    "clicking the address bar or any suggestion row",
                    "pressing enter before paste_text",
                    f"paste_text of bare google / {_NAVIGATION_TEXT}",
                ],
            )
        return (
            "The address bar is already focused. "
            f"Paste exactly '{_NAVIGATION_TEXT}' with paste_text now "
            "(never paste bare 'google' — that opens Bing/Edge search results).",
            [
                "hotkey ctrl+l (the address bar is already focused)",
                "clicking the address bar or any suggestion row again",
                "paste_text of bare google / Google (must be google.com)",
            ],
        )
    if phase == PHASE_QUERY_ENTERED:
        if leg == LEG_SEARCH:
            return (
                "The task query is already in the Google search box. "
                "Submit it with press key=enter now.",
                [
                    "hotkey ctrl+l",
                    "paste_text / type_text again",
                    "clicking Google search / result links before enter",
                ],
            )
        return (
            f"'{_NAVIGATION_TEXT}' is already in the address bar. "
            "Press key=enter now to open the Google homepage.",
            [
                "hotkey ctrl+l",
                "paste_text / type_text again",
                "paste_text of bare google",
            ],
        )
    if phase == PHASE_NAV_SUBMITTED:
        return (
            "Navigation was submitted. Re-observe the page: the task succeeds when "
            "BOTH the Google logo and the Google central search box are visible. "
            "If you are on Bing/Edge results for the keyword Google, do NOT click "
            f"result links — Ctrl+L and paste '{_NAVIGATION_TEXT}' again.",
            [
                "paste_text of bare google / Google as a search keyword",
                "clicking Bing/Edge result links titled Google",
                "pressing enter again without re-focusing the address bar",
                "treating a Bing search-results page as the Google homepage",
            ],
        )
    if phase == PHASE_HOMEPAGE_REACHED:
        return (
            "The Google homepage is open (logo + central search box). "
            "The website-open task is complete — finish.",
            [
                "paste_text / type_text",
                "pressing enter again",
                "hotkey ctrl+l",
                "clicking Bing or Google search result links",
            ],
        )
    if phase == PHASE_SEARCH_SUBMITTED:
        return (
            "The search query was submitted. Wait for / judge the results page; "
            "do not reopen Google via a keyword search.",
            [
                "paste_text / type_text",
                "pressing enter again",
                "hotkey ctrl+l",
                "clicking Bing results titled Google",
            ],
        )
    return ("", [])


def is_address_bar_focus_target(text: Any) -> bool:
    """Return True when *text* looks like an address-bar placeholder label.

    Short fragments such as bare ``Search`` must not match long markers like
    ``searchorenterwebaddress``; otherwise every Google homepage search box is
    treated as the address bar.
    """

    normalised = normalise_target_text(text)
    if not normalised:
        return False
    for marker in _ADDRESS_BAR_PLACEHOLDER_MARKERS:
        if normalised == marker:
            return True
        if marker in normalised and len(marker) >= 6:
            return True
        # Require a substantial fragment before accepting "text is contained in
        # marker"; otherwise "search" matches every English address-bar label.
        if normalised in marker and len(normalised) >= max(10, (len(marker) + 1) // 2):
            return True
    return False


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


def _action_data(action: Any) -> dict[str, Any]:
    if action is None:
        return {}
    data = coerce_action_mapping(action)
    nested = data.get("parameters")
    if isinstance(nested, Mapping):
        data = {**data, **dict(nested)}
    return data


def _action_type(data: Mapping[str, Any]) -> str:
    raw = data.get("type") or data.get("action_type")
    if hasattr(raw, "value"):
        raw = raw.value
    return str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")


def _key_parts(data: Mapping[str, Any]) -> set[str]:
    keys = data.get("keys")
    if keys is None:
        keys = data.get("key")
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
    return {part.replace("_", "") for part in parts}


def is_address_bar_focus_action(action: Any) -> bool:
    """True for Ctrl+L hotkeys or clicks on address-bar placeholder text."""

    if action is None:
        return False
    data = _action_data(action)
    action_type = _action_type(data)
    if action_type == "hotkey":
        return _key_parts(data) in {
            frozenset({"ctrl", "l"}),
            frozenset({"control", "l"}),
            frozenset({"cmd", "l"}),
            frozenset({"command", "l"}),
        }
    if action_type in CLICK_ACTION_TYPES:
        return is_address_bar_focus_target(_action_target_text(action))
    return False


def is_submit_action(action: Any) -> bool:
    """True for Enter presses that submit the query in a focused input."""

    if action is None:
        return False
    data = _action_data(action)
    if _action_type(data) not in {"press", "press_key", "key", "hotkey"}:
        return False
    return bool(_key_parts(data) & {"enter", "return"})


def is_query_input_action(action: Any) -> bool:
    """True for actions that put the query text into the focused input."""

    if action is None:
        return False
    return _action_type(_action_data(action)) in _PASTE_ACTION_TYPES


def _step_index(state: Any) -> int | None:
    for holder in (state, getattr(state, "runtime", None)):
        value = getattr(holder, "step_index", None)
        if isinstance(value, int):
            return value
    return None


def search_progress(state: Any) -> dict[str, Any] | None:
    """Return the persisted search stage, if any."""

    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    data = metadata.get(SEARCH_PROGRESS_KEY)
    return dict(data) if isinstance(data, Mapping) else None


def clear_search_progress(state: Any) -> None:
    """Drop the persisted search stage (e.g. when a new task starts)."""

    metadata = getattr(state, "metadata", None)
    if isinstance(metadata, dict):
        metadata.pop(SEARCH_PROGRESS_KEY, None)


def record_search_phase(
    state: Any,
    phase: str,
    *,
    evidence: Iterable[str] | None = None,
    note: str | None = None,
    leg: str | None = None,
) -> dict[str, Any] | None:
    """Persist *phase* so later planner turns keep the confirmed progress.

    Scheme B has two legs (navigate then search). Phases never move backwards
    inside a leg. ``nav_submitted → input_focused`` starts the search leg.
    ``search_submitted`` is terminal for the workflow.
    """

    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, dict) or phase not in _PHASE_RANK:
        return None

    current = search_progress(state)
    current_phase = str(current.get("phase")) if current else ""
    current_leg = str(current.get("leg") or LEG_NAVIGATE) if current else LEG_NAVIGATE

    if current_phase == PHASE_SEARCH_SUBMITTED and phase != PHASE_SEARCH_SUBMITTED:
        return current
    if current_phase == PHASE_HOMEPAGE_REACHED and phase != PHASE_HOMEPAGE_REACHED:
        return current

    if current_phase == PHASE_NAV_SUBMITTED and phase == PHASE_INPUT_FOCUSED:
        resolved_leg = LEG_SEARCH
    elif current_phase == PHASE_NAV_SUBMITTED and phase == PHASE_HOMEPAGE_REACHED:
        resolved_leg = LEG_NAVIGATE
    elif current_phase == PHASE_NAV_SUBMITTED and phase != PHASE_NAV_SUBMITTED:
        # Only homepage success or a confirmed search-box focus may leave nav_submitted.
        return current
    elif (
        current_phase
        and current_phase not in _TERMINAL_PHASES
        and current_phase != PHASE_NAV_SUBMITTED
        and _PHASE_RANK.get(phase, 0) < _PHASE_RANK.get(current_phase, 0)
    ):
        return current
    else:
        resolved_leg = leg or current_leg or LEG_NAVIGATE
        if phase == PHASE_INPUT_FOCUSED and not current:
            resolved_leg = LEG_NAVIGATE
        if phase == PHASE_NAV_SUBMITTED:
            resolved_leg = LEG_NAVIGATE
        if phase == PHASE_HOMEPAGE_REACHED:
            resolved_leg = LEG_NAVIGATE
        if phase == PHASE_SEARCH_SUBMITTED:
            resolved_leg = LEG_SEARCH

    next_action, forbidden = _guidance_for(phase, resolved_leg)
    payload: dict[str, Any] = {
        "phase": phase,
        "leg": resolved_leg,
        "next_action": next_action,
        "must_not_repeat": list(forbidden),
    }
    step = _step_index(state)
    if step is not None:
        payload["confirmed_at_step"] = step
    if evidence:
        payload["evidence"] = [str(item) for item in evidence][:4]
    if note:
        payload["note"] = note

    metadata[SEARCH_PROGRESS_KEY] = payload
    return payload


def record_phase_for_executed_action(
    state: Any,
    action: Any,
) -> dict[str, Any] | None:
    """Advance the persisted stage after an action executed successfully.

    Focus itself is confirmed by the verifier, not here. Paste/enter are only
    accepted from the matching prior phase so a stray paste after navigation
    cannot rewind the stage back into another paste→enter loop.
    """

    current = search_progress(state)
    current_phase = str(current.get("phase")) if current else ""
    current_leg = str(current.get("leg") or LEG_NAVIGATE) if current else LEG_NAVIGATE

    if is_query_input_action(action):
        if current_phase == PHASE_INPUT_FOCUSED:
            note = (
                "Task search query was pasted into the central search box."
                if current_leg == LEG_SEARCH
                else "Navigation word was pasted into the address bar."
            )
            return record_search_phase(
                state,
                PHASE_QUERY_ENTERED,
                note=note,
                leg=current_leg,
            )
        # Ignore paste while already entered, or after a submit — never rewind.
        return current

    if is_submit_action(action):
        if current_phase != PHASE_QUERY_ENTERED:
            return current
        if current_leg == LEG_SEARCH:
            return record_search_phase(
                state,
                PHASE_SEARCH_SUBMITTED,
                note="Enter was pressed to submit the search query.",
                leg=LEG_SEARCH,
            )
        return record_search_phase(
            state,
            PHASE_NAV_SUBMITTED,
            note="Enter was pressed to open the Google homepage.",
            leg=LEG_NAVIGATE,
        )

    if is_address_bar_focus_action(action):
        return current

    if current and current_phase in _FOCUS_SENSITIVE_PHASES:
        if _action_type(_action_data(action)) in _FOCUS_BREAKING_ACTION_TYPES:
            # A click after nav_submitted is the intended next step (central box);
            # only drop focus-sensitive phases when focus is actually at risk.
            clear_search_progress(state)
    return None


def advance_search_phase_after_verify(
    state: Any,
    action: Any,
    verify_data: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Advance nav_submitted → search input_focused after a successful focus click.

    Address-bar focus is handled by ``record_search_phase(PHASE_INPUT_FOCUSED)``
    from the focus detector. The central Google box sits outside the top chrome
    band, so a normal successful click verify must still start the search leg.
    """

    data = dict(verify_data or {})
    if not (
        bool(data.get("action_effective"))
        and str(data.get("status") or "").lower() == "success"
    ):
        return None

    current = search_progress(state)
    if not current or str(current.get("phase")) != PHASE_NAV_SUBMITTED:
        return None

    if is_address_bar_focus_action(action):
        return None

    action_type = _action_type(_action_data(action))
    if action_type not in CLICK_ACTION_TYPES:
        return None

    return record_search_phase(
        state,
        PHASE_INPUT_FOCUSED,
        note="Central search box focus confirmed after Google navigation.",
        leg=LEG_SEARCH,
        evidence=list(data.get("evidence") or [])[:4] or None,
    )


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

    width, height = screen_size(observation)
    boxes = iter_labeled_boxes(observation)
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


def looks_like_bing_results(observation: Any) -> bool:
    """True when the page is Bing/Edge search results rather than Google.com."""

    texts = normalised_texts(observation)
    if not texts:
        return False
    blob = " ".join(texts)
    return any(marker in blob for marker in _BING_PAGE_MARKERS)


def detect_google_homepage(observation: Any) -> tuple[bool, list[str]]:
    """Detect the Google homepage: brand logo + a central Google search box.

    Rejects Bing/Edge SERP pages that merely contain the keyword "Google".
    """

    if observation is None:
        return False, ["No observation available."]

    if looks_like_bing_results(observation):
        return False, [
            "Bing/Edge search-results chrome detected; not the Google homepage."
        ]

    width, height = screen_size(observation)
    boxes = iter_labeled_boxes(observation)
    if not boxes:
        return False, ["No OCR/UI boxes to judge the Google homepage."]

    logo_hits: list[str] = []
    for text, bbox in boxes:
        normalised = normalise_target_text(text)
        if normalised not in _GOOGLE_BRAND_MARKERS:
            continue
        _left, top, _right, bottom = bbox
        mid_y = (top + bottom) / 2.0
        # Homepage logo sits in the upper-middle band, not the tiny tab strip.
        if height * 0.12 <= mid_y <= height * 0.55:
            logo_hits.append(text or normalised)

    if not logo_hits:
        return False, ["Google brand/logo not found in the homepage band."]

    search_hits: list[str] = []
    for text, bbox in boxes:
        left, top, right, bottom = bbox
        box_w = right - left
        box_h = bottom - top
        mid_y = (top + bottom) / 2.0
        # Central Google search box (not the top browser address bar).
        if not (height * 0.28 <= mid_y <= height * 0.70):
            continue
        if box_w < width * 0.22 or box_h > max(120.0, height * 0.12):
            continue
        if top <= height * 0.12:
            continue
        search_hits.append(text or "<google-search-box>")

    if not search_hits:
        return False, [
            "Google logo seen but no central Google search box "
            "(page may still be loading, or this is a SERP)."
        ]

    evidence = [
        f"Google homepage: logo={logo_hits[:3]!r}, "
        f"central_search={search_hits[:3]!r}."
    ]
    return True, evidence


def build_homepage_success_verify(
    *,
    evidence: Iterable[str],
) -> dict[str, Any]:
    """Build a verifier payload that completes a Google website-open task."""

    notes = list(evidence)
    return {
        "status": "success",
        "action_effective": True,
        "task_complete": True,
        "evidence": notes,
        "reason": (
            "Google homepage reached: Google logo and the central Google "
            "search box are both visible."
        ),
        "confidence": 0.92,
        "recommended_next": "finish",
    }


def apply_homepage_verify_override(
    state: Any,
    verify_data: Mapping[str, Any],
    *,
    after: Any,
) -> tuple[dict[str, Any], bool]:
    """Mark the website-open task complete when the Google homepage is visible.

    Also blocks a false success on Bing/Edge keyword results for 'Google'.
    """

    data = dict(verify_data)
    detected, evidence = detect_google_homepage(after)
    if detected:
        success = build_homepage_success_verify(evidence=evidence)
        merged = dict(data)
        merged.update(success)
        prior = list(data.get("evidence") or [])
        if prior:
            merged["evidence"] = prior + list(success["evidence"])
        record_search_phase(
            state,
            PHASE_HOMEPAGE_REACHED,
            evidence=evidence,
            note="Google homepage confirmed (logo + central search box).",
            leg=LEG_NAVIGATE,
        )
        return merged, True

    if looks_like_bing_results(after):
        # Never treat Bing SERP as task completion for a Google website open.
        if bool(data.get("task_complete")):
            data["task_complete"] = False
            data["recommended_next"] = "continue"
            notes = list(data.get("evidence") or [])
            notes.append(
                "Rejected Bing/Edge search-results page as Google homepage."
            )
            data["evidence"] = notes
            data["reason"] = (
                str(data.get("reason") or "")
                + " Page looks like Bing/Edge results, not google.com."
            ).strip()
            return data, True
    return data, False


def focus_evidence_present(
    before: Any,
    after: Any,
    *,
    min_delta: int = _FOCUS_ELEMENT_DELTA,
) -> bool:
    """Detect a focus dropdown via a sharp rise in detected UI/OCR items."""

    before_elements, before_ocr = observation_counts(before)
    after_elements, after_ocr = observation_counts(after)
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

    before_elements, before_ocr = observation_counts(before)
    after_elements, after_ocr = observation_counts(after)
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
    "LEG_NAVIGATE",
    "LEG_SEARCH",
    "PHASE_HOMEPAGE_REACHED",
    "PHASE_INPUT_FOCUSED",
    "PHASE_NAV_SUBMITTED",
    "PHASE_QUERY_ENTERED",
    "PHASE_QUERY_SUBMITTED",
    "PHASE_SEARCH_SUBMITTED",
    "SEARCH_PROGRESS_KEY",
    "advance_search_phase_after_verify",
    "apply_focus_verify_override",
    "apply_homepage_verify_override",
    "build_focus_success_verify",
    "build_homepage_success_verify",
    "clear_search_progress",
    "detect_google_homepage",
    "detect_search_box_focus",
    "focus_evidence_present",
    "is_address_bar_focus_action",
    "is_address_bar_focus_target",
    "is_forbidden_suggestion_target",
    "is_query_input_action",
    "is_submit_action",
    "looks_like_bing_results",
    "maybe_rewrite_address_bar_click",
    "record_phase_for_executed_action",
    "record_search_phase",
    "search_progress",
]
