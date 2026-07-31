"""
chat_send
Deterministic ChatGPT (and similar) chat-compose workflow.

Caret pixels are unreliable for OCR, so focus is never judged by a visible
cursor. The pipeline is:

    click chat input → paste_text(message) → verify pasted text in the box
    → press enter → detect a thinking / generating marker → finish

State lives in ``state.metadata["chat_progress"]`` so the planner cannot rewind
to another input click after focus was accepted.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Any

from src.common.target_validation import (
    CLICK_ACTION_TYPES,
    coerce_action_mapping,
    normalise_target_text,
)

from .document_tasks import task_instruction
from .observation_utils import iter_labeled_boxes, iter_texts, screen_size
from .verify_policy import latest_action_type

logger = logging.getLogger(__name__)

CHAT_PROGRESS_KEY: str = "chat_progress"

PHASE_IDLE: str = "idle"
PHASE_INPUT_FOCUSED: str = "input_focused"
PHASE_TEXT_PASTED: str = "text_pasted"
PHASE_TEXT_VERIFIED: str = "text_verified"
PHASE_SUBMITTED: str = "submitted"
PHASE_THINKING: str = "thinking"

_PHASE_RANK: dict[str, int] = {
    PHASE_IDLE: 0,
    PHASE_INPUT_FOCUSED: 1,
    PHASE_TEXT_PASTED: 2,
    PHASE_TEXT_VERIFIED: 3,
    PHASE_SUBMITTED: 4,
    PHASE_THINKING: 5,
}

_INPUT_PLACEHOLDERS: tuple[str, ...] = (
    "问问chatgpt",
    "问问 chatgpt",
    "askanything",
    "ask chatgpt",
    "messagechatgpt",
    "message chatgpt",
    "sendamessage",
    "给chatgpt发送消息",
    "给 chatgpt 发送消息",
    "对chatgpt说点什么",
)

_THINKING_MARKERS: tuple[str, ...] = (
    "正在思考",
    "思考中",
    "推理中",
    "thinking",
    "reasoning",
    "stopgenerating",
    "停止生成",
    "停止",
    "stop",
)

_PASTE_ACTION_TYPES: frozenset[str] = frozenset(
    {"paste_text", "paste", "type_text", "type", "write"}
)
_SUBMIT_ACTION_TYPES: frozenset[str] = frozenset(
    {"press", "press_key", "key", "hotkey"}
)

# Quoted payload: 「…」 “…” "…" '…'
_MESSAGE_PATTERN = re.compile(
    r"[「“\"']([^」”\"']+)[」”\"']"
)

_TOKEN_PATTERN = re.compile(r"[0-9a-zA-Z]+|[\u4e00-\u9fff]+")
_FINGERPRINT_MIN: int = 4
_FINGERPRINT_MAX: int = 12
_REGION_PAD_X: int = 48
_REGION_PAD_Y: int = 96


def is_chat_send_task(instruction: Any) -> bool:
    """True when the task asks to type/send into a ChatGPT-style chat box."""

    text = str(instruction or "").strip()
    if not text:
        return False
    lowered = text.casefold()
    has_app = any(
        marker in lowered for marker in ("chatgpt", "chat gpt", "聊天框", "对话框")
    )
    if not has_app:
        return False
    if extract_chat_message(text):
        return True
    normalised = normalise_target_text(text)
    return any(
        marker in normalised
        for marker in ("输入", "发送", "发消息", "paste", "type", "send", "message")
    )


def extract_chat_message(instruction: Any) -> str | None:
    """Pull the quoted message body out of the task instruction."""

    text = str(instruction or "").strip()
    if not text:
        return None
    match = _MESSAGE_PATTERN.search(text)
    if match:
        message = match.group(1).strip()
        return message or None
    # Fallback: text after the last 输入/发送.
    for marker in ("输入", "发送", "type", "send", "message"):
        index = text.casefold().rfind(marker.casefold())
        if index < 0:
            continue
        tail = text[index + len(marker) :].strip(" ：:，,")
        tail = tail.strip("「」“”\"'")
        if len(tail) >= 2:
            return tail
    return None


def message_fingerprint(message: Any) -> str:
    """Compact normalised prefix used to OCR-confirm a paste landed."""

    normalised = normalise_target_text(message)
    if not normalised:
        return ""
    tokens = _TOKEN_PATTERN.findall(str(message or ""))
    if tokens:
        joined = "".join(normalise_target_text(token) for token in tokens)
        normalised = joined or normalised
    if len(normalised) <= _FINGERPRINT_MAX:
        return normalised
    return normalised[:_FINGERPRINT_MAX]


def is_chat_input_target(text: Any) -> bool:
    normalised = normalise_target_text(text)
    if not normalised:
        return False
    if normalised.startswith("+"):
        normalised = normalised[1:]
    placeholders = [normalise_target_text(item) for item in _INPUT_PLACEHOLDERS]
    return any(
        marker and (marker in normalised or normalised in marker)
        for marker in placeholders
    )


def chat_progress(state: Any) -> dict[str, Any] | None:
    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    data = metadata.get(CHAT_PROGRESS_KEY)
    return dict(data) if isinstance(data, Mapping) else None


def _guidance_for(phase: str, message: str) -> str:
    preview = message if len(message) <= 24 else message[:24] + "…"
    if phase in {PHASE_IDLE, ""}:
        return (
            "Click the ChatGPT composer placeholder "
            "(问问 ChatGPT / Ask anything / Message ChatGPT); "
            "do not paste yet."
        )
    if phase == PHASE_INPUT_FOCUSED:
        return (
            f"Composer is focused. paste_text the exact message {preview!r} now. "
            "Do not click the composer again."
        )
    if phase == PHASE_TEXT_PASTED:
        return (
            "Message was pasted; wait for OCR to confirm the text is inside "
            "the composer, then press enter."
        )
    if phase == PHASE_TEXT_VERIFIED:
        return (
            "Pasted text is confirmed in the composer. Press key=enter to send. "
            "Do not paste again."
        )
    if phase == PHASE_SUBMITTED:
        return (
            "Enter was pressed. Re-observe for a thinking / generating marker "
            "(正在思考 / Thinking / 停止生成 / Stop)."
        )
    if phase == PHASE_THINKING:
        return "ChatGPT is thinking/generating. The send task is complete — finish."
    return ""


def ensure_chat_progress(state: Any) -> dict[str, Any] | None:
    """Create or refresh the chat-send stage for a compose task."""

    instruction = task_instruction(state)
    if not is_chat_send_task(instruction):
        return None
    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, MutableMapping):
        return None

    current = chat_progress(state) or {}
    message = str(current.get("message") or extract_chat_message(instruction) or "")
    phase = str(current.get("phase") or PHASE_IDLE)
    if phase not in _PHASE_RANK:
        phase = PHASE_IDLE
    fingerprint = str(current.get("fingerprint") or message_fingerprint(message))
    progress = {
        "phase": phase,
        "message": message,
        "fingerprint": fingerprint,
        "next_action": _guidance_for(phase, message),
    }
    for key in ("input_bbox", "input_target", "evidence"):
        if key in current:
            progress[key] = current[key]
    metadata[CHAT_PROGRESS_KEY] = progress
    return progress


def record_chat_phase(
    state: Any,
    phase: str,
    *,
    note: str | None = None,
    input_bbox: Sequence[Any] | None = None,
    input_target: str | None = None,
) -> dict[str, Any] | None:
    """Advance chat_progress, never rewinding to an earlier phase."""

    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, MutableMapping):
        return None
    current = ensure_chat_progress(state) or {}
    current_phase = str(current.get("phase") or PHASE_IDLE)
    if _PHASE_RANK.get(phase, -1) < _PHASE_RANK.get(current_phase, 0):
        return current

    message = str(current.get("message") or "")
    progress = {
        **current,
        "phase": phase,
        "next_action": _guidance_for(phase, message),
    }
    if input_bbox is not None and len(input_bbox) == 4:
        try:
            progress["input_bbox"] = [int(round(float(v))) for v in input_bbox]
        except (TypeError, ValueError):
            pass
    if input_target:
        progress["input_target"] = input_target
    if note:
        evidence = list(progress.get("evidence") or [])
        evidence.append(note)
        progress["evidence"] = evidence[-8:]
    metadata[CHAT_PROGRESS_KEY] = progress
    if current_phase != phase:
        logger.info(
            "chat_progress phase %s → %s | message=%r | note=%s",
            current_phase,
            phase,
            message,
            note or "",
        )
    return progress


def forced_chat_action(state: Any) -> tuple[str, dict[str, Any]] | None:
    """Return the deterministic next chat action for the current stage.

    Bypasses the VLM once the composer is focused or the pasted text is
    confirmed, so the agent cannot rewind into another click loop.
    """

    instruction = task_instruction(state)
    if not is_chat_send_task(instruction):
        return None
    progress = ensure_chat_progress(state) or {}
    phase = str(progress.get("phase") or PHASE_IDLE)
    message = str(progress.get("message") or "").strip()

    if phase == PHASE_INPUT_FOCUSED:
        if not message:
            logger.warning(
                "chat_progress input_focused but message is empty; "
                "cannot force paste_text."
            )
            return None
        logger.info(
            "chat_progress forcing paste_text | phase=%s | message=%r",
            phase,
            message,
        )
        return (
            "paste_text",
            {
                "text": message,
                "description": (
                    f"Paste the ChatGPT message {message!r} into the focused "
                    "composer."
                ),
            },
        )

    if phase == PHASE_TEXT_VERIFIED:
        logger.info("chat_progress forcing press enter | phase=%s", phase)
        return (
            "press",
            {
                "key": "enter",
                "description": "Send the ChatGPT message with Enter.",
            },
        )

    if phase == PHASE_SUBMITTED:
        logger.info("chat_progress forcing wait before thinking check | phase=%s", phase)
        return (
            "wait",
            {
                "duration": 1.0,
                "description": (
                    "Wait briefly for ChatGPT thinking / generating markers."
                ),
            },
        )

    return None


def should_finish_chat_task(state: Any) -> bool:
    """True when the chat-send stage already reached a thinking marker."""

    if not is_chat_send_task(task_instruction(state)):
        return False
    progress = chat_progress(state) or {}
    return str(progress.get("phase") or "") == PHASE_THINKING


def maybe_rewrite_chat_action(
    action_type: str,
    parameters: Mapping[str, Any],
    *,
    state: Any,
) -> tuple[str, dict[str, Any], bool]:
    """Rewrite illegal chat clicks into the stage's required action."""

    params = dict(parameters)
    normalized = str(action_type or "").strip().lower()
    instruction = task_instruction(state)
    if not is_chat_send_task(instruction):
        return normalized, params, False

    progress = ensure_chat_progress(state) or {}
    phase = str(progress.get("phase") or PHASE_IDLE)
    message = str(progress.get("message") or "").strip()
    target = str(params.get("target_text") or params.get("text") or "")

    if normalized in CLICK_ACTION_TYPES and (
        is_chat_input_target(target)
        or (
            progress.get("input_target")
            and normalise_target_text(target)
            == normalise_target_text(progress.get("input_target"))
        )
    ):
        if phase == PHASE_INPUT_FOCUSED and message:
            logger.info(
                "Rewrote chat composer re-click to paste_text | target=%r",
                target,
            )
            return (
                "paste_text",
                {
                    "text": message,
                    "description": (
                        f"Paste {message!r} (rewritten from click {target!r})."
                    ),
                },
                True,
            )
        if phase == PHASE_TEXT_VERIFIED:
            logger.info(
                "Rewrote chat composer re-click to press enter | target=%r",
                target,
            )
            return (
                "press",
                {
                    "key": "enter",
                    "description": (
                        f"Send with Enter (rewritten from click {target!r})."
                    ),
                },
                True,
            )
        if _PHASE_RANK.get(phase, 0) >= _PHASE_RANK[PHASE_SUBMITTED]:
            logger.info(
                "Rewrote post-submit chat click to wait | target=%r",
                target,
            )
            return (
                "wait",
                {
                    "duration": 1.0,
                    "description": (
                        "Wait for thinking markers (rewritten from post-submit "
                        f"click {target!r})."
                    ),
                },
                True,
            )

    forced = forced_chat_action(state)
    if forced is None:
        return normalized, params, False
    forced_type, forced_params = forced
    # If the model already planned the correct action, keep it.
    if normalized == forced_type:
        if forced_type == "paste_text" and str(params.get("text") or "").strip() == message:
            return normalized, params, False
        if forced_type == "press" and str(params.get("key") or "").casefold() in {
            "enter",
            "return",
        }:
            return normalized, params, False
        if forced_type == "wait":
            return normalized, params, False
    if phase in {PHASE_INPUT_FOCUSED, PHASE_TEXT_VERIFIED, PHASE_SUBMITTED}:
        # Any other action at these stages is replaced by the forced step.
        logger.info(
            "Rewrote chat action %s → %s | phase=%s",
            normalized,
            forced_type,
            phase,
        )
        return forced_type, forced_params, True
    return normalized, params, False


def _action_data(action: Any) -> dict[str, Any]:
    """Flatten Action / nested parameters / planner target_validation evidence.

    Planner stores click evidence under ``metadata.target_validation`` and
    strips top-level ``target_text`` / ``element_id`` before building the
    Action. Callers that only read ``target_text`` or ``bbox`` therefore miss
    the composer label — flatten the validation block so chat helpers share
    the same field names as ``validate_click_target``.
    """

    data = coerce_action_mapping(action)
    nested = data.get("parameters")
    if isinstance(nested, Mapping):
        data = {**data, **dict(nested)}
    metadata = data.get("metadata")
    if not isinstance(metadata, Mapping):
        return data
    validation = metadata.get("target_validation")
    if not isinstance(validation, Mapping):
        return data
    if not data.get("target_text") and validation.get("target_text"):
        data["target_text"] = validation["target_text"]
    if not data.get("matched_text") and validation.get("matched_text"):
        data["matched_text"] = validation["matched_text"]
    if data.get("element_id") is None and validation.get("element_id") is not None:
        data["element_id"] = validation["element_id"]
    if not data.get("bbox") and not data.get("bounding_box"):
        bbox = validation.get("matched_bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            data["bbox"] = list(bbox)
    return data


def register_chat_composer_target(
    state: Any,
    *,
    target_text: str,
    bbox: Sequence[Any] | None = None,
) -> dict[str, Any] | None:
    """Stash composer label/bbox at plan time, before Action drops them.

    Phase stays ``idle`` until execution succeeds; only the identity of the
    planned focus click is remembered so execute / skip_verify / narrow
    observe can recover it even if Action fields are stripped.
    """

    instruction = task_instruction(state)
    if not is_chat_send_task(instruction):
        return None
    label = str(target_text or "").strip()
    if not label or not is_chat_input_target(label):
        return None
    progress = ensure_chat_progress(state) or {}
    phase = str(progress.get("phase") or PHASE_IDLE)
    if _PHASE_RANK.get(phase, 0) > _PHASE_RANK[PHASE_IDLE]:
        return progress

    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, MutableMapping):
        return None
    updated = {
        **progress,
        "input_target": label,
        "next_action": _guidance_for(phase, str(progress.get("message") or "")),
    }
    if bbox is not None and len(bbox) == 4:
        try:
            updated["input_bbox"] = [int(round(float(v))) for v in bbox]
        except (TypeError, ValueError):
            pass
    evidence = list(updated.get("evidence") or [])
    note = f"Planner selected chat composer {label!r}."
    if note not in evidence:
        evidence.append(note)
        updated["evidence"] = evidence[-8:]
    metadata[CHAT_PROGRESS_KEY] = updated
    logger.info(
        "chat_progress registered composer target=%r bbox=%s | phase=%s",
        label,
        updated.get("input_bbox"),
        phase,
    )
    return updated


def is_chat_input_focus_action(action: Any, state: Any | None = None) -> bool:
    if latest_action_type(action) not in CLICK_ACTION_TYPES:
        return False
    data = _action_data(action)
    target = (
        data.get("target_text")
        or data.get("matched_text")
        or data.get("text")
        or ""
    )
    if is_chat_input_target(target):
        return True
    if state is None:
        return False
    progress = chat_progress(state) or {}
    stored = str(progress.get("input_target") or "")
    if not stored:
        return False
    if target and normalise_target_text(target) == normalise_target_text(stored):
        return True
    # Planner registered the composer this turn but Action evidence was stripped;
    # treat any click whose coordinates fall inside the stored bbox as focus.
    if is_chat_input_target(stored) and _click_inside_input_bbox(data, progress):
        return True
    return False


def _click_inside_input_bbox(
    data: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> bool:
    bbox = progress.get("input_bbox")
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        return False
    x, y = data.get("x"), data.get("y")
    if x is None or y is None:
        return False
    try:
        left, top, right, bottom = (float(v) for v in bbox)
        xi, yi = float(x), float(y)
    except (TypeError, ValueError):
        return False
    return left <= xi <= right and top <= yi <= bottom


def is_chat_paste_action(action: Any) -> bool:
    return latest_action_type(action) in _PASTE_ACTION_TYPES


def is_chat_submit_action(action: Any) -> bool:
    action_type = latest_action_type(action)
    if action_type not in _SUBMIT_ACTION_TYPES:
        return False
    data = _action_data(action)
    if action_type == "hotkey":
        keys = [
            str(item).strip().casefold()
            for item in (data.get("keys") or [])
            if str(item).strip()
        ]
        return keys == ["enter"]
    key = str(data.get("key") or "").strip().casefold()
    return key in {"enter", "return"}


def record_chat_phase_for_executed_action(
    state: Any,
    action: Any,
) -> dict[str, Any] | None:
    """Advance the chat stage after a successful execution."""

    instruction = task_instruction(state)
    if not is_chat_send_task(instruction):
        return None
    progress = ensure_chat_progress(state) or {}
    phase = str(progress.get("phase") or PHASE_IDLE)

    if is_chat_input_focus_action(action, state):
        data = _action_data(action)
        bbox = data.get("bbox") or data.get("bounding_box")
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            stored_bbox = progress.get("input_bbox")
            if isinstance(stored_bbox, Sequence) and len(stored_bbox) == 4:
                bbox = stored_bbox
        target = str(
            data.get("target_text")
            or data.get("matched_text")
            or data.get("text")
            or progress.get("input_target")
            or ""
        )
        return record_chat_phase(
            state,
            PHASE_INPUT_FOCUSED,
            note=f"Clicked chat composer {target!r}.",
            input_bbox=bbox if isinstance(bbox, Sequence) else None,
            input_target=target or None,
        )

    if is_chat_paste_action(action):
        if phase in {PHASE_INPUT_FOCUSED, PHASE_TEXT_PASTED}:
            data = _action_data(action)
            text = str(data.get("text") or "")
            return record_chat_phase(
                state,
                PHASE_TEXT_PASTED,
                note=f"Pasted chat message ({len(text)} chars).",
            )
        return progress

    if is_chat_submit_action(action):
        if phase in {PHASE_TEXT_VERIFIED, PHASE_TEXT_PASTED, PHASE_SUBMITTED}:
            return record_chat_phase(
                state,
                PHASE_SUBMITTED,
                note="Pressed enter to send the chat message.",
            )
        return progress

    return progress


def validate_chat_action(
    action_type: str,
    parameters: Mapping[str, Any],
    state: Any,
) -> None:
    """Reject illegal chat-compose actions for the current phase."""

    instruction = task_instruction(state)
    if not is_chat_send_task(instruction):
        return
    progress = ensure_chat_progress(state) or {}
    phase = str(progress.get("phase") or PHASE_IDLE)
    message = str(progress.get("message") or "")
    normalized = str(action_type or "").strip().lower()
    params = dict(parameters)

    if normalized in CLICK_ACTION_TYPES:
        target = params.get("target_text") or params.get("text") or ""
        if is_chat_input_target(target) and _PHASE_RANK.get(phase, 0) >= _PHASE_RANK[
            PHASE_INPUT_FOCUSED
        ]:
            raise ValueError(
                "Chat composer is already focused; do not click it again. "
                f"paste_text the message {message!r} or press enter."
            )
        return

    if normalized in _PASTE_ACTION_TYPES:
        if _PHASE_RANK.get(phase, 0) < _PHASE_RANK[PHASE_INPUT_FOCUSED]:
            raise ValueError(
                "Chat composer is not focused yet; click the input placeholder first."
            )
        if _PHASE_RANK.get(phase, 0) >= _PHASE_RANK[PHASE_TEXT_VERIFIED]:
            raise ValueError(
                "Message text is already verified; press enter instead of pasting again."
            )
        if not message:
            raise ValueError(
                "Chat task does not contain a quoted message to paste."
            )
        text = str(params.get("text") or params.get("value") or "")
        if text.strip() != message.strip():
            raise ValueError(
                f"paste_text must use the exact task message {message!r}, "
                f"got {text!r}."
            )
        return

    if normalized in {"press", "press_key", "key"} or (
        normalized == "hotkey"
        and [
            str(item).strip().casefold()
            for item in (params.get("keys") or [])
            if str(item).strip()
        ]
        == ["enter"]
    ):
        if normalized != "hotkey":
            key = str(params.get("key") or "").strip().casefold()
            if key not in {"enter", "return"}:
                return
        if phase == PHASE_TEXT_VERIFIED:
            return
        if phase == PHASE_TEXT_PASTED:
            # Soft allow: verify may still promote to text_verified this turn.
            return
        if _PHASE_RANK.get(phase, 0) >= _PHASE_RANK[PHASE_SUBMITTED]:
            raise ValueError(
                "Enter was already sent; wait for a thinking marker or finish."
            )
        raise ValueError(
            "Do not press enter until the pasted message is OCR-confirmed "
            "inside the composer."
        )


def chat_observe_options(state: Any, action: Any) -> dict[str, Any] | None:
    """Return observe kwargs for a narrow composer band after paste/submit."""

    if not is_chat_send_task(task_instruction(state)):
        return None
    progress = chat_progress(state) or {}
    phase = str(progress.get("phase") or PHASE_IDLE)
    action_type = latest_action_type(action)
    useful = (
        is_chat_paste_action(action)
        or is_chat_submit_action(action)
        or (
            action_type == "wait"
            and phase in {PHASE_SUBMITTED, PHASE_TEXT_PASTED}
        )
    )
    if not useful:
        return None
    bbox = progress.get("input_bbox")
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = (int(round(float(v))) for v in bbox)
    except (TypeError, ValueError):
        return None
    width_s, height_s = screen_size(getattr(state, "observation", None))
    # Keep the band usable even when the current observation lacks screen
    # metrics (screen_size then returns 1x1 and would collapse the region).
    width_s = max(float(width_s), float(right) + _REGION_PAD_X, 1.0)
    height_s = max(float(height_s), float(bottom) + _REGION_PAD_Y, 1.0)
    # After paste: tight band around the composer. After submit/wait: taller
    # band so thinking markers above the box are still visible.
    pad_y = _REGION_PAD_Y
    if is_chat_submit_action(action) or (
        action_type == "wait" and phase == PHASE_SUBMITTED
    ):
        pad_y = max(int(height_s * 0.45), _REGION_PAD_Y * 3)
    region_left = max(0, left - _REGION_PAD_X)
    region_top = max(0, top - pad_y)
    region_right = min(int(width_s), right + _REGION_PAD_X)
    region_bottom = min(int(height_s), bottom + _REGION_PAD_Y)
    region_width = max(1, region_right - region_left)
    region_height = max(1, region_bottom - region_top)
    return {
        "region": [region_left, region_top, region_width, region_height],
    }


def detect_pasted_message(
    observation: Any,
    *,
    message: str,
    fingerprint: str | None = None,
    input_bbox: Sequence[Any] | None = None,
) -> tuple[bool, list[str]]:
    """True when the pasted message (or its fingerprint) is visible."""

    if observation is None or not message:
        return False, ["No observation or message."]
    needle = fingerprint or message_fingerprint(message)
    if not needle or len(needle) < _FINGERPRINT_MIN:
        needle = normalise_target_text(message)
    if not needle:
        return False, ["Message fingerprint is empty."]

    evidence: list[str] = []
    box: tuple[float, float, float, float] | None = None
    if isinstance(input_bbox, Sequence) and len(input_bbox) == 4:
        try:
            box = tuple(float(v) for v in input_bbox)  # type: ignore[assignment]
        except (TypeError, ValueError):
            box = None

    for text, bbox in iter_labeled_boxes(observation):
        normalised = normalise_target_text(text)
        if not normalised:
            continue
        if needle not in normalised and normalised not in needle:
            # Also accept when the full message appears.
            if normalise_target_text(message) not in normalised:
                continue
        if box is not None:
            mid_x = (bbox[0] + bbox[2]) / 2.0
            mid_y = (bbox[1] + bbox[3]) / 2.0
            # Allow a generous vertical band around the composer.
            if not (
                box[0] - _REGION_PAD_X <= mid_x <= box[2] + _REGION_PAD_X
                and box[1] - _REGION_PAD_Y * 2 <= mid_y <= box[3] + _REGION_PAD_Y * 2
            ):
                continue
        evidence.append(f"Composer OCR contains pasted text {text!r}.")
        break

    if not evidence:
        # Fallback: any OCR text on screen containing the fingerprint.
        corpus = " ".join(normalise_target_text(t) for t in iter_texts(observation))
        if needle in corpus or normalise_target_text(message) in corpus:
            evidence.append(
                f"Screen OCR contains message fingerprint {needle!r}."
            )

    if not evidence:
        return False, [f"Pasted fingerprint {needle!r} not found in OCR."]

    # Placeholder disappearing is supporting evidence, not required.
    placeholders_left = [
        text
        for text, _bbox in iter_labeled_boxes(observation)
        if is_chat_input_target(text)
    ]
    if not placeholders_left:
        evidence.append("Composer placeholder is no longer visible.")
    return True, evidence


def detect_thinking_marker(observation: Any) -> tuple[bool, list[str]]:
    """True when ChatGPT shows a thinking / stop-generating affordance."""

    if observation is None:
        return False, ["No observation."]
    evidence: list[str] = []
    for text, _bbox in iter_labeled_boxes(observation):
        normalised = normalise_target_text(text)
        if not normalised:
            continue
        for marker in _THINKING_MARKERS:
            marker_n = normalise_target_text(marker)
            if marker_n and marker_n in normalised:
                evidence.append(f"Thinking marker visible: {text!r}.")
                return True, evidence
    # Also scan raw joined OCR for short markers like Stop.
    corpus = normalise_target_text(
        " ".join(iter_texts(observation))
    )
    for marker in ("正在思考", "thinking", "停止生成", "stopgenerating", "reasoning"):
        marker_n = normalise_target_text(marker)
        if marker_n and marker_n in corpus:
            evidence.append(f"Thinking marker in OCR corpus: {marker!r}.")
            return True, evidence
    return False, ["No thinking / generating marker detected."]


def build_chat_text_verified_verify(*, evidence: Iterable[str]) -> dict[str, Any]:
    return {
        "status": "success",
        "action_effective": True,
        "task_complete": False,
        "evidence": list(evidence),
        "reason": (
            "The chat message is visible inside the composer; press enter next."
        ),
        "confidence": 0.9,
        "recommended_next": "continue",
    }


def build_chat_thinking_success_verify(*, evidence: Iterable[str]) -> dict[str, Any]:
    return {
        "status": "success",
        "action_effective": True,
        "task_complete": True,
        "evidence": list(evidence),
        "reason": (
            "ChatGPT shows a thinking/generating marker after the message was sent."
        ),
        "confidence": 0.92,
        "recommended_next": "finish",
    }


def apply_chat_verify_override(
    state: Any,
    verify_data: Mapping[str, Any],
    *,
    before: Any,
    after: Any,
) -> tuple[dict[str, Any], bool]:
    """Advance / complete a chat-send task from OCR evidence."""

    del before  # reserved for future before/after diffs
    data = dict(verify_data)
    instruction = task_instruction(state)
    if not is_chat_send_task(instruction):
        return data, False

    progress = ensure_chat_progress(state) or {}
    phase = str(progress.get("phase") or PHASE_IDLE)
    message = str(progress.get("message") or "")
    fingerprint = str(progress.get("fingerprint") or "")
    action = getattr(state, "latest_action", None)

    # After paste: confirm text landed, then mark text_verified.
    if phase == PHASE_TEXT_PASTED or (
        phase == PHASE_INPUT_FOCUSED and is_chat_paste_action(action)
    ):
        detected, evidence = detect_pasted_message(
            after,
            message=message,
            fingerprint=fingerprint,
            input_bbox=progress.get("input_bbox"),
        )
        if detected:
            record_chat_phase(
                state,
                PHASE_TEXT_VERIFIED,
                note="OCR confirmed pasted chat message.",
            )
            prior = list(data.get("evidence") or [])
            data.update(
                build_chat_text_verified_verify(evidence=prior + evidence)
            )
            return data, True
        return data, False

    # After enter / wait: look for thinking / generating chrome.
    if (
        phase == PHASE_SUBMITTED
        or is_chat_submit_action(action)
        or latest_action_type(action) == "wait"
    ):
        detected, evidence = detect_thinking_marker(after)
        if detected:
            record_chat_phase(
                state,
                PHASE_THINKING,
                note="Thinking marker detected after send.",
            )
            prior = list(data.get("evidence") or [])
            data.update(
                build_chat_thinking_success_verify(evidence=prior + evidence)
            )
            return data, True
        # Soft success: user bubble with the message appeared and placeholder
        # returned — covers instant replies where thinking is too brief.
        if message:
            pasted, paste_evidence = detect_pasted_message(
                after,
                message=message,
                fingerprint=fingerprint,
                input_bbox=None,
            )
            placeholders = [
                text
                for text, _bbox in iter_labeled_boxes(after)
                if is_chat_input_target(text)
            ]
            if pasted and placeholders:
                evidence = paste_evidence + [
                    "Composer placeholder returned after send.",
                ]
                record_chat_phase(
                    state,
                    PHASE_THINKING,
                    note="Message bubble present and composer cleared.",
                )
                prior = list(data.get("evidence") or [])
                data.update(
                    build_chat_thinking_success_verify(evidence=prior + evidence)
                )
                return data, True
        return data, False

    return data, False


def should_skip_chat_focus_verification(action: Any, state: Any) -> bool:
    """Skip verify after a chat-composer click; treat it as focus.

    Call this *before* ``record_chat_phase_for_executed_action`` advances the
    stage to ``input_focused``, otherwise the phase check always fails.
    """

    if not is_chat_send_task(task_instruction(state)):
        return False
    if not is_chat_input_focus_action(action, state):
        return False
    progress = chat_progress(state) or ensure_chat_progress(state) or {}
    phase = str(progress.get("phase") or PHASE_IDLE)
    # Only the first focus click skips; later stages must not re-click.
    if _PHASE_RANK.get(phase, 0) > _PHASE_RANK[PHASE_IDLE]:
        return False
    logger.info(
        "chat_progress skip_verify=True for composer focus click | phase=%s",
        phase,
    )
    return True


__all__ = [
    "CHAT_PROGRESS_KEY",
    "PHASE_IDLE",
    "PHASE_INPUT_FOCUSED",
    "PHASE_TEXT_PASTED",
    "PHASE_TEXT_VERIFIED",
    "PHASE_SUBMITTED",
    "PHASE_THINKING",
    "apply_chat_verify_override",
    "build_chat_text_verified_verify",
    "build_chat_thinking_success_verify",
    "chat_observe_options",
    "chat_progress",
    "detect_pasted_message",
    "detect_thinking_marker",
    "ensure_chat_progress",
    "extract_chat_message",
    "forced_chat_action",
    "is_chat_input_focus_action",
    "is_chat_input_target",
    "is_chat_paste_action",
    "is_chat_send_task",
    "is_chat_submit_action",
    "maybe_rewrite_chat_action",
    "message_fingerprint",
    "record_chat_phase",
    "record_chat_phase_for_executed_action",
    "register_chat_composer_target",
    "should_finish_chat_task",
    "should_skip_chat_focus_verification",
    "validate_chat_action",
]
