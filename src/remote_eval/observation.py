"""Build AgentState / ObservationState from remote client payloads."""

from __future__ import annotations

import base64
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.agent.state import AgentState, ObservationSource, ObservationState


def decode_image_b64(image_b64: str, *, suffix: str = ".png") -> Path:
    """Decode base64 image to a temp file; caller may keep the path for VLM."""

    raw = image_b64.strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    data = base64.b64decode(raw)
    path = Path(tempfile.gettempdir()) / f"gui_remote_{uuid.uuid4().hex}{suffix}"
    path.write_bytes(data)
    return path


def observation_from_remote(
    *,
    image_path: str | Path,
    screen_width: int,
    screen_height: int,
    ocr_text: str = "",
    gui_elements: Sequence[Mapping[str, Any]] | None = None,
    window_title: str | None = None,
    application_name: str | None = None,
) -> ObservationState:
    elements = [dict(item) for item in (gui_elements or [])]
    return ObservationState(
        screenshot_path=str(image_path),
        screen_width=int(screen_width) or None,
        screen_height=int(screen_height) or None,
        window_title=window_title,
        application_name=application_name,
        ocr_text=ocr_text or "",
        gui_elements=elements,
        source=ObservationSource.MANUAL,
        metadata={"remote_eval": True},
    )


def state_from_remote_plan(
    *,
    task: str,
    image_path: str | Path,
    screen_width: int,
    screen_height: int,
    ocr_text: str = "",
    gui_elements: Sequence[Mapping[str, Any]] | None = None,
    window_title: str | None = None,
    application_name: str | None = None,
    max_steps: int = 12,
    step_index: int = 0,
) -> AgentState:
    state = AgentState.create(task=task, max_steps=max_steps)
    state.begin()
    if step_index > 0:
        state.step_index = int(step_index)
    obs = observation_from_remote(
        image_path=image_path,
        screen_width=screen_width,
        screen_height=screen_height,
        ocr_text=ocr_text,
        gui_elements=gui_elements,
        window_title=window_title,
        application_name=application_name,
    )
    state.update_observation(obs)
    return state


def action_to_executor_dict(action: Any) -> dict[str, Any] | None:
    """Normalize planner Action / mapping into Executor-compatible flat dict."""

    if action is None:
        return None
    if isinstance(action, Mapping):
        data = dict(action)
    elif hasattr(action, "to_dict"):
        data = dict(action.to_dict())
    elif hasattr(action, "model_dump"):
        data = dict(action.model_dump())
    else:
        data = {
            "type": getattr(action, "type", None)
            or getattr(action, "action_type", None),
            "parameters": getattr(action, "parameters", None),
            "metadata": getattr(action, "metadata", None),
        }
    # Flatten planner-style {"type","parameters":{...}}
    params = data.get("parameters")
    if isinstance(params, Mapping):
        flat = {**params}
        for key in ("type", "action_type", "metadata", "description"):
            if key in data and data[key] is not None:
                flat[key] = data[key]
        if "type" not in flat and data.get("type"):
            flat["type"] = data["type"]
        data = flat
    if "type" not in data and data.get("action_type"):
        data["type"] = data["action_type"]
    # Drop Nones
    return {k: v for k, v in data.items() if v is not None}
