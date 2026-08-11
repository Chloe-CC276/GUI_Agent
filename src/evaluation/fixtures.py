"""Task list + offline case loading; build AgentState for planner-only runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.agent.state import AgentState, ObservationSource, ObservationState


def load_tasks(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks = data.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError(f"tasks.yaml must contain a list under 'tasks': {path}")
    return tasks


def load_offline_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def _element(
    *,
    element_id: str,
    text: str,
    bbox: list[float],
    control_type: str = "Button",
) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    return {
        "element_id": element_id,
        "text": text,
        "name": text,
        "label": text,
        "control_type": control_type,
        "bbox": bbox,
        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
        "x": (x1 + x2) / 2,
        "y": (y1 + y2) / 2,
    }


def state_from_offline_case(case: dict[str, Any]) -> AgentState:
    """Build a PLANNING-ready AgentState from an offline fixture case."""

    instruction = str(case.get("instruction") or case.get("task_id") or "task")
    state = AgentState.create(task=instruction)
    state.begin()
    elements = case.get("gui_elements")
    if not elements and case.get("target_bbox") and case.get("expected_target_text"):
        elements = [
            _element(
                element_id="e0",
                text=str(case["expected_target_text"]),
                bbox=list(case["target_bbox"]),
            )
        ]
    # Optional distractors for close-document interference tests
    for idx, item in enumerate(case.get("distractor_elements") or []):
        elements = list(elements or [])
        elements.append(
            _element(
                element_id=f"d{idx}",
                text=str(item.get("text") or item),
                bbox=list(item.get("bbox") or [0, 0, 10, 10]),
                control_type=str(item.get("control_type") or "Text"),
            )
        )

    width = int(case.get("screen_width") or 1280)
    height = int(case.get("screen_height") or 720)
    obs = ObservationState(
        screenshot_path=case.get("image_path"),
        screen_width=width,
        screen_height=height,
        window_title=case.get("window_title"),
        application_name=case.get("application_name"),
        ocr_text=case.get("ocr_text") or "",
        ocr_items=case.get("ocr_items") or [],
        gui_elements=elements or [],
        source=ObservationSource.MANUAL,
        metadata={"case_id": case.get("case_id"), "fixture": True},
    )
    state.update_observation(obs)
    return state
