"""
prompts/context_builder
Build bounded, JSON-safe prompt context from GUI Agent runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .config import PromptConfig
from .formatters import (
    compact_whitespace,
    format_element,
    format_elements,
    format_error,
    format_history,
    safe_json_dumps,
    truncate_text,
)


_MISSING = object()


def _read(value: Any, *names: str, default: Any = None) -> Any:
    """Read the first available key or attribute from an arbitrary object."""

    if value is None:
        return default
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        item = getattr(value, name, _MISSING)
        if item is not _MISSING:
            return item
    return default


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _clean_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove empty values while retaining meaningful false and zero values."""

    return {
        str(key): item
        for key, item in value.items()
        if item is not None and item != "" and item != [] and item != {}
    }


def gui_element_to_dict(
    element: Any,
    *,
    index: int | None = None,
    config: PromptConfig | None = None,
) -> dict[str, Any]:
    """Convert one GUI element into the standard prompt element dictionary."""

    cfg = config or PromptConfig()
    return format_element(
        element,
        index=index,
        max_text_chars=cfg.max_element_text_chars,
        coordinate_precision=cfg.coordinate_precision,
    )


def compact_history(
    history: Sequence[Any] | Iterable[Any] | None,
    *,
    config: PromptConfig | None = None,
) -> str:
    """Return a bounded, chronological text summary of recent state events."""

    cfg = config or PromptConfig()
    if not cfg.include_history:
        return "(history omitted)"
    return format_history(
        history,
        limit=cfg.history_limit,
        max_item_chars=cfg.max_history_item_chars,
    )


def build_observation_context(
    observation: Any,
    *,
    config: PromptConfig | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Convert an ObservationState-like object into a compact context mapping."""

    cfg = config or PromptConfig()
    if observation is None:
        return {"available": False}

    width = _read(observation, "screen_width")
    height = _read(observation, "screen_height")
    screen_size = [width, height] if width is not None and height is not None else None

    context: dict[str, Any] = {
        "available": True,
        "observation_id": _read(observation, "observation_id", "id"),
        "label": label,
        "source": _enum_value(_read(observation, "source")),
        "captured_at": _read(observation, "captured_at", "timestamp"),
        "application_name": _read(observation, "application_name", "application"),
        "window_title": _read(observation, "window_title", "title"),
        "screen_size": screen_size,
        "cursor_position": _read(observation, "cursor_position", "cursor"),
    }

    if cfg.include_screenshot:
        path = _read(observation, "screenshot_path", "image_path")
        image = _read(observation, "screenshot", "image")
        context["screenshot"] = {
            "available": path is not None or image is not None,
            "path": path,
        }

    if cfg.include_ocr:
        ocr_text = _read(observation, "ocr_text", "text", default="")
        ocr_items = _read(observation, "ocr_items", "texts", default=[])
        context["ocr"] = {
            "text": truncate_text(ocr_text, cfg.max_ocr_chars, compact=True),
            "item_count": len(ocr_items) if hasattr(ocr_items, "__len__") else None,
        }

    if cfg.include_elements:
        elements = _read(observation, "gui_elements", "elements", "ui_elements", default=[])
        formatted = format_elements(
            elements,
            max_items=cfg.max_elements,
            max_text_chars=cfg.max_element_text_chars,
            coordinate_precision=cfg.coordinate_precision,
        )
        total = len(elements) if hasattr(elements, "__len__") else len(formatted)
        context["elements"] = formatted
        context["element_count"] = total
        context["elements_truncated"] = total > len(formatted)

    metadata = _read(observation, "metadata", default={}) or {}
    if isinstance(metadata, Mapping) and metadata.get("search_focus_detected"):
        context["search_focus_detected"] = True
        context["search_focus_evidence"] = metadata.get("search_focus_evidence")
        context["focus_next_action_hint"] = (
            "Address/search box is focused; next action should be paste_text then press enter."
        )

    return _clean_mapping(context)


def _task_context(task: Any) -> dict[str, Any]:
    if task is None:
        return {}
    if isinstance(task, str):
        return {"instruction": compact_whitespace(task)}
    return _clean_mapping(
        {
            "task_id": _read(task, "task_id", "id"),
            "instruction": compact_whitespace(_read(task, "instruction", "task", default="")),
            "source": _read(task, "source"),
            "language": _read(task, "language"),
            "current_subgoal": _read(task, "subgoal", "current_subgoal"),
            "completed_subgoals": _read(task, "completed_subgoals", default=[]),
            "constraints": _read(task, "constraints", default=[]),
            "success_criteria": _read(task, "success_criteria", default=[]),
        }
    )


def _result_context(result: Any, *, config: PromptConfig) -> dict[str, Any] | None:
    if result is None:
        return None
    data = _clean_mapping(
        {
            "status": _enum_value(_read(result, "status")),
            "decision": _enum_value(_read(result, "decision")),
            "tool_name": _read(result, "tool_name"),
            "action": _read(result, "action"),
            "message": _read(result, "message", "reason", "finish_message"),
            "output": _read(result, "output"),
            "confidence": _read(result, "confidence"),
        }
    )
    error = _read(result, "error")
    if error is not None:
        data["error"] = format_error(error, max_chars=config.max_error_chars)
    return data


def build_agent_context(
    state: Any,
    *,
    config: PromptConfig | None = None,
    memory: Any = None,
) -> dict[str, Any]:
    """Convert an AgentState-like object into the complete planner context."""

    cfg = config or PromptConfig()
    if state is None:
        raise ValueError("state must not be None")

    runtime = _read(state, "runtime")
    current = _read(state, "observation", "current_observation")
    previous = _read(state, "previous_observation")
    task = _read(state, "task")

    context: dict[str, Any] = {
        "run": _clean_mapping(
            {
                "run_id": _read(state, "run_id"),
                "state_id": _read(state, "state_id"),
                "phase": _enum_value(_read(state, "phase")),
                "step_index": _read(runtime, "step_index", default=_read(state, "step_index")),
                "max_steps": _read(runtime, "max_steps"),
                "remaining_steps": _read(runtime, "remaining_steps"),
                "retry_count": _read(runtime, "retry_count"),
                "max_retries": _read(runtime, "max_retries"),
                "consecutive_failures": _read(runtime, "consecutive_failures"),
                "repeated_action_count": _read(runtime, "repeated_action_count"),
            }
        ),
        "task": _task_context(task),
        "observation": build_observation_context(current, config=cfg, label="current"),
    }

    if cfg.include_previous_observation:
        context["previous_observation"] = build_observation_context(
            previous, config=cfg, label="previous"
        )
    if cfg.include_history:
        context["history"] = compact_history(_read(state, "history", "completed_steps"), config=cfg)
    if cfg.include_execution_result:
        context["last_planner_result"] = _result_context(
            _read(state, "last_planner_result"), config=cfg
        )
        context["last_execution_result"] = _result_context(
            _read(state, "last_execution_result"), config=cfg
        )
        context["last_verification_result"] = _result_context(
            _read(state, "last_verification_result"), config=cfg
        )
    if cfg.include_memory:
        resolved_memory = memory
        if resolved_memory is None:
            resolved_memory = _read(state, "memory")
        if resolved_memory is None:
            metadata = _read(state, "metadata", default={})
            resolved_memory = _read(metadata, "memory")
        if resolved_memory:
            text = (
                resolved_memory
                if isinstance(resolved_memory, str)
                else safe_json_dumps(resolved_memory, indent=None, ensure_ascii=cfg.ensure_ascii)
            )
            context["memory"] = truncate_text(text, cfg.max_memory_chars)

    state_error = _read(state, "error")
    if state_error is not None:
        context["error"] = format_error(state_error, max_chars=cfg.max_error_chars)
    return _clean_mapping(context)


@dataclass(slots=True)
class ContextBuilder:
    """Reusable facade that applies one PromptConfig to all context builders."""

    config: PromptConfig | None = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = PromptConfig()
        elif not isinstance(self.config, PromptConfig):
            raise TypeError("config must be a PromptConfig instance")

    def element(self, element: Any, *, index: int | None = None) -> dict[str, Any]:
        return gui_element_to_dict(element, index=index, config=self.config)

    def observation(self, observation: Any, *, label: str | None = None) -> dict[str, Any]:
        return build_observation_context(observation, config=self.config, label=label)

    def history(self, history: Sequence[Any] | Iterable[Any] | None) -> str:
        return compact_history(history, config=self.config)

    def agent(self, state: Any, *, memory: Any = None) -> dict[str, Any]:
        return build_agent_context(state, config=self.config, memory=memory)

    def to_json(self, context: Any) -> str:
        """Serialize context using the package-wide JSON display settings."""

        assert self.config is not None
        text = safe_json_dumps(
            context,
            indent=self.config.json_indent,
            ensure_ascii=self.config.ensure_ascii,
            sort_keys=self.config.sort_json_keys,
        )
        return truncate_text(text, self.config.max_prompt_chars)

    def agent_json(self, state: Any, *, memory: Any = None) -> str:
        return self.to_json(self.agent(state, memory=memory))


__all__ = [
    "ContextBuilder",
    "build_agent_context",
    "build_observation_context",
    "compact_history",
    "gui_element_to_dict",
]