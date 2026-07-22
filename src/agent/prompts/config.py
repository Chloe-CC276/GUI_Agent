"""
prompts/config
Shared configuration types for the GUI Agent prompt package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class PromptLanguage(str, Enum):
    """Language used for prompt instructions and human-readable output."""

    AUTO = "auto"
    EN = "en"
    ZH = "zh"

    @classmethod
    def coerce(cls, value: "PromptLanguage | str") -> "PromptLanguage":
        if isinstance(value, cls):
            return value

        normalized = str(value).strip().lower().replace("_", "-")
        aliases = {
            "auto": cls.AUTO,
            "en": cls.EN,
            "en-us": cls.EN,
            "en-gb": cls.EN,
            "english": cls.EN,
            "zh": cls.ZH,
            "zh-cn": cls.ZH,
            "zh-hans": cls.ZH,
            "cn": cls.ZH,
            "chinese": cls.ZH,
            "中文": cls.ZH,
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            raise ValueError(f"Unsupported prompt language: {value!r}") from error


class PromptFormat(str, Enum):
    """Expected model response representation."""

    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"

    @classmethod
    def coerce(cls, value: "PromptFormat | str") -> "PromptFormat":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            raise ValueError(f"Unsupported prompt format: {value!r}") from error


class PromptKind(str, Enum):
    """Kinds of prompts provided by the package."""

    PLANNER = "planner"
    VERIFY = "verify"
    REPAIR = "repair"
    REFLECTION = "reflection"
    MEMORY_SUMMARY = "memory_summary"
    OBSERVATION_SUMMARY = "observation_summary"

    @classmethod
    def coerce(cls, value: "PromptKind | str") -> "PromptKind":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            raise ValueError(f"Unsupported prompt kind: {value!r}") from error


# Keep this list aligned with planner.ActionName.  It is defined locally to
# avoid importing planner.py and creating a planner <-> prompts import cycle.
DEFAULT_ALLOWED_ACTIONS: tuple[str, ...] = (
    "move_to",
    "move_by",
    "click",
    "double_click",
    "right_click",
    "middle_click",
    "mouse_down",
    "mouse_up",
    "drag_to",
    "drag_by",
    "scroll",
    "horizontal_scroll",
    "press",
    "hotkey",
    "type_text",
    "wait",
    "screenshot",
    "finish",
    "retry",
    "fail",
)


_DEFAULT_KIND_FORMATS: Mapping[PromptKind, PromptFormat] = MappingProxyType(
    {
        PromptKind.PLANNER: PromptFormat.JSON,
        PromptKind.VERIFY: PromptFormat.JSON,
        PromptKind.REPAIR: PromptFormat.JSON,
        PromptKind.REFLECTION: PromptFormat.JSON,
        PromptKind.MEMORY_SUMMARY: PromptFormat.JSON,
        PromptKind.OBSERVATION_SUMMARY: PromptFormat.JSON,
    }
)


def _unique_actions(values: Sequence[str]) -> tuple[str, ...]:
    actions: list[str] = []
    for value in values:
        action = str(value).strip().lower()
        if action and action not in actions:
            actions.append(action)
    if not actions:
        raise ValueError("allowed_actions must contain at least one action")
    return tuple(actions)


def _validate_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(slots=True)
class PromptConfig:
    """Global configuration shared by every specialised prompt builder.

    The defaults target structured GUI planning with a screenshot, compact OCR
    and element context, and bounded recent history.  Fields are intentionally
    provider-independent; model generation settings remain in ``PlannerConfig``
    or the VLM generation configuration.
    """

    language: PromptLanguage = PromptLanguage.AUTO
    format: PromptFormat = PromptFormat.JSON
    allowed_actions: tuple[str, ...] = DEFAULT_ALLOWED_ACTIONS

    include_system_prompt: bool = True
    include_rules: bool = True
    include_schema: bool = True
    include_screenshot: bool = True
    include_previous_observation: bool = True
    include_elements: bool = True
    include_ocr: bool = True
    include_history: bool = True
    include_execution_result: bool = True
    include_memory: bool = True

    history_limit: int = 10
    max_elements: int = 100
    max_ocr_chars: int = 2_000
    max_element_text_chars: int = 240
    max_history_item_chars: int = 800
    max_error_chars: int = 2_000
    max_memory_chars: int = 4_000
    max_prompt_chars: int = 30_000

    require_reason: bool = True
    require_confidence: bool = False
    confidence_default: float | None = None
    expose_chain_of_thought: bool = False

    coordinate_precision: int = 2
    json_indent: int | None = 2
    ensure_ascii: bool = False
    sort_json_keys: bool = False

    strict_schema: bool = True
    allow_unknown_actions: bool = False
    add_generation_notice: bool = True

    kind_formats: Mapping[PromptKind, PromptFormat] = field(
        default_factory=lambda: dict(_DEFAULT_KIND_FORMATS)
    )
    system_overrides: Mapping[PromptKind, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.language = PromptLanguage.coerce(self.language)
        self.format = PromptFormat.coerce(self.format)
        self.allowed_actions = _unique_actions(self.allowed_actions)

        _validate_non_negative("history_limit", self.history_limit)
        _validate_positive("max_elements", self.max_elements)
        _validate_positive("max_ocr_chars", self.max_ocr_chars)
        _validate_positive("max_element_text_chars", self.max_element_text_chars)
        _validate_positive("max_history_item_chars", self.max_history_item_chars)
        _validate_positive("max_error_chars", self.max_error_chars)
        _validate_positive("max_memory_chars", self.max_memory_chars)
        _validate_positive("max_prompt_chars", self.max_prompt_chars)

        if not 0 <= self.coordinate_precision <= 8:
            raise ValueError("coordinate_precision must be between 0 and 8")
        if self.json_indent is not None and not 0 <= self.json_indent <= 8:
            raise ValueError("json_indent must be between 0 and 8, or None")
        if self.confidence_default is not None:
            if not 0.0 <= float(self.confidence_default) <= 1.0:
                raise ValueError("confidence_default must be between 0 and 1")
            self.confidence_default = float(self.confidence_default)

        formats: dict[PromptKind, PromptFormat] = dict(_DEFAULT_KIND_FORMATS)
        for kind, output_format in dict(self.kind_formats).items():
            formats[PromptKind.coerce(kind)] = PromptFormat.coerce(output_format)
        self.kind_formats = MappingProxyType(formats)

        overrides: dict[PromptKind, str] = {}
        for kind, prompt in dict(self.system_overrides).items():
            text = str(prompt).strip()
            if text:
                overrides[PromptKind.coerce(kind)] = text
        self.system_overrides = MappingProxyType(overrides)
        self.metadata = MappingProxyType(dict(self.metadata))

    def response_format_for(self, kind: PromptKind | str) -> PromptFormat:
        """Return the configured output format for one prompt kind."""

        resolved = PromptKind.coerce(kind)
        return self.kind_formats.get(resolved, self.format)

    def system_override_for(self, kind: PromptKind | str) -> str | None:
        """Return a custom system prompt, if one was configured."""

        return self.system_overrides.get(PromptKind.coerce(kind))

    def resolve_language(self, task_language: str | None = None) -> PromptLanguage:
        """Resolve ``AUTO`` using the task language, defaulting to English."""

        if self.language is not PromptLanguage.AUTO:
            return self.language
        if task_language:
            try:
                resolved = PromptLanguage.coerce(task_language)
                return PromptLanguage.EN if resolved is PromptLanguage.AUTO else resolved
            except ValueError:
                pass
        return PromptLanguage.EN

    def with_overrides(self, **changes: Any) -> "PromptConfig":
        """Return a validated copy with selected fields replaced."""

        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for logs and metadata."""

        return {
            "language": self.language.value,
            "format": self.format.value,
            "allowed_actions": list(self.allowed_actions),
            "include_system_prompt": self.include_system_prompt,
            "include_rules": self.include_rules,
            "include_schema": self.include_schema,
            "include_screenshot": self.include_screenshot,
            "include_previous_observation": self.include_previous_observation,
            "include_elements": self.include_elements,
            "include_ocr": self.include_ocr,
            "include_history": self.include_history,
            "include_execution_result": self.include_execution_result,
            "include_memory": self.include_memory,
            "history_limit": self.history_limit,
            "max_elements": self.max_elements,
            "max_ocr_chars": self.max_ocr_chars,
            "max_element_text_chars": self.max_element_text_chars,
            "max_history_item_chars": self.max_history_item_chars,
            "max_error_chars": self.max_error_chars,
            "max_memory_chars": self.max_memory_chars,
            "max_prompt_chars": self.max_prompt_chars,
            "require_reason": self.require_reason,
            "require_confidence": self.require_confidence,
            "confidence_default": self.confidence_default,
            "expose_chain_of_thought": self.expose_chain_of_thought,
            "coordinate_precision": self.coordinate_precision,
            "json_indent": self.json_indent,
            "ensure_ascii": self.ensure_ascii,
            "sort_json_keys": self.sort_json_keys,
            "strict_schema": self.strict_schema,
            "allow_unknown_actions": self.allow_unknown_actions,
            "add_generation_notice": self.add_generation_notice,
            "kind_formats": {
                kind.value: output_format.value
                for kind, output_format in self.kind_formats.items()
            },
            "system_overrides": {
                kind.value: prompt
                for kind, prompt in self.system_overrides.items()
            },
            "metadata": dict(self.metadata),
        }


__all__ = [
    "DEFAULT_ALLOWED_ACTIONS",
    "PromptConfig",
    "PromptFormat",
    "PromptKind",
    "PromptLanguage",
]