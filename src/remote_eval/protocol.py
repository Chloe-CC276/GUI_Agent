"""Shared HTTP protocol for Colab inference ↔ Windows executor dual-mode eval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


PROTOCOL_VERSION = "1.0"


@dataclass
class PlanRequest:
    """Local client → Colab: ask Planner for the next action."""

    task: str
    image_b64: str
    screen_width: int
    screen_height: int
    variant: str = "adapter_v1_optimized"
    session_id: str | None = None
    step_index: int = 0
    ocr_text: str = ""
    gui_elements: list[dict[str, Any]] = field(default_factory=list)
    window_title: str | None = None
    application_name: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    protocol_version: str = PROTOCOL_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlanRequest":
        return cls(
            task=str(data.get("task") or ""),
            image_b64=str(data.get("image_b64") or ""),
            screen_width=int(data.get("screen_width") or 0),
            screen_height=int(data.get("screen_height") or 0),
            variant=str(data.get("variant") or "adapter_v1_optimized"),
            session_id=data.get("session_id"),
            step_index=int(data.get("step_index") or 0),
            ocr_text=str(data.get("ocr_text") or ""),
            gui_elements=list(data.get("gui_elements") or []),
            window_title=data.get("window_title"),
            application_name=data.get("application_name"),
            history=list(data.get("history") or []),
            protocol_version=str(
                data.get("protocol_version") or PROTOCOL_VERSION
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanResponse:
    """Colab → local: planner decision + executor-ready action."""

    ok: bool
    session_id: str
    decision: str
    action: dict[str, Any] | None = None
    reason: str | None = None
    confidence: float | None = None
    schema_valid: bool | None = None
    latency_seconds: float | None = None
    total_tokens: int | None = None
    raw_output: str | None = None
    error: str | None = None
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerifyRequest:
    """Local client → Colab: verify after a real action."""

    task: str
    after_image_b64: str
    last_action: dict[str, Any]
    variant: str = "adapter_v1_optimized"
    session_id: str | None = None
    before_image_b64: str | None = None
    screen_width: int = 0
    screen_height: int = 0
    ocr_text: str = ""
    gui_elements: list[dict[str, Any]] = field(default_factory=list)
    step_index: int = 0
    protocol_version: str = PROTOCOL_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerifyRequest":
        return cls(
            task=str(data.get("task") or ""),
            after_image_b64=str(data.get("after_image_b64") or ""),
            last_action=dict(data.get("last_action") or {}),
            variant=str(data.get("variant") or "adapter_v1_optimized"),
            session_id=data.get("session_id"),
            before_image_b64=data.get("before_image_b64"),
            screen_width=int(data.get("screen_width") or 0),
            screen_height=int(data.get("screen_height") or 0),
            ocr_text=str(data.get("ocr_text") or ""),
            gui_elements=list(data.get("gui_elements") or []),
            step_index=int(data.get("step_index") or 0),
            protocol_version=str(
                data.get("protocol_version") or PROTOCOL_VERSION
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerifyResponse:
    ok: bool
    session_id: str
    status: str = "uncertain"
    action_effective: bool | None = None
    task_complete: bool | None = None
    recommended_next: str = "continue"
    reason: str | None = None
    confidence: float | None = None
    evidence: list[str] = field(default_factory=list)
    latency_seconds: float | None = None
    total_tokens: int | None = None
    error: str | None = None
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
