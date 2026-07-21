"""
state

Runtime state model for the GUI Agent.

Typical lifecycle
-----------------
state = AgentState.create(task="Open the executor folder")

state.begin()
state.update_observation(...)
state.update_planner_result(...)
state.update_execution_result(...)
state.commit_step(...)
state.finish(...)
"""

from __future__ import annotations

import copy
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .result import (
    AgentRunResult,
    AgentStepResult,
    ErrorInfo,
    PlannerResult,
    ResultStatus,
    RunTerminationReason,
    ToolResult,
    UsageInfo,
    to_json_safe,
)


# ============================================================
# Exceptions
# ============================================================


class AgentStateError(ValueError):
    """Base exception raised by invalid Agent state operations."""


class AgentStateTransitionError(AgentStateError):
    """Raised when an invalid state transition is requested."""


class AgentStateSerializationError(AgentStateError):
    """Raised when AgentState cannot be serialized."""


# ============================================================
# Enumerations
# ============================================================


class AgentPhase(str, Enum):
    """Current phase of the Agent execution loop."""

    CREATED = "created"
    OBSERVING = "observing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ObservationSource(str, Enum):
    """Origin of the current observation."""

    PERCEPTION = "perception"
    SCREENSHOT = "screenshot"
    ACCESSIBILITY = "accessibility"
    OCR = "ocr"
    MANUAL = "manual"
    UNKNOWN = "unknown"


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _normalise_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _ensure_non_negative_integer(
    name: str,
    value: int,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise AgentStateError(
            f"{name} must be a non-negative integer."
        )


def _ensure_positive_integer(
    name: str,
    value: int,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise AgentStateError(
            f"{name} must be a positive integer."
        )


def _coerce_datetime(
    value: datetime | str | None,
) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value)
        except ValueError as error:
            raise AgentStateError(
                f"Invalid datetime value: {value!r}"
            ) from error
    else:
        raise AgentStateError(
            "Datetime field must be datetime, ISO string, or None."
        )

    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)

    return result


# ============================================================
# Task state
# ============================================================


@dataclass(slots=True)
class TaskState:
    """Static and progress-related information about the current task."""

    instruction: str
    task_id: str = field(
        default_factory=lambda: _new_id("task")
    )
    source: str = "user"
    language: str | None = None
    subgoal: str | None = None
    completed_subgoals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.instruction = self.instruction.strip()

        if not self.instruction:
            raise AgentStateError(
                "TaskState.instruction must not be empty."
            )

        self.source = self.source.strip() or "user"
        self.language = _normalise_text(self.language)
        self.subgoal = _normalise_text(self.subgoal)

        self.completed_subgoals = [
            str(item).strip()
            for item in self.completed_subgoals
            if str(item).strip()
        ]
        self.constraints = [
            str(item).strip()
            for item in self.constraints
            if str(item).strip()
        ]
        self.success_criteria = [
            str(item).strip()
            for item in self.success_criteria
            if str(item).strip()
        ]

    def set_subgoal(self, subgoal: str | None) -> None:
        self.subgoal = _normalise_text(subgoal)

    def complete_subgoal(
        self,
        subgoal: str | None = None,
    ) -> None:
        resolved = _normalise_text(subgoal) or self.subgoal

        if resolved is None:
            raise AgentStateError(
                "No subgoal is available to complete."
            )

        if resolved not in self.completed_subgoals:
            self.completed_subgoals.append(resolved)

        if self.subgoal == resolved:
            self.subgoal = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "source": self.source,
            "language": self.language,
            "subgoal": self.subgoal,
            "completed_subgoals": list(self.completed_subgoals),
            "constraints": list(self.constraints),
            "success_criteria": list(self.success_criteria),
            "metadata": to_json_safe(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "TaskState":
        return cls(
            task_id=str(
                data.get("task_id")
                or _new_id("task")
            ),
            instruction=str(
                data.get("instruction", "")
            ),
            source=str(data.get("source", "user")),
            language=data.get("language"),
            subgoal=data.get("subgoal"),
            completed_subgoals=list(
                data.get("completed_subgoals") or []
            ),
            constraints=list(
                data.get("constraints") or []
            ),
            success_criteria=list(
                data.get("success_criteria") or []
            ),
            metadata=dict(data.get("metadata") or {}),
        )


# ============================================================
# Observation state
# ============================================================


@dataclass(slots=True)
class ObservationState:
    """
    Current GUI observation.

    The object stores both normalized fields used by Planner and raw provider
    output used for debugging. No concrete PerceptionResult import is needed.
    """

    screenshot: Any = None
    screenshot_path: str | None = None
    screen_width: int | None = None
    screen_height: int | None = None
    window_title: str | None = None
    application_name: str | None = None
    ocr_text: str | None = None
    ocr_items: list[Any] = field(default_factory=list)
    gui_elements: list[Any] = field(default_factory=list)
    accessibility_tree: Any = None
    cursor_position: tuple[int, int] | None = None
    source: ObservationSource = ObservationSource.UNKNOWN
    captured_at: datetime = field(default_factory=utc_now)
    raw_observation: Any = None
    observation_id: str = field(
        default_factory=lambda: _new_id("obs")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.source, str):
            self.source = ObservationSource(self.source)

        self.screenshot_path = _normalise_text(
            self.screenshot_path
        )
        self.window_title = _normalise_text(
            self.window_title
        )
        self.application_name = _normalise_text(
            self.application_name
        )
        self.ocr_text = _normalise_text(self.ocr_text)

        self.captured_at = (
            _coerce_datetime(self.captured_at)
            or utc_now()
        )

        for name in ("screen_width", "screen_height"):
            value = getattr(self, name)

            if value is not None:
                _ensure_positive_integer(name, value)

        if self.cursor_position is not None:
            if (
                not isinstance(self.cursor_position, tuple)
                or len(self.cursor_position) != 2
            ):
                raise AgentStateError(
                    "cursor_position must be a two-item tuple."
                )

            x, y = self.cursor_position

            if not isinstance(x, int) or not isinstance(y, int):
                raise AgentStateError(
                    "cursor coordinates must be integers."
                )

    @property
    def has_screenshot(self) -> bool:
        return (
            self.screenshot is not None
            or self.screenshot_path is not None
        )

    @property
    def screen_size(self) -> tuple[int, int] | None:
        if (
            self.screen_width is None
            or self.screen_height is None
        ):
            return None

        return self.screen_width, self.screen_height

    @property
    def element_count(self) -> int:
        return len(self.gui_elements)

    @property
    def ocr_item_count(self) -> int:
        return len(self.ocr_items)

    def summary(
        self,
        *,
        max_ocr_chars: int = 1000,
        max_elements: int = 50,
    ) -> dict[str, Any]:
        ocr_text = self.ocr_text

        if (
            ocr_text is not None
            and len(ocr_text) > max_ocr_chars
        ):
            ocr_text = (
                ocr_text[:max_ocr_chars]
                + "...<truncated>"
            )

        return {
            "observation_id": self.observation_id,
            "window_title": self.window_title,
            "application_name": self.application_name,
            "screen_size": self.screen_size,
            "screenshot_path": self.screenshot_path,
            "ocr_text": ocr_text,
            "ocr_items": to_json_safe(
                self.ocr_items[:max_elements]
            ),
            "gui_elements": to_json_safe(
                self.gui_elements[:max_elements]
            ),
            "element_count": self.element_count,
            "ocr_item_count": self.ocr_item_count,
            "cursor_position": self.cursor_position,
            "source": self.source.value,
            "captured_at": self.captured_at.isoformat(),
            "metadata": to_json_safe(self.metadata),
        }

    def to_dict(
        self,
        *,
        include_screenshot: bool = False,
        include_raw: bool = False,
    ) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "screenshot": (
                to_json_safe(self.screenshot)
                if include_screenshot
                else None
            ),
            "screenshot_path": self.screenshot_path,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "window_title": self.window_title,
            "application_name": self.application_name,
            "ocr_text": self.ocr_text,
            "ocr_items": to_json_safe(self.ocr_items),
            "gui_elements": to_json_safe(self.gui_elements),
            "accessibility_tree": to_json_safe(
                self.accessibility_tree
            ),
            "cursor_position": (
                list(self.cursor_position)
                if self.cursor_position is not None
                else None
            ),
            "source": self.source.value,
            "captured_at": self.captured_at.isoformat(),
            "raw_observation": (
                to_json_safe(self.raw_observation)
                if include_raw
                else None
            ),
            "metadata": to_json_safe(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ObservationState":
        cursor = data.get("cursor_position")

        return cls(
            screenshot=data.get("screenshot"),
            screenshot_path=data.get("screenshot_path"),
            screen_width=data.get("screen_width"),
            screen_height=data.get("screen_height"),
            window_title=data.get("window_title"),
            application_name=data.get("application_name"),
            ocr_text=data.get("ocr_text"),
            ocr_items=list(data.get("ocr_items") or []),
            gui_elements=list(
                data.get("gui_elements") or []
            ),
            accessibility_tree=data.get(
                "accessibility_tree"
            ),
            cursor_position=(
                tuple(cursor)
                if cursor is not None
                else None
            ),
            source=data.get(
                "source",
                ObservationSource.UNKNOWN.value,
            ),
            captured_at=data.get("captured_at") or utc_now(),
            raw_observation=data.get("raw_observation"),
            observation_id=str(
                data.get("observation_id")
                or _new_id("obs")
            ),
            metadata=dict(data.get("metadata") or {}),
        )


# ============================================================
# Runtime counters and limits
# ============================================================


@dataclass(slots=True)
class RuntimeState:
    """Mutable counters, limits, and timing data."""

    step_index: int = 0
    max_steps: int = 20
    retry_count: int = 0
    max_retries: int = 3
    consecutive_failures: int = 0
    repeated_action_count: int = 0
    started_at: datetime | None = None
    updated_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None
    deadline_at: datetime | None = None
    total_usage: UsageInfo = field(default_factory=UsageInfo)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_negative_integer(
            "step_index",
            self.step_index,
        )
        _ensure_positive_integer(
            "max_steps",
            self.max_steps,
        )
        _ensure_non_negative_integer(
            "retry_count",
            self.retry_count,
        )
        _ensure_non_negative_integer(
            "max_retries",
            self.max_retries,
        )
        _ensure_non_negative_integer(
            "consecutive_failures",
            self.consecutive_failures,
        )
        _ensure_non_negative_integer(
            "repeated_action_count",
            self.repeated_action_count,
        )

        self.started_at = _coerce_datetime(self.started_at)
        self.updated_at = (
            _coerce_datetime(self.updated_at)
            or utc_now()
        )
        self.finished_at = _coerce_datetime(
            self.finished_at
        )
        self.deadline_at = _coerce_datetime(
            self.deadline_at
        )

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.step_index)

    @property
    def reached_max_steps(self) -> bool:
        return self.step_index >= self.max_steps

    @property
    def reached_max_retries(self) -> bool:
        return self.retry_count >= self.max_retries

    @property
    def is_timed_out(self) -> bool:
        return (
            self.deadline_at is not None
            and utc_now() >= self.deadline_at
        )

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0

        endpoint = self.finished_at or utc_now()

        return max(
            0.0,
            (endpoint - self.started_at).total_seconds(),
        )

    def start(self) -> None:
        now = utc_now()

        if self.started_at is None:
            self.started_at = now

        self.updated_at = now

    def touch(self) -> None:
        self.updated_at = utc_now()

    def next_step(self) -> int:
        if self.reached_max_steps:
            raise AgentStateTransitionError(
                "Cannot advance: maximum steps reached."
            )

        self.step_index += 1
        self.touch()
        return self.step_index

    def register_retry(self) -> int:
        self.retry_count += 1
        self.touch()
        return self.retry_count

    def register_success(self) -> None:
        self.consecutive_failures = 0
        self.touch()

    def register_failure(self) -> None:
        self.consecutive_failures += 1
        self.touch()

    def finish(self) -> None:
        self.finished_at = utc_now()
        self.updated_at = self.finished_at

    def reset(
        self,
        *,
        preserve_limits: bool = True,
    ) -> None:
        max_steps = self.max_steps
        max_retries = self.max_retries

        self.step_index = 0
        self.retry_count = 0
        self.consecutive_failures = 0
        self.repeated_action_count = 0
        self.started_at = None
        self.updated_at = utc_now()
        self.finished_at = None
        self.deadline_at = None
        self.total_usage = UsageInfo()
        self.metadata.clear()

        if not preserve_limits:
            self.max_steps = 20
            self.max_retries = 3
        else:
            self.max_steps = max_steps
            self.max_retries = max_retries

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "max_steps": self.max_steps,
            "remaining_steps": self.remaining_steps,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "consecutive_failures": self.consecutive_failures,
            "repeated_action_count": self.repeated_action_count,
            "started_at": (
                self.started_at.isoformat()
                if self.started_at is not None
                else None
            ),
            "updated_at": self.updated_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat()
                if self.finished_at is not None
                else None
            ),
            "deadline_at": (
                self.deadline_at.isoformat()
                if self.deadline_at is not None
                else None
            ),
            "elapsed_seconds": self.elapsed_seconds,
            "total_usage": self.total_usage.to_dict(),
            "metadata": to_json_safe(self.metadata),
        }


# ============================================================
# History entry
# ============================================================


@dataclass(slots=True)
class StateHistoryEntry:
    """Small, serializable event stored in AgentState history."""

    event_type: str
    step_index: int
    message: str | None = None
    action: Any = None
    status: ResultStatus | None = None
    created_at: datetime = field(default_factory=utc_now)
    event_id: str = field(
        default_factory=lambda: _new_id("event")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_type = self.event_type.strip()

        if not self.event_type:
            raise AgentStateError(
                "event_type must not be empty."
            )

        _ensure_non_negative_integer(
            "step_index",
            self.step_index,
        )

        self.message = _normalise_text(self.message)
        self.created_at = (
            _coerce_datetime(self.created_at)
            or utc_now()
        )

        if isinstance(self.status, str):
            self.status = ResultStatus(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "step_index": self.step_index,
            "message": self.message,
            "action": to_json_safe(self.action),
            "status": (
                self.status.value
                if self.status is not None
                else None
            ),
            "created_at": self.created_at.isoformat(),
            "metadata": to_json_safe(self.metadata),
        }


# ============================================================
# Main Agent state
# ============================================================


@dataclass(slots=True)
class AgentState:
    """
    Shared mutable state for one GUI Agent run.

    This class is the only object that AgentGraph needs to pass between
    observe, plan, execute, and verify nodes.
    """

    task: TaskState
    phase: AgentPhase = AgentPhase.CREATED
    observation: ObservationState | None = None
    previous_observation: ObservationState | None = None
    last_planner_result: PlannerResult | None = None
    last_observation_result: ToolResult | None = None
    last_execution_result: ToolResult | None = None
    last_verification_result: ToolResult | None = None
    current_step: AgentStepResult | None = None
    completed_steps: list[AgentStepResult] = field(
        default_factory=list
    )
    history: list[StateHistoryEntry] = field(
        default_factory=list
    )
    runtime: RuntimeState = field(
        default_factory=RuntimeState
    )
    final_message: str | None = None
    termination_reason: RunTerminationReason | None = None
    error: ErrorInfo | None = None
    state_id: str = field(
        default_factory=lambda: _new_id("state")
    )
    run_id: str = field(
        default_factory=lambda: _new_id("run")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.task, str):
            self.task = TaskState(instruction=self.task)

        if not isinstance(self.task, TaskState):
            raise AgentStateError(
                "task must be TaskState or str."
            )

        if isinstance(self.phase, str):
            self.phase = AgentPhase(self.phase)

        if isinstance(self.termination_reason, str):
            self.termination_reason = RunTerminationReason(
                self.termination_reason
            )

        self.final_message = _normalise_text(
            self.final_message
        )
        self.validate()

    # --------------------------------------------------------
    # Construction
    # --------------------------------------------------------

    @classmethod
    def create(
        cls,
        task: str,
        *,
        task_id: str | None = None,
        source: str = "user",
        language: str | None = None,
        max_steps: int = 20,
        max_retries: int = 3,
        constraints: Sequence[str] | None = None,
        success_criteria: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "AgentState":
        return cls(
            task=TaskState(
                instruction=task,
                task_id=task_id or _new_id("task"),
                source=source,
                language=language,
                constraints=list(constraints or []),
                success_criteria=list(
                    success_criteria or []
                ),
            ),
            runtime=RuntimeState(
                max_steps=max_steps,
                max_retries=max_retries,
            ),
            metadata=dict(metadata or {}),
        )

    # --------------------------------------------------------
    # Properties
    # --------------------------------------------------------

    @property
    def instruction(self) -> str:
        return self.task.instruction

    @property
    def step_index(self) -> int:
        return self.runtime.step_index

    @property
    def max_steps(self) -> int:
        return self.runtime.max_steps

    @property
    def is_terminal(self) -> bool:
        return self.phase in {
            AgentPhase.FINISHED,
            AgentPhase.FAILED,
            AgentPhase.CANCELLED,
        }

    @property
    def is_finished(self) -> bool:
        return self.phase == AgentPhase.FINISHED

    @property
    def has_failed(self) -> bool:
        return self.phase == AgentPhase.FAILED

    @property
    def latest_action(self) -> Any:
        if self.last_planner_result is None:
            return None

        return self.last_planner_result.action

    @property
    def action_history(self) -> list[Any]:
        return [
            step.action
            for step in self.completed_steps
            if step.action is not None
        ]

    # --------------------------------------------------------
    # Validation and phase transitions
    # --------------------------------------------------------

    def validate(self) -> None:
        if self.is_terminal and self.termination_reason is None:
            raise AgentStateError(
                "Terminal AgentState requires termination_reason."
            )

        if (
            self.phase == AgentPhase.FINISHED
            and self.termination_reason
            != RunTerminationReason.COMPLETED
        ):
            raise AgentStateError(
                "FINISHED state requires COMPLETED termination."
            )

        if (
            self.phase == AgentPhase.FAILED
            and self.error is None
        ):
            raise AgentStateError(
                "FAILED state requires error."
            )

    def _ensure_active(self) -> None:
        if self.is_terminal:
            raise AgentStateTransitionError(
                f"State is terminal: {self.phase.value}."
            )

    def set_phase(self, phase: AgentPhase | str) -> None:
        resolved = AgentPhase(phase)

        if self.is_terminal and resolved != self.phase:
            raise AgentStateTransitionError(
                "Cannot leave a terminal phase."
            )

        self.phase = resolved
        self.runtime.touch()

    def begin(self) -> None:
        if self.phase != AgentPhase.CREATED:
            raise AgentStateTransitionError(
                "Agent can begin only from CREATED phase."
            )

        self.runtime.start()
        self.phase = AgentPhase.OBSERVING
        self.add_history(
            event_type="run_started",
            message=self.task.instruction,
            status=ResultStatus.PENDING,
        )

    # --------------------------------------------------------
    # Observation update
    # --------------------------------------------------------

    def update_observation(
        self,
        observation: ObservationState,
        *,
        tool_result: ToolResult | None = None,
    ) -> None:
        self._ensure_active()

        if not isinstance(observation, ObservationState):
            raise TypeError(
                "observation must be ObservationState."
            )

        self.previous_observation = self.observation
        self.observation = observation
        self.last_observation_result = tool_result
        self.phase = AgentPhase.PLANNING
        self.runtime.touch()

        self.add_history(
            event_type="observation_updated",
            message=observation.window_title,
            status=(
                tool_result.status
                if tool_result is not None
                else ResultStatus.SUCCESS
            ),
            metadata={
                "observation_id": observation.observation_id,
                "element_count": observation.element_count,
                "ocr_item_count": observation.ocr_item_count,
            },
        )

    # --------------------------------------------------------
    # Planner update
    # --------------------------------------------------------

    def update_planner_result(
        self,
        result: PlannerResult,
    ) -> None:
        self._ensure_active()

        if not isinstance(result, PlannerResult):
            raise TypeError(
                "result must be PlannerResult."
            )

        self.last_planner_result = result
        self.runtime.total_usage.add(result.usage)
        self.runtime.touch()

        if result.is_finished:
            self.phase = AgentPhase.VERIFYING
        elif result.should_execute:
            self.phase = AgentPhase.EXECUTING
        elif result.should_retry:
            self.phase = AgentPhase.PLANNING
            self.runtime.register_retry()
        else:
            self.phase = AgentPhase.FAILED
            self.error = result.error or ErrorInfo(
                error_type="PlannerFailure",
                message=(
                    result.reason
                    or "Planner returned an unusable result."
                ),
            )
            self.termination_reason = (
                RunTerminationReason.PLANNER_FAILED
            )
            self.runtime.finish()

        self.add_history(
            event_type="planner_result",
            message=result.reason,
            action=result.action,
            status=result.status,
            metadata={
                "decision": result.decision.value,
                "planner_id": result.planner_id,
                "confidence": result.confidence,
            },
        )

    # --------------------------------------------------------
    # Execution and verification update
    # --------------------------------------------------------

    def update_execution_result(
        self,
        result: ToolResult,
    ) -> None:
        self._ensure_active()

        if not isinstance(result, ToolResult):
            raise TypeError(
                "result must be ToolResult."
            )

        self.last_execution_result = result
        self.runtime.touch()

        if result.succeeded:
            self.runtime.register_success()
            self.phase = AgentPhase.VERIFYING
        elif result.should_retry:
            self.runtime.register_failure()
            self.runtime.register_retry()
            self.phase = AgentPhase.PLANNING
        else:
            self.runtime.register_failure()
            self.phase = AgentPhase.FAILED
            self.error = result.error or ErrorInfo(
                error_type="ExecutionFailure",
                message=(
                    result.message
                    or "Executor tool failed."
                ),
            )
            self.termination_reason = (
                RunTerminationReason.EXECUTION_FAILED
            )
            self.runtime.finish()

        self.add_history(
            event_type="execution_result",
            message=result.message,
            action=self.latest_action,
            status=result.status,
            metadata={
                "tool_name": result.tool_name,
                "tool_call_id": result.tool_call_id,
            },
        )

    def update_verification_result(
        self,
        result: ToolResult,
    ) -> None:
        self._ensure_active()

        if not isinstance(result, ToolResult):
            raise TypeError(
                "result must be ToolResult."
            )

        self.last_verification_result = result
        self.runtime.touch()

        if result.succeeded:
            self.phase = AgentPhase.OBSERVING
        elif result.should_retry:
            self.runtime.register_retry()
            self.phase = AgentPhase.PLANNING
        else:
            self.phase = AgentPhase.FAILED
            self.error = result.error or ErrorInfo(
                error_type="VerificationFailure",
                message=(
                    result.message
                    or "Verification failed."
                ),
            )
            self.termination_reason = (
                RunTerminationReason.TOOL_FAILED
            )
            self.runtime.finish()

        self.add_history(
            event_type="verification_result",
            message=result.message,
            status=result.status,
            metadata={
                "tool_name": result.tool_name,
                "tool_call_id": result.tool_call_id,
            },
        )

    # --------------------------------------------------------
    # Step lifecycle
    # --------------------------------------------------------

    def build_current_step(self) -> AgentStepResult:
        if self.last_planner_result is None:
            raise AgentStateTransitionError(
                "Cannot build step without PlannerResult."
            )

        self.current_step = AgentStepResult(
            step_index=self.runtime.step_index,
            observation_result=self.last_observation_result,
            planner_result=self.last_planner_result,
            execution_result=self.last_execution_result,
            verification_result=self.last_verification_result,
        )

        return self.current_step

    def commit_step(
        self,
        step: AgentStepResult | None = None,
    ) -> AgentStepResult:
        self._ensure_active()

        resolved_step = step or self.current_step

        if resolved_step is None:
            resolved_step = self.build_current_step()

        expected_index = len(self.completed_steps)

        if resolved_step.step_index != expected_index:
            raise AgentStateTransitionError(
                "Step indexes must be continuous: "
                f"expected {expected_index}, "
                f"got {resolved_step.step_index}."
            )

        if not resolved_step.timing.is_finished:
            resolved_step.complete()

        self.completed_steps.append(resolved_step)
        self.current_step = None

        self.add_history(
            event_type="step_committed",
            message=f"Step {resolved_step.step_index} committed.",
            action=resolved_step.action,
            status=resolved_step.status,
            metadata={
                "step_id": resolved_step.step_id,
            },
        )

        self.runtime.next_step()
        self._clear_step_results()

        if self.runtime.reached_max_steps:
            self.fail(
                error=ErrorInfo(
                    error_type="MaximumStepsExceeded",
                    message=(
                        "The agent reached the maximum "
                        f"number of steps: {self.runtime.max_steps}."
                    ),
                ),
                reason=RunTerminationReason.MAX_STEPS,
            )
        elif not self.is_terminal:
            self.phase = AgentPhase.OBSERVING

        return resolved_step

    def _clear_step_results(self) -> None:
        self.last_planner_result = None
        self.last_observation_result = None
        self.last_execution_result = None
        self.last_verification_result = None

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    def add_history(
        self,
        *,
        event_type: str,
        message: str | None = None,
        action: Any = None,
        status: ResultStatus | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> StateHistoryEntry:
        entry = StateHistoryEntry(
            event_type=event_type,
            step_index=self.runtime.step_index,
            message=message,
            action=action,
            status=(
                ResultStatus(status)
                if status is not None
                else None
            ),
            metadata=dict(metadata or {}),
        )
        self.history.append(entry)
        return entry

    def recent_history(
        self,
        limit: int = 10,
    ) -> list[StateHistoryEntry]:
        _ensure_non_negative_integer("limit", limit)

        if limit == 0:
            return []

        return self.history[-limit:]

    # --------------------------------------------------------
    # Terminal operations
    # --------------------------------------------------------

    def finish(
        self,
        message: str | None = None,
    ) -> None:
        self._ensure_active()

        self.phase = AgentPhase.FINISHED
        self.final_message = _normalise_text(message)
        self.termination_reason = (
            RunTerminationReason.COMPLETED
        )
        self.error = None
        self.runtime.finish()

        self.add_history(
            event_type="run_finished",
            message=self.final_message,
            status=ResultStatus.SUCCESS,
        )

    def fail(
        self,
        *,
        error: ErrorInfo,
        reason: RunTerminationReason | str = (
            RunTerminationReason.UNKNOWN
        ),
        message: str | None = None,
    ) -> None:
        if self.is_terminal:
            return

        self.phase = AgentPhase.FAILED
        self.error = error
        self.final_message = _normalise_text(message)
        self.termination_reason = RunTerminationReason(
            reason
        )
        self.runtime.finish()

        self.add_history(
            event_type="run_failed",
            message=(
                self.final_message
                or error.message
            ),
            status=ResultStatus.FAILED,
            metadata={
                "error": error.to_dict(),
                "termination_reason": (
                    self.termination_reason.value
                ),
            },
        )

    def cancel(
        self,
        message: str | None = None,
    ) -> None:
        if self.is_terminal:
            return

        self.phase = AgentPhase.CANCELLED
        self.final_message = _normalise_text(message)
        self.termination_reason = (
            RunTerminationReason.USER_CANCELLED
        )
        self.runtime.finish()

        self.add_history(
            event_type="run_cancelled",
            message=self.final_message,
            status=ResultStatus.CANCELLED,
        )

    # --------------------------------------------------------
    # Planner context
    # --------------------------------------------------------

    def planner_context(
        self,
        *,
        history_limit: int = 10,
        max_ocr_chars: int = 1500,
        max_elements: int = 80,
    ) -> dict[str, Any]:
        """
        Return the normalized context that planner.py can place into prompts.
        """
        return {
            "run_id": self.run_id,
            "task": self.task.to_dict(),
            "phase": self.phase.value,
            "step_index": self.runtime.step_index,
            "max_steps": self.runtime.max_steps,
            "remaining_steps": self.runtime.remaining_steps,
            "retry_count": self.runtime.retry_count,
            "observation": (
                self.observation.summary(
                    max_ocr_chars=max_ocr_chars,
                    max_elements=max_elements,
                )
                if self.observation is not None
                else None
            ),
            "previous_observation": (
                self.previous_observation.summary(
                    max_ocr_chars=max_ocr_chars,
                    max_elements=max_elements,
                )
                if self.previous_observation is not None
                else None
            ),
            "history": [
                entry.to_dict()
                for entry in self.recent_history(
                    history_limit
                )
            ],
            "last_error": (
                self.error.to_dict()
                if self.error is not None
                else None
            ),
            "metadata": to_json_safe(self.metadata),
        }

    # --------------------------------------------------------
    # Run result conversion
    # --------------------------------------------------------

    def to_run_result(self) -> AgentRunResult:
        run_result = AgentRunResult(
            run_id=self.run_id,
            task=self.task.instruction,
            status=self._result_status(),
            termination_reason=self.termination_reason,
            final_message=self.final_message,
            steps=list(self.completed_steps),
            error=self.error,
            metadata={
                "state_id": self.state_id,
                "task": self.task.to_dict(),
                **to_json_safe(self.metadata),
            },
        )

        if self.runtime.started_at is not None:
            run_result.timing.started_at = (
                self.runtime.started_at
            )

        if self.runtime.finished_at is not None:
            run_result.timing.finished_at = (
                self.runtime.finished_at
            )
            run_result.timing.latency_seconds = (
                self.runtime.elapsed_seconds
            )

        run_result.recalculate_usage()
        return run_result

    def _result_status(self) -> ResultStatus:
        mapping = {
            AgentPhase.CREATED: ResultStatus.PENDING,
            AgentPhase.OBSERVING: ResultStatus.PENDING,
            AgentPhase.PLANNING: ResultStatus.PENDING,
            AgentPhase.EXECUTING: ResultStatus.PENDING,
            AgentPhase.VERIFYING: ResultStatus.PENDING,
            AgentPhase.FINISHED: ResultStatus.SUCCESS,
            AgentPhase.FAILED: ResultStatus.FAILED,
            AgentPhase.CANCELLED: ResultStatus.CANCELLED,
        }
        return mapping[self.phase]

    # --------------------------------------------------------
    # Reset and clone
    # --------------------------------------------------------

    def clone(self) -> "AgentState":
        """Return a deep copy useful for testing and checkpointing."""
        return copy.deepcopy(self)

    def reset(
        self,
        *,
        preserve_task: bool = True,
        preserve_metadata: bool = True,
    ) -> None:
        task = self.task
        metadata = dict(self.metadata)

        if not preserve_task:
            raise AgentStateError(
                "reset(preserve_task=False) is unsupported; "
                "create a new AgentState instead."
            )

        self.phase = AgentPhase.CREATED
        self.observation = None
        self.previous_observation = None
        self.last_planner_result = None
        self.last_observation_result = None
        self.last_execution_result = None
        self.last_verification_result = None
        self.current_step = None
        self.completed_steps.clear()
        self.history.clear()
        self.runtime.reset(preserve_limits=True)
        self.final_message = None
        self.termination_reason = None
        self.error = None
        self.run_id = _new_id("run")
        self.state_id = _new_id("state")
        self.task = task
        self.metadata = (
            metadata
            if preserve_metadata
            else {}
        )

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------

    def to_dict(
        self,
        *,
        include_observation_raw: bool = False,
        include_screenshot: bool = False,
        include_completed_steps: bool = True,
        include_result_raw_output: bool = False,
        include_tool_output: bool = False,
    ) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "run_id": self.run_id,
            "phase": self.phase.value,
            "task": self.task.to_dict(),
            "observation": (
                self.observation.to_dict(
                    include_screenshot=include_screenshot,
                    include_raw=include_observation_raw,
                )
                if self.observation is not None
                else None
            ),
            "previous_observation": (
                self.previous_observation.to_dict(
                    include_screenshot=include_screenshot,
                    include_raw=include_observation_raw,
                )
                if self.previous_observation is not None
                else None
            ),
            "last_planner_result": (
                self.last_planner_result.to_dict(
                    include_raw_output=include_result_raw_output
                )
                if self.last_planner_result is not None
                else None
            ),
            "last_observation_result": (
                self.last_observation_result.to_dict(
                    include_output=include_tool_output
                )
                if self.last_observation_result is not None
                else None
            ),
            "last_execution_result": (
                self.last_execution_result.to_dict(
                    include_output=include_tool_output
                )
                if self.last_execution_result is not None
                else None
            ),
            "last_verification_result": (
                self.last_verification_result.to_dict(
                    include_output=include_tool_output
                )
                if self.last_verification_result is not None
                else None
            ),
            "current_step": (
                self.current_step.to_dict(
                    include_raw_output=(
                        include_result_raw_output
                    ),
                    include_tool_output=include_tool_output,
                )
                if self.current_step is not None
                else None
            ),
            "completed_steps": (
                [
                    step.to_dict(
                        include_raw_output=(
                            include_result_raw_output
                        ),
                        include_tool_output=(
                            include_tool_output
                        ),
                    )
                    for step in self.completed_steps
                ]
                if include_completed_steps
                else []
            ),
            "history": [
                entry.to_dict()
                for entry in self.history
            ],
            "runtime": self.runtime.to_dict(),
            "final_message": self.final_message,
            "termination_reason": (
                self.termination_reason.value
                if self.termination_reason is not None
                else None
            ),
            "error": (
                self.error.to_dict()
                if self.error is not None
                else None
            ),
            "metadata": to_json_safe(self.metadata),
        }

    def to_json(
        self,
        *,
        indent: int | None = 2,
        ensure_ascii: bool = False,
        **to_dict_options: Any,
    ) -> str:
        try:
            return json.dumps(
                self.to_dict(**to_dict_options),
                indent=indent,
                ensure_ascii=ensure_ascii,
            )
        except (TypeError, ValueError) as error:
            raise AgentStateSerializationError(
                f"Unable to serialize AgentState: {error}"
            ) from error

    def save_json(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
        ensure_ascii: bool = False,
        **to_dict_options: Any,
    ) -> Path:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            self.to_json(
                indent=indent,
                ensure_ascii=ensure_ascii,
                **to_dict_options,
            ),
            encoding="utf-8",
        )
        return output_path

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"run_id={self.run_id!r}, "
            f"phase={self.phase.value!r}, "
            f"step_index={self.runtime.step_index}, "
            f"task={self.task.instruction!r}"
            f")"
        )


__all__ = [
    "AgentStateError",
    "AgentStateTransitionError",
    "AgentStateSerializationError",
    "AgentPhase",
    "ObservationSource",
    "TaskState",
    "ObservationState",
    "RuntimeState",
    "StateHistoryEntry",
    "AgentState",
    "utc_now",
]