"""
result

Result hierarchy
----------------
PlannerResult
    Result of one planner invocation.

ToolResult
    Result of one tool invocation, such as observe or execute.

AgentStepResult
    Complete record of one observe -> plan -> execute iteration.

AgentRunResult
    Final result of one GUI Agent task.

Design goals
------------
1. Stable, serializable result schema.
2. No dependency on a concrete Qwen or executor implementation.
3. Preserve raw objects for debugging while exporting safe JSON summaries.
4. Explicit success, failure, retry, and finish states.
5. Accumulate latency and token usage across a complete run.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# Exceptions
# ============================================================


class AgentResultError(ValueError):
    """Base exception raised by invalid result objects."""


class ResultSerializationError(AgentResultError):
    """Raised when a result cannot be converted into JSON-safe data."""


# ============================================================
# Enumerations
# ============================================================


class ResultStatus(str, Enum):
    """
    General execution status.

    PENDING:
        The operation has been created but has not completed.

    SUCCESS:
        The operation completed successfully.

    FAILED:
        The operation completed with an error.

    SKIPPED:
        The operation was intentionally not performed.

    RETRY:
        The operation failed temporarily and should be attempted again.

    CANCELLED:
        The operation or run was cancelled by the caller.

    TIMEOUT:
        The operation exceeded its allowed time.
    """

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class PlannerDecision(str, Enum):
    """Meaning of the planner output."""

    ACT = "act"
    FINISH = "finish"
    RETRY = "retry"
    FAIL = "fail"


class RunTerminationReason(str, Enum):
    """Reason why an Agent run stopped."""

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    PLANNER_FAILED = "planner_failed"
    TOOL_FAILED = "tool_failed"
    EXECUTION_FAILED = "execution_failed"
    USER_CANCELLED = "user_cancelled"
    TIMEOUT = "timeout"
    SAFETY_BLOCKED = "safety_blocked"
    UNKNOWN = "unknown"


# ============================================================
# General helpers
# ============================================================


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _validate_non_negative_number(
    name: str,
    value: int | float,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentResultError(f"{name} must be a number.")

    if not math.isfinite(float(value)) or value < 0:
        raise AgentResultError(
            f"{name} must be finite and non-negative."
        )


def _validate_optional_non_negative_integer(
    name: str,
    value: int | None,
) -> None:
    if value is None:
        return

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise AgentResultError(
            f"{name} must be a non-negative integer or None."
        )


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def to_json_safe(
    value: Any,
    *,
    include_private: bool = False,
    max_depth: int = 12,
    _depth: int = 0,
) -> Any:
    """
    Convert common project objects into JSON-safe values.

    Conversion order:
    - primitive values
    - Enum
    - datetime
    - pathlib.Path
    - mappings and sequences
    - dataclasses
    - object's ``to_dict`` method
    - object's public ``__dict__``
    - string fallback

    Raw screenshots, SDK responses, and executor objects can therefore remain
    attached to result objects without breaking normal JSON export.
    """
    if _depth > max_depth:
        return "<max-depth-reached>"

    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value

        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "size": len(value),
        }

    if isinstance(value, Mapping):
        return {
            str(key): to_json_safe(
                item,
                include_private=include_private,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for key, item in value.items()
            if include_private or not str(key).startswith("_")
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            to_json_safe(
                item,
                include_private=include_private,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    if dataclasses.is_dataclass(value):
        return {
            field_info.name: to_json_safe(
                getattr(value, field_info.name),
                include_private=include_private,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for field_info in dataclasses.fields(value)
            if include_private
            or not field_info.name.startswith("_")
        }

    to_dict_method = getattr(value, "to_dict", None)

    if callable(to_dict_method):
        try:
            converted = to_dict_method()
        except TypeError:
            converted = None
        except Exception:
            converted = None

        if converted is not None:
            return to_json_safe(
                converted,
                include_private=include_private,
                max_depth=max_depth,
                _depth=_depth + 1,
            )

    object_dict = getattr(value, "__dict__", None)

    if isinstance(object_dict, Mapping):
        return {
            str(key): to_json_safe(
                item,
                include_private=include_private,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for key, item in object_dict.items()
            if include_private or not str(key).startswith("_")
        }

    return str(value)


# ============================================================
# Error information
# ============================================================


@dataclass(slots=True)
class ErrorInfo:
    """Serializable error description."""

    error_type: str
    message: str
    code: str | int | None = None
    retryable: bool = False
    traceback: str | None = None
    error_class: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.error_type = self.error_type.strip()
        self.message = self.message.strip()

        if not self.error_type:
            raise AgentResultError(
                "ErrorInfo.error_type must not be empty."
            )

        if not self.message:
            raise AgentResultError(
                "ErrorInfo.message must not be empty."
            )

    @classmethod
    def from_exception(
        cls,
        error: BaseException,
        *,
        retryable: bool = False,
        code: str | int | None = None,
        traceback_text: str | None = None,
        details: Mapping[str, Any] | None = None,
        error_class: str | None = None,
    ) -> "ErrorInfo":
        resolved_code = code

        if resolved_code is None:
            resolved_code = (
                getattr(error, "status_code", None)
                or getattr(error, "code", None)
            )

        return cls(
            error_type=type(error).__name__,
            message=str(error) or type(error).__name__,
            code=resolved_code,
            retryable=retryable,
            traceback=traceback_text,
            error_class=error_class,
            details=dict(details or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "code": self.code,
            "retryable": self.retryable,
            "traceback": self.traceback,
            "error_class": self.error_class,
            "details": to_json_safe(self.details),
        }


# ============================================================
# Usage and timing
# ============================================================


@dataclass(slots=True)
class UsageInfo:
    """Provider-independent token and cost information."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    image_tokens: int | None = None
    cost: float | None = None
    currency: str | None = None
    model: str | None = None
    provider: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "image_tokens",
        ):
            _validate_optional_non_negative_integer(
                name,
                getattr(self, name),
            )

        if self.cost is not None:
            _validate_non_negative_number("cost", self.cost)

        self.normalise_total()

    def normalise_total(self) -> None:
        if self.total_tokens is None:
            known = [
                value
                for value in (
                    self.input_tokens,
                    self.output_tokens,
                )
                if value is not None
            ]

            if known:
                self.total_tokens = sum(known)

    @classmethod
    def from_vlm_usage(
        cls,
        usage: Any,
        *,
        model: str | None = None,
        provider: str | None = None,
        request_id: str | None = None,
    ) -> "UsageInfo":
        if usage is None:
            return cls(
                model=model,
                provider=provider,
                request_id=request_id,
            )

        def read(name: str) -> Any:
            if isinstance(usage, Mapping):
                return usage.get(name)

            return getattr(usage, name, None)

        return cls(
            input_tokens=read("input_tokens"),
            output_tokens=read("output_tokens"),
            total_tokens=read("total_tokens"),
            image_tokens=read("image_tokens"),
            cost=read("cost"),
            currency=read("currency"),
            model=model,
            provider=provider,
            request_id=request_id,
            metadata=dict(read("metadata") or {}),
        )

    def add(self, other: "UsageInfo | None") -> None:
        if other is None:
            return

        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "image_tokens",
        ):
            current = getattr(self, name)
            incoming = getattr(other, name)

            if incoming is not None:
                setattr(
                    self,
                    name,
                    (current or 0) + incoming,
                )

        if other.cost is not None:
            self.cost = (self.cost or 0.0) + other.cost

        if self.currency is None:
            self.currency = other.currency

    def to_dict(self) -> dict[str, Any]:
        self.normalise_total()

        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "image_tokens": self.image_tokens,
            "cost": self.cost,
            "currency": self.currency,
            "model": self.model,
            "provider": self.provider,
            "request_id": self.request_id,
            "metadata": to_json_safe(self.metadata),
        }


@dataclass(slots=True)
class TimingInfo:
    """Timing information for one operation."""

    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None
    latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        _validate_non_negative_number(
            "latency_seconds",
            self.latency_seconds,
        )

        if self.started_at.tzinfo is None:
            self.started_at = self.started_at.replace(
                tzinfo=timezone.utc
            )

        if (
            self.finished_at is not None
            and self.finished_at.tzinfo is None
        ):
            self.finished_at = self.finished_at.replace(
                tzinfo=timezone.utc
            )

    def finish(self) -> None:
        self.finished_at = utc_now()
        self.latency_seconds = max(
            0.0,
            (
                self.finished_at
                - self.started_at
            ).total_seconds(),
        )

    @property
    def is_finished(self) -> bool:
        return self.finished_at is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat()
                if self.finished_at is not None
                else None
            ),
            "latency_seconds": self.latency_seconds,
        }


# ============================================================
# Planner result
# ============================================================


@dataclass(slots=True)
class PlannerResult:
    """
    Result of one planner call.

    ``action`` deliberately uses ``Any``. The concrete executor Action class
    can be stored directly without importing it here, avoiding a circular
    dependency between ``agent`` and ``executor`` packages.
    """

    decision: PlannerDecision = PlannerDecision.ACT
    status: ResultStatus = ResultStatus.SUCCESS
    action: Any = None
    thought: str | None = None
    reason: str | None = None
    observation_summary: str | None = None
    goal_progress: str | None = None
    finish_message: str | None = None
    confidence: float | None = None
    raw_output: str | None = None
    parsed_output: dict[str, Any] = field(default_factory=dict)
    usage: UsageInfo = field(default_factory=UsageInfo)
    timing: TimingInfo = field(default_factory=TimingInfo)
    error: ErrorInfo | None = None
    planner_id: str = field(
        default_factory=lambda: _new_id("plan")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.decision, str):
            self.decision = PlannerDecision(self.decision)

        if isinstance(self.status, str):
            self.status = ResultStatus(self.status)

        if self.confidence is not None:
            if (
                isinstance(self.confidence, bool)
                or not isinstance(self.confidence, (int, float))
                or not math.isfinite(float(self.confidence))
                or not 0.0 <= self.confidence <= 1.0
            ):
                raise AgentResultError(
                    "confidence must be between 0 and 1."
                )

        self.thought = _string_or_none(self.thought)
        self.reason = _string_or_none(self.reason)
        self.observation_summary = _string_or_none(
            self.observation_summary
        )
        self.goal_progress = _string_or_none(self.goal_progress)
        self.finish_message = _string_or_none(
            self.finish_message
        )
        self.raw_output = _string_or_none(self.raw_output)

        self.validate()

    def validate(self) -> None:
        if self.decision == PlannerDecision.ACT:
            if self.action is None and self.status == ResultStatus.SUCCESS:
                raise AgentResultError(
                    "ACT decision requires an action."
                )

        if self.decision == PlannerDecision.FINISH:
            if self.action is not None:
                raise AgentResultError(
                    "FINISH decision must not contain an action."
                )

        if self.decision == PlannerDecision.FAIL:
            if self.error is None and not self.reason:
                raise AgentResultError(
                    "FAIL decision requires error or reason."
                )

        if self.status in {
            ResultStatus.FAILED,
            ResultStatus.TIMEOUT,
        } and self.error is None:
            raise AgentResultError(
                f"{self.status.value} planner result requires error."
            )

    @property
    def should_execute(self) -> bool:
        return (
            self.status == ResultStatus.SUCCESS
            and self.decision == PlannerDecision.ACT
            and self.action is not None
        )

    @property
    def is_finished(self) -> bool:
        return (
            self.status == ResultStatus.SUCCESS
            and self.decision == PlannerDecision.FINISH
        )

    @property
    def should_retry(self) -> bool:
        return (
            self.status == ResultStatus.RETRY
            or self.decision == PlannerDecision.RETRY
            or (
                self.error is not None
                and self.error.retryable
            )
        )

    @classmethod
    def act(
        cls,
        action: Any,
        *,
        reason: str | None = None,
        thought: str | None = None,
        confidence: float | None = None,
        **kwargs: Any,
    ) -> "PlannerResult":
        return cls(
            decision=PlannerDecision.ACT,
            status=ResultStatus.SUCCESS,
            action=action,
            reason=reason,
            thought=thought,
            confidence=confidence,
            **kwargs,
        )

    @classmethod
    def finish(
        cls,
        message: str,
        *,
        reason: str | None = None,
        **kwargs: Any,
    ) -> "PlannerResult":
        return cls(
            decision=PlannerDecision.FINISH,
            status=ResultStatus.SUCCESS,
            finish_message=message,
            reason=reason,
            **kwargs,
        )

    @classmethod
    def retry(
        cls,
        reason: str,
        *,
        error: ErrorInfo | None = None,
        **kwargs: Any,
    ) -> "PlannerResult":
        return cls(
            decision=PlannerDecision.RETRY,
            status=ResultStatus.RETRY,
            reason=reason,
            error=error,
            **kwargs,
        )

    @classmethod
    def failed(
        cls,
        error: ErrorInfo,
        *,
        reason: str | None = None,
        **kwargs: Any,
    ) -> "PlannerResult":
        return cls(
            decision=PlannerDecision.FAIL,
            status=ResultStatus.FAILED,
            error=error,
            reason=reason,
            **kwargs,
        )

    def to_dict(
        self,
        *,
        include_raw_output: bool = True,
        include_action: bool = True,
    ) -> dict[str, Any]:
        return {
            "planner_id": self.planner_id,
            "decision": self.decision.value,
            "status": self.status.value,
            "action": (
                to_json_safe(self.action)
                if include_action
                else None
            ),
            "thought": self.thought,
            "reason": self.reason,
            "observation_summary": self.observation_summary,
            "goal_progress": self.goal_progress,
            "finish_message": self.finish_message,
            "confidence": self.confidence,
            "raw_output": (
                self.raw_output
                if include_raw_output
                else None
            ),
            "parsed_output": to_json_safe(self.parsed_output),
            "usage": self.usage.to_dict(),
            "timing": self.timing.to_dict(),
            "error": (
                self.error.to_dict()
                if self.error is not None
                else None
            ),
            "metadata": to_json_safe(self.metadata),
        }


# ============================================================
# Tool result
# ============================================================


@dataclass(slots=True)
class ToolResult:
    """Result of a Perception, Executor, wait, or utility tool call."""

    tool_name: str
    status: ResultStatus
    output: Any = None
    message: str | None = None
    timing: TimingInfo = field(default_factory=TimingInfo)
    error: ErrorInfo | None = None
    tool_call_id: str = field(
        default_factory=lambda: _new_id("tool")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tool_name = self.tool_name.strip()

        if not self.tool_name:
            raise AgentResultError(
                "tool_name must not be empty."
            )

        if isinstance(self.status, str):
            self.status = ResultStatus(self.status)

        self.message = _string_or_none(self.message)

        if self.status in {
            ResultStatus.FAILED,
            ResultStatus.TIMEOUT,
        } and self.error is None:
            raise AgentResultError(
                f"{self.status.value} ToolResult requires error."
            )

    @property
    def succeeded(self) -> bool:
        return self.status == ResultStatus.SUCCESS

    @property
    def should_retry(self) -> bool:
        return (
            self.status == ResultStatus.RETRY
            or (
                self.error is not None
                and self.error.retryable
            )
        )

    @classmethod
    def success(
        cls,
        tool_name: str,
        *,
        output: Any = None,
        message: str | None = None,
        **kwargs: Any,
    ) -> "ToolResult":
        return cls(
            tool_name=tool_name,
            status=ResultStatus.SUCCESS,
            output=output,
            message=message,
            **kwargs,
        )

    @classmethod
    def failed(
        cls,
        tool_name: str,
        error: ErrorInfo,
        *,
        message: str | None = None,
        **kwargs: Any,
    ) -> "ToolResult":
        return cls(
            tool_name=tool_name,
            status=ResultStatus.FAILED,
            error=error,
            message=message,
            **kwargs,
        )

    def to_dict(
        self,
        *,
        include_output: bool = True,
    ) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "output": (
                to_json_safe(self.output)
                if include_output
                else None
            ),
            "message": self.message,
            "timing": self.timing.to_dict(),
            "error": (
                self.error.to_dict()
                if self.error is not None
                else None
            ),
            "metadata": to_json_safe(self.metadata),
        }


# ============================================================
# One complete Agent step
# ============================================================


@dataclass(slots=True)
class AgentStepResult:
    """
    Complete record of one Agent iteration.

    A step normally contains:
        observation -> planner_result -> execution_result
    """

    step_index: int
    planner_result: PlannerResult
    observation_result: ToolResult | None = None
    execution_result: ToolResult | None = None
    verification_result: ToolResult | None = None
    status: ResultStatus = ResultStatus.PENDING
    timing: TimingInfo = field(default_factory=TimingInfo)
    error: ErrorInfo | None = None
    step_id: str = field(
        default_factory=lambda: _new_id("step")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise AgentResultError(
                "step_index must be a non-negative integer."
            )

        if isinstance(self.status, str):
            self.status = ResultStatus(self.status)

        if self.status == ResultStatus.PENDING:
            self.status = self._infer_status()

        if self.status in {
            ResultStatus.FAILED,
            ResultStatus.TIMEOUT,
        } and self.error is None:
            self.error = self._first_error()

    def _first_error(self) -> ErrorInfo | None:
        candidates = (
            self.observation_result,
            self.planner_result,
            self.execution_result,
            self.verification_result,
        )

        for candidate in candidates:
            if candidate is not None and candidate.error is not None:
                return candidate.error

        return None

    def _infer_status(self) -> ResultStatus:
        for result in (
            self.observation_result,
            self.planner_result,
            self.execution_result,
            self.verification_result,
        ):
            if result is None:
                continue

            if result.status in {
                ResultStatus.FAILED,
                ResultStatus.TIMEOUT,
                ResultStatus.CANCELLED,
            }:
                return result.status

            if result.status == ResultStatus.RETRY:
                return ResultStatus.RETRY

        if self.planner_result.is_finished:
            return ResultStatus.SUCCESS

        if self.planner_result.should_execute:
            if self.execution_result is None:
                return ResultStatus.PENDING

            return self.execution_result.status

        return self.planner_result.status

    @property
    def action(self) -> Any:
        return self.planner_result.action

    @property
    def succeeded(self) -> bool:
        return self.status == ResultStatus.SUCCESS

    @property
    def is_finished(self) -> bool:
        return self.planner_result.is_finished

    @property
    def should_retry(self) -> bool:
        return (
            self.status == ResultStatus.RETRY
            or self.planner_result.should_retry
            or (
                self.execution_result is not None
                and self.execution_result.should_retry
            )
        )

    @property
    def usage(self) -> UsageInfo:
        return self.planner_result.usage

    def complete(
        self,
        *,
        status: ResultStatus | str | None = None,
        error: ErrorInfo | None = None,
    ) -> None:
        if status is not None:
            self.status = ResultStatus(status)
        else:
            self.status = self._infer_status()

        if error is not None:
            self.error = error
        elif self.error is None:
            self.error = self._first_error()

        self.timing.finish()

    def to_dict(
        self,
        *,
        include_raw_output: bool = True,
        include_tool_output: bool = True,
    ) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_index": self.step_index,
            "status": self.status.value,
            "observation_result": (
                self.observation_result.to_dict(
                    include_output=include_tool_output
                )
                if self.observation_result is not None
                else None
            ),
            "planner_result": self.planner_result.to_dict(
                include_raw_output=include_raw_output
            ),
            "execution_result": (
                self.execution_result.to_dict(
                    include_output=include_tool_output
                )
                if self.execution_result is not None
                else None
            ),
            "verification_result": (
                self.verification_result.to_dict(
                    include_output=include_tool_output
                )
                if self.verification_result is not None
                else None
            ),
            "timing": self.timing.to_dict(),
            "error": (
                self.error.to_dict()
                if self.error is not None
                else None
            ),
            "metadata": to_json_safe(self.metadata),
        }


# ============================================================
# Complete Agent run result
# ============================================================


@dataclass(slots=True)
class AgentRunResult:
    """Final result and audit trail of one user task."""

    task: str
    status: ResultStatus = ResultStatus.PENDING
    termination_reason: RunTerminationReason | None = None
    final_message: str | None = None
    steps: list[AgentStepResult] = field(default_factory=list)
    timing: TimingInfo = field(default_factory=TimingInfo)
    usage: UsageInfo = field(default_factory=UsageInfo)
    error: ErrorInfo | None = None
    run_id: str = field(
        default_factory=lambda: _new_id("run")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task = self.task.strip()

        if not self.task:
            raise AgentResultError(
                "AgentRunResult.task must not be empty."
            )

        if isinstance(self.status, str):
            self.status = ResultStatus(self.status)

        if isinstance(self.termination_reason, str):
            self.termination_reason = RunTerminationReason(
                self.termination_reason
            )

        self.final_message = _string_or_none(
            self.final_message
        )

        existing_steps = list(self.steps)
        self.steps = []
        self.usage = UsageInfo()

        for step in existing_steps:
            self.add_step(step)

    @property
    def succeeded(self) -> bool:
        return (
            self.status == ResultStatus.SUCCESS
            and self.termination_reason
            == RunTerminationReason.COMPLETED
        )

    @property
    def completed(self) -> bool:
        return (
            self.termination_reason
            == RunTerminationReason.COMPLETED
        )

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def last_step(self) -> AgentStepResult | None:
        return self.steps[-1] if self.steps else None

    @property
    def actions(self) -> list[Any]:
        return [
            step.action
            for step in self.steps
            if step.action is not None
        ]

    def add_step(self, step: AgentStepResult) -> None:
        if not isinstance(step, AgentStepResult):
            raise TypeError(
                "step must be AgentStepResult."
            )

        expected_index = len(self.steps)

        if step.step_index != expected_index:
            raise AgentResultError(
                "Step indexes must be continuous: "
                f"expected {expected_index}, got {step.step_index}."
            )

        self.steps.append(step)
        self.usage.add(step.usage)

    def finish_success(
        self,
        message: str | None = None,
    ) -> None:
        self.status = ResultStatus.SUCCESS
        self.termination_reason = (
            RunTerminationReason.COMPLETED
        )
        self.final_message = _string_or_none(message)
        self.error = None
        self.timing.finish()

    def finish_failure(
        self,
        *,
        reason: RunTerminationReason | str,
        error: ErrorInfo | None = None,
        message: str | None = None,
        status: ResultStatus | str = ResultStatus.FAILED,
    ) -> None:
        resolved_status = ResultStatus(status)

        if resolved_status not in {
            ResultStatus.FAILED,
            ResultStatus.TIMEOUT,
            ResultStatus.CANCELLED,
        }:
            raise AgentResultError(
                "Failure status must be failed, timeout, or cancelled."
            )

        self.status = resolved_status
        self.termination_reason = RunTerminationReason(reason)
        self.error = error
        self.final_message = _string_or_none(message)
        self.timing.finish()

    def recalculate_usage(self) -> UsageInfo:
        total = UsageInfo()

        for step in self.steps:
            total.add(step.usage)

        self.usage = total
        return total

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "status": self.status.value,
            "termination_reason": (
                self.termination_reason.value
                if self.termination_reason is not None
                else None
            ),
            "final_message": self.final_message,
            "step_count": self.step_count,
            "successful_steps": sum(
                step.succeeded
                for step in self.steps
            ),
            "failed_steps": sum(
                step.status
                in {
                    ResultStatus.FAILED,
                    ResultStatus.TIMEOUT,
                }
                for step in self.steps
            ),
            "latency_seconds": self.timing.latency_seconds,
            "usage": self.usage.to_dict(),
            "error": (
                self.error.to_dict()
                if self.error is not None
                else None
            ),
        }

    def to_dict(
        self,
        *,
        include_steps: bool = True,
        include_raw_output: bool = True,
        include_tool_output: bool = True,
    ) -> dict[str, Any]:
        return {
            **self.summary(),
            "steps": (
                [
                    step.to_dict(
                        include_raw_output=include_raw_output,
                        include_tool_output=include_tool_output,
                    )
                    for step in self.steps
                ]
                if include_steps
                else []
            ),
            "timing": self.timing.to_dict(),
            "metadata": to_json_safe(self.metadata),
        }

    def to_json(
        self,
        *,
        indent: int | None = 2,
        ensure_ascii: bool = False,
        include_steps: bool = True,
        include_raw_output: bool = True,
        include_tool_output: bool = True,
    ) -> str:
        try:
            return json.dumps(
                self.to_dict(
                    include_steps=include_steps,
                    include_raw_output=include_raw_output,
                    include_tool_output=include_tool_output,
                ),
                ensure_ascii=ensure_ascii,
                indent=indent,
            )
        except (TypeError, ValueError) as error:
            raise ResultSerializationError(
                f"Unable to serialize AgentRunResult: {error}"
            ) from error

    def save_json(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
        ensure_ascii: bool = False,
        include_steps: bool = True,
        include_raw_output: bool = True,
        include_tool_output: bool = True,
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
                include_steps=include_steps,
                include_raw_output=include_raw_output,
                include_tool_output=include_tool_output,
            ),
            encoding="utf-8",
        )
        return output_path


__all__ = [
    "AgentResultError",
    "ResultSerializationError",
    "ResultStatus",
    "PlannerDecision",
    "RunTerminationReason",
    "ErrorInfo",
    "UsageInfo",
    "TimingInfo",
    "PlannerResult",
    "ToolResult",
    "AgentStepResult",
    "AgentRunResult",
    "to_json_safe",
    "utc_now",
]