"""
planner

Planner implementation for the GUI Agent.

The Planner is the decision-making layer between AgentState and BaseVLM:

    AgentState
        -> build prompt
        -> call VLM
        -> parse structured response
        -> create Action
        -> return PlannerResult
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence, TYPE_CHECKING

from ..common.target_validation import normalise_target_text, texts_match
from .browser_search import (
    is_forbidden_suggestion_target,
    maybe_rewrite_address_bar_click,
)
from .chat_send import (
    ensure_chat_progress,
    is_chat_send_task,
    maybe_rewrite_chat_action,
    register_chat_composer_target,
    validate_chat_action,
)
from .document_tasks import (
    ensure_close_progress,
    forced_close_hotkey,
    is_close_task,
    is_forbidden_close_target,
    is_status_close_decoy,
    maybe_rewrite_close_action,
    task_instruction,
    validate_close_click_params,
    validate_close_hotkey,
)
from .result import (
    ErrorInfo,
    PlannerDecision,
    PlannerResult,
    ResultStatus,
    TimingInfo,
    UsageInfo,
)
from .state import AgentPhase, AgentState
from .prompts import PromptBuilder, PromptKind

if TYPE_CHECKING:
    from collections.abc import Awaitable


# ============================================================
# Exceptions
# ============================================================


class PlannerError(RuntimeError):
    """Base exception raised by Planner."""


class PlannerConfigurationError(PlannerError):
    """Raised when Planner configuration is invalid."""


class PlannerStateError(PlannerError):
    """Raised when AgentState is not ready for planning."""


class PlannerResponseError(PlannerError):
    """Raised when the VLM response is empty or unusable."""


class PlannerParseError(PlannerResponseError):
    """Raised when the VLM response cannot be parsed as JSON."""


class PlannerValidationError(PlannerResponseError):
    """Raised when parsed planner output violates the schema."""


class PlannerActionError(PlannerResponseError):
    """Raised when an action cannot be created or validated."""


# ============================================================
# Protocols
# ============================================================


class VLMProtocol(Protocol):
    """
    Minimal interface expected from BaseVLM/QwenVLM.

    The concrete model may expose either:
    - generate_json(...)
    - generate(...)
    - generate_request(...)

    Planner resolves the best available method dynamically.
    """

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        ...


class ActionFactoryProtocol(Protocol):
    """Convert normalized action data into a concrete action object."""

    def __call__(
        self,
        action_type: str,
        parameters: Mapping[str, Any],
    ) -> Any:
        ...


# ============================================================
# Enums and constants
# ============================================================


class PlannerOutputMode(str, Enum):
    """Preferred VLM output mode."""

    JSON = "json"
    TEXT = "text"


class InvalidResponsePolicy(str, Enum):
    """What Planner should return after all local attempts fail."""

    RETRY = "retry"
    FAIL = "fail"


class ActionName(str, Enum):
    """Canonical GUI action names understood by the default schema."""

    MOVE_TO = "move_to"
    MOVE_BY = "move_by"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    DRAG_TO = "drag_to"
    DRAG_BY = "drag_by"
    SCROLL = "scroll"
    HORIZONTAL_SCROLL = "horizontal_scroll"
    PRESS = "press"
    HOTKEY = "hotkey"
    TYPE_TEXT = "type_text"
    PASTE_TEXT = "paste_text"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    FINISH = "finish"
    RETRY = "retry"
    FAIL = "fail"


DEFAULT_ALLOWED_ACTIONS: tuple[str, ...] = tuple(
    action.value for action in ActionName
)

FINISH_ALIASES = {
    "finish",
    "finished",
    "done",
    "complete",
    "completed",
    "success",
    "stop",
}

RETRY_ALIASES = {
    "retry",
    "replan",
    "observe_again",
    "try_again",
}

FAIL_ALIASES = {
    "fail",
    "failed",
    "abort",
    "error",
}


# ============================================================
# Configuration
# ============================================================


@dataclass(slots=True)
class PlannerConfig:
    """Configuration for prompt building, parsing, retry, and validation."""

    system_prompt: str | None = None
    output_mode: PlannerOutputMode = PlannerOutputMode.JSON
    invalid_response_policy: InvalidResponsePolicy = (
        InvalidResponsePolicy.RETRY
    )

    max_attempts: int = 2
    retry_on_empty_response: bool = True
    retry_on_parse_error: bool = True
    retry_on_validation_error: bool = True
    retry_on_action_error: bool = False

    temperature: float = 0.0
    max_tokens: int | None = 1200
    timeout_seconds: float | None = None

    include_screenshot: bool = True
    include_previous_observation: bool = True
    history_limit: int = 10
    max_ocr_chars: int = 2000
    max_elements: int = 100

    require_reason: bool = True
    require_confidence: bool = False
    confidence_default: float | None = None

    allowed_actions: tuple[str, ...] = DEFAULT_ALLOWED_ACTIONS
    allow_unknown_actions: bool = False
    validate_coordinates: bool = True
    clamp_coordinates: bool = False
    allow_negative_relative_coordinates: bool = True
    allow_empty_text: bool = False
    max_text_length: int = 10000

    expose_thought: bool = False
    send_thought_field: bool = False
    include_raw_response: bool = True
    include_prompt_in_metadata: bool = False

    model_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.output_mode, str):
            self.output_mode = PlannerOutputMode(self.output_mode)

        if isinstance(self.invalid_response_policy, str):
            self.invalid_response_policy = InvalidResponsePolicy(
                self.invalid_response_policy
            )

        if self.max_attempts <= 0:
            raise PlannerConfigurationError(
                "max_attempts must be greater than zero."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise PlannerConfigurationError(
                "temperature must be between 0 and 2."
            )

        if self.max_tokens is not None and self.max_tokens <= 0:
            raise PlannerConfigurationError(
                "max_tokens must be positive or None."
            )

        if (
            self.timeout_seconds is not None
            and self.timeout_seconds <= 0
        ):
            raise PlannerConfigurationError(
                "timeout_seconds must be positive or None."
            )

        if self.history_limit < 0:
            raise PlannerConfigurationError(
                "history_limit must be non-negative."
            )

        if self.max_ocr_chars <= 0:
            raise PlannerConfigurationError(
                "max_ocr_chars must be positive."
            )

        if self.max_elements <= 0:
            raise PlannerConfigurationError(
                "max_elements must be positive."
            )

        if self.max_text_length <= 0:
            raise PlannerConfigurationError(
                "max_text_length must be positive."
            )

        allowed = []

        for item in self.allowed_actions:
            normalized = str(item).strip().lower()

            if normalized and normalized not in allowed:
                allowed.append(normalized)

        if not allowed:
            raise PlannerConfigurationError(
                "allowed_actions must not be empty."
            )

        self.allowed_actions = tuple(allowed)

        if self.confidence_default is not None:
            if not 0.0 <= self.confidence_default <= 1.0:
                raise PlannerConfigurationError(
                    "confidence_default must be between 0 and 1."
                )


# ============================================================
# Parsed response model
# ============================================================


@dataclass(slots=True)
class ParsedPlannerOutput:
    """Normalized representation of one VLM planner response."""

    decision: PlannerDecision
    action_type: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    thought: str | None = None
    observation_summary: str | None = None
    goal_progress: str | None = None
    finish_message: str | None = None
    confidence: float | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "action_type": self.action_type,
            "parameters": dict(self.parameters),
            "reason": self.reason,
            "thought": self.thought,
            "observation_summary": self.observation_summary,
            "goal_progress": self.goal_progress,
            "finish_message": self.finish_message,
            "confidence": self.confidence,
            "raw_data": dict(self.raw_data),
        }


# ============================================================
# Default action factory
# ============================================================


def dictionary_action_factory(
    action_type: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Default action factory.

    It returns a normalized dictionary. Projects with a concrete Action class
    should inject a custom factory into Planner.
    """
    return {
        "action_type": action_type,
        **dict(parameters),
    }


# ============================================================
# Planner
# ============================================================


class Planner:
    """
    GUI Agent planner.

    Parameters
    ----------
    vlm:
        BaseVLM-compatible instance, such as QwenVLM.

    config:
        Planner configuration.

    action_factory:
        Callable converting normalized action type and parameters into the
        concrete object accepted by Executor.

    prompt_builder:
        Optional custom prompt builder. Signature:
            prompt_builder(state, config) -> str

    response_parser:
        Optional custom response parser. Signature:
            response_parser(response) -> Mapping[str, Any]

    logger:
        Optional logger.
    """

    def __init__(
        self,
        vlm: VLMProtocol,
        *,
        config: PlannerConfig | None = None,
        action_factory: ActionFactoryProtocol | None = None,
        prompt_builder: (
            PromptBuilder
            | Callable[[AgentState, PlannerConfig], str]
            | None
        ) = None,
        response_parser: Callable[
            [Any],
            Mapping[str, Any],
        ]
        | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if vlm is None:
            raise PlannerConfigurationError(
                "vlm must not be None."
            )

        self.vlm = vlm
        self.config = config or PlannerConfig()
        self.action_factory = (
            action_factory or dictionary_action_factory
        )
        self.prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else PromptBuilder(self.config)
        )
        self.response_parser = response_parser
        self.logger = logger or logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def plan(self, state: AgentState) -> PlannerResult:
        """
        Generate the next GUI action or a terminal decision.

        This method never raises ordinary provider/parser exceptions to the
        caller. It returns PlannerResult.retry(...) or PlannerResult.failed(...)
        according to PlannerConfig.
        """
        timing = TimingInfo()
        started = time.perf_counter()
        last_error: ErrorInfo | None = None
        prompt: str | None = None
        raw_response: Any = None
        last_diagnostic: dict[str, Any] | None = None

        try:
            self._validate_state(state)
            if is_close_task(task_instruction(state)):
                ensure_close_progress(state)
            if is_chat_send_task(task_instruction(state)):
                ensure_chat_progress(state)
            prompt = self.build_prompt(state)
            images = self._collect_images(state)
        except Exception as error:
            self.logger.exception("Planner input validation/prompt building failed")
            timing.finish()
            return PlannerResult.failed(
                error=ErrorInfo.from_exception(
                    error,
                    retryable=False,
                ),
                reason="Planner input state is invalid.",
                timing=timing,
                metadata={
                    "attempts": 0,
                    "planner_config": self._config_metadata(),
                    "diagnostic": _exception_diagnostic(error, "planner_input"),
                },
            )

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                raw_response = self.call_vlm(
                    prompt=prompt,
                    images=images,
                    state=state,
                    attempt=attempt,
                )

                parsed_mapping = self.parse_response(raw_response)
                parsed = self.normalize_response(parsed_mapping, state)
                result = self._build_result(
                    parsed=parsed,
                    raw_response=raw_response,
                    timing=timing,
                    prompt=prompt,
                    attempt=attempt,
                )

                timing.finish()
                result.timing = timing
                result.metadata.setdefault(
                    "elapsed_perf_seconds",
                    time.perf_counter() - started,
                )
                return result

            except Exception as error:
                last_diagnostic = _exception_diagnostic(
                    error, f"planner_attempt_{attempt}"
                )
                retryable = self._is_retryable_local_error(error)
                last_error = ErrorInfo.from_exception(
                    error,
                    retryable=retryable,
                    details={
                        "attempt": attempt,
                        "max_attempts": self.config.max_attempts,
                    },
                )

                self.logger.exception(
                    "Planner attempt %s/%s failed: %s",
                    attempt,
                    self.config.max_attempts,
                    error,
                )

                if (
                    attempt >= self.config.max_attempts
                    or not retryable
                ):
                    break

                prompt = self._build_repair_prompt(
                    original_prompt=prompt,
                    error=error,
                    raw_response=raw_response,
                )

        forced = self._forced_close_hotkey_result(
            state,
            timing=timing,
            raw_response=raw_response,
            prompt=prompt,
            diagnostic=last_diagnostic,
            error=last_error,
        )
        if forced is not None:
            return forced

        timing.finish()

        return self._failure_result(
            error=last_error
            or ErrorInfo(
                error_type="PlannerUnknownError",
                message="Planner failed without an exception.",
                retryable=False,
            ),
            timing=timing,
            raw_response=raw_response,
            prompt=prompt,
            diagnostic=last_diagnostic,
        )

    def _forced_close_hotkey_result(
        self,
        state: AgentState,
        *,
        timing: TimingInfo,
        raw_response: Any,
        prompt: str | None,
        diagnostic: Mapping[str, Any] | None,
        error: ErrorInfo | None,
    ) -> PlannerResult | None:
        """Recover a close task by emitting Ctrl+W/Alt+F4 instead of retrying."""

        forced = forced_close_hotkey(state)
        if forced is None:
            return None
        action_type, parameters = forced
        timing.finish()
        action = self.create_action(action_type, parameters)
        metadata = {
            "forced_close_hotkey": True,
            "planner_config": self._config_metadata(),
            "diagnostic": dict(diagnostic or {}),
        }
        if error is not None:
            metadata["recovered_from"] = getattr(error, "message", None) or str(error)
        self.logger.info(
            "Forced close hotkey %s after planner validation failures.",
            "+".join(parameters.get("keys", ())),
        )
        return PlannerResult.act(
            action,
            reason=(
                "Close-control click was unreachable; falling back to the "
                "close hotkey while the target window is active."
            ),
            confidence=0.9,
            metadata=metadata,
            raw_output=self._extract_text(raw_response)
            if self.config.include_raw_response
            else None,
            timing=timing,
        )

    async def aplan(self, state: AgentState) -> PlannerResult:
        """
        Async planner API.

        It uses an async VLM method when available; otherwise it executes the
        synchronous plan method in a worker thread.
        """
        import asyncio

        async_method = self._resolve_async_vlm_method()

        if async_method is None:
            return await asyncio.to_thread(self.plan, state)

        timing = TimingInfo()
        prompt: str | None = None
        raw_response: Any = None
        last_error: ErrorInfo | None = None
        last_diagnostic: dict[str, Any] | None = None

        try:
            self._validate_state(state)
            if is_close_task(task_instruction(state)):
                ensure_close_progress(state)
            if is_chat_send_task(task_instruction(state)):
                ensure_chat_progress(state)
            prompt = self.build_prompt(state)
            images = self._collect_images(state)
        except Exception as error:
            self.logger.exception(
                "Async planner input validation/prompt building failed"
            )
            timing.finish()
            return PlannerResult.failed(
                error=ErrorInfo.from_exception(error),
                reason="Planner input state is invalid.",
                timing=timing,
                metadata={
                    "attempts": 0,
                    "planner_config": self._config_metadata(),
                    "diagnostic": _exception_diagnostic(error, "planner_input"),
                },
            )

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                kwargs = self._build_vlm_kwargs(
                    prompt=prompt,
                    images=images,
                    state=state,
                    attempt=attempt,
                )

                raw_response = await async_method(**kwargs)
                parsed_mapping = self.parse_response(raw_response)
                parsed = self.normalize_response(parsed_mapping, state)
                result = self._build_result(
                    parsed=parsed,
                    raw_response=raw_response,
                    timing=timing,
                    prompt=prompt,
                    attempt=attempt,
                )

                timing.finish()
                result.timing = timing
                return result

            except Exception as error:
                last_diagnostic = _exception_diagnostic(
                    error, f"planner_attempt_{attempt}"
                )
                self.logger.error(
                    "Async planner attempt %s/%s failed: %s",
                    attempt,
                    self.config.max_attempts,
                    error,
                    exc_info=(type(error), error, error.__traceback__),
                )
                retryable = self._is_retryable_local_error(error)
                last_error = ErrorInfo.from_exception(
                    error,
                    retryable=retryable,
                    details={
                        "attempt": attempt,
                        "max_attempts": self.config.max_attempts,
                    },
                )

                if (
                    attempt >= self.config.max_attempts
                    or not retryable
                ):
                    break

                prompt = self._build_repair_prompt(
                    original_prompt=prompt,
                    error=error,
                    raw_response=raw_response,
                )

        forced = self._forced_close_hotkey_result(
            state,
            timing=timing,
            raw_response=raw_response,
            prompt=prompt,
            diagnostic=last_diagnostic,
            error=last_error,
        )
        if forced is not None:
            return forced

        timing.finish()

        return self._failure_result(
            error=last_error
            or ErrorInfo(
                error_type="PlannerUnknownError",
                message="Planner failed without an exception.",
            ),
            timing=timing,
            raw_response=raw_response,
            prompt=prompt,
            diagnostic=last_diagnostic,
        )

    # --------------------------------------------------------
    # State validation
    # --------------------------------------------------------

    def _validate_state(self, state: AgentState) -> None:
        if not isinstance(state, AgentState):
            raise PlannerStateError(
                "state must be AgentState."
            )

        if state.is_terminal:
            raise PlannerStateError(
                f"Cannot plan from terminal phase: {state.phase.value}."
            )

        if state.phase not in {
            AgentPhase.PLANNING,
            AgentPhase.OBSERVING,
            AgentPhase.VERIFYING,
        }:
            raise PlannerStateError(
                "Planner requires PLANNING, OBSERVING, or VERIFYING "
                f"phase; got {state.phase.value}."
            )

        if state.observation is None:
            raise PlannerStateError(
                "Planner requires a current observation."
            )

        if not state.instruction.strip():
            raise PlannerStateError(
                "Agent task instruction is empty."
            )

        if state.runtime.reached_max_steps:
            raise PlannerStateError(
                "Maximum number of steps has already been reached."
            )

        if state.runtime.is_timed_out:
            raise PlannerStateError(
                "Agent runtime deadline has been reached."
            )

    # --------------------------------------------------------
    # Prompt building
    # --------------------------------------------------------

    def build_prompt(self, state: AgentState) -> str:
        """Build the planner prompt from AgentState."""
        if isinstance(self.prompt_builder, PromptBuilder):
            prompt = self.prompt_builder.build_text(
                PromptKind.PLANNER,
                state,
            )
        elif callable(self.prompt_builder):
            prompt = self.prompt_builder(state, self.config)
        else:
            raise PlannerConfigurationError(
                "prompt_builder must be PromptBuilder or callable."
            )

        if not isinstance(prompt, str) or not prompt.strip():
            raise PlannerConfigurationError(
                "prompt_builder returned an empty prompt."
            )

        return prompt.strip()

    def _build_repair_prompt(
        self,
        *,
        original_prompt: str,
        error: Exception,
        raw_response: Any,
    ) -> str:
        raw_text = self._extract_text(raw_response)

        if len(raw_text) > 3000:
            raw_text = raw_text[:3000] + "...<truncated>"

        example = json.dumps(
            {
                "decision": "act",
                "action": {
                    "type": "double_click",
                    "parameters": {"target_text": "Microsoft Edge"},
                },
                "reason": "open desktop shortcut",
                "confidence": 0.95,
            },
            ensure_ascii=False,
        )

        return (
            f"{original_prompt}\n\n"
            "## Previous response was invalid\n"
            f"Validation error: {type(error).__name__}: {error}\n"
            f"Previous response: {raw_text or '<empty>'}\n\n"
            "Return a corrected JSON object only. Do not add commentary.\n"
            "For every click-like action target_text is required and must be "
            "copied from visible OCR text or a GUI label. element_id is "
            "optional and must never be used alone. Never return x/y alone "
            "and never guess an element_id. Use double_click for desktop "
            "shortcuts, files and folders; use click for buttons, menus, tabs "
            "and taskbar icons. If the requested target is not visible, "
            "return retry.\n"
            f"Example: {example}"
        )

    # --------------------------------------------------------
    # Image and VLM call
    # --------------------------------------------------------

    def _collect_images(self, state: AgentState) -> list[Any]:
        if not self.config.include_screenshot:
            return []

        observation = state.observation

        if observation is None:
            return []

        if observation.screenshot_path is not None:
            return [observation.screenshot_path]

        if observation.screenshot is not None:
            return [observation.screenshot]

        return []


    def call_vlm(
        self,
        *,
        prompt: str,
        images: Sequence[Any],
        state: AgentState,
        attempt: int,
    ) -> Any:
        """Call the best available synchronous VLM method."""
        method = self._resolve_vlm_method()
        kwargs = self._build_vlm_kwargs(
            prompt=prompt,
            images=images,
            state=state,
            attempt=attempt,
        )

        return method(**kwargs)

    def _resolve_vlm_method(self) -> Callable[..., Any]:
        preferred_names = []

        if self.config.output_mode == PlannerOutputMode.JSON:
            preferred_names.extend(
                ["generate_json", "generate"]
            )
        else:
            preferred_names.extend(
                ["generate", "generate_json"]
            )

        preferred_names.append("generate_request")

        for name in preferred_names:
            method = getattr(self.vlm, name, None)

            if callable(method):
                return method

        raise PlannerConfigurationError(
            "VLM does not expose generate_json(), generate(), "
            "or generate_request()."
        )

    def _resolve_async_vlm_method(
        self,
    ) -> Callable[..., "Awaitable[Any]"] | None:
        names = (
            "agenerate_json",
            "agenerate",
            "generate_json_async",
            "generate_async",
        )

        for name in names:
            method = getattr(self.vlm, name, None)

            if callable(method):
                return method

        return None

    def _build_vlm_kwargs(
        self,
        *,
        prompt: str,
        images: Sequence[Any],
        state: AgentState,
        attempt: int,
    ) -> dict[str, Any]:
        method = self._resolve_vlm_method()
        signature = self._safe_signature(method)
        kwargs: dict[str, Any] = dict(
            self.config.model_kwargs
        )

        candidate_values = {
            "prompt": prompt,
            "messages": self._build_messages(prompt, images),
            "images": list(images),
            "image": images[0] if images else None,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout_seconds,
            "timeout_seconds": self.config.timeout_seconds,
            "response_format": (
                "json_object"
                if self.config.output_mode
                == PlannerOutputMode.JSON
                else None
            ),
        }

        accepts_var_kwargs = (
            signature is None
            or any(
                parameter.kind
                == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        )

        for key, value in candidate_values.items():
            if value is None:
                continue

            if (
                accepts_var_kwargs
                or signature is None
                or key in signature.parameters
            ):
                kwargs.setdefault(key, value)

        kwargs.setdefault(
            "metadata",
            {
                "run_id": state.run_id,
                "task_id": state.task.task_id,
                "step_index": state.step_index,
                "planner_attempt": attempt,
            },
        )

        if (
            signature is not None
            and "metadata" not in signature.parameters
            and not accepts_var_kwargs
        ):
            kwargs.pop("metadata", None)

        return kwargs

    def _build_messages(
        self,
        prompt: str,
        images: Sequence[Any],
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        for image in images:
            content.append(
                {
                    "type": "image",
                    "image": image,
                }
            )

        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    @staticmethod
    def _safe_signature(
        method: Callable[..., Any],
    ) -> inspect.Signature | None:
        try:
            return inspect.signature(method)
        except (TypeError, ValueError):
            return None

    # --------------------------------------------------------
    # Response parsing
    # --------------------------------------------------------

    def parse_response(
        self,
        response: Any,
    ) -> Mapping[str, Any]:
        """Extract one JSON object from provider-specific response formats."""
        if self.response_parser is not None:
            parsed = self.response_parser(response)

            if not isinstance(parsed, Mapping):
                raise PlannerParseError(
                    "Custom response_parser must return a mapping."
                )

            return parsed

        if response is None:
            raise PlannerResponseError(
                "VLM returned None."
            )

        if isinstance(response, Mapping):
            direct = self._unwrap_mapping_response(response)

            if isinstance(direct, Mapping):
                return direct

        json_method = getattr(response, "json", None)

        if callable(json_method):
            try:
                parsed_json = json_method()
            except Exception:
                parsed_json = None

            if isinstance(parsed_json, Mapping):
                return self._unwrap_mapping_response(
                    parsed_json
                )

        for attribute in (
            "parsed",
            "json_data",
            "data",
            "output_json",
        ):
            value = getattr(response, attribute, None)

            if isinstance(value, Mapping):
                return self._unwrap_mapping_response(value)

        text = self._extract_text(response)

        if not text.strip():
            raise PlannerResponseError(
                "VLM returned an empty response."
            )

        return self._parse_json_text(text)

    def _unwrap_mapping_response(
        self,
        data: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._looks_like_planner_output(data):
            return data

        for key in (
            "parsed",
            "json",
            "result",
            "output",
            "data",
            "content",
        ):
            value = data.get(key)

            if isinstance(value, Mapping):
                if self._looks_like_planner_output(value):
                    return value

            if isinstance(value, str):
                try:
                    parsed = self._parse_json_text(value)
                except PlannerParseError:
                    continue

                if self._looks_like_planner_output(parsed):
                    return parsed

        choices = data.get("choices")

        if isinstance(choices, Sequence) and choices:
            first = choices[0]

            if isinstance(first, Mapping):
                message = first.get("message")

                if isinstance(message, Mapping):
                    content = message.get("content")

                    if isinstance(content, str):
                        return self._parse_json_text(content)

        raise PlannerParseError(
            "Mapping response does not contain planner output."
        )

    @staticmethod
    def _looks_like_planner_output(
        data: Mapping[str, Any],
    ) -> bool:
        keys = {str(key).lower() for key in data}

        return bool(
            keys
            & {
                "decision",
                "action",
                "action_type",
                "type",
                "finish",
                "status",
            }
        )

    def _extract_text(self, response: Any) -> str:
        if response is None:
            return ""

        if isinstance(response, str):
            return response

        if isinstance(response, bytes):
            return response.decode(
                "utf-8",
                errors="replace",
            )

        if isinstance(response, Mapping):
            for key in (
                "text",
                "content",
                "output_text",
                "response",
                "raw_response",
            ):
                value = response.get(key)

                if isinstance(value, str):
                    return value

        for attribute in (
            "text",
            "content",
            "output_text",
            "response",
            "raw_response",
        ):
            value = getattr(response, attribute, None)

            if isinstance(value, str):
                return value

        return str(response)

    def _parse_json_text(
        self,
        text: str,
    ) -> Mapping[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            candidate = self._extract_first_json_object(cleaned)

            if candidate is None:
                raise PlannerParseError(
                    "No JSON object found in VLM response."
                )

            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as error:
                raise PlannerParseError(
                    f"Invalid planner JSON: {error}"
                ) from error

        if not isinstance(parsed, Mapping):
            raise PlannerParseError(
                "Planner response must be one JSON object."
            )

        return parsed

    @staticmethod
    def _extract_first_json_object(
        text: str,
    ) -> str | None:
        start = text.find("{")

        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

                if depth == 0:
                    return text[start : index + 1]

        return None

    # --------------------------------------------------------
    # Normalization and validation
    # --------------------------------------------------------

    def normalize_response(
        self,
        data: Mapping[str, Any],
        state: AgentState,
    ) -> ParsedPlannerOutput:
        raw = dict(data)
        decision = self._normalize_decision(raw)

        reason = self._optional_text(
            raw.get("reason")
            or raw.get("rationale")
            or raw.get("explanation")
        )
        thought = self._optional_text(
            raw.get("thought")
            or raw.get("analysis")
        )
        observation_summary = self._optional_text(
            raw.get("observation_summary")
            or raw.get("observation")
        )
        goal_progress = self._optional_text(
            raw.get("goal_progress")
            or raw.get("progress")
        )
        finish_message = self._optional_text(
            raw.get("finish_message")
            or raw.get("message")
            or raw.get("final_answer")
        )
        confidence = self._normalize_confidence(
            raw.get("confidence")
        )

        if (
            confidence is None
            and self.config.confidence_default is not None
        ):
            confidence = self.config.confidence_default

        if self.config.require_reason and not reason:
            if decision == PlannerDecision.FINISH:
                reason = "The task appears complete."
            elif decision == PlannerDecision.RETRY:
                reason = "The current observation is insufficient."
            elif decision == PlannerDecision.FAIL:
                reason = "The task cannot continue."
            else:
                raise PlannerValidationError(
                    "Planner output requires a non-empty reason."
                )

        if self.config.require_confidence and confidence is None:
            raise PlannerValidationError(
                "Planner output requires confidence."
            )

        action_type: str | None = None
        parameters: dict[str, Any] = {}

        if decision == PlannerDecision.ACT:
            action_type, parameters = self._normalize_action(raw)
            action_type = self._normalize_action_type(
                action_type
            )
            action_type, parameters, rewritten = maybe_rewrite_address_bar_click(
                action_type,
                parameters,
            )
            if rewritten:
                self.logger.info(
                    "Rewrote address-bar click to hotkey Ctrl+L for browser search focus."
                )
            if is_close_task(task_instruction(state)):
                ensure_close_progress(state)
            if is_chat_send_task(task_instruction(state)):
                ensure_chat_progress(state)
            action_type, parameters, closed = maybe_rewrite_close_action(
                action_type,
                parameters,
                state=state,
            )
            if closed:
                self.logger.info(
                    "Rewrote a close-task click to %s.",
                    "+".join(parameters.get("keys", ())),
                )
            action_type, parameters, chat_rewritten = maybe_rewrite_chat_action(
                action_type,
                parameters,
                state=state,
            )
            if chat_rewritten:
                self.logger.info(
                    "Rewrote a chat-task action to %s.",
                    action_type,
                )
            # Chat-phase validation must run before target resolution strips
            # target_text into metadata, or its click checks see empty labels.
            try:
                validate_chat_action(action_type, parameters, state)
            except ValueError as error:
                raise PlannerValidationError(str(error)) from error
            parameters = self._validate_action_parameters(
                action_type,
                parameters,
                state,
            )

        elif decision == PlannerDecision.FINISH:
            if not finish_message:
                finish_message = reason or "Task completed."

        return ParsedPlannerOutput(
            decision=decision,
            action_type=action_type,
            parameters=parameters,
            reason=reason,
            thought=(
                thought
                if self.config.expose_thought
                else None
            ),
            observation_summary=observation_summary,
            goal_progress=goal_progress,
            finish_message=finish_message,
            confidence=confidence,
            raw_data=raw,
        )

    def _normalize_decision(
        self,
        data: Mapping[str, Any],
    ) -> PlannerDecision:
        raw_decision = (
            data.get("decision")
            or data.get("status")
            or data.get("result")
        )

        finish_flag = data.get("finish")

        if isinstance(finish_flag, bool) and finish_flag:
            return PlannerDecision.FINISH

        if raw_decision is not None:
            normalized = str(raw_decision).strip().lower()

            if normalized in {"act", "action", "continue"}:
                return PlannerDecision.ACT

            if normalized in FINISH_ALIASES:
                return PlannerDecision.FINISH

            if normalized in RETRY_ALIASES:
                return PlannerDecision.RETRY

            if normalized in FAIL_ALIASES:
                return PlannerDecision.FAIL

        action_data = data.get("action")

        if isinstance(action_data, Mapping):
            action_type = (
                action_data.get("type")
                or action_data.get("action_type")
                or action_data.get("name")
            )

            if action_type is not None:
                normalized = str(action_type).strip().lower()

                if normalized in FINISH_ALIASES:
                    return PlannerDecision.FINISH

                if normalized in RETRY_ALIASES:
                    return PlannerDecision.RETRY

                if normalized in FAIL_ALIASES:
                    return PlannerDecision.FAIL

                return PlannerDecision.ACT

        if any(
            key in data
            for key in (
                "action_type",
                "type",
                "x",
                "y",
                "text",
                "key",
                "keys",
            )
        ):
            action_type = (
                data.get("action_type")
                or data.get("type")
            )

            if str(action_type).strip().lower() in FINISH_ALIASES:
                return PlannerDecision.FINISH

            return PlannerDecision.ACT

        raise PlannerValidationError(
            "Unable to determine planner decision."
        )

    def _normalize_action(
        self,
        data: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        action = data.get("action")

        if isinstance(action, str):
            return action, self._extract_top_level_parameters(data)

        if isinstance(action, Mapping):
            action_type = (
                action.get("type")
                or action.get("action_type")
                or action.get("name")
            )

            parameters = action.get("parameters")

            if isinstance(parameters, Mapping):
                result = dict(parameters)
            else:
                result = {
                    key: value
                    for key, value in action.items()
                    if key
                    not in {
                        "type",
                        "action_type",
                        "name",
                        "parameters",
                    }
                }

            if action_type is None:
                raise PlannerValidationError(
                    "Action object requires type."
                )

            return str(action_type), result

        action_type = (
            data.get("action_type")
            or data.get("type")
            or data.get("name")
        )

        if action_type is None:
            raise PlannerValidationError(
                "ACT decision requires an action type."
            )

        return (
            str(action_type),
            self._extract_top_level_parameters(data),
        )

    @staticmethod
    def _extract_top_level_parameters(
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        ignored = {
            "decision",
            "status",
            "result",
            "action",
            "action_type",
            "type",
            "name",
            "reason",
            "rationale",
            "explanation",
            "thought",
            "analysis",
            "observation",
            "observation_summary",
            "goal_progress",
            "progress",
            "finish",
            "finish_message",
            "message",
            "final_answer",
            "confidence",
        }

        parameters = data.get("parameters")

        if isinstance(parameters, Mapping):
            result = dict(parameters)
        else:
            result = {}

        for key, value in data.items():
            if key not in ignored:
                result.setdefault(str(key), value)

        return result

    def _normalize_action_type(
        self,
        action_type: str,
    ) -> str:
        normalized = (
            action_type.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        aliases = {
            "tap": "click",
            "left_click": "click",
            "doubleclick": "double_click",
            "double-click": "double_click",
            "dblclick": "double_click",
            "dbl_click": "double_click",
            "rightclick": "right_click",
            "middleclick": "middle_click",
            "move": "move_to",
            "drag": "drag_to",
            "drag_and_drop": "drag_to",
            "type": "type_text",
            "input": "type_text",
            "write": "type_text",
            "keyboard_input": "type_text",
            "paste": "paste_text",
            "paste_text": "paste_text",
            "clipboard_paste": "paste_text",
            "key_press": "press",
            "shortcut": "hotkey",
            "vertical_scroll": "scroll",
            "scroll_vertical": "scroll",
            "scroll_horizontal": "horizontal_scroll",
        }

        normalized = aliases.get(normalized, normalized)

        if (
            normalized not in self.config.allowed_actions
            and not self.config.allow_unknown_actions
        ):
            raise PlannerValidationError(
                f"Unsupported action type: {normalized!r}."
            )

        return normalized

    def _validate_action_parameters(
        self,
        action_type: str,
        parameters: Mapping[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        params = dict(parameters)

        position_actions = {
            "move_to",
            "click",
            "double_click",
            "right_click",
            "middle_click",
            "mouse_down",
            "mouse_up",
            "drag_to",
        }

        relative_actions = {
            "move_by",
            "drag_by",
        }

        if action_type in position_actions:
            if action_type in {
                "click",
                "double_click",
                "right_click",
                "middle_click",
                "mouse_down",
                "mouse_up",
            } and (
                "x" not in params
                and "y" not in params
            ):
                pass
            else:
                params["x"] = self._required_integer(
                    params,
                    "x",
                )
                params["y"] = self._required_integer(
                    params,
                    "y",
                )
                params = self._validate_absolute_coordinates(
                    params,
                    state,
                )

        params = self._resolve_action_target(
            action_type=action_type,
            params=params,
            state=state,
        )

        if action_type in relative_actions:
            params["dx"] = self._required_integer(
                params,
                "dx",
            )
            params["dy"] = self._required_integer(
                params,
                "dy",
            )

            if not self.config.allow_negative_relative_coordinates:
                if params["dx"] < 0 or params["dy"] < 0:
                    raise PlannerValidationError(
                        "Negative relative coordinates are disabled."
                    )

        if action_type in {
            "scroll",
            "horizontal_scroll",
        }:
            amount_key = (
                "amount"
                if "amount" in params
                else "clicks"
                if "clicks" in params
                else "delta"
            )
            params["amount"] = self._required_integer(
                params,
                amount_key,
            )

            for alias in ("clicks", "delta"):
                if alias != "amount":
                    params.pop(alias, None)

        if action_type in {"type_text", "paste_text"}:
            text = params.get("text")

            if text is None:
                text = params.get("value")

            if text is None:
                raise PlannerValidationError(
                    f"{action_type} requires text."
                )

            text = str(text)

            if not text and not self.config.allow_empty_text:
                raise PlannerValidationError(
                    f"{action_type} text must not be empty."
                )

            if len(text) > self.config.max_text_length:
                raise PlannerValidationError(
                    f"{action_type} exceeds max_text_length."
                )

            params["text"] = text
            params.pop("value", None)

        if action_type == "press":
            key = params.get("key")

            if key is None:
                raise PlannerValidationError(
                    "press requires key."
                )

            params["key"] = str(key).strip()

            if not params["key"]:
                raise PlannerValidationError(
                    "press key must not be empty."
                )

        if action_type == "hotkey":
            keys = params.get("keys")

            if isinstance(keys, str):
                keys = [
                    item.strip()
                    for item in re.split(r"[+,]", keys)
                    if item.strip()
                ]

            if not isinstance(keys, Sequence) or not keys:
                raise PlannerValidationError(
                    "hotkey requires a non-empty keys list."
                )

            params["keys"] = [
                str(item).strip()
                for item in keys
                if str(item).strip()
            ]

            if not params["keys"]:
                raise PlannerValidationError(
                    "hotkey keys must not be empty."
                )
            try:
                validate_close_hotkey(params["keys"], state)
            except ValueError as error:
                raise PlannerValidationError(str(error)) from error

        if action_type == "wait":
            duration = (
                params.get("duration")
                if "duration" in params
                else params.get("seconds", 1.0)
            )

            try:
                duration = float(duration)
            except (TypeError, ValueError) as error:
                raise PlannerValidationError(
                    "wait duration must be numeric."
                ) from error

            if not math.isfinite(duration) or duration < 0:
                raise PlannerValidationError(
                    "wait duration must be finite and non-negative."
                )

            params["duration"] = duration
            params.pop("seconds", None)

        if "duration" in params and action_type != "wait":
            try:
                duration = float(params["duration"])
            except (TypeError, ValueError) as error:
                raise PlannerValidationError(
                    "duration must be numeric."
                ) from error

            if not math.isfinite(duration) or duration < 0:
                raise PlannerValidationError(
                    "duration must be finite and non-negative."
                )

            params["duration"] = duration

        if "clicks" in params:
            clicks = self._coerce_integer(
                params["clicks"],
                "clicks",
            )

            if clicks <= 0:
                raise PlannerValidationError(
                    "clicks must be positive."
                )

            params["clicks"] = clicks

        return params

    @staticmethod
    def _point_in_bbox(
        x: int,
        y: int,
        bbox: Sequence[int | float],
    ) -> bool:
        if len(bbox) != 4:
            return False
        left, top, right, bottom = bbox
        return left <= x <= right and top <= y <= bottom

    @staticmethod
    def _element_value(element: Any, name: str, default: Any = None) -> Any:
        if isinstance(element, Mapping):
            return element.get(name, default)
        return getattr(element, name, default)

    @classmethod
    def _element_labels(cls, element: Any) -> list[str]:
        """Return the non-empty text/label/name of a detected element."""

        values = (
            cls._element_value(element, "text", ""),
            cls._element_value(element, "label", ""),
            cls._element_value(element, "name", ""),
        )
        return [str(value).strip() for value in values if str(value).strip()]

    @classmethod
    def _observation_elements(cls, observation: Any) -> list[Any]:
        for name in (
            "elements",
            "merged_elements",
            "gui_elements",
            "ocr_elements",
        ):
            value = getattr(observation, name, None)
            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                return list(value)

        metadata = getattr(observation, "metadata", None)
        if isinstance(metadata, Mapping):
            for name in (
                "target_candidates",
                "elements",
                "merged_elements",
                "gui_elements",
                "ocr_elements",
            ):
                value = metadata.get(name)
                if isinstance(value, Sequence) and not isinstance(
                    value, (str, bytes, bytearray)
                ):
                    return list(value)
        return []

    def _resolve_action_target(
        self,
        *,
        action_type: str,
        params: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        target_actions = {
            "click",
            "double_click",
            "right_click",
            "middle_click",
            "mouse_down",
            "mouse_up",
        }
        if action_type not in target_actions:
            return params

        target_text = str(params.get("target_text", "")).strip()
        raw_element_id = params.get("element_id")
        element_id: int | None = None
        if raw_element_id is not None:
            element_id = self._coerce_integer(raw_element_id, "element_id")
            if element_id < 0:
                raise PlannerValidationError(
                    "element_id must be a non-negative integer."
                )

        if not target_text:
            raise PlannerValidationError(
                f"{action_type} requires target_text. "
                "element_id alone cannot prove that the selected element "
                "matches the task target. Return the visible OCR text or GUI "
                "label and optionally include element_id."
            )

        if is_forbidden_suggestion_target(target_text):
            raise PlannerValidationError(
                f"Refusing to click suggestion/OCR junk target {target_text!r}. "
                "For browser search use hotkey Ctrl+L to focus the address bar, "
                "then paste_text and press enter."
            )

        if is_close_task(task_instruction(state)):
            if is_status_close_decoy(target_text):
                raise PlannerValidationError(
                    f"Refusing to click status chip {target_text!r} for a close "
                    "task. Prefer hotkey Ctrl+W (or Alt+F4); click a close "
                    "control only when OCR shows ✕ / × / 关闭 / Close."
                )
            if is_forbidden_close_target(target_text):
                raise PlannerValidationError(
                    f"Refusing to click menu/panel target {target_text!r} for a "
                    "close task. Prefer hotkey Ctrl+W (or Alt+F4)."
                )

        observation = state.observation
        if observation is None:
            raise PlannerValidationError(
                "Cannot validate a target without an observation."
            )

        elements = self._observation_elements(observation)
        if not elements:
            raise PlannerValidationError(
                "Observation contains no detected target elements."
            )

        target: Any | None = None
        resolved_element_id: int | None = None

        if element_id is not None:
            for index, element in enumerate(elements):
                candidate_id = self._element_value(
                    element, "element_id", index
                )
                try:
                    candidate_id = self._coerce_integer(
                        candidate_id, "detected element_id"
                    )
                except PlannerValidationError:
                    continue
                if candidate_id == element_id:
                    target = element
                    resolved_element_id = candidate_id
                    break
            # An element_id is only an index inside one observation, so an id
            # copied from an earlier step can be absent or point at an unrelated
            # element. target_text stays the authority: drop the id and resolve
            # by text instead of failing the whole plan.
            if target is not None and not any(
                texts_match(target_text, label)
                for label in self._element_labels(target)
            ):
                self.logger.warning(
                    "Ignoring element_id=%s: it does not match target_text=%r "
                    "in the current observation.",
                    element_id,
                    target_text,
                )
                target = None
                resolved_element_id = None
            elif target is None:
                self.logger.warning(
                    "element_id=%s is not present in the current observation; "
                    "resolving target_text=%r instead.",
                    element_id,
                    target_text,
                )

        if target is None and target_text:
            expected = normalise_target_text(target_text)
            matches: list[tuple[int, int, float, int, int, Any]] = []
            
            # 获取屏幕高度用于桌面区域判断
            screen_height = observation.screen_height if observation else 1080
            # 桌面图标偏好只适用于启动类动作（双击桌面快捷方式）。
            prefers_desktop_icon = action_type == "double_click"
            
            for index, element in enumerate(elements):
                # element_type 只是启发式类别，不能作为语义证据，因此不参与匹配：
                # 否则选中的元素会缺少 matched_text，动作在执行前必然被拒绝。
                labels = [
                    label
                    for label in self._element_labels(element)
                    if normalise_target_text(label)
                ]
                score = 0
                if any(normalise_target_text(label) == expected for label in labels):
                    score = 3
                elif any(texts_match(target_text, label) for label in labels):
                    score = 2
                if score:
                    confidence = float(
                        self._element_value(
                            element, "confidence", 0.0
                        ) or 0.0
                    )
                    candidate_id = self._element_value(
                        element, "element_id", index
                    )
                    try:
                        candidate_id = self._coerce_integer(
                            candidate_id, "detected element_id"
                        )
                    except PlannerValidationError:
                        candidate_id = index
                    
                    # 计算bbox大小（用于优先选择小图标）
                    desktop_bonus = 0
                    bbox = self._element_value(element, "bbox", [0, 0, 0, 0])
                    if isinstance(bbox, Sequence) and len(bbox) == 4:
                        width = abs(bbox[2] - bbox[0])
                        height = abs(bbox[3] - bbox[1])
                        bbox_size = width * height
                        
                        # 桌面区域（屏幕下半部分）的小图标：仅在启动类动作里
                        # 作为同分候选之间的次级偏好，绝不能压过精确文本匹配。
                        center_y = (bbox[1] + bbox[3]) / 2
                        is_desktop_area = center_y > screen_height * 0.5
                        is_small_icon = width < 100 and height < 100
                        if prefers_desktop_icon and is_desktop_area and is_small_icon:
                            desktop_bonus = 1
                    else:
                        bbox_size = 999999  # 无效bbox给最大值
                    
                    matches.append(
                        (score, desktop_bonus, confidence, bbox_size, candidate_id, element)
                    )

            if not matches:
                raise PlannerValidationError(
                    f"Target element was not detected: {target_text!r}."
                )
            # 优先级：score最高 -> 桌面图标偏好 -> confidence最高 -> bbox最小
            _, _, _, _, resolved_element_id, target = max(
                matches, key=lambda item: (item[0], item[1], item[2], -item[3])
            )

        if target is None:
            raise PlannerValidationError("No detected target was resolved.")

        detected_text = next(iter(self._element_labels(target)), "")

        # 保留OCR/GUI检测的真实结果，不用target_text覆盖
        # matched_text必须是真实识别结果，后续验证通过标准化和包含关系接受部分匹配
        if not target_text:
            raise PlannerValidationError(
                f"{action_type} requires target_text; "
                "element_id alone is not accepted."
            )

        if not detected_text:
            raise PlannerValidationError(
                f"Resolved element (element_id={resolved_element_id}) has no "
                "text, label or name, so it cannot be verified as "
                f"target_text={target_text!r}."
            )

        if not texts_match(target_text, detected_text):
            raise PlannerValidationError(
                f"target_text={target_text!r} is not supported by detected "
                f"text={detected_text!r} (element_id={resolved_element_id})."
            )
        bbox_value = self._element_value(target, "bbox")
        if not isinstance(bbox_value, Sequence) or len(bbox_value) != 4:
            raise PlannerValidationError(
                f"Detected target {target_text!r} has no valid bbox."
            )

        try:
            bbox = tuple(int(round(float(value))) for value in bbox_value)
        except (TypeError, ValueError) as error:
            raise PlannerValidationError(
                f"Detected target {target_text!r} has an invalid bbox."
            ) from error

        left, top, right, bottom = bbox
        if right < left or bottom < top:
            raise PlannerValidationError(
                f"Detected target {target_text!r} has an inverted bbox={bbox}."
            )

        try:
            validate_close_click_params(
                target_text=target_text,
                bbox=bbox,
                state=state,
            )
        except ValueError as error:
            raise PlannerValidationError(str(error)) from error

        params["x"] = (left + right) // 2
        params["y"] = (top + bottom) // 2
        # Remember the composer identity before Action strips target_text.
        # Phase stays idle until execute succeeds; this only stores label/bbox.
        if is_chat_send_task(task_instruction(state)):
            register_chat_composer_target(
                state,
                target_text=target_text,
                bbox=bbox,
            )
        params.pop("target_text", None)
        params.pop("element_id", None)
        validation = {
            "target_text": target_text,
            "matched_text": detected_text,
            "element_id": resolved_element_id,
            "matched_bbox": list(bbox),
            "center_x": params["x"],
            "center_y": params["y"],
            "coordinate_source": "detected_element_center",
        }
        existing_metadata = params.get("metadata")
        metadata = (
            dict(existing_metadata)
            if isinstance(existing_metadata, Mapping)
            else {}
        )
        metadata["target_validation"] = validation
        params["metadata"] = metadata

        params = self._validate_absolute_coordinates(params, state)
        if not self._point_in_bbox(params["x"], params["y"], bbox):
            raise PlannerValidationError(
                "Resolved center is outside the matched target bbox."
            )
        return params

    def _validate_absolute_coordinates(
        self,
        params: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        if not self.config.validate_coordinates:
            return params

        observation = state.observation

        if observation is None:
            return params

        width = observation.screen_width
        height = observation.screen_height

        if width is None or height is None:
            return params

        x = params["x"]
        y = params["y"]

        if self.config.clamp_coordinates:
            params["x"] = min(max(0, x), width - 1)
            params["y"] = min(max(0, y), height - 1)
            return params

        if not 0 <= x < width:
            raise PlannerValidationError(
                f"x={x} is outside screen width {width}."
            )

        if not 0 <= y < height:
            raise PlannerValidationError(
                f"y={y} is outside screen height {height}."
            )

        return params

    @staticmethod
    def _required_integer(
        data: Mapping[str, Any],
        key: str,
    ) -> int:
        if key not in data:
            raise PlannerValidationError(
                f"Action requires {key}."
            )

        return Planner._coerce_integer(
            data[key],
            key,
        )

    @staticmethod
    def _coerce_integer(
        value: Any,
        name: str,
    ) -> int:
        if isinstance(value, bool):
            raise PlannerValidationError(
                f"{name} must be an integer."
            )

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            if value.is_integer():
                return int(value)

            raise PlannerValidationError(
                f"{name} must be an integer."
            )

        if isinstance(value, str):
            stripped = value.strip()

            if re.fullmatch(r"-?\d+", stripped):
                return int(stripped)

        raise PlannerValidationError(
            f"{name} must be an integer."
        )

    @staticmethod
    def _normalize_confidence(
        value: Any,
    ) -> float | None:
        if value is None or value == "":
            return None

        try:
            confidence = float(value)
        except (TypeError, ValueError) as error:
            raise PlannerValidationError(
                "confidence must be numeric."
            ) from error

        if not math.isfinite(confidence):
            raise PlannerValidationError(
                "confidence must be finite."
            )

        if confidence > 1.0 and confidence <= 100.0:
            confidence = confidence / 100.0

        if not 0.0 <= confidence <= 1.0:
            raise PlannerValidationError(
                "confidence must be between 0 and 1."
            )

        return confidence

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None

        text = str(value).strip()
        return text or None

    # --------------------------------------------------------
    # Action creation and result construction
    # --------------------------------------------------------

    def create_action(
        self,
        action_type: str,
        parameters: Mapping[str, Any],
    ) -> Any:
        try:
            action = self.action_factory(
                action_type,
                parameters,
            )
        except Exception as error:
            raise PlannerActionError(
                f"Action factory failed for {action_type}: {error}"
            ) from error

        if action is None:
            raise PlannerActionError(
                "Action factory returned None."
            )

        validate_method = getattr(action, "validate", None)

        if callable(validate_method):
            try:
                validate_method()
            except Exception as error:
                raise PlannerActionError(
                    f"Concrete action validation failed: {error}"
                ) from error

        return action

    def _build_result(
        self,
        *,
        parsed: ParsedPlannerOutput,
        raw_response: Any,
        timing: TimingInfo,
        prompt: str,
        attempt: int,
    ) -> PlannerResult:
        usage = self._extract_usage(raw_response)
        raw_text = self._extract_text(raw_response)
        metadata = {
            "attempt": attempt,
            "max_attempts": self.config.max_attempts,
            "planner_config": self._config_metadata(),
        }

        if self.config.include_prompt_in_metadata:
            metadata["prompt"] = prompt

        metadata.update(self.config.metadata)

        common = {
            "reason": parsed.reason,
            "thought": parsed.thought,
            "observation_summary": parsed.observation_summary,
            "goal_progress": parsed.goal_progress,
            "confidence": parsed.confidence,
            "raw_output": (
                raw_text
                if self.config.include_raw_response
                else None
            ),
            "parsed_output": parsed.to_dict(),
            "usage": usage,
            "timing": timing,
            "metadata": metadata,
        }

        if parsed.decision == PlannerDecision.ACT:
            if parsed.action_type is None:
                raise PlannerActionError(
                    "ACT decision has no action type."
                )

            action = self.create_action(
                parsed.action_type,
                parsed.parameters,
            )

            return PlannerResult.act(
                action=action,
                **common,
            )

        if parsed.decision == PlannerDecision.FINISH:
            return PlannerResult.finish(
                message=(
                    parsed.finish_message
                    or "Task completed."
                ),
                **common,
            )

        if parsed.decision == PlannerDecision.RETRY:
            return PlannerResult.retry(
                reason=(
                    parsed.reason
                    or "Planner requested another observation."
                ),
                **{
                    key: value
                    for key, value in common.items()
                    if key != "reason"
                },
            )

        error = ErrorInfo(
            error_type="PlannerDeclaredFailure",
            message=(
                parsed.reason
                or "Planner declared that the task cannot continue."
            ),
            retryable=False,
            details=parsed.to_dict(),
        )

        return PlannerResult.failed(
            error=error,
            reason=parsed.reason,
            **{
                key: value
                for key, value in common.items()
                if key != "reason"
            },
        )

    def _failure_result(
        self,
        *,
        error: ErrorInfo,
        timing: TimingInfo,
        raw_response: Any,
        prompt: str | None,
        diagnostic: Mapping[str, Any] | None = None,
    ) -> PlannerResult:
        metadata = {
            "attempts": self.config.max_attempts,
            "planner_config": self._config_metadata(),
            "diagnostic": dict(diagnostic or {}),
        }

        if self.config.include_prompt_in_metadata:
            metadata["prompt"] = prompt

        raw_text = self._extract_text(raw_response)

        if (
            self.config.invalid_response_policy
            == InvalidResponsePolicy.RETRY
        ):
            return PlannerResult.retry(
                reason=(
                    "Planner could not produce a valid action. "
                    "A new observation or another planning attempt is needed."
                ),
                error=error,
                raw_output=(
                    raw_text
                    if self.config.include_raw_response
                    else None
                ),
                timing=timing,
                metadata=metadata,
            )

        return PlannerResult.failed(
            error=error,
            reason="Planner failed to produce a valid result.",
            raw_output=(
                raw_text
                if self.config.include_raw_response
                else None
            ),
            timing=timing,
            metadata=metadata,
        )

    # --------------------------------------------------------
    # Usage extraction
    # --------------------------------------------------------

    def _extract_usage(
        self,
        response: Any,
    ) -> UsageInfo:
        usage = None
        model = None
        provider = None
        request_id = None

        if isinstance(response, Mapping):
            usage = response.get("usage")
            model = response.get("model")
            provider = response.get("provider")
            request_id = (
                response.get("request_id")
                or response.get("id")
            )
        elif response is not None:
            usage = getattr(response, "usage", None)
            model = getattr(response, "model", None)
            provider = getattr(response, "provider", None)
            request_id = (
                getattr(response, "request_id", None)
                or getattr(response, "id", None)
            )

        return UsageInfo.from_vlm_usage(
            usage,
            model=model,
            provider=provider,
            request_id=request_id,
        )

    # --------------------------------------------------------
    # Retry policy
    # --------------------------------------------------------

    def _is_retryable_local_error(
        self,
        error: Exception,
    ) -> bool:
        if isinstance(error, PlannerResponseError):
            if isinstance(error, PlannerParseError):
                return self.config.retry_on_parse_error

            if isinstance(error, PlannerValidationError):
                return self.config.retry_on_validation_error

            if isinstance(error, PlannerActionError):
                return self.config.retry_on_action_error

            return self.config.retry_on_empty_response

        retryable_attr = getattr(error, "retryable", None)

        if isinstance(retryable_attr, bool):
            return retryable_attr

        status_code = (
            getattr(error, "status_code", None)
            or getattr(error, "code", None)
        )

        if status_code in {
            408,
            409,
            425,
            429,
            500,
            502,
            503,
            504,
        }:
            return True

        return False

    # --------------------------------------------------------
    # Metadata and representation
    # --------------------------------------------------------

    def _config_metadata(self) -> dict[str, Any]:
        return {
            "output_mode": self.config.output_mode.value,
            "invalid_response_policy": (
                self.config.invalid_response_policy.value
            ),
            "max_attempts": self.config.max_attempts,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "include_screenshot": self.config.include_screenshot,
            "history_limit": self.config.history_limit,
            "allowed_actions": list(
                self.config.allowed_actions
            ),
            "validate_coordinates": (
                self.config.validate_coordinates
            ),
            "clamp_coordinates": (
                self.config.clamp_coordinates
            ),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vlm={self.vlm.__class__.__name__}, "
            f"output_mode={self.config.output_mode.value!r}, "
            f"max_attempts={self.config.max_attempts}"
            f")"
        )


# ============================================================
# Optional concrete Action adapter
# ============================================================


def build_executor_action_factory(
    action_class: type[Any],
    *,
    action_type_enum: type[Enum] | None = None,
    action_type_field: str = "action_type",
) -> ActionFactoryProtocol:
    """
    Build an action factory for the project's concrete executor Action class.

    Examples
    --------
    factory = build_executor_action_factory(
        Action,
        action_type_enum=ActionType,
    )

    planner = Planner(
        vlm=qwen,
        action_factory=factory,
    )

    The helper supports either:
    - Action(action_type=ActionType.CLICK, x=..., y=...)
    - Action(type=..., ...)
    - a dataclass-like constructor accepting normalized parameters.
    """

    def factory(
        action_type: str,
        parameters: Mapping[str, Any],
    ) -> Any:
        resolved_type: Any = action_type

        if action_type_enum is not None:
            try:
                resolved_type = action_type_enum(action_type)
            except ValueError:
                try:
                    resolved_type = action_type_enum[
                        action_type.upper()
                    ]
                except (KeyError, TypeError) as error:
                    raise PlannerActionError(
                        "Unable to map action type "
                        f"{action_type!r} to {action_type_enum.__name__}."
                    ) from error

        kwargs = dict(parameters)
        kwargs[action_type_field] = resolved_type

        try:
            return action_class(**kwargs)
        except TypeError as first_error:
            alternate_field = (
                "type"
                if action_type_field != "type"
                else "action_type"
            )
            alternate_kwargs = dict(parameters)
            alternate_kwargs[alternate_field] = resolved_type

            try:
                return action_class(**alternate_kwargs)
            except TypeError:
                raise PlannerActionError(
                    "Unable to construct concrete Action. "
                    f"Primary error: {first_error}"
                ) from first_error

    return factory


def _exception_diagnostic(
    error: BaseException,
    stage: str,
) -> dict[str, Any]:
    """Capture the original stack and chained causes before returning failed."""
    causes: list[dict[str, str]] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        causes.append(
            {
                "type": type(current).__name__,
                "message": str(current),
            }
        )
        current = current.__cause__ or current.__context__
    return {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
        "cause": causes,
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }


__all__ = [
    "PlannerError",
    "PlannerConfigurationError",
    "PlannerStateError",
    "PlannerResponseError",
    "PlannerParseError",
    "PlannerValidationError",
    "PlannerActionError",
    "VLMProtocol",
    "ActionFactoryProtocol",
    "PlannerOutputMode",
    "InvalidResponsePolicy",
    "ActionName",
    "PlannerConfig",
    "ParsedPlannerOutput",
    "dictionary_action_factory",
    "build_executor_action_factory",
    "Planner",
]