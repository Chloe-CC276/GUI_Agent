"""
agent_chain
LangChain orchestration for the GUI Agent.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Mapping, Protocol, TypedDict

from ..common.target_validation import (
    coerce_action_mapping,
    validate_click_target,
)
from .memory import AgentMemory, MemoryImportance, MemoryKind
from .result import ErrorInfo, ResultStatus, RunTerminationReason, ToolResult
from .state import AgentPhase, AgentState, ObservationState
from .tools import AgentTools
from .verify_policy import latest_action_type, should_skip_verification
from .browser_search import (
    apply_focus_verify_override,
    build_focus_success_verify,
    detect_search_box_focus,
    is_address_bar_focus_action,
)

from .prompts import PromptBuilder, PromptKind
from .prompts.schemas import REFLECTION_RESPONSE_SCHEMA, VERIFY_RESPONSE_SCHEMA

logger = logging.getLogger(__name__)


try:
    from langchain_core.runnables import Runnable, RunnableLambda, RunnableSequence
except ImportError:
    Runnable = Any  # type: ignore[assignment,misc]
    RunnableLambda = None  # type: ignore[assignment]
    RunnableSequence = None  # type: ignore[assignment]


class AgentChainError(RuntimeError):
    """Base error raised by LangChain orchestration."""


class AgentChainDependencyError(AgentChainError):
    """Raised when ``langchain-core`` is unavailable."""


class ModelResponseError(AgentChainError):
    """Raised when a verifier/reflection response cannot be used."""


class PlannerProtocol(Protocol):
    def plan(self, state: AgentState) -> Any: ...
    async def aplan(self, state: AgentState) -> Any: ...


class VLMProtocol(Protocol):
    def generate(self, *args: Any, **kwargs: Any) -> Any: ...


class ChainStage(str, Enum):
    OBSERVE = "observe"
    PLAN = "plan"
    EXECUTE = "execute"
    OBSERVE_AFTER = "observe_after"
    VERIFY = "verify"
    REFLECT = "reflect"
    MEMORY = "memory"
    FINISH = "finish"
    FAIL = "fail"


class ChainState(TypedDict, total=False):
    """Mutable envelope passed between LCEL stages."""

    agent_state: AgentState
    stage: str
    verify_data: dict[str, Any]
    reflection_data: dict[str, Any]
    last_prompt: str
    last_raw_response: Any
    chain_error: str
    failed_stage: str
    error_type: str
    traceback: str
    error_details: dict[str, Any]


@dataclass(slots=True)
class AgentChainConfig:
    """Runtime policy for the independent LangChain loop."""

    observe_options: dict[str, Any] = field(default_factory=dict)
    post_action_wait_seconds: float = 0.5
    verify_confidence_threshold: float = 0.55
    max_reflections: int = 2
    max_model_repairs: int = 1
    max_planner_retries: int = 2
    summarize_memory: bool = True
    fail_on_execution_error: bool = False
    max_chain_iterations: int = 20

    def __post_init__(self) -> None:
        if not 0 <= self.post_action_wait_seconds <= 60:
            raise ValueError("post_action_wait_seconds must be between 0 and 60")
        if not 0 <= self.verify_confidence_threshold <= 1:
            raise ValueError("verify_confidence_threshold must be between 0 and 1")
        if (
            self.max_reflections < 0
            or self.max_model_repairs < 0
            or self.max_planner_retries < 0
        ):
            raise ValueError(
                "reflection, repair and planner retry limits "
                "must be non-negative"
            )

        if self.max_chain_iterations <= 0:
            raise ValueError(
                "max_chain_iterations must be positive"
            )


class AgentChain:
    """Orchestrate the GUI Agent exclusively with LangChain Runnables."""

    def __init__(
        self,
        *,
        planner: PlannerProtocol,
        tools: AgentTools,
        vlm: VLMProtocol,
        prompt_builder: PromptBuilder | None = None,
        memory: AgentMemory | None = None,
        memory_summarizer: Callable[..., Any] | None = None,
        config: AgentChainConfig | None = None,
    ) -> None:
        if planner is None or tools is None or vlm is None:
            raise ValueError("planner, tools, and vlm are required")
        self.planner = planner
        self.tools = tools
        self.vlm = vlm
        self.prompts = prompt_builder or PromptBuilder()
        self.memory = memory or AgentMemory()
        self.memory_summarizer = memory_summarizer
        self.config = config or AgentChainConfig()
        self._chain: Runnable | None = None

    def build(self) -> Runnable:
        """Build ``prepare | feedback_loop | finalise`` as an LCEL chain."""
        if RunnableLambda is None or RunnableSequence is None:
            raise AgentChainDependencyError(
                "LangChain Core is not installed. Run: pip install langchain-core"
            )
        prepare = RunnableLambda(self._prepare_input, name="prepare_agent_state")
        loop = RunnableLambda(
            self._run_loop_sync,
            afunc=self._run_loop,
            name="gui_agent_feedback_loop",
        )
        finalise = RunnableLambda(self._finalise_output, name="build_run_result")
        return RunnableSequence(prepare, loop, finalise)

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            self._chain = self.build()
        return self._chain

    def invoke(self, value: AgentState | Mapping[str, Any], **kwargs: Any) -> Any:
        return self.chain.invoke(value, **kwargs)

    async def ainvoke(
        self, value: AgentState | Mapping[str, Any], **kwargs: Any
    ) -> Any:
        return await self.chain.ainvoke(value, **kwargs)

    def run(self, state: AgentState, **kwargs: Any) -> Any:
        return self.invoke(state, **kwargs)

    async def arun(self, state: AgentState, **kwargs: Any) -> Any:
        return await self.ainvoke(state, **kwargs)

    def batch(self, states: list[AgentState], **kwargs: Any) -> list[Any]:
        return self.chain.batch(states, **kwargs)

    def stream_steps(self, state: AgentState) -> Iterator[ChainState]:
        """Yield each completed stage without requiring LangGraph streaming."""
        context = self._prepare_input(state)
        for _ in range(self.config.max_chain_iterations):
            if self._terminal(context):
                break
            completed_stage = str(context.get("stage", ChainStage.FAIL.value))
            stage_started = time.perf_counter()
            context = asyncio.run(self._dispatch(context))
            context["completed_stage"] = completed_stage
            context["stage_elapsed_seconds"] = time.perf_counter() - stage_started
            yield context
        else:
            yield self._iteration_limit_failure(context)

    def _prepare_input(self, value: AgentState | Mapping[str, Any]) -> ChainState:
        state = value.get("agent_state") if isinstance(value, Mapping) else value
        if not isinstance(state, AgentState):
            raise TypeError("input must be AgentState or {'agent_state': AgentState}")
        if state.phase == AgentPhase.CREATED:
            state.begin()
        elif state.is_terminal:
            raise AgentChainError("Cannot run a terminal AgentState")
        return {"agent_state": state, "stage": ChainStage.OBSERVE.value}

    async def _run_loop(self, context: ChainState) -> ChainState:
        for _ in range(self.config.max_chain_iterations):
            if self._terminal(context):
                return context
            context = await self._dispatch(context)
        return self._iteration_limit_failure(context)

    def _run_loop_sync(self, context: ChainState) -> ChainState:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._run_loop(context))
        raise AgentChainError(
            "invoke() cannot run inside an active event loop; use await ainvoke()"
        )

    async def _dispatch(self, context: ChainState) -> ChainState:
        handlers = {
            ChainStage.OBSERVE.value: self._observe,
            ChainStage.PLAN.value: self._plan,
            ChainStage.EXECUTE.value: self._execute,
            ChainStage.OBSERVE_AFTER.value: self._observe_after,
            ChainStage.VERIFY.value: self._verify,
            ChainStage.REFLECT.value: self._reflect,
            ChainStage.MEMORY.value: self._update_memory,
            ChainStage.FINISH.value: self._finish,
            ChainStage.FAIL.value: self._fail,
        }
        stage = str(context.get("stage", ChainStage.FAIL.value))
        handler = handlers.get(stage)
        if handler is None:
            return self._fail_update(
                context["agent_state"], f"Unknown AgentChain stage: {stage!r}"
            )
        try:
            return await handler(context)
        except Exception as error:
            logger.exception("AgentChain stage failed unexpectedly: stage=%s", stage)
            return self._exception_failure(
                context["agent_state"], error, stage
            )

    @staticmethod
    def _terminal(context: ChainState) -> bool:
        state = context["agent_state"]
        return state.is_terminal and context.get("stage") in {
            ChainStage.FINISH.value,
            ChainStage.FAIL.value,
        }

    async def _observe(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        if self._limit_reached(state):
            return self._fail_update(state, "Agent reached its step/time limit.")
        result = await self.tools.acall("observe", self.config.observe_options)
        if not result.succeeded or not isinstance(result.output, ObservationState):
            return self._tool_failure(state, result, "observation")
        state.update_observation(result.output, tool_result=result)
        return {"agent_state": state, "stage": ChainStage.PLAN.value}

    async def _plan(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        if self._limit_reached(state):
            return self._fail_update(state, "Agent reached its step/time limit.")
        try:
            async_method = getattr(self.planner, "aplan", None)
            result = (
                await async_method(state)
                if callable(async_method)
                else await asyncio.to_thread(self.planner.plan, state)
            )
            state.update_planner_result(result)
        except Exception as error:
            return self._exception_failure(state, error, "planning")
        if state.is_terminal:
            return self._planner_failure_update(state, result)
        if result.is_finished:
            state.metadata["planner_retry_count"] = 0

            return {
                "agent_state": state,
                "stage": ChainStage.FINISH.value,
            }
        if result.should_execute:
            state.metadata["planner_retry_count"] = 0

            return {
                "agent_state": state,
                "stage": ChainStage.EXECUTE.value,
            }
        if result.should_retry:
            retry_count = int(
                state.metadata.get("planner_retry_count", 0)
            ) + 1

            state.metadata["planner_retry_count"] = retry_count

            retry_reason = (
                result.reason
                or "Planner requested another observation."
            )

            state.add_history(
                event_type="planner_retry",
                message=retry_reason,
                status=ResultStatus.RETRY,
                metadata={
                    "retry_count": retry_count,
                    "max_retries": self.config.max_planner_retries,
                    "raw_output": getattr(result, "raw_output", None),
                },
            )

            if retry_count > self.config.max_planner_retries:
                return self._fail_update(
                    state,
                    (
                        "Planner retry limit reached: "
                        f"{retry_count - 1}/"
                        f"{self.config.max_planner_retries}. "
                        f"Last reason: {retry_reason}"
                    ),
                    getattr(result, "error", None),
                )

            return {
                "agent_state": state,
                "stage": ChainStage.OBSERVE.value,
                "chain_error": retry_reason,
                "planner_retry_count": retry_count,
            }

        return self._fail_update(
            state,
            getattr(result, "reason", None)
            or "Planner returned no usable decision.",
            getattr(result, "error", None),
        )

    @staticmethod
    def _action_mapping(action: Any) -> dict[str, Any]:
        return coerce_action_mapping(action)

    @staticmethod
    def _validate_action_target(state: AgentState) -> str | None:
        return validate_click_target(state.latest_action, require_evidence=True)

    async def _execute(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        validation_error = self._validate_action_target(state)

        if validation_error:
            logger.warning(
                "Executor未调用：目标校验失败 | error=%s | action=%r",
                validation_error,
                state.latest_action,
            )

            target_rejection_count = (
                int(state.metadata.get("target_rejection_count", 0)) + 1
            )
            state.metadata["target_rejection_count"] = target_rejection_count

            state.add_history(
                event_type="action_rejected",
                message=validation_error,
                status=ResultStatus.RETRY,
                metadata={
                    "rejection_count": target_rejection_count,
                    "max_rejections": 2,
                    "action": self._action_mapping(state.latest_action),
                },
            )

            if target_rejection_count > 2:
                return self._fail_update(
                    state,
                    "Target validation failed repeatedly: "
                    f"{validation_error}",
                )

            return {
                "agent_state": state,
                "stage": ChainStage.OBSERVE.value,
                "chain_error": validation_error,
            }

        # 目标校验成功，清零连续拒绝次数
        state.metadata["target_rejection_count"] = 0

        result = await self.tools.acall(
            "execute_action",
            {"action": state.latest_action},
        )

        if not result.succeeded and not self.config.fail_on_execution_error:
            result.status = ResultStatus.RETRY
            if result.error is not None:
                result.error.retryable = True

        state.update_execution_result(result)

        if state.is_terminal:
            return {
                "agent_state": state,
                "stage": ChainStage.FAIL.value,
            }

        if result.succeeded and self._should_skip_verification(state.latest_action):
            action_type = self._latest_action_type(state.latest_action)
            state.add_history(
                event_type="verify_skipped",
                message=(
                    f"Skipped verification after {action_type}; "
                    "continuing to plan the next search step."
                ),
                status=ResultStatus.SUCCESS,
                metadata={"action_type": action_type},
            )
            self._commit_current_step(state)
            if state.is_terminal:
                return {
                    "agent_state": state,
                    "stage": ChainStage.FAIL.value,
                }
            # Keep the pre-paste observation so the planner can submit immediately.
            state.set_phase(
                AgentPhase.PLANNING if state.observation is not None else AgentPhase.OBSERVING
            )
            return {
                "agent_state": state,
                "stage": (
                    ChainStage.PLAN.value
                    if state.observation is not None
                    else ChainStage.OBSERVE.value
                ),
            }

        stage = (
            ChainStage.OBSERVE_AFTER
            if result.succeeded
            else ChainStage.REFLECT
        )

        return {
            "agent_state": state,
            "stage": stage.value,
        }

    @staticmethod
    def _latest_action_type(action: Any) -> str:
        return latest_action_type(action)

    @classmethod
    def _should_skip_verification(cls, action: Any) -> bool:
        return should_skip_verification(action)

    async def _observe_after(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        if self.config.post_action_wait_seconds:
            await self.tools.acall(
                "wait", {"seconds": self.config.post_action_wait_seconds}
            )
        result = await self.tools.acall("observe", self.config.observe_options)
        if not result.succeeded or not isinstance(result.output, ObservationState):
            return self._tool_failure(state, result, "post-action observation")
        state.update_observation(result.output, tool_result=result)
        state.set_phase(AgentPhase.VERIFYING)
        return {"agent_state": state, "stage": ChainStage.VERIFY.value}

    async def _verify(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        prompt = ""
        raw: Any = None
        try:
            if is_address_bar_focus_action(state.latest_action):
                detected, evidence = detect_search_box_focus(state.observation)
                if detected:
                    data = build_focus_success_verify(
                        evidence=evidence,
                        before=state.previous_observation,
                        after=state.observation,
                    )
                    state.add_history(
                        event_type="verify_focus_detected",
                        message=str(data["reason"]),
                        status=ResultStatus.SUCCESS,
                        metadata={"verify": data, "evidence": evidence},
                    )
                    metadata = dict(getattr(state.observation, "metadata", None) or {})
                    metadata["search_focus_detected"] = True
                    metadata["search_focus_evidence"] = list(evidence)
                    state.observation.metadata = metadata
                else:
                    prompt = self.prompts.build_text(PromptKind.VERIFY, state)
                    data, raw = await self._call_structured(
                        prompt, VERIFY_RESPONSE_SCHEMA, PromptKind.VERIFY
                    )
                    self._validate_verify(data)
                    data, overridden = apply_focus_verify_override(
                        data,
                        action=state.latest_action,
                        before=state.previous_observation,
                        after=state.observation,
                    )
                    if overridden:
                        state.add_history(
                            event_type="verify_focus_override",
                            message=str(data.get("reason") or "Focus verify override"),
                            status=ResultStatus.SUCCESS,
                            metadata={"verify": data},
                        )
                        if state.observation is not None:
                            metadata = dict(getattr(state.observation, "metadata", None) or {})
                            metadata["search_focus_detected"] = True
                            state.observation.metadata = metadata
            else:
                prompt = self.prompts.build_text(PromptKind.VERIFY, state)
                data, raw = await self._call_structured(
                    prompt, VERIFY_RESPONSE_SCHEMA, PromptKind.VERIFY
                )
                self._validate_verify(data)
                data, overridden = apply_focus_verify_override(
                    data,
                    action=state.latest_action,
                    before=state.previous_observation,
                    after=state.observation,
                )
                if overridden:
                    state.add_history(
                        event_type="verify_focus_override",
                        message=str(data.get("reason") or "Focus verify override"),
                        status=ResultStatus.SUCCESS,
                        metadata={"verify": data},
                    )
                    if state.observation is not None:
                        metadata = dict(getattr(state.observation, "metadata", None) or {})
                        metadata["search_focus_detected"] = True
                        state.observation.metadata = metadata
        except Exception as error:
            return self._exception_reflection(state, error, "verification")

        succeeded = bool(data["action_effective"]) and data["status"] == "success"
        complete = bool(data["task_complete"])
        confidence = float(data["confidence"])
        verification = ToolResult(
            tool_name="verify_action",
            status=ResultStatus.SUCCESS if succeeded else ResultStatus.RETRY,
            output=data,
            message=str(data["reason"]),
            metadata={"raw_response": raw},
        )
        state.update_verification_result(verification)
        shared: ChainState = {
            "agent_state": state,
            "verify_data": data,
            "last_prompt": prompt,
            "last_raw_response": raw,
        }
        if complete and confidence >= self.config.verify_confidence_threshold:
            self._commit_current_step(state)
            shared["stage"] = ChainStage.FINISH.value
        elif succeeded and str(data["recommended_next"]) in {"continue", "replan"}:
            self._commit_current_step(state)
            shared["stage"] = ChainStage.MEMORY.value
        else:
            shared["stage"] = ChainStage.REFLECT.value
        return shared

    async def _reflect(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        count = int(state.metadata.get("reflection_count", 0)) + 1
        state.metadata["reflection_count"] = count
        if count > self.config.max_reflections:
            return self._fail_update(state, "Maximum reflection count reached.")
        try:
            prompt = self.prompts.build_text(PromptKind.REFLECTION, state)
            data, raw = await self._call_structured(
                prompt, REFLECTION_RESPONSE_SCHEMA, PromptKind.REFLECTION
            )
            self._validate_reflection(data)
        except Exception as error:
            return self._exception_failure(state, error, "reflection")

        summary = str(data["summary"])
        state.add_history(
            event_type="reflection",
            message=summary,
            status=ResultStatus.RETRY,
            metadata={"reflection": data},
        )
        self.memory.add(
            MemoryKind.REFLECTION,
            summary,
            step_index=state.step_index,
            importance=MemoryImportance.HIGH,
            evidence=data.get("evidence", []),
            payload=data,
            tags=[str(data.get("failure_type", "unknown"))],
        )
        self.memory.attach_to_state(state)
        self._commit_current_step(state)
        if state.is_terminal or self._limit_reached(state):
            return self._fail_update(state, "Agent reached its step/time limit.")
        stage = ChainStage.PLAN if bool(data["should_replan"]) else ChainStage.OBSERVE
        if stage is ChainStage.PLAN and state.observation is None:
            stage = ChainStage.OBSERVE
        state.set_phase(
            AgentPhase.PLANNING if stage is ChainStage.PLAN else AgentPhase.OBSERVING
        )
        return {
            "agent_state": state,
            "stage": stage.value,
            "reflection_data": data,
            "last_prompt": prompt,
            "last_raw_response": raw,
        }

    async def _update_memory(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        self.memory.ingest_state(state, recent_only=20)
        if (
            self.config.summarize_memory
            and self.memory_summarizer is not None
            and self.memory.should_summarize()
        ):
            try:
                await self.memory.asummarize(state, self.memory_summarizer)
            except Exception as error:
                state.add_history(
                    event_type="memory_summary_failed",
                    message=str(error),
                    status=ResultStatus.RETRY,
                )
        self.memory.attach_to_state(state)
        if state.is_terminal:
            stage = ChainStage.FINISH if state.is_finished else ChainStage.FAIL
        elif self._limit_reached(state):
            return self._fail_update(state, "Agent reached its step/time limit.")
        else:
            state.set_phase(AgentPhase.OBSERVING)
            stage = ChainStage.OBSERVE
        return {"agent_state": state, "stage": stage.value}

    async def _finish(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        if not state.is_terminal:
            planner_result = state.last_planner_result
            verify = context.get("verify_data", {})
            message = (
                verify.get("reason")
                or getattr(planner_result, "finish_message", None)
                or getattr(planner_result, "reason", None)
                or "Task completed."
            )
            state.finish(str(message))
        self.memory.ingest_state(state, recent_only=20)
        self.memory.attach_to_state(state)
        return {"agent_state": state, "stage": ChainStage.FINISH.value}

    async def _fail(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        if not state.is_terminal:
            message = context.get("chain_error") or "Agent chain failed."
            state.fail(
                error=ErrorInfo(error_type="AgentChainFailure", message=message),
                reason=RunTerminationReason.UNKNOWN,
                message=message,
            )
        self.memory.ingest_state(state, recent_only=20)
        self.memory.attach_to_state(state)
        return {"agent_state": state, "stage": ChainStage.FAIL.value}

    async def _call_structured(
        self,
        prompt: str,
        schema: Mapping[str, Any],
        target_kind: PromptKind,
    ) -> tuple[dict[str, Any], Any]:
        current_prompt = prompt
        raw: Any = None
        last_error: Exception | None = None
        for attempt in range(self.config.max_model_repairs + 1):
            try:
                raw = await self._call_vlm(current_prompt, schema)
                return _parse_json_object(raw), raw
            except Exception as error:
                last_error = error
                if attempt >= self.config.max_model_repairs:
                    break
                current_prompt = self.prompts.build_text(
                    PromptKind.REPAIR,
                    prompt,
                    invalid_response=raw,
                    validation_error=error,
                    target_kind=target_kind,
                    response_schema=schema,
                )
        raise ModelResponseError(str(last_error or "Model returned no usable response"))

    async def _call_vlm(self, prompt: str, schema: Mapping[str, Any]) -> Any:
        for name in ("agenerate_json", "agenerate", "generate_json", "generate"):
            method = getattr(self.vlm, name, None)
            if not callable(method):
                continue
            value = method(**_supported_kwargs(method, prompt=prompt, schema=dict(schema)))
            return await value if inspect.isawaitable(value) else value
        raise AgentChainError("VLM exposes no supported generate method")

    @staticmethod
    def _validate_verify(data: Mapping[str, Any]) -> None:
        _require_fields(
            data,
            {
                "status", "action_effective", "task_complete", "evidence",
                "reason", "confidence", "recommended_next",
            },
            "verify",
        )
        if data["status"] not in {"success", "failure", "uncertain"}:
            raise ModelResponseError("Invalid verify status")
        if not 0 <= float(data["confidence"]) <= 1:
            raise ModelResponseError("Verify confidence must be between 0 and 1")

    @staticmethod
    def _validate_reflection(data: Mapping[str, Any]) -> None:
        _require_fields(
            data,
            {
                "failure_type", "summary", "evidence", "likely_cause",
                "avoid", "strategy", "should_replan", "confidence",
            },
            "reflection",
        )
        if not isinstance(data["should_replan"], bool):
            raise ModelResponseError("should_replan must be boolean")

    @staticmethod
    def _commit_current_step(state: AgentState) -> None:
        if state.last_planner_result is None:
            return
        try:
            state.commit_step()
        except Exception:
            if not state.last_planner_result.is_finished:
                raise

    @staticmethod
    def _limit_reached(state: AgentState) -> bool:
        return bool(state.runtime.reached_max_steps or state.runtime.is_timed_out)

    def _tool_failure(
        self, state: AgentState, result: ToolResult, stage: str
    ) -> ChainState:
        message = result.message or f"{stage} failed"
        if result.should_retry and not self._limit_reached(state):
            state.add_history(
                event_type=f"{stage}_retry",
                message=message,
                status=ResultStatus.RETRY,
            )
            return {
                "agent_state": state,
                "stage": ChainStage.REFLECT.value,
                "chain_error": message,
            }
        update = self._fail_update(state, message, result.error)
        update["failed_stage"] = stage
        update["error_type"] = getattr(
            result.error, "error_type", "ToolFailure"
        )
        update["error_details"] = _error_info_details(result.error)
        return update

    def _exception_reflection(
        self, state: AgentState, error: BaseException, stage: str
    ) -> ChainState:
        state.add_history(
            event_type=f"{stage}_error",
            message=str(error),
            status=ResultStatus.RETRY,
        )
        diagnostics = _exception_diagnostics(error, stage)
        logger.error(
            "%s failed and will enter reflection: %s",
            stage,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
        return {
            "agent_state": state,
            "stage": ChainStage.REFLECT.value,
            "chain_error": str(error),
            **diagnostics,
        }

    def _exception_failure(
        self, state: AgentState, error: BaseException, stage: str
    ) -> ChainState:
        diagnostics = _exception_diagnostics(error, stage)
        logger.error(
            "%s failed: %s",
            stage,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
        update = self._fail_update(
            state,
            f"{stage} failed: {error}",
            ErrorInfo.from_exception(error, retryable=False),
        )
        update.update(diagnostics)
        return update

    def _planner_failure_update(
        self,
        state: AgentState,
        result: Any,
    ) -> ChainState:
        error = getattr(result, "error", None)
        message = (
            getattr(error, "message", None)
            or getattr(result, "reason", None)
            or "Planner failed."
        )
        update: ChainState = {
            "agent_state": state,
            "stage": ChainStage.FAIL.value,
            "chain_error": str(message),
            "failed_stage": ChainStage.PLAN.value,
            "error_type": str(
                getattr(error, "error_type", "PlannerFailure")
            ),
            "error_details": _error_info_details(error),
        }
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, Mapping):
            diagnostic = metadata.get("diagnostic")
            if isinstance(diagnostic, Mapping):
                update["error_details"] = {
                    **update["error_details"],
                    **dict(diagnostic),
                }
                if diagnostic.get("traceback"):
                    update["traceback"] = str(diagnostic["traceback"])
        raw_output = getattr(result, "raw_output", None)
        if raw_output is not None:
            update["last_raw_response"] = raw_output
        return update

    @staticmethod
    def _fail_update(
        state: AgentState,
        message: str,
        error: ErrorInfo | None = None,
    ) -> ChainState:
        if not state.is_terminal:
            state.fail(
                error=error or ErrorInfo(
                    error_type="AgentChainFailure", message=message
                ),
                reason=(
                    RunTerminationReason.MAX_STEPS
                    if state.runtime.reached_max_steps
                    else RunTerminationReason.TIMEOUT
                    if state.runtime.is_timed_out
                    else RunTerminationReason.UNKNOWN
                ),
                message=message,
            )
        return {
            "agent_state": state,
            "stage": ChainStage.FAIL.value,
            "chain_error": message,
        }

    def _iteration_limit_failure(self, context: ChainState) -> ChainState:
        return self._fail_update(
            context["agent_state"],
            f"AgentChain exceeded {self.config.max_chain_iterations} stages.",
        )

    @staticmethod
    def _finalise_output(context: ChainState) -> Any:
        return context["agent_state"].to_run_result()


def create_agent_chain(
    *,
    planner: PlannerProtocol,
    tools: AgentTools,
    vlm: VLMProtocol,
    prompt_builder: PromptBuilder | None = None,
    memory: AgentMemory | None = None,
    memory_summarizer: Callable[..., Any] | None = None,
    config: AgentChainConfig | None = None,
) -> AgentChain:
    """Create an independent AgentChain and eagerly build its LCEL pipeline."""
    agent = AgentChain(
        planner=planner,
        tools=tools,
        vlm=vlm,
        prompt_builder=prompt_builder,
        memory=memory,
        memory_summarizer=memory_summarizer,
        config=config,
    )
    agent._chain = agent.build()
    return agent


def _supported_kwargs(method: Callable[..., Any], **values: Any) -> dict[str, Any]:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return {"prompt": values["prompt"]}
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return values
    result = {name: value for name, value in values.items() if name in parameters}
    if not result and parameters:
        result[next(iter(parameters))] = values["prompt"]
    return result


def _parse_json_object(response: Any) -> dict[str, Any]:
    if isinstance(response, tuple) and response:
        response = response[0]
    if isinstance(response, Mapping):
        for key in ("parsed", "json", "output", "content", "text"):
            value = response.get(key)
            if isinstance(value, Mapping):
                return dict(value)
        if response:
            return dict(response)
    for attribute in ("parsed", "json", "output", "content", "text"):
        value = getattr(response, attribute, None)
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str) and value.strip():
            response = value
            break
    if not isinstance(response, str) or not response.strip():
        raise ModelResponseError("Model response is empty or not JSON-compatible")
    text = response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ModelResponseError(f"Invalid JSON response: {error}") from error
    if not isinstance(value, Mapping):
        raise ModelResponseError("Model response must be one JSON object")
    return dict(value)


def _require_fields(data: Mapping[str, Any], fields: set[str], name: str) -> None:
    missing = sorted(fields.difference(data))
    if missing:
        raise ModelResponseError(
            f"{name} response missing fields: {', '.join(missing)}"
        )


def _exception_diagnostics(
    error: BaseException,
    stage: str,
) -> dict[str, Any]:
    """Preserve the complete exception chain before it is converted to state."""
    return {
        "failed_stage": stage,
        "error_type": type(error).__name__,
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
        "error_details": {
            "message": str(error),
            "cause": _exception_chain(error),
        },
    }


def _exception_chain(error: BaseException) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(
            {
                "type": type(current).__name__,
                "message": str(current),
            }
        )
        current = current.__cause__ or current.__context__
    return chain


def _error_info_details(error: Any) -> dict[str, Any]:
    if error is None:
        return {}
    details = getattr(error, "details", None)
    result = dict(details) if isinstance(details, Mapping) else {}
    for name in ("error_type", "message", "retryable"):
        value = getattr(error, name, None)
        if value is not None:
            result[name] = value
    return result


__all__ = [
    "AgentChain",
    "AgentChainConfig",
    "AgentChainDependencyError",
    "AgentChainError",
    "ChainStage",
    "ChainState",
    "ModelResponseError",
    "PlannerProtocol",
    "VLMProtocol",
    "create_agent_chain",
]