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
from .robustness import (
    action_fingerprint,
    apply_error_class,
    bump_retry_budget,
    classify_error,
    decompose_instruction,
    ensure_retry_budget,
    is_empty_observation,
    mark_recoverable,
    observation_fingerprint,
    robustness_snapshot,
    sub_task_text,
    sync_step_budget,
)
from .state import AgentPhase, AgentState, ObservationState
from .tools import AgentTools

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
    DECOMPOSE = "decompose"
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
    error_class: str
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
    # When True, skip VLM verify after dry-run executes (UI cannot change).
    synthetic_verify_on_dry_run: bool = False
    enable_decompose: bool = True
    max_sub_tasks: int = 8
    max_target_rejections: int = 2
    max_repeated_actions: int = 2
    max_empty_ocr_retries: int = 2

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
        if (
            self.max_sub_tasks <= 0
            or self.max_target_rejections < 0
            or self.max_repeated_actions < 0
            or self.max_empty_ocr_retries < 0
        ):
            raise ValueError(
                "max_sub_tasks must be positive; rejection/repeat/OCR "
                "limits must be non-negative"
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
        ensure_retry_budget(state.metadata)
        stage = (
            ChainStage.DECOMPOSE.value
            if self.config.enable_decompose
            else ChainStage.OBSERVE.value
        )
        return {"agent_state": state, "stage": stage}

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
            ChainStage.DECOMPOSE.value: self._decompose,
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
        started = time.perf_counter()
        try:
            next_context = await handler(context)
        except Exception as error:
            logger.exception("AgentChain stage failed unexpectedly: stage=%s", stage)
            return self._exception_failure(
                context["agent_state"], error, stage
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        state = next_context.get("agent_state") or context["agent_state"]
        error_class = next_context.get("error_class") or state.metadata.get(
            "last_error_class"
        )
        logger.info(
            "task_id=%s step=%s stage=%s next=%s error_class=%s "
            "latency_ms=%.0f retry_count=%s",
            getattr(state, "run_id", None),
            getattr(getattr(state, "runtime", None), "step_index", None),
            stage,
            next_context.get("stage"),
            error_class or "-",
            elapsed_ms,
            getattr(getattr(state, "runtime", None), "retry_count", 0),
        )
        next_context["error_class"] = error_class
        next_context["latency_ms"] = elapsed_ms
        return next_context

    @staticmethod
    def _terminal(context: ChainState) -> bool:
        state = context["agent_state"]
        return state.is_terminal and context.get("stage") in {
            ChainStage.FINISH.value,
            ChainStage.FAIL.value,
        }

    async def _decompose(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        existing = state.metadata.get("sub_tasks")
        if isinstance(existing, list) and existing:
            sub_tasks = existing
        else:
            sub_tasks = decompose_instruction(
                state.task.instruction,
                max_sub_tasks=self.config.max_sub_tasks,
            )
            state.metadata["sub_tasks"] = sub_tasks
        state.metadata["sub_task_index"] = int(state.metadata.get("sub_task_index") or 0)
        current = (
            sub_tasks[state.metadata["sub_task_index"]]
            if sub_tasks
            else None
        )
        text = sub_task_text(current)
        if text:
            state.task.set_subgoal(text)
        state.add_history(
            event_type="decompose",
            message=text or state.task.instruction,
            status=ResultStatus.SUCCESS,
            metadata={
                "sub_task_count": len(sub_tasks),
                "sub_tasks": [sub_task_text(item) for item in sub_tasks],
            },
        )
        logger.info(
            "task_id=%s stage=decompose sub_task_count=%s current=%s",
            state.run_id,
            len(sub_tasks),
            text,
        )
        return {"agent_state": state, "stage": ChainStage.OBSERVE.value}

    async def _observe(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        if self._limit_reached(state):
            return self._fail_update(state, "Agent reached its step/time limit.")
        result = await self.tools.acall("observe", self.config.observe_options)
        if not result.succeeded or not isinstance(result.output, ObservationState):
            return self._tool_failure(state, result, "observation")
        state.update_observation(result.output, tool_result=result)
        if is_empty_observation(result.output):
            count = bump_retry_budget(state, "empty_ocr")
            mark_recoverable(state, "empty_ocr")
            state.add_history(
                event_type="empty_ocr_retry",
                message="Observation has no OCR text or GUI elements.",
                status=ResultStatus.RETRY,
                metadata={"retry_count": count},
            )
            if count > self.config.max_empty_ocr_retries:
                return self._fail_update(
                    state,
                    "Empty OCR/UI observation retry limit reached.",
                    error_class="empty_ocr",
                )
            return {
                "agent_state": state,
                "stage": ChainStage.OBSERVE.value,
                "error_class": "empty_ocr",
                "chain_error": "empty_ocr",
            }
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
            bump_retry_budget(state, "planner")
            mark_recoverable(state, "parse")

            retry_reason = (
                result.reason
                or "Planner requested another observation."
            )
            error_class = classify_error(
                getattr(result, "error", None), retry_reason, stage="plan"
            )
            state.metadata["last_error_class"] = error_class

            state.add_history(
                event_type="planner_retry",
                message=retry_reason,
                status=ResultStatus.RETRY,
                metadata={
                    "retry_count": retry_count,
                    "max_retries": self.config.max_planner_retries,
                    "error_class": error_class,
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
                    error_class="planner_retry_exhausted",
                )

            return {
                "agent_state": state,
                "stage": ChainStage.OBSERVE.value,
                "chain_error": retry_reason,
                "planner_retry_count": retry_count,
                "error_class": error_class,
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
            bump_retry_budget(state, "target_rejection")
            mark_recoverable(state, "target_rejected")

            state.add_history(
                event_type="action_rejected",
                message=validation_error,
                status=ResultStatus.RETRY,
                metadata={
                    "rejection_count": target_rejection_count,
                    "max_rejections": self.config.max_target_rejections,
                    "error_class": "target_rejected",
                    "action": self._action_mapping(state.latest_action),
                },
            )

            if target_rejection_count > self.config.max_target_rejections:
                return self._fail_update(
                    state,
                    "Target validation failed repeatedly: "
                    f"{validation_error}",
                    error_class="target_rejected",
                )

            return {
                "agent_state": state,
                "stage": ChainStage.OBSERVE.value,
                "chain_error": validation_error,
                "error_class": "target_rejected",
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

        stage = (
            ChainStage.OBSERVE_AFTER
            if result.succeeded
            else ChainStage.REFLECT
        )

        return {
            "agent_state": state,
            "stage": stage.value,
        }

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
        stalled = self._update_progress_guard(state)
        if stalled:
            if state.runtime.repeated_action_count > self.config.max_repeated_actions:
                return self._fail_update(
                    state,
                    "Repeated action produced no visible UI change.",
                    error_class="no_progress",
                )
            mark_recoverable(state, "no_progress")
            state.add_history(
                event_type="no_progress",
                message="Same action and unchanged observation; forcing reflection.",
                status=ResultStatus.RETRY,
                metadata={
                    "repeated_action_count": state.runtime.repeated_action_count,
                    "error_class": "no_progress",
                },
            )
            return {
                "agent_state": state,
                "stage": ChainStage.REFLECT.value,
                "error_class": "no_progress",
                "chain_error": "no_progress",
            }
        state.set_phase(AgentPhase.VERIFYING)
        return {"agent_state": state, "stage": ChainStage.VERIFY.value}

    async def _verify(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        prompt = ""
        raw: Any = None
        dry_run = bool(
            state.metadata.get("dry_run")
            or self.config.synthetic_verify_on_dry_run
        )
        if dry_run:
            data = {
                "status": "uncertain",
                "action_effective": False,
                "task_complete": False,
                "evidence": [
                    "dry_run: executor did not apply side effects; UI cannot prove success"
                ],
                "reason": (
                    "Synthetic verification for dry-run: action was planned/logged "
                    "but not applied to the desktop."
                ),
                "confidence": 0.0,
                "recommended_next": "continue",
            }
            prompt = "(synthetic dry-run verify)"
            raw = data
        else:
            try:
                prompt = self.prompts.build_text(PromptKind.VERIFY, state)
                data, raw = await self._call_structured(
                    prompt, VERIFY_RESPONSE_SCHEMA, PromptKind.VERIFY, state=state
                )
                data = _coerce_verify_data(data)
                self._validate_verify(data)
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
            metadata={"raw_response": raw, "synthetic": dry_run},
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
            self._advance_sub_task(state)
            shared["stage"] = (
                ChainStage.MEMORY.value
                if self._has_remaining_sub_tasks(state)
                else ChainStage.FINISH.value
            )
        elif dry_run:
            # Dry-run cannot prove task completion; keep stepping until planner finish/max.
            self._commit_current_step(state)
            shared["stage"] = ChainStage.MEMORY.value
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
            return self._fail_update(
                state,
                "Maximum reflection count reached.",
                error_class="reflection_exhausted",
            )
        try:
            prompt = self.prompts.build_text(PromptKind.REFLECTION, state)
            data, raw = await self._call_structured(
                prompt, REFLECTION_RESPONSE_SCHEMA, PromptKind.REFLECTION, state=state
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
        if bool(data.get("should_replan")):
            state.metadata["replan_count"] = int(
                state.metadata.get("replan_count") or 0
            ) + 1
            mark_recoverable(state)
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
        if state.metadata.get("recoverable_failure"):
            state.metadata["auto_recovered"] = True
        state.metadata.update(robustness_snapshot(state))
        sync_step_budget(state)
        self.memory.ingest_state(state, recent_only=20)
        self.memory.attach_to_state(state)
        return {"agent_state": state, "stage": ChainStage.FINISH.value}

    async def _fail(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        if not state.is_terminal:
            message = context.get("chain_error") or "Agent chain failed."
            state.fail(
                error=ErrorInfo(
                    error_type="AgentChainFailure",
                    message=message,
                    error_class=str(
                        context.get("error_class")
                        or state.metadata.get("last_error_class")
                        or "unknown"
                    ),
                ),
                reason=RunTerminationReason.UNKNOWN,
                message=message,
            )
        state.metadata.update(robustness_snapshot(state))
        sync_step_budget(state)
        self.memory.ingest_state(state, recent_only=20)
        self.memory.attach_to_state(state)
        return {"agent_state": state, "stage": ChainStage.FAIL.value}

    async def _call_structured(
        self,
        prompt: str,
        schema: Mapping[str, Any],
        target_kind: PromptKind,
        state: AgentState | None = None,
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
                if state is not None:
                    bump_retry_budget(state, "repair")
                    mark_recoverable(state, "parse")
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
    def _has_remaining_sub_tasks(state: AgentState) -> bool:
        tasks = state.metadata.get("sub_tasks") or []
        index = int(state.metadata.get("sub_task_index") or 0)
        return bool(tasks) and index < len(tasks)

    def _advance_sub_task(self, state: AgentState) -> None:
        tasks = list(state.metadata.get("sub_tasks") or [])
        if not tasks:
            return
        index = int(state.metadata.get("sub_task_index") or 0)
        current = tasks[index] if index < len(tasks) else None
        current_text = sub_task_text(current)
        if current_text:
            try:
                state.task.complete_subgoal(current_text)
            except Exception:
                if current_text not in state.task.completed_subgoals:
                    state.task.completed_subgoals.append(current_text)
                state.task.set_subgoal(None)
        index += 1
        state.metadata["sub_task_index"] = index
        if index < len(tasks):
            state.task.set_subgoal(sub_task_text(tasks[index]))
            state.metadata["sub_tasks_complete"] = False
        else:
            state.task.set_subgoal(None)
            state.metadata["sub_tasks_complete"] = True

    def _update_progress_guard(self, state: AgentState) -> bool:
        action_fp = action_fingerprint(state.latest_action)
        obs_fp = observation_fingerprint(state.observation)
        previous_action = state.metadata.get("last_action_fingerprint")
        previous_obs = state.metadata.get("last_observation_fingerprint")
        stalled = bool(
            previous_action
            and previous_action == action_fp
            and previous_obs == obs_fp
        )
        if stalled:
            state.runtime.repeated_action_count += 1
            bump_retry_budget(state, "repeated_action")
            state.metadata["last_error_class"] = "no_progress"
        else:
            state.runtime.repeated_action_count = 0
        state.metadata["last_action_fingerprint"] = action_fp
        state.metadata["last_observation_fingerprint"] = obs_fp
        return stalled

    @staticmethod
    def _limit_reached(state: AgentState) -> bool:
        return bool(state.runtime.reached_max_steps or state.runtime.is_timed_out)

    def _tool_failure(
        self, state: AgentState, result: ToolResult, stage: str
    ) -> ChainState:
        message = result.message or f"{stage} failed"
        error_class = classify_error(result.error, message, stage=stage)
        if result.should_retry and not self._limit_reached(state):
            mark_recoverable(state, error_class)
            state.add_history(
                event_type=f"{stage}_retry",
                message=message,
                status=ResultStatus.RETRY,
                metadata={"error_class": error_class},
            )
            return {
                "agent_state": state,
                "stage": ChainStage.REFLECT.value,
                "chain_error": message,
                "error_class": error_class,
            }
        update = self._fail_update(
            state, message, result.error, error_class=error_class
        )
        update["failed_stage"] = stage
        update["error_type"] = getattr(
            result.error, "error_type", "ToolFailure"
        )
        update["error_details"] = _error_info_details(result.error)
        return update

    def _exception_reflection(
        self, state: AgentState, error: BaseException, stage: str
    ) -> ChainState:
        error_class = classify_error(error, str(error), stage=stage)
        mark_recoverable(state, error_class)
        state.add_history(
            event_type=f"{stage}_error",
            message=str(error),
            status=ResultStatus.RETRY,
            metadata={"error_class": error_class},
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
            "error_class": error_class,
            **diagnostics,
        }

    def _exception_failure(
        self, state: AgentState, error: BaseException, stage: str
    ) -> ChainState:
        error_class = classify_error(error, str(error), stage=stage)
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
            ErrorInfo.from_exception(error, retryable=False, error_class=error_class),
            error_class=error_class,
        )
        update.update(diagnostics)
        update["error_class"] = error_class
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
        error_class = classify_error(error, str(message), stage="plan")
        return self._fail_update(
            state, str(message), error, error_class=error_class
        )

    @staticmethod
    def _fail_update(
        state: AgentState,
        message: str,
        error: ErrorInfo | None = None,
        *,
        error_class: str | None = None,
    ) -> ChainState:
        resolved = error_class or classify_error(error, message)
        apply_error_class(error, resolved)
        state.metadata["last_error_class"] = resolved
        if not state.is_terminal:
            state.fail(
                error=error or ErrorInfo(
                    error_type="AgentChainFailure",
                    message=message,
                    error_class=resolved,
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
        state.metadata.update(robustness_snapshot(state))
        sync_step_budget(state)
        return {
            "agent_state": state,
            "stage": ChainStage.FAIL.value,
            "chain_error": message,
            "error_class": resolved,
        }

    def _iteration_limit_failure(self, context: ChainState) -> ChainState:
        return self._fail_update(
            context["agent_state"],
            f"AgentChain exceeded {self.config.max_chain_iterations} stages.",
            error_class="no_progress",
        )

    @staticmethod
    def _finalise_output(context: ChainState) -> Any:
        state = context["agent_state"]
        state.metadata.update(robustness_snapshot(state))
        sync_step_budget(state)
        return state.to_run_result()


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


_VERIFY_KEYS = {
    "status",
    "action_effective",
    "task_complete",
    "evidence",
    "reason",
    "confidence",
    "recommended_next",
}
_STRUCTURED_HINT_KEYS = _VERIFY_KEYS | {
    "decision",
    "action",
    "failure_type",
    "summary",
    "should_replan",
}


def _parse_json_object(response: Any) -> dict[str, Any]:
    if isinstance(response, tuple) and response:
        # generate_json returns (parsed_mapping, VLMResponse)
        first, second = response[0], response[1] if len(response) > 1 else None
        if isinstance(first, Mapping):
            response = first
        elif second is not None:
            response = second
        else:
            response = first
    if isinstance(response, Mapping):
        # Prefer the outer object when it already looks structured.
        if _STRUCTURED_HINT_KEYS.intersection(str(k).lower() for k in response):
            return dict(response)
        for key in ("parsed", "json", "output", "content", "text"):
            value = response.get(key)
            if isinstance(value, Mapping):
                return dict(value)
            if isinstance(value, str) and value.strip():
                return _parse_json_object(value)
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


def _coerce_verify_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize common aliases / missing optional shapes for verify JSON."""

    out = {str(k): v for k, v in dict(data).items()}
    aliases = {
        "action_effective": ("actionEffective", "effective", "acted"),
        "task_complete": ("taskComplete", "complete", "finished", "done"),
        "recommended_next": ("recommendedNext", "next", "next_action"),
        "status": ("result", "verdict"),
        "reason": ("explanation", "message"),
        "confidence": ("score",),
        "evidence": ("evidences", "proof"),
    }
    for canonical, alts in aliases.items():
        if canonical in out and out[canonical] is not None:
            continue
        for alt in alts:
            if alt in out and out[alt] is not None:
                out[canonical] = out[alt]
                break
    if "evidence" in out and isinstance(out["evidence"], str):
        out["evidence"] = [out["evidence"]]
    if "evidence" not in out:
        out["evidence"] = []
    if "confidence" in out:
        try:
            out["confidence"] = float(out["confidence"])
        except (TypeError, ValueError):
            out["confidence"] = 0.0
    return out


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
    for name in ("error_type", "message", "retryable", "error_class"):
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