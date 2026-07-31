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
from .document_tasks import (
    advance_close_phase_after_verify,
    apply_close_verify_override,
    apply_open_verify_override,
    build_window_close_observation,
    ensure_close_progress,
    forced_close_hotkey,
    is_close_action,
    is_close_task,
    prepare_close_execution,
    should_use_window_observe_after,
    task_instruction,
)
from .chat_send import (
    apply_chat_verify_override,
    chat_observe_options,
    ensure_chat_progress,
    forced_chat_action,
    is_chat_paste_action,
    is_chat_send_task,
    is_chat_submit_action,
    record_chat_phase_for_executed_action,
    should_finish_chat_task,
)
from .launch_wait import needs_additional_observation, resolve_post_action_wait
from .memory import AgentMemory, MemoryImportance, MemoryKind
from .observation_utils import (
    consume_post_action_observation,
    is_partial_observation,
    mark_post_action_observation,
)
from .result import (
    ErrorInfo,
    PlannerResult,
    ResultStatus,
    RunTerminationReason,
    ToolResult,
)
from .state import AgentPhase, AgentState, ObservationState
from .tools import AgentTools
from .verify_policy import latest_action_type, should_skip_verification
from .browser_search import (
    PHASE_INPUT_FOCUSED,
    advance_search_phase_after_verify,
    apply_focus_verify_override,
    apply_homepage_verify_override,
    build_focus_success_verify,
    detect_search_box_focus,
    is_address_bar_focus_action,
    record_phase_for_executed_action,
    record_search_phase,
    search_progress,
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
    launch_post_action_wait_seconds: float = 2.5
    launch_extra_observation_attempts: int = 1
    launch_extra_observation_wait_seconds: float = 1.5
    reuse_post_action_observation: bool = True
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
        if not 0 <= self.launch_post_action_wait_seconds <= 60:
            raise ValueError(
                "launch_post_action_wait_seconds must be between 0 and 60"
            )
        if not 0 <= self.launch_extra_observation_wait_seconds <= 60:
            raise ValueError(
                "launch_extra_observation_wait_seconds must be between 0 and 60"
            )
        if self.launch_extra_observation_attempts < 0:
            raise ValueError(
                "launch_extra_observation_attempts must be non-negative"
            )
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

        forced = self._force_chat_plan(state)
        if forced is not None:
            return forced

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

            # Close tasks: never burn another full observation on the same
            # unreachable ✕ click. Force Ctrl+W when possible, otherwise replan
            # on the existing observation.
            if is_close_task(task_instruction(state)) and state.observation is not None:
                recovered = self._recover_close_plan(state)
                if recovered is not None:
                    return recovered
                state.set_phase(AgentPhase.PLANNING)
                return {
                    "agent_state": state,
                    "stage": ChainStage.PLAN.value,
                    "chain_error": retry_reason,
                    "planner_retry_count": retry_count,
                }

            # Chat tasks: inject the deterministic stage action instead of
            # re-observing and re-clicking the composer.
            if is_chat_send_task(task_instruction(state)) and state.observation is not None:
                recovered = self._force_chat_plan(state)
                if recovered is not None:
                    return recovered
                state.set_phase(AgentPhase.PLANNING)
                return {
                    "agent_state": state,
                    "stage": ChainStage.PLAN.value,
                    "chain_error": retry_reason,
                    "planner_retry_count": retry_count,
                }

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

    def _force_chat_plan(self, state: AgentState) -> ChainState | None:
        """Inject paste/enter/wait or finish for a chat-send stage."""

        if should_finish_chat_task(state):
            state.add_history(
                event_type="chat_finish_forced",
                message="ChatGPT thinking marker already detected; finishing.",
                status=ResultStatus.SUCCESS,
                metadata={"chat_phase": "thinking"},
            )
            result = PlannerResult.finish(
                message="ChatGPT is thinking/generating after the message was sent.",
                reason="chat_progress reached thinking.",
                confidence=0.92,
                metadata={"forced_chat_finish": True},
            )
            state.update_planner_result(result)
            state.metadata["planner_retry_count"] = 0
            return {
                "agent_state": state,
                "stage": ChainStage.FINISH.value,
            }

        forced = forced_chat_action(state)
        if forced is None:
            return None
        action_type, parameters = forced
        create = getattr(self.planner, "create_action", None)
        if not callable(create):
            return None
        action = create(action_type, parameters)
        result = PlannerResult.act(
            action,
            reason=(
                f"Deterministic chat stage action: {action_type} "
                f"(skipped VLM replan)."
            ),
            confidence=0.95,
            metadata={
                "forced_chat_action": True,
                "action_type": action_type,
            },
        )
        state.update_planner_result(result)
        state.metadata["planner_retry_count"] = 0
        state.add_history(
            event_type="chat_action_forced",
            message=result.reason,
            status=ResultStatus.SUCCESS,
            metadata={
                "action_type": action_type,
                "parameters": parameters,
            },
        )
        return {
            "agent_state": state,
            "stage": ChainStage.EXECUTE.value,
        }

    def _recover_close_plan(self, state: AgentState) -> ChainState | None:
        """Inject Ctrl+W/Alt+F4 for a close task instead of re-observing."""

        forced = forced_close_hotkey(state)
        if forced is None:
            return None
        action_type, parameters = forced
        create = getattr(self.planner, "create_action", None)
        if not callable(create):
            return None
        action = create(action_type, parameters)
        result = PlannerResult.act(
            action,
            reason=(
                "Recovered close task with the close hotkey after a failed "
                "plan; skipped a redundant observation."
            ),
            confidence=0.9,
            metadata={"forced_close_hotkey": True, "recovered_in_chain": True},
        )
        state.update_planner_result(result)
        state.metadata["planner_retry_count"] = 0
        state.add_history(
            event_type="close_hotkey_recovered",
            message=result.reason,
            status=ResultStatus.SUCCESS,
            metadata={"keys": parameters.get("keys")},
        )
        return {
            "agent_state": state,
            "stage": ChainStage.EXECUTE.value,
        }

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

        if prepare_close_execution(state):
            state.add_history(
                event_type="close_window_activated",
                message="Activated the target Word window before close.",
                status=ResultStatus.SUCCESS,
                metadata={"prepare_close_execution": True},
            )

        result = await self.tools.acall(
            "execute_action",
            {"action": state.latest_action},
        )

        if not result.succeeded and not self.config.fail_on_execution_error:
            result.status = ResultStatus.RETRY
            if result.error is not None:
                result.error.retryable = True

        state.update_execution_result(result)

        # Decide skip_verify against the pre-record chat phase. Recording focus
        # first would advance idle → input_focused and make the skip check fail.
        skip_verify = False
        if result.succeeded:
            skip_verify = self._should_skip_verification(
                state.latest_action, state
            )
            record_phase_for_executed_action(state, state.latest_action)
            record_chat_phase_for_executed_action(state, state.latest_action)

        if state.is_terminal:
            return {
                "agent_state": state,
                "stage": ChainStage.FAIL.value,
            }

        if result.succeeded and skip_verify:
            action_type = self._latest_action_type(state.latest_action)
            state.add_history(
                event_type="verify_skipped",
                message=(
                    f"Skipped verification after {action_type}; "
                    "continuing to plan the next step."
                ),
                status=ResultStatus.SUCCESS,
                metadata={
                    "action_type": action_type,
                    "skip_verify": True,
                    "chat_phase": (
                        (state.metadata.get("chat_progress") or {}).get("phase")
                        if isinstance(state.metadata.get("chat_progress"), dict)
                        else None
                    ),
                },
            )
            self._commit_current_step(state)
            if state.is_terminal:
                return {
                    "agent_state": state,
                    "stage": ChainStage.FAIL.value,
                }
            # Keep the pre-paste observation so the planner can submit
            # immediately — unless it is a narrow/Win32 partial capture.
            plan_ready = state.observation is not None and not (
                is_partial_observation(state.observation)
            )
            state.set_phase(
                AgentPhase.PLANNING if plan_ready else AgentPhase.OBSERVING
            )
            return {
                "agent_state": state,
                "stage": (
                    ChainStage.PLAN.value
                    if plan_ready
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
    def _should_skip_verification(cls, action: Any, state: Any = None) -> bool:
        return should_skip_verification(action, state)

    async def _observe_after(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        action = state.latest_action
        # Still the pre-action observation: update_observation runs once, at the end.
        before = state.observation

        wait_seconds = resolve_post_action_wait(
            action,
            base_seconds=self.config.post_action_wait_seconds,
            launch_seconds=self.config.launch_post_action_wait_seconds,
        )
        if wait_seconds:
            await self.tools.acall("wait", {"seconds": wait_seconds})

        if should_use_window_observe_after(state, action):
            observation = build_window_close_observation(
                before=before,
                instruction=task_instruction(state),
            )
            result = ToolResult.success(
                "observe",
                output=observation,
                message=(
                    "Observed top-level windows for close verification "
                    f"({len(observation.gui_elements)} windows)."
                ),
                metadata={
                    "observation_id": observation.observation_id,
                    "observation_kind": "win32_windows",
                },
            )
            state.update_observation(result.output, tool_result=result)
            mark_post_action_observation(state, result.output)
            state.set_phase(AgentPhase.VERIFYING)
            return {"agent_state": state, "stage": ChainStage.VERIFY.value}

        observe_args = dict(self.config.observe_options)
        chat_opts = chat_observe_options(state, action)
        if chat_opts:
            observe_args.update(chat_opts)

        result = await self.tools.acall("observe", observe_args)
        if not result.succeeded or not isinstance(result.output, ObservationState):
            return self._tool_failure(state, result, "post-action observation")

        for attempt in range(max(0, self.config.launch_extra_observation_attempts)):
            if not needs_additional_observation(action, before, result.output):
                break
            extra_wait = self.config.launch_extra_observation_wait_seconds
            state.add_history(
                event_type="observe_after_retry",
                message=(
                    "Screen still matches the pre-action state after a launch "
                    f"action; re-observing in {extra_wait}s."
                ),
                status=ResultStatus.RETRY,
                metadata={
                    "attempt": attempt + 1,
                    "action_type": self._latest_action_type(action),
                    "wait_seconds": extra_wait,
                },
            )
            if extra_wait:
                await self.tools.acall("wait", {"seconds": extra_wait})
            retry = await self.tools.acall("observe", observe_args)
            if not retry.succeeded or not isinstance(retry.output, ObservationState):
                break
            result = retry

        state.update_observation(result.output, tool_result=result)
        mark_post_action_observation(state, result.output)
        state.set_phase(AgentPhase.VERIFYING)
        return {"agent_state": state, "stage": ChainStage.VERIFY.value}

    @staticmethod
    def _persist_focus_stage(state: AgentState, evidence: Any = ()) -> None:
        """Remember a confirmed focus so the next planner turn pastes instead
        of focusing the input again."""

        notes = [str(item) for item in (evidence or ())]
        record_search_phase(state, PHASE_INPUT_FOCUSED, evidence=notes)
        if state.observation is not None:
            metadata = dict(getattr(state.observation, "metadata", None) or {})
            metadata["search_focus_detected"] = True
            if notes:
                metadata["search_focus_evidence"] = notes
            state.observation.metadata = metadata

    def _apply_focus_override(
        self,
        state: AgentState,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        # Focus evidence only means something inside a browser-search flow;
        # on other tasks it would overturn a legitimate VLM verdict.
        if search_progress(state) is None and not is_address_bar_focus_action(
            state.latest_action
        ):
            return data
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
            self._persist_focus_stage(state, data.get("evidence") or ())
        return data

    async def _verify(self, context: ChainState) -> ChainState:
        state = context["agent_state"]
        prompt = ""
        raw: Any = None
        try:
            instruction = task_instruction(state)
            if is_close_task(instruction) and is_close_action(
                state.latest_action, instruction
            ):
                ensure_close_progress(state)
                meta = getattr(state.observation, "metadata", None) or {}
                code_only = str(meta.get("observation_kind") or "") == "win32_windows"
                if code_only:
                    data = {
                        "status": "retry",
                        "action_effective": False,
                        "task_complete": False,
                        "evidence": [],
                        "reason": (
                            "Close verification uses Win32 window identity."
                        ),
                        "confidence": 0.5,
                        "recommended_next": "continue",
                    }
                else:
                    prompt = self.prompts.build_text(PromptKind.VERIFY, state)
                    data, raw = await self._call_structured(
                        prompt, VERIFY_RESPONSE_SCHEMA, PromptKind.VERIFY
                    )
                    self._validate_verify(data)
                data, closed_hit = apply_close_verify_override(
                    state,
                    data,
                    before=state.previous_observation,
                    after=state.observation,
                )
                if closed_hit:
                    state.add_history(
                        event_type="verify_document_closed",
                        message=str(data.get("reason") or "Document closed"),
                        status=ResultStatus.SUCCESS,
                        metadata={"verify": data},
                    )
            elif is_chat_send_task(instruction) and (
                is_chat_paste_action(state.latest_action)
                or is_chat_submit_action(state.latest_action)
                or (
                    # Only submitted-phase waits belong to the chat flow; a
                    # generic wait earlier must go through normal VLM verify.
                    self._latest_action_type(state.latest_action) == "wait"
                    and str(
                        (state.metadata.get("chat_progress") or {}).get("phase")
                        or ""
                    )
                    == "submitted"
                )
            ):
                ensure_chat_progress(state)
                data = {
                    "status": "retry",
                    "action_effective": False,
                    "task_complete": False,
                    "evidence": [],
                    "reason": "Chat send verification uses composer OCR.",
                    "confidence": 0.5,
                    "recommended_next": "continue",
                }
                data, chat_hit = apply_chat_verify_override(
                    state,
                    data,
                    before=state.previous_observation,
                    after=state.observation,
                )
                if chat_hit:
                    event = (
                        "verify_chat_thinking"
                        if bool(data.get("task_complete"))
                        else "verify_chat_text"
                    )
                    state.add_history(
                        event_type=event,
                        message=str(data.get("reason") or "Chat verify override"),
                        status=ResultStatus.SUCCESS,
                        metadata={"verify": data},
                    )
                elif is_chat_submit_action(state.latest_action) or (
                    self._latest_action_type(state.latest_action) == "wait"
                ):
                    data = {
                        "status": "success",
                        "action_effective": True,
                        "task_complete": False,
                        "evidence": list(data.get("evidence") or []),
                        "reason": (
                            "Enter/wait completed; thinking marker not visible yet."
                        ),
                        "confidence": 0.55,
                        "recommended_next": "continue",
                    }
                else:
                    data = {
                        "status": "success",
                        "action_effective": True,
                        "task_complete": False,
                        "evidence": list(data.get("evidence") or []),
                        "reason": (
                            "Paste was delivered; composer OCR has not confirmed "
                            "the message yet — paste again or re-observe."
                        ),
                        "confidence": 0.55,
                        "recommended_next": "continue",
                    }
            elif is_address_bar_focus_action(state.latest_action):
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
                    self._persist_focus_stage(state, evidence)
                else:
                    prompt = self.prompts.build_text(PromptKind.VERIFY, state)
                    data, raw = await self._call_structured(
                        prompt, VERIFY_RESPONSE_SCHEMA, PromptKind.VERIFY
                    )
                    self._validate_verify(data)
                    data = self._apply_focus_override(state, data)
                data, homepage_hit = apply_homepage_verify_override(
                    state, data, after=state.observation
                )
                if homepage_hit:
                    state.add_history(
                        event_type="verify_homepage_detected",
                        message=str(
                            data.get("reason") or "Google homepage reached"
                        ),
                        status=ResultStatus.SUCCESS,
                        metadata={"verify": data},
                    )
            else:
                prompt = self.prompts.build_text(PromptKind.VERIFY, state)
                data, raw = await self._call_structured(
                    prompt, VERIFY_RESPONSE_SCHEMA, PromptKind.VERIFY
                )
                self._validate_verify(data)
                data = self._apply_focus_override(state, data)
                data, homepage_hit = apply_homepage_verify_override(
                    state, data, after=state.observation
                )
                if homepage_hit:
                    state.add_history(
                        event_type="verify_homepage_detected",
                        message=str(
                            data.get("reason") or "Google homepage reached"
                        ),
                        status=ResultStatus.SUCCESS,
                        metadata={"verify": data},
                    )
                data, document_hit = apply_open_verify_override(
                    state, data, after=state.observation
                )
                if document_hit:
                    state.add_history(
                        event_type="verify_document_opened",
                        message=str(data.get("reason") or "Document opened"),
                        status=ResultStatus.SUCCESS,
                        metadata={"verify": data},
                    )
                data, closed_hit = apply_close_verify_override(
                    state,
                    data,
                    before=state.previous_observation,
                    after=state.observation,
                )
                if closed_hit:
                    state.add_history(
                        event_type="verify_document_closed",
                        message=str(data.get("reason") or "Document closed"),
                        status=ResultStatus.SUCCESS,
                        metadata={"verify": data},
                    )
                data, chat_hit = apply_chat_verify_override(
                    state,
                    data,
                    before=state.previous_observation,
                    after=state.observation,
                )
                if chat_hit:
                    event = (
                        "verify_chat_thinking"
                        if bool(data.get("task_complete"))
                        else "verify_chat_text"
                    )
                    state.add_history(
                        event_type=event,
                        message=str(data.get("reason") or "Chat verify override"),
                        status=ResultStatus.SUCCESS,
                        metadata={"verify": data},
                    )
        except Exception as error:
            return self._exception_reflection(state, error, "verification")

        succeeded = bool(data["action_effective"]) and data["status"] == "success"
        if succeeded:
            advance_search_phase_after_verify(
                state, state.latest_action, data
            )
            advance_close_phase_after_verify(
                state, state.latest_action, data
            )
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
            # final=True: completing on the last step is success, not MAX_STEPS.
            self._commit_current_step(state, final=True)
            shared["stage"] = ChainStage.FINISH.value
        elif succeeded and str(data["recommended_next"]) in {
            "continue",
            "replan",
            "finish",
        }:
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
        if stage is ChainStage.PLAN and (
            state.observation is None
            or is_partial_observation(state.observation)
        ):
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
        elif self._reuse_post_action_observation(state):
            state.set_phase(AgentPhase.PLANNING)
            stage = ChainStage.PLAN
        else:
            state.set_phase(AgentPhase.OBSERVING)
            stage = ChainStage.OBSERVE
        return {"agent_state": state, "stage": stage.value}

    def _reuse_post_action_observation(self, state: AgentState) -> bool:
        """Plan on the post-action capture instead of observing the same screen
        twice: nothing touched the GUI between the two observations."""

        if not self.config.reuse_post_action_observation:
            return False
        # Narrow-band / Win32-only captures cover a sliver of the screen and
        # must never seed the next plan (they poison screen size and targets).
        if is_partial_observation(state.observation):
            return False
        if not consume_post_action_observation(state, state.observation):
            return False
        state.add_history(
            event_type="observation_reused",
            message="Planning on the post-action observation; skipped a re-observe.",
            status=ResultStatus.SUCCESS,
            metadata={
                "observation_id": getattr(state.observation, "observation_id", None)
            },
        )
        return True

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
    def _commit_current_step(state: AgentState, *, final: bool = False) -> None:
        if state.last_planner_result is None:
            return
        try:
            state.commit_step(final=final)
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