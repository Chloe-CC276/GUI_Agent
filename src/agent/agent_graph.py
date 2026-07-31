"""
agent_graph
LangGraph orchestration for the GUI Agent.

Flow
----
observe -> plan -> execute -> observe_after -> verify
                                  |              |
                                  |              +-> finish
                                  |              +-> memory -> observe
                                  |              +-> reflection -> plan
                                  +------------------------------^
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, TypedDict

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
)

try:
    from .prompts import PromptBuilder, PromptKind
    from .prompts.schemas import (
        REFLECTION_RESPONSE_SCHEMA,
        VERIFY_RESPONSE_SCHEMA,
    )
except ImportError:  # package installed below another project root
    from ..prompts import PromptBuilder, PromptKind  # type: ignore
    from ..prompts.schemas import (  # type: ignore
        REFLECTION_RESPONSE_SCHEMA,
        VERIFY_RESPONSE_SCHEMA,
    )

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # optional until build()/compile() is called
    END = "__end__"
    START = "__start__"
    StateGraph = None  # type: ignore[assignment]


class AgentGraphError(RuntimeError):
    """Base graph orchestration error."""


class AgentGraphDependencyError(AgentGraphError):
    """Raised when the optional LangGraph dependency is unavailable."""


class ModelResponseError(AgentGraphError):
    """Raised when a verifier/reflection model response cannot be used."""


class PlannerProtocol(Protocol):
    def plan(self, state: AgentState) -> Any: ...
    async def aplan(self, state: AgentState) -> Any: ...


class VLMProtocol(Protocol):
    def generate(self, *args: Any, **kwargs: Any) -> Any: ...


class GraphRoute(str, Enum):
    OBSERVE = "observe"
    PLAN = "plan"
    EXECUTE = "execute"
    OBSERVE_AFTER = "observe_after"
    VERIFY = "verify"
    REFLECT = "reflect"
    MEMORY = "memory"
    FINISH = "finish"
    FAIL = "fail"


class GraphState(TypedDict, total=False):
    """LangGraph envelope; the canonical mutable state remains AgentState."""

    agent_state: AgentState
    route: str
    verify_data: dict[str, Any]
    reflection_data: dict[str, Any]
    last_prompt: str
    last_raw_response: Any
    graph_error: str


@dataclass(slots=True)
class AgentGraphConfig:
    """Runtime policy for the orchestration graph."""

    observe_options: dict[str, Any] = field(default_factory=dict)
    post_action_wait_seconds: float = 0.5
    launch_post_action_wait_seconds: float = 2.5
    launch_extra_observation_attempts: int = 1
    launch_extra_observation_wait_seconds: float = 1.5
    reuse_post_action_observation: bool = True
    verify_confidence_threshold: float = 0.55
    max_reflections: int = 3
    max_model_repairs: int = 1
    summarize_memory: bool = True
    fail_on_execution_error: bool = False
    recursion_limit: int = 200

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
        if self.max_reflections < 0 or self.max_model_repairs < 0:
            raise ValueError("reflection and repair limits must be non-negative")
        if self.recursion_limit <= 0:
            raise ValueError("recursion_limit must be positive")


class AgentGraph:
    """Build and run the complete GUI-Agent feedback loop."""

    def __init__(
        self,
        *,
        planner: PlannerProtocol,
        tools: AgentTools,
        vlm: VLMProtocol,
        prompt_builder: PromptBuilder | None = None,
        memory: AgentMemory | None = None,
        memory_summarizer: Callable[..., Any] | None = None,
        config: AgentGraphConfig | None = None,
    ) -> None:
        if planner is None or tools is None or vlm is None:
            raise ValueError("planner, tools, and vlm are required")
        self.planner = planner
        self.tools = tools
        self.vlm = vlm
        self.prompts = prompt_builder or PromptBuilder()
        self.memory = memory or AgentMemory()
        self.memory_summarizer = memory_summarizer
        self.config = config or AgentGraphConfig()
        self._compiled: Any = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build(self) -> Any:
        if StateGraph is None:
            raise AgentGraphDependencyError(
                "LangGraph is not installed. Run: pip install langgraph"
            )
        graph = StateGraph(GraphState)
        graph.add_node(GraphRoute.OBSERVE.value, self.observe_node)
        graph.add_node(GraphRoute.PLAN.value, self.plan_node)
        graph.add_node(GraphRoute.EXECUTE.value, self.execute_node)
        graph.add_node(GraphRoute.OBSERVE_AFTER.value, self.observe_after_node)
        graph.add_node(GraphRoute.VERIFY.value, self.verify_node)
        graph.add_node(GraphRoute.REFLECT.value, self.reflect_node)
        graph.add_node(GraphRoute.MEMORY.value, self.memory_node)
        graph.add_node(GraphRoute.FINISH.value, self.finish_node)
        graph.add_node(GraphRoute.FAIL.value, self.fail_node)

        graph.add_edge(START, GraphRoute.OBSERVE.value)
        graph.add_conditional_edges(
            GraphRoute.OBSERVE.value, self._route,
            self._route_map(GraphRoute.PLAN, GraphRoute.FAIL),
        )
        graph.add_conditional_edges(
            GraphRoute.PLAN.value, self._route,
            self._route_map(
                GraphRoute.EXECUTE, GraphRoute.OBSERVE, GraphRoute.REFLECT,
                GraphRoute.FINISH, GraphRoute.FAIL,
            ),
        )
        graph.add_conditional_edges(
            GraphRoute.EXECUTE.value, self._route,
            self._route_map(GraphRoute.OBSERVE_AFTER, GraphRoute.REFLECT, GraphRoute.FAIL),
        )
        graph.add_conditional_edges(
            GraphRoute.OBSERVE_AFTER.value, self._route,
            self._route_map(GraphRoute.VERIFY, GraphRoute.REFLECT, GraphRoute.FAIL),
        )
        graph.add_conditional_edges(
            GraphRoute.VERIFY.value, self._route,
            self._route_map(
                GraphRoute.MEMORY, GraphRoute.REFLECT,
                GraphRoute.FINISH, GraphRoute.FAIL,
            ),
        )
        graph.add_conditional_edges(
            GraphRoute.REFLECT.value, self._route,
            self._route_map(GraphRoute.PLAN, GraphRoute.OBSERVE, GraphRoute.FAIL),
        )
        graph.add_conditional_edges(
            GraphRoute.MEMORY.value, self._route,
            self._route_map(
                GraphRoute.OBSERVE, GraphRoute.PLAN,
                GraphRoute.FINISH, GraphRoute.FAIL,
            ),
        )
        graph.add_edge(GraphRoute.FINISH.value, END)
        graph.add_edge(GraphRoute.FAIL.value, END)
        return graph

    def compile(self, **kwargs: Any) -> Any:
        self._compiled = self.build().compile(**kwargs)
        return self._compiled

    @property
    def app(self) -> Any:
        return self._compiled or self.compile()

    @staticmethod
    def _route(value: GraphState) -> str:
        return str(value.get("route", GraphRoute.FAIL.value))

    @staticmethod
    def _route_map(*routes: GraphRoute) -> dict[str, str]:
        return {route.value: route.value for route in routes}

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------

    def run(self, state: AgentState, **invoke_kwargs: Any) -> Any:
        self._prepare_state(state)
        config = dict(invoke_kwargs.pop("config", {}) or {})
        config.setdefault("recursion_limit", self.config.recursion_limit)
        output = self.app.invoke({"agent_state": state}, config=config, **invoke_kwargs)
        return output["agent_state"].to_run_result()

    async def arun(self, state: AgentState, **invoke_kwargs: Any) -> Any:
        self._prepare_state(state)
        config = dict(invoke_kwargs.pop("config", {}) or {})
        config.setdefault("recursion_limit", self.config.recursion_limit)
        output = await self.app.ainvoke(
            {"agent_state": state}, config=config, **invoke_kwargs
        )
        return output["agent_state"].to_run_result()

    def stream(self, state: AgentState, **kwargs: Any) -> Any:
        self._prepare_state(state)
        config = dict(kwargs.pop("config", {}) or {})
        config.setdefault("recursion_limit", self.config.recursion_limit)
        return self.app.stream({"agent_state": state}, config=config, **kwargs)

    @staticmethod
    def _prepare_state(state: AgentState) -> None:
        if not isinstance(state, AgentState):
            raise TypeError("state must be AgentState")
        if state.phase == AgentPhase.CREATED:
            state.begin()
        elif state.is_terminal:
            raise AgentGraphError("Cannot run a terminal AgentState")

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def observe_node(self, graph: GraphState) -> GraphState:
        state = graph["agent_state"]
        if self._limit_reached(state):
            return self._fail_update(state, "Agent reached its step/time limit.")
        result = await self.tools.acall("observe", self.config.observe_options)
        if not result.succeeded or not isinstance(result.output, ObservationState):
            return self._tool_failure_route(state, result, stage="observation")
        state.update_observation(result.output, tool_result=result)
        return {"agent_state": state, "route": GraphRoute.PLAN.value}

    async def plan_node(self, graph: GraphState) -> GraphState:
        state = graph["agent_state"]
        if self._limit_reached(state):
            return self._fail_update(state, "Agent reached its step/time limit.")

        forced = self._force_chat_plan(state)
        if forced is not None:
            return forced

        try:
            method = getattr(self.planner, "aplan", None)
            result = await method(state) if callable(method) else await asyncio.to_thread(
                self.planner.plan, state
            )
            state.update_planner_result(result)
        except Exception as error:
            return self._exception_failure(state, error, "planning")

        if state.is_terminal:
            return {"agent_state": state, "route": GraphRoute.FAIL.value}
        if result.is_finished:
            return {"agent_state": state, "route": GraphRoute.FINISH.value}
        if result.should_execute:
            return {"agent_state": state, "route": GraphRoute.EXECUTE.value}
        if result.should_retry:
            if is_close_task(task_instruction(state)) and state.observation is not None:
                recovered = self._recover_close_plan(state)
                if recovered is not None:
                    return recovered
                # Replan on the same observation; do not pay for another OCR pass.
                return {"agent_state": state, "route": GraphRoute.REFLECT.value}
            if is_chat_send_task(task_instruction(state)) and state.observation is not None:
                recovered = self._force_chat_plan(state)
                if recovered is not None:
                    return recovered
                return {"agent_state": state, "route": GraphRoute.REFLECT.value}
            route = GraphRoute.REFLECT if state.runtime.consecutive_failures else GraphRoute.OBSERVE
            return {"agent_state": state, "route": route.value}
        return self._fail_update(state, result.reason or "Planner returned no usable decision.")

    def _force_chat_plan(self, state: AgentState) -> GraphState | None:
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
            return {"agent_state": state, "route": GraphRoute.FINISH.value}

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
        state.add_history(
            event_type="chat_action_forced",
            message=result.reason,
            status=ResultStatus.SUCCESS,
            metadata={
                "action_type": action_type,
                "parameters": parameters,
            },
        )
        return {"agent_state": state, "route": GraphRoute.EXECUTE.value}

    def _recover_close_plan(self, state: AgentState) -> GraphState | None:
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
            metadata={"forced_close_hotkey": True, "recovered_in_graph": True},
        )
        state.update_planner_result(result)
        state.add_history(
            event_type="close_hotkey_recovered",
            message=result.reason,
            status=ResultStatus.SUCCESS,
            metadata={"keys": parameters.get("keys")},
        )
        return {"agent_state": state, "route": GraphRoute.EXECUTE.value}

    async def execute_node(self, graph: GraphState) -> GraphState:
        state = graph["agent_state"]
        if prepare_close_execution(state):
            state.add_history(
                event_type="close_window_activated",
                message="Activated the target Word window before close.",
                status=ResultStatus.SUCCESS,
                metadata={"prepare_close_execution": True},
            )
        result = await self.tools.acall(
            "execute_action", {"action": state.latest_action}
        )
        # Preserve retryability for the reflection loop when configured.
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
            return {"agent_state": state, "route": GraphRoute.FAIL.value}
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
                return {"agent_state": state, "route": GraphRoute.FAIL.value}
            if state.observation is not None:
                state.set_phase(AgentPhase.PLANNING)
                return {"agent_state": state, "route": GraphRoute.PLAN.value}
            state.set_phase(AgentPhase.OBSERVING)
            return {"agent_state": state, "route": GraphRoute.OBSERVE.value}
        route = GraphRoute.OBSERVE_AFTER if result.succeeded else GraphRoute.REFLECT
        return {"agent_state": state, "route": route.value}

    @staticmethod
    def _latest_action_type(action: Any) -> str:
        return latest_action_type(action)

    @classmethod
    def _should_skip_verification(cls, action: Any, state: Any = None) -> bool:
        return should_skip_verification(action, state)

    async def observe_after_node(self, graph: GraphState) -> GraphState:
        state = graph["agent_state"]
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
            return {"agent_state": state, "route": GraphRoute.VERIFY.value}

        observe_args = dict(self.config.observe_options)
        chat_opts = chat_observe_options(state, action)
        if chat_opts:
            observe_args.update(chat_opts)

        result = await self.tools.acall("observe", observe_args)
        if not result.succeeded or not isinstance(result.output, ObservationState):
            return self._tool_failure_route(state, result, stage="post-action observation")

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
        return {"agent_state": state, "route": GraphRoute.VERIFY.value}

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

    async def verify_node(self, graph: GraphState) -> GraphState:
        state = graph["agent_state"]
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
                        prompt, VERIFY_RESPONSE_SCHEMA, target_kind=PromptKind.VERIFY
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
                or self._latest_action_type(state.latest_action) == "wait"
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
                        prompt, VERIFY_RESPONSE_SCHEMA, target_kind=PromptKind.VERIFY
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
                    prompt, VERIFY_RESPONSE_SCHEMA, target_kind=PromptKind.VERIFY
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
        task_complete = bool(data["task_complete"])
        confidence = float(data["confidence"])
        recommended = str(data["recommended_next"])
        message = str(data["reason"])
        status = ResultStatus.SUCCESS if succeeded else ResultStatus.RETRY
        verification = ToolResult(
            tool_name="verify_action",
            status=status,
            output=data,
            message=message,
            metadata={"raw_response": raw},
        )
        state.update_verification_result(verification)

        if task_complete and confidence >= self.config.verify_confidence_threshold:
            self._commit_current_step(state)
            return {
                "agent_state": state, "route": GraphRoute.FINISH.value,
                "verify_data": data, "last_prompt": prompt,
                "last_raw_response": raw,
            }
        if succeeded and recommended in {"continue", "replan", "finish"}:
            self._commit_current_step(state)
            return {
                "agent_state": state, "route": GraphRoute.MEMORY.value,
                "verify_data": data, "last_prompt": prompt,
                "last_raw_response": raw,
            }
        return {
            "agent_state": state, "route": GraphRoute.REFLECT.value,
            "verify_data": data, "last_prompt": prompt,
            "last_raw_response": raw,
        }

    async def reflect_node(self, graph: GraphState) -> GraphState:
        state = graph["agent_state"]
        count = int(state.metadata.get("reflection_count", 0)) + 1
        state.metadata["reflection_count"] = count
        if count > self.config.max_reflections:
            return self._fail_update(state, "Maximum reflection count reached.")
        try:
            prompt = self.prompts.build_text(PromptKind.REFLECTION, state)
            data, raw = await self._call_structured(
                prompt, REFLECTION_RESPONSE_SCHEMA,
                target_kind=PromptKind.REFLECTION,
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
            MemoryKind.REFLECTION, summary,
            step_index=state.step_index,
            importance=MemoryImportance.HIGH,
            evidence=data.get("evidence", []),
            payload=data,
            tags=[str(data.get("failure_type", "unknown"))],
        )
        self.memory.attach_to_state(state)
        # Failed attempts still consume a step and become part of the audit trail.
        self._commit_current_step(state)
        if state.is_terminal or self._limit_reached(state):
            return self._fail_update(state, "Agent reached its step/time limit.")
        route = GraphRoute.PLAN if bool(data["should_replan"]) else GraphRoute.OBSERVE
        if route is GraphRoute.PLAN and state.observation is None:
            route = GraphRoute.OBSERVE
        else:
            state.set_phase(AgentPhase.PLANNING if route is GraphRoute.PLAN else AgentPhase.OBSERVING)
        return {
            "agent_state": state, "route": route.value,
            "reflection_data": data, "last_prompt": prompt,
            "last_raw_response": raw,
        }

    async def memory_node(self, graph: GraphState) -> GraphState:
        state = graph["agent_state"]
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
                    event_type="memory_summary_failed", message=str(error),
                    status=ResultStatus.RETRY,
                )
        self.memory.attach_to_state(state)
        if state.is_terminal:
            route = GraphRoute.FINISH if state.is_finished else GraphRoute.FAIL
        elif self._limit_reached(state):
            return self._fail_update(state, "Agent reached its step/time limit.")
        elif self._reuse_post_action_observation(state):
            state.set_phase(AgentPhase.PLANNING)
            route = GraphRoute.PLAN
        else:
            state.set_phase(AgentPhase.OBSERVING)
            route = GraphRoute.OBSERVE
        return {"agent_state": state, "route": route.value}

    def _reuse_post_action_observation(self, state: AgentState) -> bool:
        """Plan on the post-action capture instead of observing the same screen
        twice: nothing touched the GUI between the two observations."""

        if not self.config.reuse_post_action_observation:
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

    async def finish_node(self, graph: GraphState) -> GraphState:
        state = graph["agent_state"]
        if not state.is_terminal:
            planner_result = state.last_planner_result
            verify = graph.get("verify_data", {})
            message = (
                verify.get("reason")
                or getattr(planner_result, "finish_message", None)
                or getattr(planner_result, "reason", None)
                or "Task completed."
            )
            state.finish(str(message))
        self.memory.ingest_state(state, recent_only=20)
        self.memory.attach_to_state(state)
        return {"agent_state": state, "route": GraphRoute.FINISH.value}

    async def fail_node(self, graph: GraphState) -> GraphState:
        state = graph["agent_state"]
        if not state.is_terminal:
            message = graph.get("graph_error") or "Agent graph failed."
            state.fail(
                error=ErrorInfo(error_type="AgentGraphFailure", message=message),
                reason=RunTerminationReason.UNKNOWN,
                message=message,
            )
        self.memory.ingest_state(state, recent_only=20)
        self.memory.attach_to_state(state)
        return {"agent_state": state, "route": GraphRoute.FAIL.value}

    # ------------------------------------------------------------------
    # Model response and state helpers
    # ------------------------------------------------------------------

    async def _call_structured(
        self,
        prompt: str,
        schema: Mapping[str, Any],
        *,
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
        methods = ("agenerate_json", "agenerate", "generate_json", "generate")
        for name in methods:
            method = getattr(self.vlm, name, None)
            if not callable(method):
                continue
            kwargs = _supported_kwargs(
                method,
                prompt=prompt,
                schema=dict(schema),
            )
            value = method(**kwargs)
            return await value if inspect.isawaitable(value) else value
        raise AgentGraphError("VLM exposes no supported generate method")

    @staticmethod
    def _validate_verify(data: Mapping[str, Any]) -> None:
        required = {
            "status", "action_effective", "task_complete", "evidence",
            "reason", "confidence", "recommended_next",
        }
        _require_fields(data, required, "verify")
        if data["status"] not in {"success", "failure", "uncertain"}:
            raise ModelResponseError("Invalid verify status")
        confidence = float(data["confidence"])
        if not 0 <= confidence <= 1:
            raise ModelResponseError("Verify confidence must be between 0 and 1")

    @staticmethod
    def _validate_reflection(data: Mapping[str, Any]) -> None:
        required = {
            "failure_type", "summary", "evidence", "likely_cause", "avoid",
            "strategy", "should_replan", "confidence",
        }
        _require_fields(data, required, "reflection")
        if not isinstance(data["should_replan"], bool):
            raise ModelResponseError("should_replan must be boolean")

    @staticmethod
    def _commit_current_step(state: AgentState) -> None:
        if state.last_planner_result is None:
            return
        try:
            state.commit_step()
        except Exception:
            # FINISH planner decisions do not always form an executable step.
            if not state.last_planner_result.is_finished:
                raise

    @staticmethod
    def _limit_reached(state: AgentState) -> bool:
        return bool(state.runtime.reached_max_steps or state.runtime.is_timed_out)

    def _tool_failure_route(
        self, state: AgentState, result: ToolResult, *, stage: str
    ) -> GraphState:
        message = result.message or f"{stage} failed"
        if result.should_retry and not self._limit_reached(state):
            state.add_history(
                event_type=f"{stage}_retry", message=message,
                status=ResultStatus.RETRY,
            )
            return {
                "agent_state": state, "route": GraphRoute.REFLECT.value,
                "graph_error": message,
            }
        return self._fail_update(state, message, error=result.error)

    def _exception_reflection(
        self, state: AgentState, error: BaseException, stage: str
    ) -> GraphState:
        state.add_history(
            event_type=f"{stage}_error", message=str(error),
            status=ResultStatus.RETRY,
        )
        return {
            "agent_state": state, "route": GraphRoute.REFLECT.value,
            "graph_error": str(error),
        }

    def _exception_failure(
        self, state: AgentState, error: BaseException, stage: str
    ) -> GraphState:
        return self._fail_update(
            state, f"{stage} failed: {error}",
            error=ErrorInfo.from_exception(error, retryable=False),
        )

    @staticmethod
    def _fail_update(
        state: AgentState,
        message: str,
        *,
        error: ErrorInfo | None = None,
    ) -> GraphState:
        if not state.is_terminal:
            state.fail(
                error=error or ErrorInfo(
                    error_type="AgentGraphFailure", message=message
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
            "route": GraphRoute.FAIL.value,
            "graph_error": message,
        }


def create_agent_graph(
    *,
    planner: PlannerProtocol,
    tools: AgentTools,
    vlm: VLMProtocol,
    prompt_builder: PromptBuilder | None = None,
    memory: AgentMemory | None = None,
    memory_summarizer: Callable[..., Any] | None = None,
    config: AgentGraphConfig | None = None,
    compile_kwargs: Mapping[str, Any] | None = None,
) -> AgentGraph:
    """Create an ``AgentGraph`` and compile its LangGraph application."""

    graph = AgentGraph(
        planner=planner,
        tools=tools,
        vlm=vlm,
        prompt_builder=prompt_builder,
        memory=memory,
        memory_summarizer=memory_summarizer,
        config=config,
    )
    graph.compile(**dict(compile_kwargs or {}))
    return graph


def _supported_kwargs(method: Callable[..., Any], **candidates: Any) -> dict[str, Any]:
    """Pass only arguments accepted by a provider method."""

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return {"prompt": candidates["prompt"]}
    parameters = signature.parameters
    accepts_any = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_any:
        return candidates
    result = {key: value for key, value in candidates.items() if key in parameters}
    if not result:
        # Bound provider callables commonly expose one positional text argument.
        first = next(iter(parameters.values()), None)
        if first is not None:
            result[first.name] = candidates["prompt"]
    return result


def _parse_json_object(response: Any) -> dict[str, Any]:
    # BaseVLM.generate_json returns ``(parsed_value, VLMResponse)``.
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
            text = text[start:end + 1]
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
        raise ModelResponseError(f"{name} response missing fields: {', '.join(missing)}")


__all__ = [
    "AgentGraph",
    "AgentGraphConfig",
    "AgentGraphDependencyError",
    "AgentGraphError",
    "GraphRoute",
    "GraphState",
    "ModelResponseError",
    "PlannerProtocol",
    "VLMProtocol",
    "create_agent_graph",
]