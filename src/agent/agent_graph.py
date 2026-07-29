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

from .memory import AgentMemory, MemoryImportance, MemoryKind
from .result import ErrorInfo, ResultStatus, RunTerminationReason, ToolResult
from .state import AgentPhase, AgentState, ObservationState
from .tools import AgentTools
from .verify_policy import latest_action_type, should_skip_verification

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
    verify_confidence_threshold: float = 0.55
    max_reflections: int = 3
    max_model_repairs: int = 1
    summarize_memory: bool = True
    fail_on_execution_error: bool = False
    recursion_limit: int = 200

    def __post_init__(self) -> None:
        if not 0 <= self.post_action_wait_seconds <= 60:
            raise ValueError("post_action_wait_seconds must be between 0 and 60")
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
            self._route_map(GraphRoute.OBSERVE, GraphRoute.FINISH, GraphRoute.FAIL),
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
            route = GraphRoute.REFLECT if state.runtime.consecutive_failures else GraphRoute.OBSERVE
            return {"agent_state": state, "route": route.value}
        return self._fail_update(state, result.reason or "Planner returned no usable decision.")

    async def execute_node(self, graph: GraphState) -> GraphState:
        state = graph["agent_state"]
        result = await self.tools.acall(
            "execute_action", {"action": state.latest_action}
        )
        # Preserve retryability for the reflection loop when configured.
        if not result.succeeded and not self.config.fail_on_execution_error:
            result.status = ResultStatus.RETRY
            if result.error is not None:
                result.error.retryable = True
        state.update_execution_result(result)
        if state.is_terminal:
            return {"agent_state": state, "route": GraphRoute.FAIL.value}
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
    def _should_skip_verification(cls, action: Any) -> bool:
        return should_skip_verification(action)

    async def observe_after_node(self, graph: GraphState) -> GraphState:
        state = graph["agent_state"]
        if self.config.post_action_wait_seconds:
            await self.tools.acall(
                "wait", {"seconds": self.config.post_action_wait_seconds}
            )
        result = await self.tools.acall("observe", self.config.observe_options)
        if not result.succeeded or not isinstance(result.output, ObservationState):
            return self._tool_failure_route(state, result, stage="post-action observation")
        state.update_observation(result.output, tool_result=result)
        state.set_phase(AgentPhase.VERIFYING)
        return {"agent_state": state, "route": GraphRoute.VERIFY.value}

    async def verify_node(self, graph: GraphState) -> GraphState:
        state = graph["agent_state"]
        try:
            prompt = self.prompts.build_text(PromptKind.VERIFY, state)
            data, raw = await self._call_structured(
                prompt, VERIFY_RESPONSE_SCHEMA, target_kind=PromptKind.VERIFY
            )
            self._validate_verify(data)
        except Exception as error:
            return self._exception_reflection(state, error, "verification")

        succeeded = bool(data["action_effective"]) and data["status"] == "success"
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
        if succeeded and recommended in {"continue", "replan"}:
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
        else:
            state.set_phase(AgentPhase.OBSERVING)
            route = GraphRoute.OBSERVE
        return {"agent_state": state, "route": route.value}

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