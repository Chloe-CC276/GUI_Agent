"""
agent/tools
Unified tool layer for the GUI Agent.

"""

from __future__ import annotations

import inspect
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .result import ErrorInfo, ResultStatus, TimingInfo, ToolResult, to_json_safe
from .state import ObservationSource, ObservationState


class ToolError(RuntimeError):
    """Base exception raised by the Agent tool layer."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool has not been registered."""


class ToolValidationError(ToolError):
    """Raised when a tool call contains invalid arguments."""


class ToolRegistrationError(ToolError):
    """Raised when a tool definition conflicts with the registry."""


class PerceptionProtocol(Protocol):
    def capture_and_run(self, region: Any = None, **kwargs: Any) -> Any: ...
    def process_image(self, image: Any, **kwargs: Any) -> Any: ...


class ExecutorProtocol(Protocol):
    def execute(self, action: Any) -> Any: ...


ToolHandler = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Metadata and callable for one Agent-facing tool."""

    name: str
    description: str
    handler: ToolHandler = field(repr=False, compare=False)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    category: str = "utility"
    requires_confirmation: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name or not name.replace("_", "").isalnum():
            raise ToolRegistrationError(
                "Tool name must contain only letters, numbers, and underscores."
            )
        if not callable(self.handler):
            raise ToolRegistrationError("Tool handler must be callable.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", self.description.strip())

    def schema(self) -> dict[str, Any]:
        """Return an OpenAI-style function definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters) or {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }


class ToolRegistry:
    """Register, inspect, and invoke named tool handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._lock = threading.RLock()

    def register(self, spec: ToolSpec, *, replace: bool = False) -> ToolSpec:
        if not isinstance(spec, ToolSpec):
            raise TypeError("spec must be a ToolSpec.")
        with self._lock:
            if spec.name in self._tools and not replace:
                raise ToolRegistrationError(
                    f"Tool {spec.name!r} is already registered."
                )
            self._tools[spec.name] = spec
        return spec

    def tool(
        self,
        name: str,
        description: str,
        *,
        parameters: Mapping[str, Any] | None = None,
        category: str = "utility",
        requires_confirmation: bool = False,
        replace: bool = False,
    ) -> Callable[[ToolHandler], ToolHandler]:
        """Decorator that registers a callable."""
        def decorator(handler: ToolHandler) -> ToolHandler:
            self.register(
                ToolSpec(
                    name=name,
                    description=description,
                    handler=handler,
                    parameters=parameters or {},
                    category=category,
                    requires_confirmation=requires_confirmation,
                ),
                replace=replace,
            )
            return handler
        return decorator

    def unregister(self, name: str) -> ToolSpec:
        with self._lock:
            try:
                return self._tools.pop(name)
            except KeyError as error:
                raise ToolNotFoundError(f"Unknown tool: {name!r}.") from error

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolNotFoundError(f"Unknown tool: {name!r}.") from error

    def names(self, *, enabled_only: bool = True) -> tuple[str, ...]:
        return tuple(
            name for name, spec in self._tools.items()
            if spec.enabled or not enabled_only
        )

    def schemas(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in self.names(enabled_only=enabled_only)]

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        spec = self.get(name)
        if not spec.enabled:
            raise ToolValidationError(f"Tool {name!r} is disabled.")
        kwargs = dict(arguments or {})
        if not isinstance(arguments, (Mapping, type(None))):
            raise ToolValidationError("Tool arguments must be a mapping or None.")
        _validate_handler_arguments(spec.handler, kwargs)
        return spec.handler(**kwargs)

    async def acall(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        value = self.call(name, arguments)
        if inspect.isawaitable(value):
            return await value
        return value

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


class AgentTools:
    """Default GUI tools backed by a perception pipeline and executor."""

    def __init__(
        self,
        *,
        perception: PerceptionProtocol | None = None,
        executor: ExecutorProtocol | None = None,
        registry: ToolRegistry | None = None,
        retryable_exceptions: tuple[type[BaseException], ...] = (
            TimeoutError,
            ConnectionError,
        ),
    ) -> None:
        self.perception = perception
        self.executor = executor
        self.registry = registry or ToolRegistry()
        self.retryable_exceptions = retryable_exceptions
        self._register_defaults()

    def _register_defaults(self) -> None:
        definitions = (
            ToolSpec(
                "observe",
                "Capture and analyse the current GUI screen.",
                self.observe,
                _OBSERVE_SCHEMA,
                "perception",
            ),
            ToolSpec(
                "observe_image",
                "Analyse an existing screenshot or image.",
                self.observe_image,
                _OBSERVE_IMAGE_SCHEMA,
                "perception",
            ),
            ToolSpec(
                "execute_action",
                "Execute one validated GUI action.",
                self.execute_action,
                _EXECUTE_SCHEMA,
                "execution",
                True,
            ),
            ToolSpec(
                "wait",
                "Wait briefly before the next observation.",
                self.wait,
                _WAIT_SCHEMA,
                "utility",
            ),
        )
        for spec in definitions:
            if spec.name not in self.registry:
                self.registry.register(spec)

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        """Invoke a registered tool and always return ``ToolResult``."""
        started = time.perf_counter()
        try:
            value = self.registry.call(name, arguments)
            result = value if isinstance(value, ToolResult) else ToolResult.success(
                name, output=value, message=f"Tool {name!r} completed."
            )
        except Exception as error:  # tool boundary intentionally normalises errors
            result = self._failed_result(name, error, started)
        _finish_timing(result, started)
        return result

    async def acall(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        started = time.perf_counter()
        try:
            value = await self.registry.acall(name, arguments)
            result = value if isinstance(value, ToolResult) else ToolResult.success(
                name, output=value, message=f"Tool {name!r} completed."
            )
        except Exception as error:
            result = self._failed_result(name, error, started)
        _finish_timing(result, started)
        return result

    def observe(
        self,
        region: Sequence[int] | None = None,
        **options: Any,
    ) -> ToolResult:
        if self.perception is None:
            raise ToolValidationError("No perception pipeline is configured.")
        resolved_region = _normalise_region(region)
        raw = self.perception.capture_and_run(region=resolved_region, **options)
        observation = perception_to_observation(raw)
        return ToolResult.success(
            "observe",
            output=observation,
            message=f"Observed {observation.element_count} GUI elements.",
            metadata={"observation_id": observation.observation_id},
        )

    def observe_image(self, image: Any, **options: Any) -> ToolResult:
        if self.perception is None:
            raise ToolValidationError("No perception pipeline is configured.")
        if image is None or (isinstance(image, str) and not image.strip()):
            raise ToolValidationError("image must not be empty.")
        raw = self.perception.process_image(image, **options)
        observation = perception_to_observation(raw, screenshot_path=image if isinstance(image, (str, Path)) else None)
        return ToolResult.success(
            "observe_image", output=observation,
            message=f"Analysed image with {observation.element_count} GUI elements.",
        )

    def execute_action(self, action: Any) -> ToolResult:
        if self.executor is None:
            raise ToolValidationError("No executor is configured.")
        if action is None:
            raise ToolValidationError("action must not be None.")
        raw = self.executor.execute(action)
        success = bool(getattr(raw, "success", False))
        output = raw.summary() if hasattr(raw, "summary") else to_json_safe(raw)
        if success:
            return ToolResult.success(
                "execute_action", output=output,
                message=getattr(raw, "message", None) or "Action executed.",
                metadata={"raw_result": raw},
            )
        error_text = getattr(raw, "error", None) or getattr(raw, "message", None) or "Action execution failed."
        return ToolResult.failed(
            "execute_action",
            ErrorInfo(error_type="ExecutionError", message=str(error_text), retryable=False),
            message=str(error_text),
            output=output,
            metadata={"raw_result": raw},
        )

    @staticmethod
    def wait(seconds: float = 1.0) -> ToolResult:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ToolValidationError("seconds must be a number.")
        if seconds < 0 or seconds > 60:
            raise ToolValidationError("seconds must be between 0 and 60.")
        time.sleep(float(seconds))
        return ToolResult.success("wait", output={"seconds": float(seconds)}, message=f"Waited {seconds:g} seconds.")

    def _failed_result(self, name: str, error: BaseException, started: float) -> ToolResult:
        retryable = isinstance(error, self.retryable_exceptions)
        status = ResultStatus.RETRY if retryable else ResultStatus.FAILED
        info = ErrorInfo.from_exception(error, retryable=retryable)
        return ToolResult(
            tool_name=name,
            status=status,
            error=info if status == ResultStatus.FAILED else info,
            message=str(error),
            metadata={"elapsed_seconds": time.perf_counter() - started},
        )


def perception_to_observation(
    result: Any,
    *,
    screenshot_path: str | Path | None = None,
) -> ObservationState:
    """Convert a concrete ``PerceptionResult`` into shared Agent state."""
    if result is None:
        raise ToolValidationError("Perception returned no result.")
    image = getattr(result, "original_image", None)
    shape = getattr(image, "shape", None)
    height = int(shape[0]) if shape is not None and len(shape) >= 2 else None
    width = int(shape[1]) if shape is not None and len(shape) >= 2 else None
    elements = list(getattr(result, "merged_elements", None) or [])
    ocr_items = list(getattr(result, "ocr_elements", None) or [])
    texts = [str(getattr(item, "text", "")).strip() for item in ocr_items]
    metadata = dict(getattr(result, "metadata", None) or {})
    metadata.update({
        "capture_region": getattr(result, "capture_region", None),
        "perception_elapsed_seconds": getattr(result, "elapsed_time", None),
    })
    return ObservationState(
        screenshot=image,
        screenshot_path=str(screenshot_path) if screenshot_path is not None else None,
        screen_width=width,
        screen_height=height,
        ocr_text="\n".join(text for text in texts if text) or None,
        ocr_items=ocr_items,
        gui_elements=elements,
        source=ObservationSource.PERCEPTION,
        raw_observation=result,
        metadata=metadata,
    )


def _validate_handler_arguments(handler: ToolHandler, arguments: Mapping[str, Any]) -> None:
    try:
        inspect.signature(handler).bind(**arguments)
    except (TypeError, ValueError) as error:
        raise ToolValidationError(f"Invalid tool arguments: {error}") from error


def _normalise_region(region: Sequence[int] | None) -> tuple[int, int, int, int] | None:
    if region is None:
        return None
    if isinstance(region, (str, bytes)) or len(region) != 4:
        raise ToolValidationError("region must contain left, top, width, height.")
    values = tuple(region)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ToolValidationError("region values must be integers.")
    if values[2] <= 0 or values[3] <= 0:
        raise ToolValidationError("region width and height must be positive.")
    return values  # type: ignore[return-value]


def _finish_timing(result: ToolResult, started: float) -> None:
    timing = getattr(result, "timing", None)
    if isinstance(timing, TimingInfo) and not timing.is_finished:
        timing.finish()
    result.metadata.setdefault("elapsed_seconds", time.perf_counter() - started)


_OBSERVE_SCHEMA = {
    "type": "object",
    "properties": {
        "region": {"type": ["array", "null"], "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
        "enable_preprocessing": {"type": "boolean"},
        "enable_ocr": {"type": "boolean"},
        "enable_ui_detection": {"type": "boolean"},
        "merge_results": {"type": "boolean"},
    },
    "additionalProperties": True,
}

_OBSERVE_IMAGE_SCHEMA = {
    "type": "object",
    "properties": {"image": {"description": "Image array or local image path."}},
    "required": ["image"],
    "additionalProperties": True,
}

_EXECUTE_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": ["object", "string"]}},
    "required": ["action"],
    "additionalProperties": False,
}

_WAIT_SCHEMA = {
    "type": "object",
    "properties": {"seconds": {"type": "number", "minimum": 0, "maximum": 60, "default": 1}},
    "additionalProperties": False,
}


__all__ = [
    "AgentTools",
    "ExecutorProtocol",
    "PerceptionProtocol",
    "ToolError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolRegistry",
    "ToolSpec",
    "ToolValidationError",
    "perception_to_observation",
]