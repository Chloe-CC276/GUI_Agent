"""
prompts/builder
Unified entry point for all GUI Agent prompt modules.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import PromptConfig, PromptKind
from .context_builder import ContextBuilder
from .memory_prompt import (
    MemoryPrompt,
    build_memory_messages,
    build_observation_summary_messages,
)
from .planner_prompt import (
    PlannerPrompt,
    build_planner_messages,
    prompt_config_from_planner_config,
)
from .reflection_prompt import ReflectionPrompt, build_reflection_messages
from .repair_prompt import RepairPrompt, build_repair_messages
from .schemas import get_response_schema
from .templates import PromptTemplate, get_template
from .verify_prompt import VerifyPrompt, build_verify_messages


PromptObject = PlannerPrompt | VerifyPrompt | RepairPrompt | ReflectionPrompt | MemoryPrompt


class PromptBuildError(ValueError):
    """Raised when the unified builder receives invalid or missing inputs."""


def _required(name: str, value: Any, kind: PromptKind) -> Any:
    if value is None:
        raise PromptBuildError(f"{kind.value} prompt requires {name!r}")
    return value


@dataclass(slots=True)
class PromptBuilder:
    """Build every supported prompt with one shared :class:`PromptConfig`.

    ``config`` may also be an existing ``PlannerConfig``-like object; it is
    converted once during initialisation to keep the current Planner callback
    interface compatible.
    """

    config: PromptConfig | Any | None = None
    _context_builder: ContextBuilder = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.config = prompt_config_from_planner_config(self.config)
        self._context_builder = ContextBuilder(self.config)

    def with_config(self, config: PromptConfig | Any | None) -> "PromptBuilder":
        """Return a new builder without mutating this builder's configuration."""

        return type(self)(config)

    def with_overrides(self, **changes: Any) -> "PromptBuilder":
        """Return a new builder with validated ``PromptConfig`` overrides."""

        return type(self)(self.config.with_overrides(**changes))

    def template(self, kind: PromptKind | str) -> PromptTemplate:
        """Return the registered template for ``kind``."""

        return get_template(kind)

    def schema(self, kind: PromptKind | str, *, copy: bool = True) -> dict[str, Any]:
        """Return the response JSON Schema registered for ``kind``."""

        return get_response_schema(kind, copy=copy)

    def context(self, state: Any, *, memory: Any = None) -> dict[str, Any]:
        """Convert AgentState-like input into the standard prompt context."""

        return self._context_builder.agent(state, memory=memory)

    def context_json(self, state: Any, *, memory: Any = None) -> str:
        """Return the standard AgentState context as bounded JSON text."""

        return self._context_builder.agent_json(state, memory=memory)

    def build(
        self,
        kind: PromptKind | str,
        source: Any = None,
        **kwargs: Any,
    ) -> PromptObject:
        """Build a separated system/user prompt for any supported kind.

        Common usage passes AgentState-like data as ``source``.  For repair,
        ``source`` is the original request and ``invalid_response`` plus
        ``validation_error`` are required.  For observation summaries,
        ``source`` is the observation and ``task`` is required.
        """

        resolved = PromptKind.coerce(kind)

        try:
            if resolved is PromptKind.PLANNER:
                return build_planner_messages(
                    _required("state", source, resolved), self.config, **kwargs
                )

            if resolved is PromptKind.VERIFY:
                return build_verify_messages(
                    _required("state", source, resolved), self.config, **kwargs
                )

            if resolved is PromptKind.REFLECTION:
                return build_reflection_messages(
                    _required("state", source, resolved), self.config, **kwargs
                )

            if resolved is PromptKind.MEMORY_SUMMARY:
                return build_memory_messages(
                    _required("state", source, resolved), self.config, **kwargs
                )

            if resolved is PromptKind.OBSERVATION_SUMMARY:
                task = _required("task", kwargs.pop("task", None), resolved)
                return build_observation_summary_messages(
                    _required("observation", source, resolved),
                    task,
                    self.config,
                    **kwargs,
                )

            if resolved is PromptKind.REPAIR:
                invalid_response = _required(
                    "invalid_response", kwargs.pop("invalid_response", None), resolved
                )
                validation_error = _required(
                    "validation_error", kwargs.pop("validation_error", None), resolved
                )
                return build_repair_messages(
                    _required("original_request", source, resolved),
                    invalid_response,
                    validation_error,
                    self.config,
                    **kwargs,
                )
        except PromptBuildError:
            raise
        except (TypeError, ValueError) as error:
            raise PromptBuildError(
                f"Failed to build {resolved.value} prompt: {error}"
            ) from error

        raise PromptBuildError(f"No builder registered for {resolved.value!r}")

    def build_text(
        self, kind: PromptKind | str, source: Any = None, **kwargs: Any
    ) -> str:
        """Build ``kind`` and return its combined system/user text."""

        return self.build(kind, source, **kwargs).text

    def build_messages(
        self, kind: PromptKind | str, source: Any = None, **kwargs: Any
    ) -> list[dict[str, str]]:
        """Build ``kind`` and return provider-neutral chat messages."""

        return self.build(kind, source, **kwargs).as_messages()

    def planner(self, state: Any, **kwargs: Any) -> PlannerPrompt:
        return build_planner_messages(state, self.config, **kwargs)

    def verify(self, state: Any, **kwargs: Any) -> VerifyPrompt:
        return build_verify_messages(state, self.config, **kwargs)

    def repair(
        self,
        original_request: Any,
        invalid_response: Any,
        validation_error: Any,
        **kwargs: Any,
    ) -> RepairPrompt:
        return build_repair_messages(
            original_request,
            invalid_response,
            validation_error,
            self.config,
            **kwargs,
        )

    def reflection(self, state: Any, **kwargs: Any) -> ReflectionPrompt:
        return build_reflection_messages(state, self.config, **kwargs)

    def memory(self, state: Any, **kwargs: Any) -> MemoryPrompt:
        return build_memory_messages(state, self.config, **kwargs)

    def observation_summary(
        self, observation: Any, task: Any, **kwargs: Any
    ) -> MemoryPrompt:
        return build_observation_summary_messages(
            observation, task, self.config, **kwargs
        )


def build_prompt(
    kind: PromptKind | str,
    source: Any = None,
    config: PromptConfig | Any | None = None,
    *,
    as_messages: bool = False,
    **kwargs: Any,
) -> str | list[dict[str, str]]:
    """Stateless convenience function for unified prompt construction."""

    builder = PromptBuilder(config)
    if as_messages:
        return builder.build_messages(kind, source, **kwargs)
    return builder.build_text(kind, source, **kwargs)


def build_prompt_object(
    kind: PromptKind | str,
    source: Any = None,
    config: PromptConfig | Any | None = None,
    **kwargs: Any,
) -> PromptObject:
    """Stateless helper returning the typed separated-message object."""

    return PromptBuilder(config).build(kind, source, **kwargs)


__all__ = [
    "PromptBuildError",
    "PromptBuilder",
    "PromptObject",
    "build_prompt",
    "build_prompt_object",
]