"""
memory
Working-memory management for the GUI Agent.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import threading
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


class MemoryError(RuntimeError):
    """Base exception raised by the memory layer."""


class MemoryValidationError(MemoryError, ValueError):
    """Raised when memory input is structurally invalid."""


class MemorySerializationError(MemoryError):
    """Raised when persisted memory cannot be encoded or decoded."""


class MemorySummarizationError(MemoryError):
    """Raised when a summarizer returns an invalid result."""


class MemoryKind(str, Enum):
    OBSERVATION = "observation"
    PLAN = "plan"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    REFLECTION = "reflection"
    FACT = "fact"
    SUCCESS = "success"
    FAILURE = "failure"
    SYSTEM = "system"


class MemoryImportance(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    if _depth > 12:
        return "<max-depth-reached>"
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v, _depth=_depth + 1) for v in value]
    if is_dataclass(value):
        return _json_safe(asdict(value), _depth=_depth + 1)
    method = getattr(value, "to_dict", None)
    if callable(method):
        try:
            return _json_safe(method(), _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            k: _json_safe(v, _depth=_depth + 1)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return str(value)


def _coerce_datetime(value: Any) -> datetime:
    if value is None:
        return utc_now()
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        result = datetime.fromisoformat(str(value))
    except ValueError as error:
        raise MemoryValidationError(f"Invalid datetime: {value!r}") from error
    return result if result.tzinfo else result.replace(tzinfo=timezone.utc)


def _as_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise MemoryValidationError(f"{field_name} must be a sequence of strings.")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _normalise_text(item)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


@dataclass(slots=True)
class ImportantElement:
    element_id: str | int | None = None
    text: str = ""
    role: str = ""
    location: str = ""
    state: str = ""

    @classmethod
    def from_value(cls, value: Any) -> "ImportantElement":
        if isinstance(value, cls):
            return cls(**value.to_dict())
        if not isinstance(value, Mapping):
            raise MemoryValidationError("important_elements entries must be mappings.")
        return cls(
            element_id=value.get("element_id"),
            text=_normalise_text(value.get("text")),
            role=_normalise_text(value.get("role")),
            location=_normalise_text(value.get("location")),
            state=_normalise_text(value.get("state")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "text": self.text,
            "role": self.role,
            "location": self.location,
            "state": self.state,
        }


@dataclass(slots=True)
class MemorySummary:
    """Durable task memory matching ``MEMORY_SUMMARY_SCHEMA``."""

    task_goal: str = ""
    current_state: str = ""
    completed_steps: list[str] = field(default_factory=list)
    verified_facts: list[str] = field(default_factory=list)
    successful_methods: list[str] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)
    open_issues: list[str] = field(default_factory=list)
    next_focus: list[str] = field(default_factory=list)
    important_elements: list[ImportantElement] = field(default_factory=list)
    revision: int = 0
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.task_goal = _normalise_text(self.task_goal)
        self.current_state = _normalise_text(self.current_state)
        for name in (
            "completed_steps", "verified_facts", "successful_methods",
            "failed_attempts", "open_issues", "next_focus",
        ):
            setattr(self, name, _as_string_list(getattr(self, name), name))
        self.important_elements = [
            ImportantElement.from_value(item) for item in self.important_elements
        ]
        if not isinstance(self.revision, int) or self.revision < 0:
            raise MemoryValidationError("revision must be a non-negative integer.")
        self.updated_at = _coerce_datetime(self.updated_at)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemorySummary":
        if not isinstance(data, Mapping):
            raise MemoryValidationError("Memory summary must be a mapping.")
        return cls(
            task_goal=data.get("task_goal", ""),
            current_state=data.get("current_state", ""),
            completed_steps=data.get("completed_steps", []),
            verified_facts=data.get("verified_facts", []),
            successful_methods=data.get("successful_methods", []),
            failed_attempts=data.get("failed_attempts", []),
            open_issues=data.get("open_issues", []),
            next_focus=data.get("next_focus", []),
            important_elements=data.get("important_elements", []),
            revision=int(data.get("revision", 0)),
            updated_at=data.get("updated_at"),
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return only fields allowed by the model response schema."""
        return {
            "task_goal": self.task_goal,
            "current_state": self.current_state,
            "completed_steps": list(self.completed_steps),
            "verified_facts": list(self.verified_facts),
            "successful_methods": list(self.successful_methods),
            "failed_attempts": list(self.failed_attempts),
            "open_issues": list(self.open_issues),
            "next_focus": list(self.next_focus),
            "important_elements": [item.to_dict() for item in self.important_elements],
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_prompt_dict()
        value.update(revision=self.revision, updated_at=self.updated_at.isoformat())
        return value

    def is_empty(self) -> bool:
        return not any((
            self.task_goal, self.current_state, self.completed_steps,
            self.verified_facts, self.successful_methods, self.failed_attempts,
            self.open_issues, self.next_focus, self.important_elements,
        ))


@dataclass(slots=True)
class MemoryItem:
    """One bounded short-term memory event."""

    kind: MemoryKind
    content: str
    step_index: int = 0
    importance: MemoryImportance = MemoryImportance.NORMAL
    evidence: list[str] = field(default_factory=list)
    payload: Any = None
    tags: list[str] = field(default_factory=list)
    verified: bool = False
    item_id: str = field(default_factory=lambda: _new_id("memory"))
    created_at: datetime = field(default_factory=utc_now)
    fingerprint: str = ""

    def __post_init__(self) -> None:
        self.kind = MemoryKind(self.kind)
        self.importance = MemoryImportance(self.importance)
        self.content = _normalise_text(self.content)
        if not self.content:
            raise MemoryValidationError("MemoryItem.content must not be empty.")
        if not isinstance(self.step_index, int) or self.step_index < 0:
            raise MemoryValidationError("step_index must be non-negative.")
        self.evidence = _as_string_list(self.evidence, "evidence")
        self.tags = _as_string_list(self.tags, "tags")
        self.created_at = _coerce_datetime(self.created_at)
        if not self.fingerprint:
            raw = f"{self.kind.value}|{self.step_index}|{self.content.casefold()}"
            self.fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryItem":
        return cls(
            item_id=str(data.get("item_id") or _new_id("memory")),
            kind=data.get("kind", MemoryKind.SYSTEM.value),
            content=data.get("content", ""),
            step_index=int(data.get("step_index", 0)),
            importance=data.get("importance", MemoryImportance.NORMAL.value),
            evidence=data.get("evidence", []),
            payload=data.get("payload"),
            tags=data.get("tags", []),
            verified=bool(data.get("verified", False)),
            created_at=data.get("created_at"),
            fingerprint=str(data.get("fingerprint", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "kind": self.kind.value,
            "content": self.content,
            "step_index": self.step_index,
            "importance": self.importance.value,
            "evidence": list(self.evidence),
            "payload": _json_safe(self.payload),
            "tags": list(self.tags),
            "verified": self.verified,
            "created_at": self.created_at.isoformat(),
            "fingerprint": self.fingerprint,
        }


@dataclass(slots=True)
class MemoryConfig:
    max_short_term_items: int = 80
    recent_items_for_prompt: int = 20
    summarise_after_items: int = 30
    retain_after_summary: int = 10
    deduplicate: bool = True
    auto_attach_to_state: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_short_term_items", "recent_items_for_prompt",
            "summarise_after_items", "retain_after_summary",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise MemoryValidationError(f"{name} must be non-negative.")
        if self.retain_after_summary > self.max_short_term_items:
            raise MemoryValidationError(
                "retain_after_summary cannot exceed max_short_term_items."
            )


class SummarizerProtocol(Protocol):
    def __call__(
        self,
        *,
        state: Any,
        existing_memory: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any] | MemorySummary: ...


class AgentMemory:
    """Thread-safe short-term buffer and long-term task summary."""

    def __init__(
        self,
        *,
        config: MemoryConfig | None = None,
        summary: MemorySummary | Mapping[str, Any] | None = None,
        items: Iterable[MemoryItem | Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.summary = (
            summary if isinstance(summary, MemorySummary)
            else MemorySummary.from_dict(summary) if summary is not None
            else MemorySummary()
        )
        self._items: list[MemoryItem] = []
        self._fingerprints: set[str] = set()
        self._lock = threading.RLock()
        for item in items or ():
            self.add_item(item)

    @property
    def items(self) -> tuple[MemoryItem, ...]:
        with self._lock:
            return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        kind: MemoryKind | str,
        content: Any,
        *,
        step_index: int = 0,
        importance: MemoryImportance | str = MemoryImportance.NORMAL,
        evidence: Sequence[str] | None = None,
        payload: Any = None,
        tags: Sequence[str] | None = None,
        verified: bool = False,
    ) -> MemoryItem | None:
        return self.add_item(MemoryItem(
            kind=MemoryKind(kind), content=content, step_index=step_index,
            importance=MemoryImportance(importance), evidence=list(evidence or []),
            payload=payload, tags=list(tags or []), verified=verified,
        ))

    def add_item(self, item: MemoryItem | Mapping[str, Any]) -> MemoryItem | None:
        resolved = item if isinstance(item, MemoryItem) else MemoryItem.from_dict(item)
        with self._lock:
            if self.config.deduplicate and resolved.fingerprint in self._fingerprints:
                return None
            self._items.append(resolved)
            self._fingerprints.add(resolved.fingerprint)
            self._trim()
        return resolved

    def _trim(self) -> None:
        overflow = len(self._items) - self.config.max_short_term_items
        if overflow <= 0:
            return
        removed = self._items[:overflow]
        del self._items[:overflow]
        for item in removed:
            self._fingerprints.discard(item.fingerprint)

    def record_state_event(self, event: Any) -> MemoryItem | None:
        """Copy one ``StateHistoryEntry`` or mapping into short-term memory."""
        data = _json_safe(event)
        if not isinstance(data, Mapping):
            data = {"message": str(data)}
        event_type = _normalise_text(data.get("event_type", "system"))
        kind = _EVENT_KIND_MAP.get(event_type, MemoryKind.SYSTEM)
        message = data.get("message") or data.get("reason") or event_type
        status = str(data.get("status") or "").lower()
        importance = (
            MemoryImportance.HIGH
            if status in {"failed", "failure", "retry", "timeout"}
            else MemoryImportance.NORMAL
        )
        return self.add(
            kind, message, step_index=int(data.get("step_index", 0)),
            importance=importance, payload=data,
            verified=kind in {MemoryKind.VERIFICATION, MemoryKind.FACT},
        )

    def ingest_state(self, state: Any, *, recent_only: int | None = None) -> int:
        history = getattr(state, "history", None)
        if history is None and isinstance(state, Mapping):
            history = state.get("history", [])
        values = list(history or [])
        if recent_only is not None:
            values = values[-max(0, recent_only):]
        before = len(self)
        for event in values:
            self.record_state_event(event)
        return len(self) - before

    def recent(
        self,
        limit: int | None = None,
        *,
        kinds: Iterable[MemoryKind | str] | None = None,
        verified_only: bool = False,
    ) -> list[MemoryItem]:
        allowed = {MemoryKind(k) for k in kinds} if kinds is not None else None
        with self._lock:
            selected = [
                item for item in self._items
                if (allowed is None or item.kind in allowed)
                and (not verified_only or item.verified)
            ]
        resolved_limit = self.config.recent_items_for_prompt if limit is None else limit
        if not isinstance(resolved_limit, int) or resolved_limit < 0:
            raise MemoryValidationError("limit must be non-negative.")
        return selected[-resolved_limit:] if resolved_limit else []

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        kinds: Iterable[MemoryKind | str] | None = None,
    ) -> list[MemoryItem]:
        """Simple deterministic token search suitable for an in-process store."""
        tokens = {token.casefold() for token in _normalise_text(query).split() if token}
        if not tokens:
            return self.recent(limit, kinds=kinds)
        candidates = self.recent(len(self), kinds=kinds)
        scored: list[tuple[int, float, MemoryItem]] = []
        for item in candidates:
            haystack = f"{item.content} {' '.join(item.tags)} {' '.join(item.evidence)}".casefold()
            score = sum(token in haystack for token in tokens)
            if score:
                scored.append((score, item.created_at.timestamp(), item))
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [row[2] for row in scored[:limit]]

    def should_summarize(self) -> bool:
        return len(self) >= self.config.summarise_after_items > 0

    def summarize(self, state: Any, summarizer: SummarizerProtocol) -> MemorySummary:
        """Update long-term memory using an injected synchronous summarizer."""
        history = [item.to_dict() for item in self.recent()]
        try:
            value = summarizer(
                state=state,
                existing_memory=self.summary.to_prompt_dict(),
                history=history,
            )
        except Exception as error:
            raise MemorySummarizationError("Memory summarizer failed.") from error
        if inspect.isawaitable(value):
            raise MemorySummarizationError("Use asummarize() with an async summarizer.")
        result = _coerce_summary(value, previous_revision=self.summary.revision)
        self._commit_summary(result, state)
        return result

    async def asummarize(self, state: Any, summarizer: SummarizerProtocol) -> MemorySummary:
        history = [item.to_dict() for item in self.recent()]
        try:
            value = summarizer(
                state=state,
                existing_memory=self.summary.to_prompt_dict(),
                history=history,
            )
            if inspect.isawaitable(value):
                value = await value
        except Exception as error:
            raise MemorySummarizationError("Memory summarizer failed.") from error
        result = _coerce_summary(value, previous_revision=self.summary.revision)
        self._commit_summary(result, state)
        return result

    def _commit_summary(self, result: MemorySummary, state: Any = None) -> None:
        with self._lock:
            self.summary = result
            keep = self.config.retain_after_summary
            removed = self._items[:-keep] if keep else list(self._items)
            self._items = self._items[-keep:] if keep else []
            for item in removed:
                self._fingerprints.discard(item.fingerprint)
        if self.config.auto_attach_to_state and state is not None:
            self.attach_to_state(state)

    def attach_to_state(self, state: Any) -> None:
        """Expose the long-term summary through ``state.metadata['memory']``."""
        if isinstance(state, Mapping):
            metadata = state.setdefault("metadata", {})  # type: ignore[attr-defined]
        else:
            metadata = getattr(state, "metadata", None)
            if metadata is None:
                raise MemoryValidationError("State has no mutable metadata mapping.")
        if not isinstance(metadata, dict):
            raise MemoryValidationError("state.metadata must be a mutable dictionary.")
        metadata["memory"] = self.summary.to_prompt_dict()

    def clear(self, *, keep_summary: bool = False) -> None:
        with self._lock:
            self._items.clear()
            self._fingerprints.clear()
            if not keep_summary:
                self.summary = MemorySummary()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "config": asdict(self.config),
                "summary": self.summary.to_dict(),
                "items": [item.to_dict() for item in self._items],
            }

    def to_json(self, *, indent: int | None = 2) -> str:
        try:
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        except (TypeError, ValueError) as error:
            raise MemorySerializationError("Could not serialize AgentMemory.") from error

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentMemory":
        if int(data.get("version", 1)) != 1:
            raise MemorySerializationError("Unsupported memory format version.")
        return cls(
            config=MemoryConfig(**dict(data.get("config", {}))),
            summary=data.get("summary"),
            items=data.get("items", []),
        )

    @classmethod
    def from_json(cls, text: str) -> "AgentMemory":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise MemorySerializationError("Invalid memory JSON.") from error
        if not isinstance(data, Mapping):
            raise MemorySerializationError("Memory JSON must contain an object.")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(self.to_json(), encoding="utf-8")
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "AgentMemory":
        try:
            return cls.from_json(Path(path).read_text(encoding="utf-8"))
        except OSError as error:
            raise MemorySerializationError(f"Could not read memory file: {path}") from error


def _coerce_summary(value: Any, *, previous_revision: int) -> MemorySummary:
    if isinstance(value, MemorySummary):
        data = value.to_prompt_dict()
    else:
        data = _extract_mapping(value)
    required = {
        "task_goal", "current_state", "completed_steps", "verified_facts",
        "successful_methods", "failed_attempts", "open_issues", "next_focus",
        "important_elements",
    }
    missing = required.difference(data)
    if missing:
        raise MemorySummarizationError(
            f"Memory summary is missing fields: {', '.join(sorted(missing))}."
        )
    clean = {key: data[key] for key in required}
    clean["revision"] = previous_revision + 1
    clean["updated_at"] = utc_now()
    return MemorySummary.from_dict(clean)


def _extract_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "parsed") and isinstance(value.parsed, Mapping):
        return value.parsed
    if hasattr(value, "json") and isinstance(value.json, Mapping):
        return value.json
    if hasattr(value, "content"):
        value = value.content
    elif hasattr(value, "text"):
        value = value.text
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:-1]
            if lines and lines[0].strip().lower() == "json":
                lines = lines[1:]
            text = "\n".join(lines)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise MemorySummarizationError("Summarizer did not return valid JSON.") from error
        if isinstance(parsed, Mapping):
            return parsed
    raise MemorySummarizationError("Summarizer did not return a mapping.")


def make_vlm_summarizer(vlm: Any, *, prompt_config: Any = None) -> SummarizerProtocol:
    """Adapt the existing Memory Prompt and a VLM into a summarizer callable."""
    def summarizer(
        *, state: Any, existing_memory: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> Any:
        try:
            from .prompts.memory_prompt import build_memory_messages
        except ImportError as error:
            raise MemorySummarizationError(
                "src.agent.prompts.memory_prompt is unavailable."
            ) from error
        prompt = build_memory_messages(
            state, prompt_config, existing_memory=existing_memory, history=history,
        )
        messages = prompt.as_messages()
        generate = getattr(vlm, "generate", None)
        if not callable(generate):
            raise MemorySummarizationError("VLM must provide generate().")
        attempts = (
            lambda: generate(messages=messages),
            lambda: generate(prompt.text),
            lambda: generate(prompt=prompt.text),
        )
        last_error: Exception | None = None
        for call in attempts:
            try:
                return call()
            except TypeError as error:
                last_error = error
        raise MemorySummarizationError("Unsupported VLM generate() signature.") from last_error
    return summarizer


def summarize_with_vlm(
    memory: AgentMemory, state: Any, vlm: Any, *, prompt_config: Any = None,
) -> MemorySummary:
    return memory.summarize(
        state, make_vlm_summarizer(vlm, prompt_config=prompt_config)
    )


# Keys must match the event_type strings emitted by AgentState.add_history.
_EVENT_KIND_MAP: dict[str, MemoryKind] = {
    "observation_updated": MemoryKind.OBSERVATION,
    "observation": MemoryKind.OBSERVATION,
    "planner_result": MemoryKind.PLAN,
    "planned": MemoryKind.PLAN,
    "execution_result": MemoryKind.EXECUTION,
    "executed": MemoryKind.EXECUTION,
    "verification_result": MemoryKind.VERIFICATION,
    "verified": MemoryKind.VERIFICATION,
    "reflection": MemoryKind.REFLECTION,
    "run_finished": MemoryKind.SUCCESS,
    "run_failed": MemoryKind.FAILURE,
}


__all__ = [
    "AgentMemory", "ImportantElement", "MemoryConfig", "MemoryError",
    "MemoryImportance", "MemoryItem", "MemoryKind", "MemorySerializationError",
    "MemorySummarizationError", "MemorySummary", "MemoryValidationError",
    "SummarizerProtocol", "make_vlm_summarizer", "summarize_with_vlm",
]