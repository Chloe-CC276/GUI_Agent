"""LangChain-side robustness helpers for AgentChain.

Keeps error taxonomy, task decomposition, stall detection, and retry-budget
accounting in one place so AgentChain stages stay thin.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .result import ErrorInfo


ERROR_CLASSES: tuple[str, ...] = (
    "parse",
    "validation",
    "empty_ocr",
    "wrong_window",
    "no_progress",
    "timeout",
    "vlm_provider",
    "target_rejected",
    "planner_retry_exhausted",
    "reflection_exhausted",
    "unknown",
)

RETRY_BUDGET_KEYS: tuple[str, ...] = (
    "planner",
    "vlm",
    "repair",
    "step",
    "task",
    "target_rejection",
    "repeated_action",
    "empty_ocr",
)

_SPLIT_PATTERN = re.compile(
    r"(?:然后请|然后再|然后|接着|再请|之后|and then|after that|then)",
    flags=re.IGNORECASE,
)
_VERB_AND_PATTERN = re.compile(
    r"(?:，|\s)?并(?=打开|点击|输入|搜索|关闭|发送|提交|选择|填写)"
)


def empty_retry_budget() -> dict[str, int]:
    return {key: 0 for key in RETRY_BUDGET_KEYS}


def ensure_retry_budget(metadata: dict[str, Any]) -> dict[str, int]:
    budget = metadata.get("retry_budget_used")
    if not isinstance(budget, dict):
        budget = empty_retry_budget()
        metadata["retry_budget_used"] = budget
        return budget
    for key in RETRY_BUDGET_KEYS:
        budget.setdefault(key, 0)
    return budget


def bump_retry_budget(state: Any, key: str, amount: int = 1) -> int:
    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, dict):
        return 0
    budget = ensure_retry_budget(metadata)
    if key not in budget:
        budget[key] = 0
    budget[key] = int(budget[key] or 0) + amount
    return budget[key]


def sync_step_budget(state: Any) -> dict[str, int]:
    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, dict):
        return empty_retry_budget()
    budget = ensure_retry_budget(metadata)
    runtime = getattr(state, "runtime", None)
    budget["step"] = int(getattr(runtime, "retry_count", 0) or 0)
    budget["repeated_action"] = int(
        getattr(runtime, "repeated_action_count", 0) or 0
    )
    return budget


def mark_recoverable(state: Any, error_class: str | None = None) -> None:
    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, dict):
        return
    metadata["recoverable_failure"] = True
    metadata["recoverable_failure_count"] = int(
        metadata.get("recoverable_failure_count") or 0
    ) + 1
    if error_class:
        metadata["last_error_class"] = error_class


def classify_error(
    error: Any = None,
    message: str = "",
    *,
    stage: str = "",
) -> str:
    """Map an exception / message onto the shared AgentChain error taxonomy."""

    error_class = getattr(error, "error_class", None)
    if isinstance(error_class, str) and error_class.strip():
        return error_class.strip()

    type_name = type(error).__name__ if error is not None else ""
    details = getattr(error, "message", "") if error is not None else ""
    text = f"{type_name} {details} {message} {stage}".lower()

    if "timeout" in text or "timed out" in text:
        return "timeout"
    if any(token in text for token in ("json", "parse", "modelresponseerror", "schema")):
        return "parse"
    if "empty" in text and "ocr" in text:
        return "empty_ocr"
    if "no_progress" in text or "repeated" in text or "stall" in text:
        return "no_progress"
    if "window" in text:
        return "wrong_window"
    if "validation" in text or "validate" in text or "evidence" in text:
        return "validation"
    if "vlm" in text or "provider" in text:
        return "vlm_provider"
    if "reflection" in text:
        return "reflection_exhausted"
    if "planner retry" in text:
        return "planner_retry_exhausted"
    return "unknown"


def apply_error_class(error: ErrorInfo | None, error_class: str) -> ErrorInfo | None:
    if error is None:
        return None
    if not getattr(error, "error_class", None):
        error.error_class = error_class
    return error


def decompose_instruction(
    instruction: str,
    *,
    max_sub_tasks: int = 8,
) -> list[dict[str, str]]:
    """Split a complex goal into ordered sub-tasks (heuristic, no VLM)."""

    text = " ".join(str(instruction or "").split()).strip()
    if not text:
        return []

    chunks = [part.strip(" ，,;；") for part in _SPLIT_PATTERN.split(text) if part]
    expanded: list[str] = []
    for chunk in chunks:
        pieces = [part.strip(" ，,;；") for part in _VERB_AND_PATTERN.split(chunk) if part]
        expanded.extend(pieces or [chunk])

    unique: list[str] = []
    for item in expanded:
        if item and item not in unique:
            unique.append(item)
    if len(unique) <= 1:
        unique = [text]
    unique = unique[: max(1, max_sub_tasks)]
    return [
        {
            "instruction": item,
            "success_criteria": item,
        }
        for item in unique
    ]


def sub_task_text(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("instruction") or item.get("success_criteria") or "").strip()
    return str(item or "").strip()


def action_fingerprint(action: Any) -> str:
    if action is None:
        return ""
    data = action
    if hasattr(action, "to_dict"):
        try:
            data = action.to_dict()
        except Exception:
            data = action
    if not isinstance(data, Mapping):
        data = {
            "type": getattr(action, "type", None) or getattr(action, "action_type", None),
            "parameters": getattr(action, "parameters", None),
        }
    action_type = data.get("type") or data.get("action_type") or ""
    if hasattr(action_type, "value"):
        action_type = action_type.value
    params = data.get("parameters") if isinstance(data.get("parameters"), Mapping) else data
    keys = ("target_text", "element_id", "x", "y", "text", "key", "keys")
    parts = [str(action_type).lower()]
    for key in keys:
        value = params.get(key) if isinstance(params, Mapping) else None
        if value is not None:
            parts.append(f"{key}={value}")
    return "|".join(parts)


def observation_fingerprint(observation: Any) -> str:
    if observation is None:
        return ""
    title = getattr(observation, "window_title", None) or ""
    app = getattr(observation, "application_name", None) or ""
    ocr = getattr(observation, "ocr_text", None) or ""
    elements = getattr(observation, "gui_elements", None) or []
    labels: list[str] = []
    for item in list(elements)[:40]:
        if isinstance(item, Mapping):
            labels.append(str(item.get("text") or item.get("name") or item.get("label") or ""))
        else:
            labels.append(
                str(
                    getattr(item, "text", None)
                    or getattr(item, "name", None)
                    or getattr(item, "label", None)
                    or ""
                )
            )
    return f"{app}|{title}|{ocr}|{'/'.join(labels)}"


def is_empty_observation(observation: Any) -> bool:
    if observation is None:
        return True
    ocr = str(getattr(observation, "ocr_text", None) or "").strip()
    elements = getattr(observation, "gui_elements", None) or []
    return not ocr and len(list(elements)) == 0


def robustness_snapshot(state: Any) -> dict[str, Any]:
    """Fields copied into evaluation JSONL / run metadata."""

    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, Mapping):
        metadata = {}
    runtime = getattr(state, "runtime", None)
    error = getattr(state, "error", None)
    sub_tasks = list(metadata.get("sub_tasks") or [])
    budget = dict(metadata.get("retry_budget_used") or empty_retry_budget())
    if runtime is not None:
        budget["step"] = int(getattr(runtime, "retry_count", 0) or 0)
        budget["repeated_action"] = int(
            getattr(runtime, "repeated_action_count", 0) or 0
        )
    error_class = (
        metadata.get("last_error_class")
        or getattr(error, "error_class", None)
    )
    return {
        "planner_retry_count": int(metadata.get("planner_retry_count") or 0),
        "target_rejection_count": int(metadata.get("target_rejection_count") or 0),
        "repeated_action_count": int(
            getattr(runtime, "repeated_action_count", 0) or 0
        ),
        "replan_count": int(metadata.get("replan_count") or 0),
        "sub_task_count": len(sub_tasks),
        "sub_task_index": int(metadata.get("sub_task_index") or 0),
        "sub_tasks": [sub_task_text(item) for item in sub_tasks],
        "retry_budget_used": budget,
        "error_class": error_class,
        "recoverable_failure": bool(metadata.get("recoverable_failure")),
        "recoverable_failure_count": int(
            metadata.get("recoverable_failure_count") or 0
        ),
        "auto_recovered": bool(metadata.get("auto_recovered")),
    }


def robustness_fields_from_run(run_result: Any) -> dict[str, Any]:
    metadata = getattr(run_result, "metadata", None)
    if isinstance(metadata, Mapping) and metadata.get("retry_budget_used") is not None:
        return {
            "planner_retry_count": int(metadata.get("planner_retry_count") or 0),
            "target_rejection_count": int(metadata.get("target_rejection_count") or 0),
            "repeated_action_count": int(metadata.get("repeated_action_count") or 0),
            "replan_count": int(metadata.get("replan_count") or 0),
            "sub_task_count": int(metadata.get("sub_task_count") or 0),
            "sub_task_index": int(metadata.get("sub_task_index") or 0),
            "retry_budget_used": metadata.get("retry_budget_used"),
            "error_class": metadata.get("error_class") or metadata.get("last_error_class"),
            "recoverable_failure": bool(metadata.get("recoverable_failure")),
            "recoverable_failure_count": int(
                metadata.get("recoverable_failure_count") or 0
            ),
            "auto_recovered": bool(metadata.get("auto_recovered")),
        }
    state = getattr(run_result, "agent_state", None) or run_result
    return robustness_snapshot(state)


__all__ = [
    "ERROR_CLASSES",
    "RETRY_BUDGET_KEYS",
    "action_fingerprint",
    "apply_error_class",
    "bump_retry_budget",
    "classify_error",
    "decompose_instruction",
    "empty_retry_budget",
    "ensure_retry_budget",
    "is_empty_observation",
    "mark_recoverable",
    "observation_fingerprint",
    "robustness_fields_from_run",
    "robustness_snapshot",
    "sub_task_text",
    "sync_step_budget",
]
