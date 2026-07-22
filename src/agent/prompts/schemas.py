"""
prompts/schemas
JSON response schemas used by the GUI Agent prompt builders.
"""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, Mapping

from .config import DEFAULT_ALLOWED_ACTIONS, PromptKind


JSONSchema = dict[str, Any]


def _nullable(schema: JSONSchema) -> JSONSchema:
    """Return a schema that accepts either ``schema`` or JSON null."""

    return {"anyOf": [schema, {"type": "null"}]}


def _string_list(description: str, *, max_items: int = 20) -> JSONSchema:
    """Build a bounded, unique list-of-strings schema."""

    return {
        "type": "array",
        "description": description,
        "items": {"type": "string", "minLength": 1},
        "maxItems": max_items,
        "uniqueItems": True,
    }


_ACTION_PARAMETERS_SCHEMA: JSONSchema = {
    "type": "object",
    "description": (
        "Parameters for the selected action. Include only fields needed by "
        "that action; coordinates are screenshot pixel coordinates."
    ),
    "properties": {
        "x": {"type": "number", "description": "Absolute or relative x coordinate."},
        "y": {"type": "number", "description": "Absolute or relative y coordinate."},
        "duration": {"type": "number", "minimum": 0},
        "button": {"type": "string", "enum": ["left", "middle", "right"]},
        "clicks": {"type": "integer", "minimum": 1},
        "interval": {"type": "number", "minimum": 0},
        "amount": {"type": "number", "description": "Scroll amount."},
        "key": {"type": "string", "minLength": 1},
        "keys": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        },
        "text": {"type": "string"},
        "seconds": {"type": "number", "minimum": 0},
        "message": {"type": "string"},
        "region": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        },
        "element_id": {"type": ["string", "integer"]},
    },
    "additionalProperties": False,
}


PLANNER_RESPONSE_SCHEMA: JSONSchema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://gui-agent.local/schemas/planner-response.json",
    "title": "PlannerResponse",
    "description": "One safe next GUI action or a terminal planner decision.",
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["act", "finish", "retry", "fail"],
        },
        "action": _nullable(
            {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": list(DEFAULT_ALLOWED_ACTIONS),
                    },
                    "parameters": deepcopy(_ACTION_PARAMETERS_SCHEMA),
                },
                "required": ["type", "parameters"],
                "additionalProperties": False,
            }
        ),
        "reason": {
            "type": "string",
            "minLength": 1,
            "description": "Short evidence-based explanation; never hidden reasoning.",
        },
        "confidence": _nullable({"type": "number", "minimum": 0, "maximum": 1}),
    },
    "required": ["decision", "action", "reason", "confidence"],
    "additionalProperties": False,
    "allOf": [
        {
            "if": {"properties": {"decision": {"const": "act"}}},
            "then": {"properties": {"action": {"type": "object"}}},
        },
        {
            "if": {
                "properties": {
                    "decision": {"enum": ["finish", "retry", "fail"]}
                }
            },
            "then": {"properties": {"action": {"type": "null"}}},
        },
    ],
}


VERIFY_RESPONSE_SCHEMA: JSONSchema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://gui-agent.local/schemas/verify-response.json",
    "title": "VerifyResponse",
    "description": "Evidence-based assessment of the latest executed GUI action.",
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["success", "failure", "uncertain"],
        },
        "action_effective": {"type": "boolean"},
        "task_complete": {"type": "boolean"},
        "evidence": _string_list("Visible evidence supporting the result.", max_items=10),
        "reason": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "recommended_next": {
            "type": "string",
            "enum": ["continue", "retry", "replan", "finish", "fail"],
        },
    },
    "required": [
        "status",
        "action_effective",
        "task_complete",
        "evidence",
        "reason",
        "confidence",
        "recommended_next",
    ],
    "additionalProperties": False,
}


REFLECTION_RESPONSE_SCHEMA: JSONSchema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://gui-agent.local/schemas/reflection-response.json",
    "title": "ReflectionResponse",
    "description": "Compact diagnosis of stalled or failed GUI-agent progress.",
    "type": "object",
    "properties": {
        "failure_type": {
            "type": "string",
            "enum": [
                "none",
                "wrong_target",
                "invalid_action",
                "no_effect",
                "repeated_action",
                "missing_evidence",
                "interface_changed",
                "blocked",
                "unknown",
            ],
        },
        "summary": {"type": "string", "minLength": 1},
        "evidence": _string_list("Observed facts supporting the diagnosis.", max_items=10),
        "likely_cause": {"type": "string", "minLength": 1},
        "avoid": _string_list("Actions or assumptions that should not be repeated."),
        "strategy": _string_list("High-level adjustments for the next planning cycle."),
        "should_replan": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "failure_type",
        "summary",
        "evidence",
        "likely_cause",
        "avoid",
        "strategy",
        "should_replan",
        "confidence",
    ],
    "additionalProperties": False,
}


MEMORY_SUMMARY_SCHEMA: JSONSchema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://gui-agent.local/schemas/memory-summary.json",
    "title": "MemorySummary",
    "description": "Structured, evidence-aware summary of reusable task memory.",
    "type": "object",
    "properties": {
        "task_goal": {"type": "string"},
        "current_state": {"type": "string"},
        "completed_steps": _string_list("Steps confirmed as completed."),
        "verified_facts": _string_list("Facts directly supported by observations."),
        "successful_methods": _string_list("Approaches that visibly worked."),
        "failed_attempts": _string_list("Failed approaches and their observed effects."),
        "open_issues": _string_list("Unresolved blockers or uncertainties."),
        "next_focus": _string_list("Useful priorities for the next planning cycle."),
        "important_elements": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "element_id": _nullable({"type": ["string", "integer"]}),
                    "text": {"type": "string"},
                    "role": {"type": "string"},
                    "location": {"type": "string"},
                    "state": {"type": "string"},
                },
                "required": ["element_id", "text", "role", "location", "state"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "task_goal",
        "current_state",
        "completed_steps",
        "verified_facts",
        "successful_methods",
        "failed_attempts",
        "open_issues",
        "next_focus",
        "important_elements",
    ],
    "additionalProperties": False,
}


RESPONSE_SCHEMAS: Mapping[PromptKind, JSONSchema] = MappingProxyType(
    {
        PromptKind.PLANNER: PLANNER_RESPONSE_SCHEMA,
        PromptKind.VERIFY: VERIFY_RESPONSE_SCHEMA,
        PromptKind.REPAIR: PLANNER_RESPONSE_SCHEMA,
        PromptKind.REFLECTION: REFLECTION_RESPONSE_SCHEMA,
        PromptKind.MEMORY_SUMMARY: MEMORY_SUMMARY_SCHEMA,
        PromptKind.OBSERVATION_SUMMARY: MEMORY_SUMMARY_SCHEMA,
    }
)


def get_response_schema(
    kind: PromptKind | str,
    *,
    copy: bool = True,
) -> JSONSchema:
    """Return the response schema registered for ``kind``.

    A deep copy is returned by default so callers may customize a schema
    without mutating the shared module-level definition.
    """

    schema = RESPONSE_SCHEMAS[PromptKind.coerce(kind)]
    return deepcopy(schema) if copy else schema


__all__ = [
    "PLANNER_RESPONSE_SCHEMA",
    "VERIFY_RESPONSE_SCHEMA",
    "REFLECTION_RESPONSE_SCHEMA",
    "MEMORY_SUMMARY_SCHEMA",
    "RESPONSE_SCHEMAS",
    "get_response_schema",
]