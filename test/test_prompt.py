"""
Tests for prompt construction, delivery, repair and reflection feedback.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.agent.prompts import PromptBuilder, PromptConfig, PromptLanguage


class RecordingVLM:
    """A deterministic model double that records every received message."""

    def __init__(self, responses: list[dict[str, Any] | str]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict[str, str]]] = []

    def generate(self, *, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        if not self.responses:
            raise AssertionError("RecordingVLM has no scripted response left")
        response = self.responses.pop(0)
        return response if isinstance(response, str) else json.dumps(response)


def call_model(vlm: RecordingVLM, messages: list[dict[str, str]]) -> str:
    """The same boundary used by production code to send built messages."""

    return vlm.generate(messages=messages)


def validate_required_fields(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    """Small dependency-free check; production may replace this with jsonschema."""

    missing = [name for name in schema.get("required", []) if name not in payload]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder(PromptConfig(language=PromptLanguage.ZH))


@pytest.fixture
def base_state() -> dict[str, Any]:
    return {
        "instruction": "点击 Submit 按钮",
        "phase": "planning",
        "runtime": {"step_index": 1, "consecutive_failures": 0},
        "observation": {
            "screenshot_path": "step_01.png",
            "screen_size": [1280, 720],
            "ocr_text": "Name  Submit  Clear",
            "elements": [
                {
                    "id": "submit_button",
                    "text": "Submit",
                    "element_type": "button",
                    "bbox": [500, 320, 590, 360],
                    "confidence": 0.98,
                }
            ],
        },
        "history": [],
    }


def test_all_prompt_kinds_can_be_built(
    builder: PromptBuilder, base_state: dict[str, Any]
) -> None:
    """Basic unit test: all developed prompt builders return usable messages."""

    previous = {
        "screenshot_path": "before.png",
        "ocr_text": "Name Submit Clear",
        "elements": [],
    }
    current = {
        "screenshot_path": "after.png",
        "ocr_text": "Submitted successfully",
        "elements": [],
    }
    verify_state = {
        **base_state,
        "previous_observation": previous,
        "observation": current,
        "last_planner_result": {
            "decision": "act",
            "action": {"type": "click", "parameters": {"x": 545, "y": 340}},
        },
        "last_execution_result": {"success": True, "message": "click delivered"},
    }

    cases = {
        "planner": builder.build_messages("planner", base_state),
        "verify": builder.build_messages("verify", verify_state),
        "reflection": builder.build_messages(
            "reflection",
            {**base_state, "runtime": {"consecutive_failures": 2}},
            failure_context={"error": "click produced no visible change"},
        ),
        "memory_summary": builder.build_messages("memory_summary", base_state),
        "observation_summary": builder.build_messages(
            "observation_summary",
            base_state["observation"],
            task=base_state["instruction"],
        ),
        "repair": builder.build_messages(
            "repair",
            "ORIGINAL_REQUEST_MARKER",
            invalid_response='{"decision":"act"}',
            validation_error="action is required",
        ),
    }

    assert set(cases) == {
        "planner", "verify", "reflection", "memory_summary",
        "observation_summary", "repair",
    }
    for messages in cases.values():
        assert messages
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"].strip()


def test_planner_prompt_is_really_sent_to_model(
    builder: PromptBuilder, base_state: dict[str, Any]
) -> None:
    """Integration boundary: generated task/context must reach VLM.generate."""

    vlm = RecordingVLM([
        {
            "decision": "act",
            "action": {"type": "click", "parameters": {"x": 545, "y": 340}},
            "reason": "Submit 按钮可见",
            "confidence": 0.96,
        }
    ])
    messages = builder.build_messages("planner", base_state)
    raw = call_model(vlm, messages)
    result = json.loads(raw)

    assert len(vlm.requests) == 1
    sent_text = "\n".join(item["content"] for item in vlm.requests[0])
    assert "点击 Submit 按钮" in sent_text
    assert "Submit" in sent_text
    assert result["action"]["type"] == "click"


def test_invalid_response_is_fed_into_repair_prompt(
    builder: PromptBuilder, base_state: dict[str, Any]
) -> None:
    """First response is invalid; the second request must contain repair data."""

    invalid_response = '{"decision":"act","reason":"click it","confidence":0.7}'
    corrected_response = {
        "decision": "act",
        "action": {"type": "click", "parameters": {"x": 545, "y": 340}},
        "reason": "补全缺失的 action",
        "confidence": 0.7,
    }
    vlm = RecordingVLM([invalid_response, corrected_response])

    planner_messages = builder.build_messages("planner", base_state)
    original_request = "\n\n".join(m["content"] for m in planner_messages)
    raw = call_model(vlm, planner_messages)

    try:
        parsed = json.loads(raw)
        validate_required_fields(parsed, builder.schema("planner"))
    except (json.JSONDecodeError, ValueError) as error:
        repair_messages = builder.build_messages(
            "repair",
            original_request,
            invalid_response=raw,
            validation_error=error,
            target_kind="planner",
        )
        raw = call_model(vlm, repair_messages)

    repaired = json.loads(raw)
    validate_required_fields(repaired, builder.schema("planner"))

    assert len(vlm.requests) == 2
    repair_text = "\n".join(m["content"] for m in vlm.requests[1])
    assert invalid_response in repair_text
    assert "missing required fields: action" in repair_text
    assert repaired["action"]["parameters"] == {"x": 545, "y": 340}


def test_reflection_is_fed_back_to_next_planner(
    builder: PromptBuilder, base_state: dict[str, Any]
) -> None:
    """A failed action produces reflection, which is added to the next plan."""

    reflection_result = {
        "failure_type": "repeated_action",
        "summary": "同一按钮被重复点击但界面没有变化",
        "evidence": ["连续两次点击后 OCR 内容相同"],
        "likely_cause": "按钮可能被遮挡或窗口未获得焦点",
        "avoid": ["继续点击同一坐标"],
        "strategy": ["重新识别窗口层级并确认按钮状态"],
        "should_replan": True,
        "confidence": 0.88,
    }
    vlm = RecordingVLM([reflection_result])
    failed_state = {
        **base_state,
        "runtime": {"step_index": 3, "consecutive_failures": 2, "repeated_action_count": 2},
        "history": [
            {"step": 1, "action": "click submit", "result": "no visible change"},
            {"step": 2, "action": "click submit", "result": "no visible change"},
        ],
        "last_execution_result": {"success": True, "message": "click delivered"},
        "last_verification_result": {
            "status": "failure",
            "action_effective": False,
            "reason": "screen unchanged",
        },
    }

    reflection_messages = builder.build_messages("reflection", failed_state)
    reflection = json.loads(call_model(vlm, reflection_messages))
    validate_required_fields(reflection, builder.schema("reflection"))
    assert reflection["should_replan"] is True

    next_state = {
        **failed_state,
        "phase": "planning",
        "history": [
            *failed_state["history"],
            {"type": "reflection", "content": reflection},
        ],
    }
    next_prompt = builder.build_text("planner", next_state)

    assert "重新识别窗口层级并确认按钮状态" in next_prompt
    assert "继续点击同一坐标" in next_prompt


def test_repair_and_reflection_have_different_responsibilities(
    builder: PromptBuilder, base_state: dict[str, Any]
) -> None:
    """Regression test preventing format repair from becoming task replanning."""

    repair = builder.build_text(
        "repair",
        "planner request",
        invalid_response="not-json",
        validation_error="invalid JSON",
    )
    reflection = builder.build_text(
        "reflection",
        base_state,
        failure_context={"error": "same click had no effect twice"},
    )

    assert "not-json" in repair
    assert "invalid JSON" in repair
    assert "same click had no effect twice" in reflection
    assert repair != reflection