"""Smoke checks for the P0-P2 contract and state-machine fixes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_action_contract() -> None:
    from src.executor.action import Action

    # wait accepts duration as an alias for seconds.
    action = Action.from_dict({"type": "wait", "duration": 2.5})
    assert action.seconds == 2.5, action.seconds

    # Nested planner parameters are flattened instead of dropped.
    action = Action.from_dict(
        {"type": "click", "parameters": {"x": 10, "y": 20}}
    )
    assert action.x == 10 and action.y == 20, (action.x, action.y)

    # action_type alias.
    action = Action.from_dict({"action_type": "wait", "seconds": 1})
    assert action.type.value == "wait"
    print("action contract OK")


def check_texts_match_symbols() -> None:
    from src.common.target_validation import texts_match

    assert texts_match("✕", "✕")
    assert texts_match("×", "✕")
    assert texts_match("✕", "x")
    assert not texts_match("✕", "关闭")
    assert not texts_match("", "")
    assert texts_match("关闭", "关闭")
    print("texts_match symbols OK")


def check_chat_input_target() -> None:
    from src.agent.chat_send import is_chat_input_target

    assert not is_chat_input_target("发送"), "send button must not match"
    assert is_chat_input_target("给ChatGPT发送消息")
    assert is_chat_input_target("+ 问问ChatGPT")
    assert is_chat_input_target("Ask anything")
    assert not is_chat_input_target("gpt")
    print("chat input target OK")


def check_thinking_markers() -> None:
    from src.agent.chat_send import detect_thinking_marker
    from src.agent.state import ObservationState

    # Bare Stop as full label counts; substring inside another word does not.
    obs = ObservationState(ocr_items=[{"text": "Stopwatch", "bbox": [0, 0, 10, 10]}])
    hit, _ = detect_thinking_marker(obs)
    assert not hit, "Stopwatch must not be a thinking marker"

    obs = ObservationState(ocr_items=[{"text": "停止", "bbox": [0, 0, 10, 10]}])
    hit, _ = detect_thinking_marker(obs)
    assert hit, "standalone 停止 is the stop button"

    obs = ObservationState(ocr_items=[{"text": "正在思考...", "bbox": [0, 0, 10, 10]}])
    hit, _ = detect_thinking_marker(obs)
    assert hit
    print("thinking markers OK")


def check_flatten_action_evidence() -> None:
    from src.common.target_validation import flatten_action_evidence

    data = flatten_action_evidence(
        {
            "type": "click",
            "parameters": {"x": 1, "y": 2},
            "metadata": {
                "target_validation": {
                    "target_text": "发送",
                    "matched_bbox": [1, 2, 3, 4],
                    "element_id": 7,
                }
            },
        }
    )
    assert data["target_text"] == "发送"
    assert data["bbox"] == [1, 2, 3, 4]
    assert data["element_id"] == 7
    assert data["x"] == 1
    print("flatten_action_evidence OK")


def check_partial_observation() -> None:
    from src.agent.observation_utils import is_partial_observation
    from src.agent.state import ObservationState

    assert not is_partial_observation(ObservationState())
    assert is_partial_observation(
        ObservationState(metadata={"observation_kind": "win32_windows"})
    )
    assert is_partial_observation(
        ObservationState(metadata={"capture_region": [0, 0, 128, 200]})
    )
    print("partial observation OK")


def check_planner_rules_language() -> None:
    from src.agent.prompts.config import PromptLanguage
    from src.agent.prompts.planner_prompt import build_planner_rules

    zh = build_planner_rules(language=PromptLanguage.ZH)
    en = build_planner_rules(language=PromptLanguage.EN)
    assert zh != en, "ZH enum must not silently fall back to EN"
    assert "不得" in zh or "必须" in zh or "禁止" in zh
    print("planner rules language OK")


def check_schema_target_text() -> None:
    from src.agent.prompts.schemas import PLANNER_RESPONSE_SCHEMA

    params = PLANNER_RESPONSE_SCHEMA["properties"]["action"]["anyOf"][0][
        "properties"
    ]["parameters"]["properties"]
    assert "target_text" in params
    assert "element_id" in params
    print("schema target_text OK")


def check_commit_final() -> None:
    from src.agent.state import AgentState

    state = AgentState(task="test")
    state.runtime.max_steps = 1
    state.begin()
    state.update_planner_result(_dummy_planner_result())
    state.commit_step(final=True)
    assert not state.is_terminal or state.phase.value != "failed", state.phase
    print("commit final OK")


def _dummy_planner_result():
    from src.agent.result import PlannerResult
    from src.executor.action import Action

    return PlannerResult.act(
        Action.from_dict({"type": "wait", "seconds": 0.1}),
        reason="smoke",
        confidence=0.9,
    )


def check_wait_budget() -> None:
    from src.agent import chat_send
    from src.agent.state import AgentState

    state = AgentState(task="给ChatGPT发送“你好”")
    state.metadata[chat_send.CHAT_PROGRESS_KEY] = {
        "phase": chat_send.PHASE_SUBMITTED,
        "message": "你好",
        "wait_attempts": 3,
    }
    forced = chat_send.forced_chat_action(state)
    assert forced is None, "wait budget exhausted must yield control"

    state.metadata[chat_send.CHAT_PROGRESS_KEY]["wait_attempts"] = 0
    forced = chat_send.forced_chat_action(state)
    assert forced is not None and forced[0] == "wait"
    print("wait budget OK")


def main() -> None:
    check_action_contract()
    check_texts_match_symbols()
    check_chat_input_target()
    check_thinking_markers()
    check_flatten_action_evidence()
    check_partial_observation()
    check_planner_rules_language()
    check_schema_target_text()
    check_commit_final()
    check_wait_budget()
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
