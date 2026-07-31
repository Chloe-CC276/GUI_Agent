"""Tests for the persistent browser-search / website-open stage."""

from __future__ import annotations

from types import SimpleNamespace

from src.agent.browser_search import (
    LEG_NAVIGATE,
    PHASE_HOMEPAGE_REACHED,
    PHASE_INPUT_FOCUSED,
    PHASE_NAV_SUBMITTED,
    PHASE_QUERY_ENTERED,
    apply_homepage_verify_override,
    detect_google_homepage,
    looks_like_bing_results,
    record_phase_for_executed_action,
    record_search_phase,
    search_progress,
)
from src.agent.prompts.config import PromptLanguage
from src.agent.prompts.context_builder import build_agent_context
from src.agent.prompts.planner_prompt import build_planner_messages
from src.agent.prompts.search_workflow import search_workflow_rules


def make_state() -> SimpleNamespace:
    return SimpleNamespace(metadata={}, step_index=3)


def make_observation(count: int) -> SimpleNamespace:
    elements = [
        SimpleNamespace(
            element_id=f"el_{index}",
            text=f"一些界面文字 element {index}",
            bbox=[10 * index, 20, 10 * index + 180, 60],
            confidence=0.93,
            source="ocr",
            element_type="text",
        )
        for index in range(count)
    ]
    return SimpleNamespace(
        screen_width=1920,
        screen_height=1080,
        gui_elements=elements,
        ocr_items=elements,
        metadata={},
    )


def make_google_homepage() -> SimpleNamespace:
    elements = [
        SimpleNamespace(
            element_id="logo",
            text="Google",
            bbox=[860, 220, 1060, 300],
            confidence=0.99,
            source="ocr",
            element_type="text",
        ),
        SimpleNamespace(
            element_id="search",
            text="",
            bbox=[560, 360, 1360, 420],
            confidence=0.95,
            source="ui",
            element_type="edit",
        ),
    ]
    return SimpleNamespace(
        screen_width=1920,
        screen_height=1080,
        gui_elements=elements,
        ocr_items=elements,
        metadata={},
    )


def make_bing_serp() -> SimpleNamespace:
    elements = [
        SimpleNamespace(
            element_id="bing",
            text="Microsoft Bing",
            bbox=[40, 20, 220, 60],
            confidence=0.99,
            source="ocr",
            element_type="text",
        ),
        SimpleNamespace(
            element_id="q",
            text="Google",
            bbox=[400, 30, 900, 70],
            confidence=0.95,
            source="ocr",
            element_type="text",
        ),
        SimpleNamespace(
            element_id="result",
            text="Google",
            bbox=[200, 200, 400, 240],
            confidence=0.95,
            source="ocr",
            element_type="text",
        ),
    ]
    return SimpleNamespace(
        screen_width=1920,
        screen_height=1080,
        gui_elements=elements,
        ocr_items=elements,
        metadata={},
    )


def test_focus_phase_requires_google_com() -> None:
    state = make_state()
    record_search_phase(state, PHASE_INPUT_FOCUSED, evidence=["dropdown below box"])

    progress = search_progress(state)
    assert progress is not None
    assert progress["phase"] == PHASE_INPUT_FOCUSED
    assert progress["leg"] == LEG_NAVIGATE
    assert "google.com" in progress["next_action"]
    assert any("bare google" in item for item in progress["must_not_repeat"])


def test_navigate_leg_paste_enter_reaches_nav_submitted() -> None:
    state = make_state()
    record_search_phase(state, PHASE_INPUT_FOCUSED)

    record_phase_for_executed_action(state, {"type": "paste_text", "text": "google.com"})
    assert search_progress(state)["phase"] == PHASE_QUERY_ENTERED

    record_phase_for_executed_action(state, {"type": "press", "key": "enter"})
    progress = search_progress(state)
    assert progress["phase"] == PHASE_NAV_SUBMITTED
    assert "Google logo" in progress["next_action"]
    assert any("Bing" in item for item in progress["must_not_repeat"])


def test_paste_after_nav_submitted_does_not_rewind() -> None:
    state = make_state()
    record_search_phase(state, PHASE_INPUT_FOCUSED)
    record_phase_for_executed_action(state, {"type": "paste_text", "text": "google.com"})
    record_phase_for_executed_action(state, {"type": "press", "key": "enter"})

    record_phase_for_executed_action(state, {"type": "paste_text", "text": "google"})
    assert search_progress(state)["phase"] == PHASE_NAV_SUBMITTED


def test_google_homepage_completes_the_task() -> None:
    state = make_state()
    record_search_phase(state, PHASE_NAV_SUBMITTED, leg=LEG_NAVIGATE)
    observation = make_google_homepage()

    ok, evidence = detect_google_homepage(observation)
    assert ok is True
    assert evidence

    data, overridden = apply_homepage_verify_override(
        state,
        {
            "status": "success",
            "action_effective": True,
            "task_complete": False,
            "confidence": 0.5,
            "recommended_next": "continue",
            "reason": "page changed",
            "evidence": [],
        },
        after=observation,
    )
    assert overridden is True
    assert data["task_complete"] is True
    assert data["recommended_next"] == "finish"
    assert search_progress(state)["phase"] == PHASE_HOMEPAGE_REACHED


def test_bing_serp_is_not_google_homepage() -> None:
    observation = make_bing_serp()
    assert looks_like_bing_results(observation) is True
    ok, evidence = detect_google_homepage(observation)
    assert ok is False
    assert any("Bing" in item for item in evidence)

    state = make_state()
    data, overridden = apply_homepage_verify_override(
        state,
        {
            "status": "success",
            "action_effective": True,
            "task_complete": True,
            "confidence": 0.9,
            "recommended_next": "finish",
            "reason": "saw Google text",
            "evidence": [],
        },
        after=observation,
    )
    assert overridden is True
    assert data["task_complete"] is False


def test_focus_is_not_downgraded_by_a_repeated_ctrl_l() -> None:
    state = make_state()
    record_search_phase(state, PHASE_INPUT_FOCUSED)
    record_phase_for_executed_action(state, {"type": "paste_text", "text": "google.com"})

    record_phase_for_executed_action(state, {"type": "hotkey", "keys": ["ctrl", "l"]})
    assert search_progress(state)["phase"] == PHASE_QUERY_ENTERED

    record_search_phase(state, PHASE_INPUT_FOCUSED)
    assert search_progress(state)["phase"] == PHASE_QUERY_ENTERED


def test_unrelated_click_drops_the_pending_focus() -> None:
    state = make_state()
    record_search_phase(state, PHASE_INPUT_FOCUSED)

    record_phase_for_executed_action(state, {"type": "click", "x": 10, "y": 10})
    assert search_progress(state) is None


def test_planner_context_exposes_the_stage() -> None:
    state = make_state()
    record_search_phase(state, PHASE_INPUT_FOCUSED)

    context = build_agent_context(state)
    assert context["search_progress"]["phase"] == PHASE_INPUT_FOCUSED
    assert list(context).index("search_progress") < list(context).index("observation")


def test_stage_section_mentions_google_com() -> None:
    state = SimpleNamespace(
        metadata={},
        step_index=1,
        observation=make_observation(30),
        previous_observation=make_observation(30),
        history=[],
        task=SimpleNamespace(instruction="在浏览器搜索Google", language="zh"),
        runtime=SimpleNamespace(step_index=1, max_steps=20),
    )
    record_search_phase(state, PHASE_INPUT_FOCUSED)

    text = build_planner_messages(state).text
    assert "已确认的搜索阶段" in text
    assert "google.com" in text


def test_prompt_rules_require_google_com_and_homepage_success() -> None:
    rules = "\n".join(search_workflow_rules(PromptLanguage.ZH, include_planner=True))
    assert "google.com" in rules
    assert "裸词 google" in rules or "禁止粘贴裸词" in rules
    assert "logo" in rules.lower() or "logo" in rules
    assert "Bing" in rules or "bing" in rules
    # Old two-leg / bare-google recipe must be gone.
    assert "粘贴导航词 google（或 google.com）" not in rules
    assert "点中央搜索框 → paste 查询词" not in rules
