"""
prompts/input_focus
Shared visual-state rules for recognizing a focused search or input box.

Desktop browsers often show a search-history dropdown after a successful focus
click, while OCR may miss the caret.  Treat the control as focused when at
least two of the listed visual signals are present together.
"""

from __future__ import annotations

from .config import PromptLanguage


INPUT_FOCUS_SIGNALS_EN: tuple[str, ...] = (
    "The search or input box is still visible at the expected location.",
    "A search-history, suggestion or autocomplete dropdown is visible.",
    "That dropdown sits directly below the box and is visually attached to it.",
    "The box shows a caret/cursor, border highlight or background change.",
)

INPUT_FOCUS_SIGNALS_ZH: tuple[str, ...] = (
    "搜索框或输入框仍出现在预期位置。",
    "出现了搜索历史、建议词或自动补全下拉列表。",
    "该下拉列表位于搜索框正下方，并与搜索框视觉关联。",
    "搜索框内出现光标、边框高亮或背景变化。",
)

INPUT_FOCUS_MIN_SIGNALS: int = 2


def input_focus_rules(language: PromptLanguage) -> tuple[str, ...]:
    """Return bilingual rules that encode the combined focus heuristic."""

    if language is PromptLanguage.ZH:
        signals = INPUT_FOCUS_SIGNALS_ZH
        intro = (
            "判断搜索框、地址栏或文本输入框是否已被选中/获得焦点时，使用组合视觉状态，"
            f"不要只依赖能否看到光标。同时检查下列信号，满足任意 "
            f"{INPUT_FOCUS_MIN_SIGNALS} 项及以上即判定为已选中："
            + "；".join(
                f"（{index}）{signal.rstrip('。')}"
                for index, signal in enumerate(signals, start=1)
            )
            + "。"
        )
        verify = (
            "若计划动作是点击搜索框/地址栏/输入框，且动作后观察按上述规则判定为已选中，"
            "则将 action_effective 设为 true，不得因为缺少单独光标证据而判定失败。"
        )
        planner = (
            "若当前观察已按上述规则判定搜索框/地址栏/输入框已选中，下一步应直接 "
            "type_text 或 paste_text 输入内容，不要仅为确认焦点而重复点击同一控件。"
        )
        reflection = (
            "诊断点击搜索框/输入框是否失败时，必须应用上述组合视觉状态规则；"
            "出现与搜索框关联的历史下拉列表通常是焦点成功证据，不得单独当作失败或未选中。"
        )
        return (intro, verify, planner, reflection)

    signals = INPUT_FOCUS_SIGNALS_EN
    intro = (
        "When judging whether a search box, address bar or text input is focused, "
        "use a combined visual state instead of relying on a visible caret alone. "
        f"Check the following signals and treat the control as focused when at least "
        f"{INPUT_FOCUS_MIN_SIGNALS} are present together: "
        + "; ".join(
            f"({index}) {signal}"
            for index, signal in enumerate(signals, start=1)
        )
        + "."
    )
    verify = (
        "If the planned action was clicking a search box, address bar or input field "
        "and the after observation satisfies the combined focus rule above, set "
        "action_effective=true; do not mark failure solely because a caret is missing."
    )
    planner = (
        "If the current observation already satisfies the combined focus rule for a "
        "search box, address bar or input field, the next step should be type_text or "
        "paste_text; do not re-click the same control only to reconfirm focus."
    )
    reflection = (
        "When diagnosing whether a click on a search or input box failed, apply the "
        "combined focus rule above; a history dropdown attached under the box is usually "
        "evidence of successful focus, not of failure."
    )
    return (intro, verify, planner, reflection)


def input_focus_rules_for(
    language: PromptLanguage,
    *,
    include_verify: bool = False,
    include_planner: bool = False,
    include_reflection: bool = False,
) -> tuple[str, ...]:
    """Select the shared intro plus role-specific follow-up rules."""

    intro, verify, planner, reflection = input_focus_rules(language)
    selected = [intro]
    if include_verify:
        selected.append(verify)
    if include_planner:
        selected.append(planner)
    if include_reflection:
        selected.append(reflection)
    return tuple(selected)


__all__ = [
    "INPUT_FOCUS_MIN_SIGNALS",
    "INPUT_FOCUS_SIGNALS_EN",
    "INPUT_FOCUS_SIGNALS_ZH",
    "input_focus_rules",
    "input_focus_rules_for",
]
