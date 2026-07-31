"""
prompts/input_focus
Shared visual-state rules for recognizing a focused search or input box.

Desktop browsers often show a search-history dropdown after a successful focus
action, while OCR may miss the caret. Treat focus as successful when the
search/address box exists together with a spatially attached history dropdown,
or when the box itself shows caret / border / background focus chrome.
"""

from __future__ import annotations

from .config import PromptLanguage


INPUT_FOCUS_SIGNALS_EN: tuple[str, ...] = (
    "The search or address box is visible.",
    "A search-history, suggestion or autocomplete dropdown is visible.",
    "That dropdown sits directly below the box and is horizontally associated with it.",
    "The box shows a caret/cursor, border highlight or background change.",
)

INPUT_FOCUS_SIGNALS_ZH: tuple[str, ...] = (
    "搜索框或地址栏仍然可见。",
    "出现了搜索历史、建议词或自动补全下拉列表。",
    "该下拉列表位于搜索框正下方，并与搜索框水平空间关联。",
    "搜索框内出现光标、边框高亮或背景变化。",
)

INPUT_FOCUS_MIN_SIGNALS: int = 2


def input_focus_rules(language: PromptLanguage) -> tuple[str, ...]:
    """Return bilingual rules that encode the combined focus heuristic."""

    if language is PromptLanguage.ZH:
        intro = (
            "判断搜索框/地址栏是否已获得焦点时，不要只依赖能否看到光标。"
            "判定成功的标准（满足其一即可）："
            "（A）搜索框存在，并且出现搜索历史/建议下拉列表，且下拉列表在搜索框下方、"
            "与搜索框存在空间关联；"
            "（B）搜索框内出现光标、边框高亮或背景变化。"
            "辅助信号包括："
            + "；".join(
                f"（{index}）{signal.rstrip('。')}"
                for index, signal in enumerate(INPUT_FOCUS_SIGNALS_ZH, start=1)
            )
            + "。"
        )
        verify = (
            "若计划动作是点击搜索框/地址栏/输入框，或 hotkey Ctrl+L / Command+L 聚焦地址栏，"
            "动作后观察只要满足上述（A）或（B），必须将 action_effective=true、status=success、"
            "recommended_next=continue；禁止因为缺少单独光标证据而判定失败。"
            "Edge 地址栏聚焦后常见现象就是历史下拉贴在地址栏下方——这就是成功证据。"
        )
        planner = (
            "浏览器内打开网站（如 Google）时，优先用 hotkey Ctrl+L（macOS 用 Command+L）"
            "聚焦地址栏，不要点击地址栏占位文案，也不要点击历史/建议下拉项；"
            "若当前观察已满足焦点成功标准，或 search_progress.phase=input_focused，"
            "下一步必须 paste_text 粘贴 google.com（禁止裸词 google），然后 press enter；"
            "若 search_progress.phase=nav_submitted，应观察是否已到 Google 首页"
            "（logo+中央搜索框），已到则结束；若误入 Bing 结果页则 Ctrl+L 后重贴 google.com；"
            "不要仅为确认焦点而重复 Ctrl+L。"
        )
        reflection = (
            "诊断聚焦是否失败时，必须应用上述（A）/（B）标准；"
            "出现贴在地址栏下方的历史下拉列表是焦点成功证据，不得判为 no_effect，"
            "也不得建议去点击下拉列表中的 OCR 噪声行；"
            "若已 input_focused，下一策略应是 paste_text google.com；"
            "若已在 Bing 结果页，下一策略应是地址栏重开 google.com，而不是点击结果链接。"
        )
        return (intro, verify, planner, reflection)

    intro = (
        "When judging whether a search box or address bar is focused, do not rely on a "
        "visible caret alone. Treat focus as successful when either "
        "(A) the search/address box is present and a history/suggestion dropdown appears "
        "directly below it with horizontal spatial association, or "
        "(B) the box shows a caret/cursor, border highlight or background change. "
        "Supporting signals: "
        + "; ".join(
            f"({index}) {signal}"
            for index, signal in enumerate(INPUT_FOCUS_SIGNALS_EN, start=1)
        )
        + "."
    )
    verify = (
        "If the planned action was clicking a search/address/input field or hotkey "
        "Ctrl+L / Command+L, and the after observation satisfies (A) or (B) above, you "
        "MUST set action_effective=true, status=success and recommended_next=continue. "
        "Never fail solely because a caret is missing. An Edge history dropdown attached "
        "under the address bar is success evidence."
    )
    planner = (
        "When opening a website such as Google, prefer hotkey Ctrl+L "
        "(Command+L on macOS) to focus the address bar — do not click address-bar "
        "placeholder text or history/suggestion dropdown rows; if the current observation "
        "already satisfies the focus success rule, or search_progress.phase=input_focused, "
        "the next step MUST be paste_text google.com (never bare google), then press enter; "
        "if search_progress.phase=nav_submitted, judge whether the Google homepage is open "
        "(logo + central search box) and finish if so; if you are on Bing results, Ctrl+L "
        "and paste google.com again; do not repeat Ctrl+L only to reconfirm focus."
    )
    reflection = (
        "When diagnosing whether address focus failed, apply rules (A)/(B) above; "
        "a history dropdown attached under the bar is evidence of successful focus, not "
        "no_effect, and must not be treated as a click target. If already input_focused, "
        "the next strategy should be paste_text google.com; if already on a Bing results "
        "page, reopen google.com via the address bar instead of clicking result links."
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
