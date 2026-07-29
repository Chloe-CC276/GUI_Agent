"""
prompts/search_workflow
Shared rules for browser search: paste query, submit with Enter, and
recognize the Google homepage without redundant clicks.
"""

from __future__ import annotations

from .config import PromptLanguage


def search_workflow_rules(
    language: PromptLanguage,
    *,
    include_verify: bool = False,
    include_planner: bool = False,
    include_reflection: bool = False,
) -> tuple[str, ...]:
    """Return bilingual search-workflow rules for the selected prompt roles."""

    if language is PromptLanguage.ZH:
        paste = (
            "向搜索框、地址栏或输入框填入搜索词时，必须使用 paste_text 通过剪贴板粘贴完整查询词"
            "（例如任务是搜索 GitHub，就 paste_text 粘贴 GitHub），禁止使用 type_text 逐字输入，"
            "也禁止依赖中文输入法候选词选词。"
        )
        submit = (
            "paste_text 成功写入搜索词后，下一步必须立即使用 press（key=enter）提交搜索，"
            "不要再次点击搜索按钮，也不要重复粘贴同一查询词。"
        )
        homepage = (
            "判断是否已进入 Google 首页时，同时看到 Google logo（或等价品牌标识）"
            "以及页面中央的搜索输入框，即可认定已成功进入首页；不要因为还没有搜索结果"
            "而继续点击 Google search 或重复打开 Google。"
        )
        planner_home = (
            "若当前界面已满足 Google 首页判定（logo + 中央搜索框），应点击中央搜索框获取焦点，"
            "然后 paste_text 查询词并 press enter；禁止在已处于首页时反复点击 Google search、"
            "结果页链接或其它无关 Google 文本。"
        )
        verify_paste = (
            "paste_text 写入搜索查询词的步骤由编排层跳过验证；若仍收到该类动作，"
            "只要执行器成功就将 action_effective 设为 true，并推荐 continue。"
        )
        verify_home = (
            "当任务目标包含打开或进入 Google 时，动作后观察若同时出现 Google logo "
            "（或等价品牌标识）与中央搜索框，则认定已成功进入首页："
            "action_effective=true；若整个任务仅要求打开 Google 首页，还可将 task_complete=true。"
        )
        reflection = (
            "诊断浏览器搜索失败时：输入查询应使用 paste_text 而非中文输入法打字；"
            "已在 Google 首页（logo + 中央搜索框）时不要建议再次点击 Google search；"
            "粘贴后的合理下一步是 press enter 提交。"
        )
        selected: list[str] = []
        if include_planner:
            selected.extend((paste, submit, homepage, planner_home))
        if include_verify:
            selected.extend((homepage, verify_paste, verify_home))
        if include_reflection:
            selected.extend((paste, homepage, reflection))
        return tuple(selected)

    paste = (
        "When filling a search box, address bar or input with a query, always use "
        "paste_text to clipboard-paste the full query (for example paste GitHub when "
        "the task is to search GitHub). Never use type_text character-by-character and "
        "never rely on IME candidate selection."
    )
    submit = (
        "After paste_text successfully inserts the query, the next action must be "
        "press with key=enter to submit the search. Do not click a search button again "
        "and do not paste the same query again."
    )
    homepage = (
        "Treat the Google homepage as reached when both the Google logo (or equivalent "
        "brand mark) and the central search input are visible. Do not keep clicking "
        "Google search or reopening Google merely because results are not yet shown."
    )
    planner_home = (
        "If the current screen already satisfies the Google homepage rule (logo + "
        "central search box), click the central search box for focus, then paste_text "
        "the query and press enter. Do not repeatedly click Google search, result links "
        "or unrelated Google text while already on the homepage."
    )
    verify_paste = (
        "paste_text of a search query is skipped by the orchestrator; if you still see "
        "such an action, set action_effective=true on executor success and recommend continue."
    )
    verify_home = (
        "When the task involves opening Google, if the after observation shows both the "
        "Google logo (or equivalent brand mark) and the central search box, treat the "
        "homepage as reached: action_effective=true; if the whole task is only to open "
        "Google, task_complete may also be true."
    )
    reflection = (
        "When diagnosing browser-search failures: queries should use paste_text rather "
        "than IME typing; once on the Google homepage (logo + central search box), do "
        "not advise clicking Google search again; after paste, the sensible next step "
        "is press enter."
    )
    selected = []
    if include_planner:
        selected.extend((paste, submit, homepage, planner_home))
    if include_verify:
        selected.extend((homepage, verify_paste, verify_home))
    if include_reflection:
        selected.extend((paste, homepage, reflection))
    return tuple(selected)


__all__ = [
    "search_workflow_rules",
]
