"""
prompts/search_workflow
Browser search recipe (方案 B): focus address bar with Ctrl+L,
paste a navigation query, Enter to the Google homepage, then use the
central search box with paste + Enter. Never click autocomplete junk.
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
        recipe = (
            "浏览器搜索必须按方案 B 顺序执行，每次只做一步："
            "（1）hotkey 使用 Ctrl+L（macOS 用 Command+L）聚焦地址栏，禁止点击"
            "「搜索或输入 Web 地址」等占位文案，也禁止点击地址栏下方的历史/建议下拉项；"
            "（2）paste_text 粘贴导航词 google（或 google.com），禁止 type_text；"
            "（3）press key=enter 进入 Google；"
            "（4）进入首页后点击页面中央搜索框获取焦点；"
            "（5）paste_text 粘贴任务查询词（例如 Google）；"
            "（6）press key=enter 提交搜索。"
        )
        no_dropdown = (
            "地址栏或搜索框下方的历史、建议、自动补全下拉列表只是焦点成功的附带现象，"
            "禁止把下拉列表中的任何一行当作点击目标（尤其是长中文句子、半截说明、"
            "含「输入法」「选词」等字样的 OCR 噪声）；聚焦后应直接 paste_text。"
        )
        paste = (
            "向地址栏或搜索框填入文本时，必须使用 paste_text 通过剪贴板粘贴完整内容，"
            "禁止使用 type_text 逐字输入。"
        )
        submit = (
            "paste_text 成功写入后，下一步必须立即 press（key=enter）提交，"
            "不要再次点击搜索按钮，也不要重复粘贴同一文本。"
        )
        homepage = (
            "判断是否已进入 Google 首页时，同时看到 Google logo（或等价品牌标识）"
            "以及页面中央的搜索输入框，即可认定已成功进入首页；不要因为还没有搜索结果"
            "而继续点击 Google search 或重复打开 Google。"
        )
        planner_home = (
            "若当前界面已满足 Google 首页判定（logo + 中央搜索框），应点击中央搜索框"
            "获取焦点，然后 paste_text 查询词并 press enter；禁止在已处于首页时反复点击"
            "Google search、结果页链接或其它无关 Google 文本。"
        )
        verify_paste = (
            "paste_text 以及用于聚焦地址栏的 hotkey（Ctrl+L / Command+L）由编排层跳过验证；"
            "若仍收到该类动作，只要执行器成功就将 action_effective 设为 true，并推荐 continue。"
        )
        verify_home = (
            "当任务目标包含打开或进入 Google 时，动作后观察若同时出现 Google logo "
            "（或等价品牌标识）与中央搜索框，则认定已成功进入首页："
            "action_effective=true；若整个任务仅要求打开 Google 首页，还可将 task_complete=true。"
        )
        reflection = (
            "诊断浏览器搜索失败时：应回到方案 B（Ctrl+L → paste google → Enter → "
            "点中央搜索框 → paste 查询词 → Enter）；不要建议点击地址栏占位文案或下拉建议行；"
            "已在 Google 首页（logo + 中央搜索框）时不要建议再次点击 Google search；"
            "粘贴后的合理下一步是 press enter。"
        )
        selected: list[str] = []
        if include_planner:
            selected.extend((recipe, no_dropdown, paste, submit, homepage, planner_home))
        if include_verify:
            selected.extend((homepage, verify_paste, verify_home, no_dropdown))
        if include_reflection:
            selected.extend((recipe, no_dropdown, paste, homepage, reflection))
        return tuple(selected)

    recipe = (
        "For browser search follow recipe B one step at a time: "
        "(1) hotkey Ctrl+L (Command+L on macOS) to focus the address bar — never click "
        "placeholder text such as 'Search or enter web address', and never click any "
        "history/suggestion dropdown row under the bar; "
        "(2) paste_text the navigation query google (or google.com), never type_text; "
        "(3) press key=enter to open Google; "
        "(4) on the homepage click the central search box; "
        "(5) paste_text the task query (e.g. Google); "
        "(6) press key=enter to submit the search."
    )
    no_dropdown = (
        "History, suggestion and autocomplete dropdowns under the address bar or search "
        "box are side-effects of successful focus — never click any dropdown row as a "
        "target (especially long Chinese sentences, truncated instructions, or OCR noise "
        "mentioning IME/candidate selection). After focus, go straight to paste_text."
    )
    paste = (
        "When filling the address bar or a search box, always use paste_text to "
        "clipboard-paste the full content. Never use type_text character-by-character."
    )
    submit = (
        "After paste_text successfully inserts text, the next action must be "
        "press with key=enter to submit. Do not click a search button again "
        "and do not paste the same text again."
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
        "paste_text and address-bar focus hotkeys (Ctrl+L / Command+L) are skipped by "
        "the orchestrator; if you still see such an action, set action_effective=true on "
        "executor success and recommend continue."
    )
    verify_home = (
        "When the task involves opening Google, if the after observation shows both the "
        "Google logo (or equivalent brand mark) and the central search box, treat the "
        "homepage as reached: action_effective=true; if the whole task is only to open "
        "Google, task_complete may also be true."
    )
    reflection = (
        "When diagnosing browser-search failures, return to recipe B "
        "(Ctrl+L → paste google → Enter → click central search box → paste query → Enter); "
        "do not advise clicking address-bar placeholders or suggestion rows; once on the "
        "Google homepage (logo + central search box), do not advise clicking Google search "
        "again; after paste, the sensible next step is press enter."
    )
    selected: list[str] = []
    if include_planner:
        selected.extend((recipe, no_dropdown, paste, submit, homepage, planner_home))
    if include_verify:
        selected.extend((homepage, verify_paste, verify_home, no_dropdown))
    if include_reflection:
        selected.extend((recipe, no_dropdown, paste, homepage, reflection))
    return tuple(selected)


__all__ = [
    "search_workflow_rules",
]
