"""
prompts/search_workflow
Open a website from the address bar (Google example):
Ctrl+L → paste google.com → Enter → re-observe → Google logo + search box
⇒ task complete. Never paste bare "google" (that becomes Bing/Edge search).
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
            "当任务是打开/搜索网站（例如「在浏览器搜索 Google」「打开 Google」）时，"
            "按下列顺序每次只做一步："
            "（1）hotkey Ctrl+L（macOS 用 Command+L）聚焦地址栏，禁止点击"
            "「搜索或输入 Web 地址」等占位文案，也禁止点击地址栏下方的历史/建议下拉项；"
            "（2）paste_text 必须粘贴完整网址 google.com，禁止粘贴裸词 google/Google"
            "（裸词会变成 Bing/Edge 关键词搜索结果页）；禁止 type_text；"
            "（3）press key=enter 导航；"
            "（4）动作后重新观察并验证：同时看到 Google logo 与 Google 中央搜索框，"
            "即任务成功，应结束，不要再粘贴查询词，也不要点击搜索结果链接。"
        )
        no_dropdown = (
            "地址栏下方的历史、建议、自动补全下拉列表只是焦点成功的附带现象，"
            "禁止把下拉列表中的任何一行当作点击目标（尤其是长中文句子、半截说明、"
            "含「输入法」「选词」等字样的 OCR 噪声）；聚焦后应直接 paste_text google.com。"
        )
        paste = (
            "向地址栏填入网站时，必须使用 paste_text 粘贴 google.com；"
            "禁止 type_text；禁止只粘贴 google。"
        )
        submit = (
            "仅当 search_progress.phase=query_entered（已粘贴 google.com）时，"
            "下一步必须 press（key=enter）。"
            "若 phase=nav_submitted，应等待观察首页，不要再次 paste/enter；"
            "若误入 Bing/Edge 结果页，应 Ctrl+L 后重新粘贴 google.com，"
            "禁止点击标题为 Google 的搜索结果链接冒充进入官网。"
        )
        homepage = (
            "Google 首页成功标准：同时看到 Google logo（或等价品牌标识）"
            "以及页面中央的 Google 搜索输入框。"
            "Bing/Edge/其它搜索引擎的「Google」关键词结果页不算成功"
            "（常见地址含 bing.com/search?q=Google）。"
        )
        planner_home = (
            "若动作后已满足 Google 首页判定（logo + 中央搜索框），应直接 finish/"
            "判定任务完成，禁止再 paste、禁止再 enter、禁止点击结果链接。"
            "若页面是 Bing/Edge 结果，下一步只能 Ctrl+L → paste_text google.com → enter。"
        )
        verify_paste = (
            "paste_text 写入步骤由编排层跳过验证；若仍收到该类动作，"
            "只要执行器成功就将 action_effective 设为 true，并推荐 continue。"
            "Ctrl+L 聚焦后必须依据动作后观察判定：地址栏存在且下方出现历史/建议下拉"
            "（空间关联），或框内高亮/光标，即视为聚焦成功。"
        )
        verify_home = (
            "当地址栏 Enter 导航之后（或任意动作后观察），若同时出现 Google logo "
            "与中央 Google 搜索框，则认定已打开 Google 官网："
            "action_effective=true，task_complete=true，recommended_next=finish。"
            "若页面像 Bing/Edge 搜索结果，则 task_complete 必须为 false，"
            "action_effective 至多表示「提交了导航/搜索」，推荐 continue 以便改用 google.com。"
        )
        reflection = (
            "诊断失败时：应回到 Ctrl+L → paste google.com → Enter → "
            "确认 logo+中央搜索框；若已在 Bing 结果页，不要建议点击 Google 结果链接，"
            "应建议重新用地址栏打开 google.com。"
        )
        progress = (
            "上下文中的 search_progress 是已确认阶段，具有最高优先级，并遵守 must_not_repeat："
            "phase=input_focused → 只能 paste_text google.com（禁止裸词 google）；"
            "phase=query_entered → 只能 press enter；"
            "phase=nav_submitted → 观察是否已到 Google 首页，禁止把 Bing 结果当成功；"
            "phase=homepage_reached → 任务已完成，应结束。"
        )
        selected: list[str] = []
        if include_planner:
            selected.extend(
                (progress, recipe, no_dropdown, paste, submit, homepage, planner_home)
            )
        if include_verify:
            selected.extend((homepage, verify_paste, verify_home, no_dropdown))
        if include_reflection:
            selected.extend((progress, recipe, no_dropdown, paste, homepage, reflection))
        return tuple(selected)

    recipe = (
        "When the task is to open/search a website (e.g. 'search Google in the browser' "
        "or 'open Google'), follow this one-step-at-a-time recipe: "
        "(1) hotkey Ctrl+L (Command+L on macOS) to focus the address bar — never click "
        "placeholder text or history/suggestion rows; "
        "(2) paste_text exactly google.com — never paste bare google/Google "
        "(that becomes a Bing/Edge keyword results page); never type_text; "
        "(3) press key=enter to navigate; "
        "(4) re-observe and verify: when BOTH the Google logo and the central Google "
        "search box are visible, the task is complete — finish. Do not paste another "
        "query and do not click search-result links."
    )
    no_dropdown = (
        "History/suggestion dropdowns under the address bar are side-effects of focus — "
        "never click any dropdown row. After focus, paste_text google.com immediately."
    )
    paste = (
        "When filling the address bar to open Google, always paste_text google.com. "
        "Never type_text. Never paste bare google."
    )
    submit = (
        "Only when search_progress.phase=query_entered (google.com already pasted) "
        "must the next action be press key=enter. "
        "If phase=nav_submitted, wait to observe the homepage — do not paste/enter again. "
        "If you landed on Bing/Edge results, Ctrl+L and paste google.com again; "
        "never click a result link titled Google as a substitute for google.com."
    )
    homepage = (
        "Google homepage success requires BOTH the Google logo (or equivalent brand mark) "
        "AND the central Google search input. A Bing/Edge keyword results page for "
        "'Google' (e.g. bing.com/search?q=Google) is NOT success."
    )
    planner_home = (
        "If the after observation already satisfies the Google homepage rule "
        "(logo + central search box), finish the task. Do not paste, do not press enter, "
        "and do not click result links. If the page is Bing/Edge results, the only recovery "
        "is Ctrl+L → paste_text google.com → enter."
    )
    verify_paste = (
        "paste_text steps are skipped by the orchestrator; if you still see such an action, "
        "set action_effective=true on executor success and recommend continue. "
        "After Ctrl+L, judge focus from the after observation: success when the address bar "
        "exists with a history/suggestion dropdown spatially below it, or caret/highlight."
    )
    verify_home = (
        "After address-bar Enter (or on any after observation), if BOTH the Google logo "
        "and the central Google search box are visible, set action_effective=true, "
        "task_complete=true and recommended_next=finish. "
        "If the page looks like Bing/Edge search results, task_complete must be false; "
        "at most mark the submit as effective and recommend continue so the agent can "
        "retry with google.com."
    )
    reflection = (
        "When diagnosing failures, return to Ctrl+L → paste google.com → Enter → "
        "confirm logo + central search box. If already on a Bing results page, do not "
        "advise clicking the Google result link — advise reopening google.com via the "
        "address bar."
    )
    progress = (
        "The context field search_progress is authoritative; obey must_not_repeat: "
        "phase=input_focused → only paste_text google.com (never bare google); "
        "phase=query_entered → only press enter; "
        "phase=nav_submitted → judge whether the Google homepage is open; "
        "never treat Bing results as success; "
        "phase=homepage_reached → the task is complete; finish."
    )
    selected: list[str] = []
    if include_planner:
        selected.extend(
            (progress, recipe, no_dropdown, paste, submit, homepage, planner_home)
        )
    if include_verify:
        selected.extend((homepage, verify_paste, verify_home, no_dropdown))
    if include_reflection:
        selected.extend((progress, recipe, no_dropdown, paste, homepage, reflection))
    return tuple(selected)


__all__ = [
    "search_workflow_rules",
]
