"""Optimized Planner rules (Week-4 failure modes). Stock rules stay untouched."""

from __future__ import annotations

OPTIMIZED_RULES_ZH: tuple[str, ...] = (
    "结合任务语义、控件类型、空间位置与当前窗口选择目标；禁止仅凭单个相同字符匹配控件。",
    "状态文字中的「关」不等于关闭按钮；关闭文档时不得把「自动保存：关」当作关闭入口。",
    "存在多个相同文案时，必须结合窗口、区域与控件类型消歧；证据不足时返回 retry，不得猜测坐标。",
    "地址栏操作优先使用 hotkey（Ctrl+L）；关闭当前文档优先 hotkey（Ctrl+W）或 close_window，"
    "若点击关闭则只能点标题栏右侧关闭控件。",
    "打开桌面文件/快捷方式使用 double_click；文本输入优先 paste_text（若允许）或一次完整 type_text，"
    "避免依赖输入法逐字选词。",
    "页面或应用仍在加载时使用 wait 或 retry，不得连续重复相同点击。",
    "每次只输出一个原子动作；任务完成证据已可见时必须 finish，不得继续操作。",
)

OPTIMIZED_RULES_EN: tuple[str, ...] = (
    "Choose targets using task semantics, control type, spatial region and active window; "
    "never match a control using only a single shared character.",
    "Status text such as Off/关 is not a close button; never treat AutoSave: Off as the document close affordance.",
    "When identical labels exist, disambiguate by window, region and control type; if evidence is weak, retry instead of guessing coordinates.",
    "Prefer hotkey Ctrl+L for the address bar; prefer hotkey Ctrl+W or close_window to close the current document; "
    "if clicking close, only the title-bar close control is allowed.",
    "Open desktop files/shortcuts with double_click; prefer paste_text (when allowed) or one complete type_text; "
    "do not rely on IME candidate picking.",
    "While a page or app is loading, wait or retry; do not repeat the same click.",
    "Emit exactly one atomic action; finish as soon as success evidence is visible.",
)


def optimized_rules(language: str = "zh") -> tuple[str, ...]:
    lang = (language or "zh").lower()
    if lang.startswith("en"):
        return OPTIMIZED_RULES_EN
    return OPTIMIZED_RULES_ZH
