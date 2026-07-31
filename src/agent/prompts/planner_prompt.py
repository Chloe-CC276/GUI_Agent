"""
prompts/planner_prompt
Build the Planner prompt for the GUI Agent.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Mapping, Sequence

from .config import (
    DEFAULT_ALLOWED_ACTIONS,
    PromptConfig,
    PromptFormat,
    PromptKind,
    PromptLanguage,
)
from .context_builder import (
    ContextBuilder,
    chat_progress_context,
    close_progress_context,
    search_progress_context,
)
from .formatters import compact_whitespace, safe_json_dumps, truncate_text
from .input_focus import input_focus_rules_for
from .schemas import PLANNER_RESPONSE_SCHEMA
from .search_workflow import search_workflow_rules
from .templates import PLANNER_TEMPLATE


PLANNER_RULES_EN: tuple[str, ...] = (
    "Return exactly one decision: act, finish, retry, or fail.",
    "When decision is act, select exactly one allowed action and provide only its required parameters.",
    "Base the action on the latest visible OCR, GUI elements, screenshot, and verified history.",
    "Do not invent elements, text, coordinates, application state, or successful outcomes.",
    "For click, double_click, right_click, middle_click, mouse_down and mouse_up, always provide target_text copied from the visible OCR text or GUI label of a detected element.",
    "element_id is optional and must be copied from the detected elements of the CURRENT observation; ids from earlier steps or history are invalid because they are renumbered on every observation.",
    "Never send element_id alone; it may only be sent together with target_text.",
    "Never send x/y alone for a click-like action; the planner resolves the selected element to its own detected center.",
    "Use retry instead of guessing when the requested target text is not present in the detected elements.",
    "Use double_click to open a desktop shortcut, file or folder; use click for buttons, menus, tabs, taskbar icons and focus.",
    "To close a document or window: prefer hotkey Ctrl+W (document) or Alt+F4 (window) once the target window is active. Click a close control only when OCR actually shows ✕ / × / 关闭 / Close in that window's title-bar right strip. Never invent ✕, and never click 关 / 开 / 自动保存, File, Search, or ribbon tabs to close. If the task names left/right and that Word window is inactive, first click its title-bar text to activate it.",
    "To send a ChatGPT message: click the composer placeholder (问问 ChatGPT / Ask anything), then paste_text the exact quoted task message, wait until OCR shows that text in the box, then press enter. Success is a thinking/generating marker (正在思考 / Thinking / 停止生成 / Stop). Never judge composer focus by a caret, and never re-click the composer after it is focused.",
    "Use screenshot pixel coordinates; prefer the center of a detected target when a reliable bounding box exists.",
    "Treat executor success as delivery evidence only; use later visible GUI evidence to confirm effects.",
    "Use finish only when the task success condition is visibly satisfied.",
    "Use retry when the observation is missing, stale, loading, ambiguous, or insufficient for a safe action.",
    "Use fail only for a clear unrecoverable blocker or an impossible/unsafe request.",
    "Avoid repeating an action that produced no visible progress unless there is new evidence that retrying will help.",
    "Keep reason and summaries concise and do not reveal private chain-of-thought.",
)


PLANNER_RULES_ZH: tuple[str, ...] = (
    "每次只返回一个决策：act、finish、retry 或 fail。",
    "decision 为 act 时，只选择一个允许动作，并且只填写该动作需要的参数。",
    "动作必须依据最新可见的 OCR、GUI 元素、截图和已验证历史。",
    "不得虚构界面元素、文本、坐标、应用状态或动作成功结果。",
    "click、double_click、right_click、middle_click、mouse_down 和 mouse_up 必须提供 target_text，取值必须抄自某个已检测元素可见的 OCR 文本或 GUI 标签。",
    "element_id 是可选的，必须抄自「本次观察」的已检测元素；每次观察都会重新编号，禁止沿用历史步骤里的 element_id。",
    "禁止只提交 element_id，它只能与 target_text 一起提交。",
    "click 类动作禁止只提交 x/y；坐标由 Planner 根据所选元素的检测中心自行解析。",
    "已检测元素中找不到所需目标文本时返回 retry，不要猜测。",
    "打开桌面快捷方式、文件或文件夹使用 double_click；按钮、菜单、标签页、任务栏图标和获取焦点使用 click。",
    "关闭文档或窗口时：目标窗口已激活后优先 hotkey Ctrl+W（关文档）或 Alt+F4（关窗口）。仅当 OCR 在标题栏右端真实检出 ✕ / × / 关闭 / Close 时才允许点击关闭控件；禁止虚构 ✕，禁止点「关 / 开 / 自动保存」、文件菜单、搜索或功能区。若任务指明左/右侧且该 Word 未激活，先点该侧标题栏文本激活。",
    "向 ChatGPT 发消息时：先点击输入框占位（问问 ChatGPT / Ask anything），再 paste_text 粘贴任务引号内的原文，OCR 确认输入框出现该文本后按 enter。成功标准是出现思考/生成标志（正在思考 / Thinking / 停止生成 / Stop）。禁止用光标判断焦点，聚焦后禁止再次点击输入框。",
    "坐标使用截图像素坐标；若目标边界框可靠，优先采用目标中心点。",
    "执行器返回成功只代表动作已发送，动作效果仍须由后续可见界面证据确认。",
    "仅在界面证据明确满足任务成功条件时使用 finish。",
    "观察缺失、过期、仍在加载、存在歧义或不足以安全决策时使用 retry。",
    "仅在存在明确且不可恢复的阻碍，或请求无法/不应执行时使用 fail。",
    "没有可见进展时不要重复相同动作，除非出现了支持重试的新证据。",
    "reason 和摘要应简短，不得输出隐藏思维链。",
)


@dataclass(frozen=True, slots=True)
class PlannerPrompt:
    """A Planner prompt split into provider-friendly message fields."""

    system: str
    user: str

    @property
    def text(self) -> str:
        """Return system and user content as one conventional prompt string."""

        return "\n\n".join(part.strip() for part in (self.system, self.user) if part.strip())

    def as_messages(self) -> list[dict[str, str]]:
        """Return OpenAI-style chat messages without provider dependencies."""

        messages: list[dict[str, str]] = []
        if self.system.strip():
            messages.append({"role": "system", "content": self.system.strip()})
        messages.append({"role": "user", "content": self.user.strip()})
        return messages


def _read(value: Any, *names: str, default: Any = None) -> Any:
    if value is None:
        return default
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _coerce_actions(values: Sequence[Any] | None) -> tuple[str, ...]:
    source = values or DEFAULT_ALLOWED_ACTIONS
    actions: list[str] = []
    for value in source:
        raw = value.value if isinstance(value, Enum) else value
        action = str(raw).strip().lower()
        if action and action not in actions:
            actions.append(action)
    if not actions:
        raise ValueError("allowed_actions must contain at least one action")
    return tuple(actions)


def prompt_config_from_planner_config(value: Any) -> PromptConfig:
    """Convert a PlannerConfig-like object into a validated PromptConfig.

    Unknown Planner fields are intentionally ignored because model generation,
    parsing and retry settings do not belong to the prompt composition layer.
    """

    if value is None:
        return PromptConfig()
    if isinstance(value, PromptConfig):
        return value

    supported = {item.name for item in fields(PromptConfig)}
    aliases = {
        "expose_thought": "expose_chain_of_thought",
    }
    data: dict[str, Any] = {}
    for target in supported:
        source = next((key for key, mapped in aliases.items() if mapped == target), target)
        item = _read(value, source)
        if item is not None:
            data[target] = item

    system_prompt = _read(value, "system_prompt")
    if system_prompt:
        data["system_overrides"] = {PromptKind.PLANNER: str(system_prompt)}

    output_mode = _read(value, "output_mode")
    if output_mode is not None:
        raw_mode = output_mode.value if isinstance(output_mode, Enum) else output_mode
        try:
            data["format"] = PromptFormat.coerce(str(raw_mode))
        except ValueError:
            pass

    return PromptConfig(**data)


def build_planner_rules(
    config: PromptConfig | None = None,
    *,
    language: PromptLanguage | str | None = None,
) -> str:
    """Build numbered Planner rules, including relevant config constraints."""

    cfg = config or PromptConfig()
    resolved = cfg.resolve_language(str(language) if language is not None else None)
    rules = list(PLANNER_RULES_ZH if resolved is PromptLanguage.ZH else PLANNER_RULES_EN)
    rules.extend(input_focus_rules_for(resolved, include_planner=True))
    rules.extend(search_workflow_rules(resolved, include_planner=True))

    if cfg.require_reason:
        rules.append("reason 字段不能为空。" if resolved is PromptLanguage.ZH else "The reason field must not be empty.")
    if cfg.require_confidence:
        rules.append(
            "必须返回 0 到 1 之间的 confidence。"
            if resolved is PromptLanguage.ZH
            else "Return confidence as a number between 0 and 1."
        )
    if not cfg.allow_unknown_actions:
        rules.append(
            "不得使用允许动作列表之外的动作。"
            if resolved is PromptLanguage.ZH
            else "Never use an action outside the allowed-action list."
        )
    return "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, start=1))


def _planner_schema(config: PromptConfig) -> dict[str, Any]:
    schema = deepcopy(PLANNER_RESPONSE_SCHEMA)
    action_schema = schema["properties"]["action"]["anyOf"][0]
    action_schema["properties"]["type"]["enum"] = list(config.allowed_actions)

    required = schema.setdefault("required", [])
    if not config.require_confidence and "confidence" in required:
        required.remove("confidence")
    if not config.require_reason and "reason" in required:
        required.remove("reason")
    return schema


def _task_text(state: Any) -> str:
    task = _read(state, "instruction", "user_instruction", "goal")
    if task is None:
        task_object = _read(state, "task")
        task = _read(task_object, "instruction", "task", "goal", default=task_object)
    text = compact_whitespace(task)
    if not text:
        raise ValueError("Planner task instruction must not be empty")
    return text


def _close_stage_block(state: Any, language: PromptLanguage) -> str:
    """Render the activate→close stage as its own prompt section."""

    progress = close_progress_context(state)
    if not progress:
        return ""
    phase = compact_whitespace(progress.get("phase"))
    if not phase:
        return ""

    side = compact_whitespace(progress.get("side"))
    next_action = compact_whitespace(progress.get("next_action"))
    active = progress.get("window_active")
    shortcut = progress.get("shortcut") or []
    keys = "+".join(str(item) for item in shortcut if str(item).strip())

    if language is PromptLanguage.ZH:
        lines = [
            "## 关闭任务阶段（权威信息）",
            f"- 当前阶段：{phase}",
        ]
        if side:
            side_label = "左侧" if side == "left" else "右侧"
            lines.append(f"- 目标区域：{side_label}")
        if isinstance(active, bool):
            lines.append(f"- 目标窗口已激活：{'是' if active else '否'}")
        visible = progress.get("close_control_visible")
        if isinstance(visible, bool):
            lines.append(
                f"- OCR 可见关闭控件：{'是' if visible else '否'}"
            )
        prefer = progress.get("prefer_hotkey")
        if isinstance(prefer, bool):
            lines.append(f"- 优先关闭快捷键：{'是' if prefer else '否'}")
        blocked = progress.get("hotkey_blocked")
        if isinstance(blocked, bool) and blocked:
            lines.append("- 二次关闭快捷键：已禁止（防误关其他文档）")
        attempts = progress.get("close_attempts")
        if isinstance(attempts, int) and attempts > 0:
            lines.append(f"- 已执行关闭次数：{attempts}")
        if next_action:
            lines.append(f"- 下一步必须：{next_action}")
        if keys:
            lines.append(f"- 允许的关闭快捷键（仅激活后）：{keys}")
        lines.append(
            "- 默认优先关闭快捷键；仅当 close_control_visible=true 时才允许点 "
            "✕ / × / 关闭 / Close。禁止虚构关闭控件，禁止点「关 / 开 / 自动保存 / 文件」。"
        )
        return "\n".join(lines)

    lines = [
        "## Close-task stage (authoritative)",
        f"- Current phase: {phase}",
    ]
    if side:
        lines.append(f"- Target side: {side}")
    if isinstance(active, bool):
        lines.append(f"- Target window active: {'yes' if active else 'no'}")
    visible = progress.get("close_control_visible")
    if isinstance(visible, bool):
        lines.append(
            f"- Close control visible in OCR: {'yes' if visible else 'no'}"
        )
    prefer = progress.get("prefer_hotkey")
    if isinstance(prefer, bool):
        lines.append(f"- Prefer close hotkey: {'yes' if prefer else 'no'}")
    blocked = progress.get("hotkey_blocked")
    if isinstance(blocked, bool) and blocked:
        lines.append(
            "- Second close hotkey: blocked (avoid closing another document)"
        )
    attempts = progress.get("close_attempts")
    if isinstance(attempts, int) and attempts > 0:
        lines.append(f"- Close attempts so far: {attempts}")
    if next_action:
        lines.append(f"- Required next: {next_action}")
    if keys:
        lines.append(f"- Allowed close hotkey (only after activation): {keys}")
    lines.append(
        "- Prefer the close hotkey by default; click ✕ / × / 关闭 / Close only "
        "when close_control_visible=true. Never invent a close glyph; never "
        "click 关 / 开 / Autosave / File."
    )
    return "\n".join(lines)


def _search_stage_block(state: Any, language: PromptLanguage) -> str:
    """Render the confirmed search stage as its own prompt section.

    It cannot live inside the JSON context: a large observation can consume the
    whole character budget, and everything after it is truncated away.
    """

    progress = search_progress_context(state)
    if not progress:
        return ""
    phase = compact_whitespace(progress.get("phase"))
    if not phase:
        return ""

    leg = compact_whitespace(progress.get("leg"))
    next_action = compact_whitespace(progress.get("next_action"))
    forbidden = [
        compact_whitespace(item)
        for item in (progress.get("must_not_repeat") or [])
        if compact_whitespace(item)
    ]
    note = compact_whitespace(progress.get("note"))
    step = progress.get("confirmed_at_step")

    if language is PromptLanguage.ZH:
        heading = "## 已确认的搜索阶段（权威信息，优先于你对截图的判断）"
        lines = [heading, f"- 当前阶段：{phase}"]
        if leg:
            leg_label = "地址栏导航" if leg == "navigate" else "首页搜索"
            lines.append(f"- 当前轮次：{leg}（{leg_label}）")
        if isinstance(step, int):
            lines.append(f"- 确认于步骤：{step}")
        if next_action:
            lines.append(f"- 下一步必须：{next_action}")
        if forbidden:
            lines.append("- 禁止重复：" + "；".join(forbidden))
        if note:
            lines.append(f"- 说明：{note}")
        lines.append(
            "- 该阶段由编排层依据动作后观察确认；即使当前截图看不到光标或高亮，"
            "也必须采信它，不要重复已经完成的步骤。"
            "打开 Google 网站时地址栏必须粘贴 google.com（禁止裸词 google）；"
            "同时看到 Google logo 与中央搜索框即任务成功；"
            "Bing/Edge 的 Google 关键词结果页不算成功。"
        )
        return "\n".join(lines)

    heading = "## Confirmed search stage (authoritative, outranks your reading of the screenshot)"
    lines = [heading, f"- Current phase: {phase}"]
    if leg:
        leg_label = (
            "address-bar navigation" if leg == "navigate" else "homepage search"
        )
        lines.append(f"- Current leg: {leg} ({leg_label})")
    if isinstance(step, int):
        lines.append(f"- Confirmed at step: {step}")
    if next_action:
        lines.append(f"- Required next action: {next_action}")
    if forbidden:
        lines.append("- Must not repeat: " + "; ".join(forbidden))
    if note:
        lines.append(f"- Note: {note}")
    lines.append(
        "- The orchestrator confirmed this stage from the after-action observation. "
        "Trust it even when no caret or highlight is visible, and never redo a "
        "step that is already done. To open Google, paste google.com in the address "
        "bar (never bare google). The task succeeds when BOTH the Google logo and "
        "the central Google search box are visible; a Bing/Edge keyword results page "
        "is not success."
    )
    return "\n".join(lines)


def _chat_stage_block(state: Any, language: PromptLanguage) -> str:
    """Render the chat-compose stage as its own prompt section."""

    progress = chat_progress_context(state)
    if not progress:
        return ""
    phase = compact_whitespace(progress.get("phase"))
    if not phase:
        return ""

    message = compact_whitespace(progress.get("message"))
    next_action = compact_whitespace(progress.get("next_action"))
    fingerprint = compact_whitespace(progress.get("fingerprint"))

    if language is PromptLanguage.ZH:
        lines = [
            "## 聊天发送阶段（权威信息）",
            f"- 当前阶段：{phase}",
        ]
        if message:
            lines.append(f"- 必须粘贴的原文：{message}")
        if fingerprint:
            lines.append(f"- OCR 校验指纹：{fingerprint}")
        if next_action:
            lines.append(f"- 下一步必须：{next_action}")
        lines.append(
            "- 禁止用光标判断焦点；聚焦后禁止再点输入框。"
            "paste_text 后须看到输入框出现原文，再按 enter。"
            "出现正在思考 / Thinking / 停止生成 即任务完成。"
        )
        return "\n".join(lines)

    lines = [
        "## Chat-send stage (authoritative)",
        f"- Current phase: {phase}",
    ]
    if message:
        lines.append(f"- Exact message to paste: {message}")
    if fingerprint:
        lines.append(f"- OCR fingerprint: {fingerprint}")
    if next_action:
        lines.append(f"- Required next: {next_action}")
    lines.append(
        "- Never judge composer focus by a caret; do not re-click the composer "
        "after focus. After paste_text, confirm the message text in the box, "
        "then press enter. Thinking / Stop generating markers complete the task."
    )
    return "\n".join(lines)


def _stage_blocks(state: Any, language: PromptLanguage) -> str:
    blocks = [
        block
        for block in (
            _search_stage_block(state, language),
            _close_stage_block(state, language),
            _chat_stage_block(state, language),
        )
        if block
    ]
    return "\n\n".join(blocks)


def _generation_notice(config: PromptConfig, language: PromptLanguage) -> str:
    if not config.add_generation_notice:
        return ""
    output_format = config.response_format_for(PromptKind.PLANNER)
    if output_format is PromptFormat.JSON:
        return (
            "只输出一个符合上述 Schema 的 JSON 对象；不要使用 Markdown 代码块，也不要添加额外说明。"
            if language is PromptLanguage.ZH
            else "Return exactly one JSON object matching the schema above. Do not use Markdown fences or add commentary."
        )
    if output_format is PromptFormat.MARKDOWN:
        return "使用简洁的 Markdown 输出。" if language is PromptLanguage.ZH else "Return concise Markdown."
    return "只输出请求的结果。" if language is PromptLanguage.ZH else "Return only the requested result."


def build_planner_messages(
    state: Any,
    config: PromptConfig | Any | None = None,
    *,
    memory: Any = None,
    task_language: str | None = None,
) -> PlannerPrompt:
    """Build separate system/user Planner messages from AgentState-like input."""

    cfg = prompt_config_from_planner_config(config)
    task = _task_text(state)
    language_hint = task_language or _read(_read(state, "task"), "language")
    language = cfg.resolve_language(language_hint)
    context_builder = ContextBuilder(cfg)
    context = context_builder.agent_json(state, memory=memory)

    rules = build_planner_rules(cfg, language=language)
    allowed_actions = safe_json_dumps(
        list(cfg.allowed_actions), indent=cfg.json_indent, ensure_ascii=cfg.ensure_ascii
    )
    response_schema = (
        safe_json_dumps(
            _planner_schema(cfg),
            indent=cfg.json_indent,
            ensure_ascii=cfg.ensure_ascii,
            sort_keys=cfg.sort_json_keys,
        )
        if cfg.include_schema
        else ("(schema omitted)" if language is PromptLanguage.EN else "（响应结构已省略）")
    )
    values = {
        "rules": rules if cfg.include_rules else ("(rules omitted)" if language is PromptLanguage.EN else "（规则已省略）"),
        "task": task,
        "search_stage": _stage_blocks(state, language),
        "context": context,
        "allowed_actions": allowed_actions,
        "response_schema": response_schema,
        "generation_notice": _generation_notice(cfg, language),
    }

    system = ""
    if cfg.include_system_prompt:
        system = cfg.system_override_for(PromptKind.PLANNER) or PLANNER_TEMPLATE.system(language)
    user = PLANNER_TEMPLATE.render(values, language=language, include_system=False)

    prompt = PlannerPrompt(system=system, user=user)
    if len(prompt.text) > cfg.max_prompt_chars:
        fixed_values = dict(values)
        fixed_values["context"] = ""
        fixed_user = PLANNER_TEMPLATE.render(fixed_values, language=language, include_system=False)
        budget = max(256, cfg.max_prompt_chars - len(system) - len(fixed_user) - 4)
        values["context"] = truncate_text(context, budget)
        user = PLANNER_TEMPLATE.render(values, language=language, include_system=False)
        prompt = PlannerPrompt(system=system, user=user)
        if len(prompt.text) > cfg.max_prompt_chars:
            raise ValueError(
                "max_prompt_chars is too small for the planner rules and response schema"
            )
    return prompt


def build_planner_prompt(
    state: Any,
    config: PromptConfig | Any | None = None,
    *,
    memory: Any = None,
    task_language: str | None = None,
    separate_messages: bool = False,
) -> str | PlannerPrompt:
    """Build the complete Planner prompt or return its separated messages.

    The first two positional parameters deliberately match the existing
    ``Planner.prompt_builder`` callback signature.
    """

    prompt = build_planner_messages(
        state,
        config,
        memory=memory,
        task_language=task_language,
    )
    return prompt if separate_messages else prompt.text


__all__ = [
    "PLANNER_RULES_EN",
    "PLANNER_RULES_ZH",
    "PlannerPrompt",
    "build_planner_messages",
    "build_planner_prompt",
    "build_planner_rules",
    "prompt_config_from_planner_config",
]