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
from .context_builder import ContextBuilder
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
    "element_id is optional, must be copied from the detected elements, and may only be sent together with target_text; never send element_id alone.",
    "Never send x/y alone for a click-like action; the planner resolves the selected element to its own detected center.",
    "Use retry instead of guessing when the requested target text is not present in the detected elements.",
    "Use double_click to open a desktop shortcut, file or folder; use click for buttons, menus, tabs, taskbar icons and focus.",
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
    "element_id 是可选的，必须抄自已检测元素，且只能与 target_text 一起提交；禁止只提交 element_id。",
    "click 类动作禁止只提交 x/y；坐标由 Planner 根据所选元素的检测中心自行解析。",
    "已检测元素中找不到所需目标文本时返回 retry，不要猜测。",
    "打开桌面快捷方式、文件或文件夹使用 double_click；按钮、菜单、标签页、任务栏图标和获取焦点使用 click。",
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