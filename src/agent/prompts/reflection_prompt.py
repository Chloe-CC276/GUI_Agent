"""
prompt/reflection_prompt
Build evidence-based reflection prompts for stalled GUI Agent runs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .config import PromptConfig, PromptFormat, PromptKind, PromptLanguage
from .context_builder import ContextBuilder, compact_history
from .formatters import compact_whitespace, format_error, safe_json_dumps, truncate_text
from .input_focus import input_focus_rules_for
from .planner_prompt import prompt_config_from_planner_config
from .schemas import REFLECTION_RESPONSE_SCHEMA
from .templates import REFLECTION_TEMPLATE


REFLECTION_RULES_EN: tuple[str, ...] = (
    "Diagnose the trajectory from observed outcomes, not from the original intention alone.",
    "Separate directly observed facts from hypotheses; include only observed facts in evidence.",
    "Use failure_type=none when progress is normal and no failure or stall is supported.",
    "Identify the single most likely immediate cause while acknowledging uncertainty through confidence.",
    "Treat repeated actions without visible progress as a signal to change strategy, target or observation method.",
    "Executor success proves only that input was delivered; it does not prove the intended GUI effect.",
    "Do not invent hidden UI state, unavailable screenshots, elements, coordinates, errors or task results.",
    "Strategy entries must be high-level planning adjustments, not executable GUI actions or coordinates.",
    "List specific assumptions or ineffective behaviours that the next planning cycle should avoid.",
    "Set should_replan=true when the current approach should change; use false only when continuation is justified by evidence.",
    "Keep the response concise and do not reveal private chain-of-thought.",
)


REFLECTION_RULES_ZH: tuple[str, ...] = (
    "必须依据实际观察结果诊断执行轨迹，不能只依据原始意图判断。",
    "区分直接观察事实与推测；evidence 中只能填写已观察到的事实。",
    "任务正常推进且没有失败或停滞证据时，使用 failure_type=none。",
    "识别一个最可能的直接原因，并通过 confidence 表达不确定程度。",
    "相同动作重复执行却没有可见进展时，应调整策略、目标或观察方式。",
    "执行器成功只表示输入已发送，不能证明预期 GUI 效果已经发生。",
    "不得虚构隐藏界面状态、不可用截图、元素、坐标、错误或任务结果。",
    "strategy 只能给出高层规划调整，不得直接输出可执行 GUI 动作或坐标。",
    "avoid 应明确列出下一轮规划需要避免的假设或无效行为。",
    "当前方法需要改变时将 should_replan 设为 true；只有证据支持继续时才设为 false。",
    "输出应简洁，不得暴露隐藏思维链。",
)


@dataclass(frozen=True, slots=True)
class ReflectionPrompt:
    """A reflection prompt split into provider-friendly message fields."""

    system: str
    user: str

    @property
    def text(self) -> str:
        """Return system and user content as one prompt string."""

        return "\n\n".join(
            part.strip() for part in (self.system, self.user) if part.strip()
        )

    def as_messages(self) -> list[dict[str, str]]:
        """Return OpenAI-style messages without importing a provider SDK."""

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


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _task_text(state: Any) -> str:
    task = _read(state, "instruction", "user_instruction", "goal")
    if task is None:
        task_object = _read(state, "task")
        task = _read(task_object, "instruction", "task", "goal", default=task_object)
    text = compact_whitespace(task)
    if not text:
        raise ValueError("Reflection task instruction must not be empty")
    return text


def _json(value: Any, config: PromptConfig, *, empty_label: str) -> str:
    if value is None or value == "" or value == {} or value == []:
        return empty_label
    return safe_json_dumps(
        value,
        indent=config.json_indent,
        ensure_ascii=config.ensure_ascii,
        sort_keys=config.sort_json_keys,
    )


def _result_summary(result: Any, config: PromptConfig) -> dict[str, Any] | None:
    if result is None:
        return None
    data = {
        "status": _enum_value(_read(result, "status")),
        "decision": _enum_value(_read(result, "decision")),
        "action": _read(result, "action"),
        "success": _read(result, "success", "succeeded"),
        "message": _read(result, "message", "reason", "finish_message"),
        "confidence": _read(result, "confidence"),
    }
    error = _read(result, "error")
    if error is not None:
        data["error"] = format_error(error, max_chars=config.max_error_chars)
    cleaned = {key: value for key, value in data.items() if value is not None and value != ""}
    return cleaned or None


def _infer_failure_context(state: Any, config: PromptConfig) -> dict[str, Any]:
    """Collect explicit runtime failure signals without inferring a diagnosis."""

    runtime = _read(state, "runtime")
    current_step = _read(state, "current_step")
    context: dict[str, Any] = {
        "phase": _enum_value(_read(state, "phase")),
        "step_index": _read(runtime, "step_index", default=_read(state, "step_index")),
        "retry_count": _read(runtime, "retry_count"),
        "max_retries": _read(runtime, "max_retries"),
        "consecutive_failures": _read(runtime, "consecutive_failures"),
        "repeated_action_count": _read(runtime, "repeated_action_count"),
        "last_planner_result": _result_summary(
            _read(state, "last_planner_result", default=_read(current_step, "planner_result")),
            config,
        ),
        "last_execution_result": _result_summary(
            _read(state, "last_execution_result", default=_read(current_step, "execution_result")),
            config,
        ),
        "last_verification_result": _result_summary(
            _read(state, "last_verification_result", default=_read(current_step, "verification_result")),
            config,
        ),
    }
    error = _read(state, "error", "last_error")
    if error is not None:
        context["state_error"] = format_error(error, max_chars=config.max_error_chars)
    metadata = _read(state, "metadata", default={})
    for name in ("failure_reason", "stall_reason", "reflection_trigger"):
        value = _read(state, name, default=_read(metadata, name))
        if value is not None and value != "":
            context[name] = value
    return {key: value for key, value in context.items() if value is not None and value != ""}


def _reflection_schema(config: PromptConfig) -> dict[str, Any]:
    schema = deepcopy(REFLECTION_RESPONSE_SCHEMA)
    required = schema.setdefault("required", [])
    if not config.require_confidence and "confidence" in required:
        required.remove("confidence")
    return schema


def build_reflection_rules(
    config: PromptConfig | None = None,
    *,
    language: PromptLanguage | str | None = None,
) -> str:
    """Build numbered, language-aware reflection rules."""

    cfg = config or PromptConfig()
    hint = language.value if isinstance(language, PromptLanguage) else language
    resolved = cfg.resolve_language(str(hint) if hint is not None else None)
    rules = list(
        REFLECTION_RULES_ZH if resolved is PromptLanguage.ZH else REFLECTION_RULES_EN
    )
    rules.extend(input_focus_rules_for(resolved, include_reflection=True))
    if cfg.require_confidence:
        rules.append(
            "必须返回 0 到 1 之间的 confidence。"
            if resolved is PromptLanguage.ZH
            else "Return confidence as a number between 0 and 1."
        )
    if cfg.strict_schema:
        rules.append(
            "不得返回响应 Schema 未定义的字段。"
            if resolved is PromptLanguage.ZH
            else "Do not return fields that are not defined by the response schema."
        )
    return "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, start=1))


def _generation_notice(config: PromptConfig, language: PromptLanguage) -> str:
    if not config.add_generation_notice:
        return ""
    output_format = config.response_format_for(PromptKind.REFLECTION)
    if output_format is PromptFormat.JSON:
        return (
            "只输出一个符合上述 Schema 的 JSON 对象；不要使用 Markdown 代码块，也不要添加解释。"
            if language is PromptLanguage.ZH
            else "Return exactly one JSON object matching the schema above. Do not use Markdown fences or add commentary."
        )
    if output_format is PromptFormat.MARKDOWN:
        return "使用简洁的 Markdown 输出。" if language is PromptLanguage.ZH else "Return concise Markdown."
    return "只输出请求的反思结果。" if language is PromptLanguage.ZH else "Return only the requested reflection."


def build_reflection_messages(
    state: Any,
    config: PromptConfig | Any | None = None,
    *,
    history: Any = None,
    failure_context: Any = None,
    memory: Any = None,
    task_language: str | None = None,
) -> ReflectionPrompt:
    """Build separate system/user reflection messages from AgentState-like input."""

    cfg = prompt_config_from_planner_config(config)
    task = _task_text(state)
    language_hint = task_language or _read(_read(state, "task"), "language")
    language = cfg.resolve_language(language_hint)
    builder = ContextBuilder(cfg)

    current_context = builder.agent(state, memory=memory)
    # History is supplied in its own section, so avoid duplicating it.
    current_context.pop("history", None)
    raw_history = history if history is not None else _read(state, "history", "completed_steps")
    history_text = compact_history(raw_history, config=cfg)
    failure = failure_context if failure_context is not None else _infer_failure_context(state, cfg)

    none_label = "（未提供）" if language is PromptLanguage.ZH else "(not provided)"
    rules_omitted = "（规则已省略）" if language is PromptLanguage.ZH else "(rules omitted)"
    schema_omitted = "（响应结构已省略）" if language is PromptLanguage.ZH else "(schema omitted)"
    values = {
        "rules": build_reflection_rules(cfg, language=language) if cfg.include_rules else rules_omitted,
        "task": task,
        "current_context": _json(current_context, cfg, empty_label=none_label),
        "history": history_text or none_label,
        "failure_context": _json(failure, cfg, empty_label=none_label),
        "response_schema": (
            _json(_reflection_schema(cfg), cfg, empty_label="null")
            if cfg.include_schema
            else schema_omitted
        ),
        "generation_notice": _generation_notice(cfg, language),
    }

    system = ""
    if cfg.include_system_prompt:
        system = cfg.system_override_for(PromptKind.REFLECTION) or REFLECTION_TEMPLATE.system(language)
    prompt = ReflectionPrompt(
        system=system,
        user=REFLECTION_TEMPLATE.render(values, language=language, include_system=False),
    )

    if len(prompt.text) > cfg.max_prompt_chars:
        fixed = dict(values)
        fixed["current_context"] = ""
        fixed["history"] = ""
        fixed["failure_context"] = ""
        fixed_user = REFLECTION_TEMPLATE.render(fixed, language=language, include_system=False)
        budget = cfg.max_prompt_chars - len(system) - len(fixed_user) - 4
        if budget < 768:
            raise ValueError("max_prompt_chars is too small for reflection rules and response schema")
        failure_budget = max(256, budget // 4)
        history_budget = max(256, budget // 3)
        context_budget = budget - failure_budget - history_budget
        values["failure_context"] = truncate_text(values["failure_context"], failure_budget)
        values["history"] = truncate_text(values["history"], history_budget)
        values["current_context"] = truncate_text(values["current_context"], context_budget)
        prompt = ReflectionPrompt(
            system=system,
            user=REFLECTION_TEMPLATE.render(values, language=language, include_system=False),
        )
        if len(prompt.text) > cfg.max_prompt_chars:
            raise ValueError("max_prompt_chars is too small for the reflection prompt")
    return prompt


def build_reflection_prompt(
    state: Any,
    config: PromptConfig | Any | None = None,
    *,
    history: Any = None,
    failure_context: Any = None,
    memory: Any = None,
    task_language: str | None = None,
    separate_messages: bool = False,
) -> str | ReflectionPrompt:
    """Build a complete reflection prompt or return separated message fields."""

    prompt = build_reflection_messages(
        state,
        config,
        history=history,
        failure_context=failure_context,
        memory=memory,
        task_language=task_language,
    )
    return prompt if separate_messages else prompt.text


__all__ = [
    "REFLECTION_RULES_EN",
    "REFLECTION_RULES_ZH",
    "ReflectionPrompt",
    "build_reflection_messages",
    "build_reflection_prompt",
    "build_reflection_rules",
]