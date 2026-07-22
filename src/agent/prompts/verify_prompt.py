"""
prompts/verify_prompt
Build evidence-based verification prompts for the GUI Agent.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .config import PromptConfig, PromptFormat, PromptKind, PromptLanguage
from .context_builder import ContextBuilder
from .formatters import compact_whitespace, format_error, safe_json_dumps, truncate_text
from .planner_prompt import prompt_config_from_planner_config
from .schemas import VERIFY_RESPONSE_SCHEMA
from .templates import VERIFY_TEMPLATE


VERIFY_RULES_EN: tuple[str, ...] = (
    "Compare the before and after observations and identify only visible, evidence-supported changes.",
    "Executor success means the input was delivered; it does not prove that the intended GUI effect occurred.",
    "Judge the planned action's immediate effect separately from completion of the whole task.",
    "Set action_effective to true only when the after observation contains evidence consistent with the intended effect.",
    "Set task_complete to true only when the task success criteria are visibly satisfied after the action.",
    "Use status=success when the intended effect is supported, failure when contrary evidence or a clear error exists, and uncertain when evidence is missing or ambiguous.",
    "Use only facts present in the supplied observations and executor result; do not invent hidden UI state, elements, text, or outcomes.",
    "A missing visual difference may be valid for actions such as move_to or wait; evaluate them according to their stated intent and available evidence.",
    "When screenshots or observations are stale, unavailable, loading, or incomparable, prefer uncertain over an unsupported success or failure.",
    "recommended_next is advisory workflow status only; do not propose coordinates or a new executable GUI action.",
    "Keep evidence and reason concise and do not reveal private chain-of-thought.",
)


VERIFY_RULES_ZH: tuple[str, ...] = (
    "比较动作前后的观察，只识别具有可见证据支持的界面变化。",
    "执行器成功仅表示输入已发送，不能证明预期 GUI 效果已经发生。",
    "应分别判断计划动作的直接效果和整个任务是否完成。",
    "仅当动作后观察包含与预期效果一致的证据时，才将 action_effective 设为 true。",
    "仅当动作后界面明确满足任务成功条件时，才将 task_complete 设为 true。",
    "预期效果有证据支持时使用 status=success；存在相反证据或明确错误时使用 failure；证据缺失或含糊时使用 uncertain。",
    "只能使用给定观察和执行结果中的事实，不得虚构隐藏界面状态、元素、文本或结果。",
    "move_to、wait 等动作可能不会产生明显界面变化，应依据动作意图和现有证据判断。",
    "截图或观察过期、缺失、仍在加载或无法比较时，应优先返回 uncertain，而不是无依据地判断成功或失败。",
    "recommended_next 只表示建议的工作流方向，不得给出坐标或新的可执行 GUI 动作。",
    "evidence 和 reason 应简洁，不得输出隐藏思维链。",
)


@dataclass(frozen=True, slots=True)
class VerifyPrompt:
    """A verifier prompt split into provider-friendly message fields."""

    system: str
    user: str

    @property
    def text(self) -> str:
        """Return system and user content as one prompt string."""

        return "\n\n".join(part.strip() for part in (self.system, self.user) if part.strip())

    def as_messages(self) -> list[dict[str, str]]:
        """Return OpenAI-style messages without depending on a provider SDK."""

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


def _task_text(state: Any) -> str:
    task = _read(state, "instruction", "user_instruction", "goal")
    if task is None:
        task_object = _read(state, "task")
        task = _read(task_object, "instruction", "task", "goal", default=task_object)
    text = compact_whitespace(task)
    if not text:
        raise ValueError("Verifier task instruction must not be empty")
    return text


def _json(value: Any, config: PromptConfig, *, empty_label: str) -> str:
    if value is None:
        return empty_label
    return safe_json_dumps(
        value,
        indent=config.json_indent,
        ensure_ascii=config.ensure_ascii,
        sort_keys=config.sort_json_keys,
    )


def _planned_action(state: Any) -> Any:
    result = _read(state, "last_planner_result")
    if result is None:
        result = _read(_read(state, "current_step"), "planner_result")
    action = _read(result, "action")
    if action is not None:
        return action
    decision = _read(result, "decision")
    reason = _read(result, "reason")
    if decision is None and reason is None:
        return None
    return {
        "decision": decision.value if isinstance(decision, Enum) else decision,
        "reason": reason,
    }


def _execution_result(state: Any) -> Any:
    result = _read(state, "last_execution_result")
    if result is None:
        result = _read(_read(state, "current_step"), "execution_result")
    return result


def _result_payload(result: Any, config: PromptConfig) -> Any:
    if result is None:
        return None
    payload = {
        "status": _read(result, "status"),
        "success": _read(result, "success", "succeeded"),
        "tool_name": _read(result, "tool_name"),
        "message": _read(result, "message", "reason"),
        "output": _read(result, "output"),
        "duration_ms": _read(result, "duration_ms", "elapsed_ms"),
    }
    error = _read(result, "error")
    if error is not None:
        payload["error"] = format_error(error, max_chars=config.max_error_chars)
    cleaned = {key: value for key, value in payload.items() if value is not None}
    return cleaned or result


def _verify_schema(config: PromptConfig) -> dict[str, Any]:
    schema = deepcopy(VERIFY_RESPONSE_SCHEMA)
    required = schema.setdefault("required", [])
    if not config.require_confidence and "confidence" in required:
        required.remove("confidence")
    if not config.require_reason and "reason" in required:
        required.remove("reason")
    return schema


def build_verify_rules(
    config: PromptConfig | None = None,
    *,
    language: PromptLanguage | str | None = None,
) -> str:
    """Build numbered, language-aware verification rules."""

    cfg = config or PromptConfig()
    hint = language.value if isinstance(language, PromptLanguage) else language
    resolved = cfg.resolve_language(str(hint) if hint is not None else None)
    rules = list(VERIFY_RULES_ZH if resolved is PromptLanguage.ZH else VERIFY_RULES_EN)
    if cfg.require_reason:
        rules.append("reason 字段不能为空。" if resolved is PromptLanguage.ZH else "The reason field must not be empty.")
    if cfg.require_confidence:
        rules.append(
            "必须返回 0 到 1 之间的 confidence。"
            if resolved is PromptLanguage.ZH
            else "Return confidence as a number between 0 and 1."
        )
    return "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, start=1))


def _generation_notice(config: PromptConfig, language: PromptLanguage) -> str:
    if not config.add_generation_notice:
        return ""
    output_format = config.response_format_for(PromptKind.VERIFY)
    if output_format is PromptFormat.JSON:
        return (
            "只输出一个符合上述 Schema 的 JSON 对象；不要使用 Markdown 代码块，也不要添加额外说明。"
            if language is PromptLanguage.ZH
            else "Return exactly one JSON object matching the schema above. Do not use Markdown fences or add commentary."
        )
    if output_format is PromptFormat.MARKDOWN:
        return "使用简洁的 Markdown 输出。" if language is PromptLanguage.ZH else "Return concise Markdown."
    return "只输出请求的结果。" if language is PromptLanguage.ZH else "Return only the requested result."


def build_verify_messages(
    state: Any,
    config: PromptConfig | Any | None = None,
    *,
    planned_action: Any = None,
    execution_result: Any = None,
    before_observation: Any = None,
    after_observation: Any = None,
    task_language: str | None = None,
) -> VerifyPrompt:
    """Build separate system/user verifier messages from AgentState-like input."""

    cfg = prompt_config_from_planner_config(config)
    task = _task_text(state)
    language_hint = task_language or _read(_read(state, "task"), "language")
    language = cfg.resolve_language(language_hint)
    context_builder = ContextBuilder(cfg)

    action = planned_action if planned_action is not None else _planned_action(state)
    result = execution_result if execution_result is not None else _execution_result(state)
    before = before_observation if before_observation is not None else _read(state, "previous_observation")
    after = after_observation if after_observation is not None else _read(state, "observation", "current_observation")

    missing = []
    if action is None:
        missing.append("planned_action")
    if before is None:
        missing.append("before_observation")
    if after is None:
        missing.append("after_observation")
    if missing:
        raise ValueError("Verifier requires: " + ", ".join(missing))

    unavailable = "（无执行结果）" if language is PromptLanguage.ZH else "(no executor result)"
    schema_omitted = "（响应结构已省略）" if language is PromptLanguage.ZH else "(schema omitted)"
    rules_omitted = "（规则已省略）" if language is PromptLanguage.ZH else "(rules omitted)"
    values = {
        "rules": build_verify_rules(cfg, language=language) if cfg.include_rules else rules_omitted,
        "task": task,
        "planned_action": _json(action, cfg, empty_label="null"),
        "execution_result": _json(_result_payload(result, cfg), cfg, empty_label=unavailable),
        "before_observation": _json(context_builder.observation(before, label="before"), cfg, empty_label="null"),
        "after_observation": _json(context_builder.observation(after, label="after"), cfg, empty_label="null"),
        "response_schema": (
            _json(_verify_schema(cfg), cfg, empty_label="null") if cfg.include_schema else schema_omitted
        ),
        "generation_notice": _generation_notice(cfg, language),
    }

    system = ""
    if cfg.include_system_prompt:
        system = cfg.system_override_for(PromptKind.VERIFY) or VERIFY_TEMPLATE.system(language)
    user = VERIFY_TEMPLATE.render(values, language=language, include_system=False)
    prompt = VerifyPrompt(system=system, user=user)

    if len(prompt.text) > cfg.max_prompt_chars:
        fixed = dict(values)
        fixed["before_observation"] = ""
        fixed["after_observation"] = ""
        fixed_user = VERIFY_TEMPLATE.render(fixed, language=language, include_system=False)
        budget = cfg.max_prompt_chars - len(system) - len(fixed_user) - 4
        if budget < 512:
            raise ValueError("max_prompt_chars is too small for verifier rules and response schema")
        before_budget = budget // 2
        values["before_observation"] = truncate_text(values["before_observation"], before_budget)
        values["after_observation"] = truncate_text(values["after_observation"], budget - before_budget)
        prompt = VerifyPrompt(
            system=system,
            user=VERIFY_TEMPLATE.render(values, language=language, include_system=False),
        )
        if len(prompt.text) > cfg.max_prompt_chars:
            raise ValueError("max_prompt_chars is too small for the verifier prompt")
    return prompt


def build_verify_prompt(
    state: Any,
    config: PromptConfig | Any | None = None,
    *,
    planned_action: Any = None,
    execution_result: Any = None,
    before_observation: Any = None,
    after_observation: Any = None,
    task_language: str | None = None,
    separate_messages: bool = False,
) -> str | VerifyPrompt:
    """Build the complete verifier prompt or return separated messages."""

    prompt = build_verify_messages(
        state,
        config,
        planned_action=planned_action,
        execution_result=execution_result,
        before_observation=before_observation,
        after_observation=after_observation,
        task_language=task_language,
    )
    return prompt if separate_messages else prompt.text


__all__ = [
    "VERIFY_RULES_EN",
    "VERIFY_RULES_ZH",
    "VerifyPrompt",
    "build_verify_messages",
    "build_verify_prompt",
    "build_verify_rules",
]