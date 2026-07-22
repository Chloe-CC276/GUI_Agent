"""
prompt/repair_prompt
Build schema-aware repair prompts for invalid GUI Agent responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import PromptConfig, PromptFormat, PromptKind, PromptLanguage
from .formatters import compact_whitespace, format_error, safe_json_dumps, truncate_text
from .planner_prompt import prompt_config_from_planner_config
from .schemas import get_response_schema
from .templates import REPAIR_TEMPLATE


REPAIR_RULES_EN: tuple[str, ...] = (
    "Preserve the valid, evidence-supported intent of the invalid response.",
    "Correct only the JSON syntax, field names, value types, required fields, enum values, or schema relationships identified by the error.",
    "Treat the required response schema as authoritative when it conflicts with the invalid response.",
    "Do not introduce a new GUI action, target, coordinate, element, observation, result, or claim that is absent from the original request and invalid response.",
    "Do not reinterpret the task or improve the plan beyond what is necessary to produce a valid response.",
    "When a required value cannot be recovered without invention, use a schema-valid conservative value such as null, retry, uncertain, an empty list, or an explicit failure value when the schema permits it.",
    "Remove Markdown fences, prose before or after the object, comments, and unsupported fields when JSON-only output is required.",
    "Return exactly one corrected response and do not explain the repair or reveal private chain-of-thought.",
)


REPAIR_RULES_ZH: tuple[str, ...] = (
    "保留无效响应中已经存在且有证据支持的真实意图。",
    "只修复错误涉及的 JSON 语法、字段名、值类型、必填字段、枚举值或字段关系。",
    "当无效响应与目标响应结构冲突时，以目标响应结构为准。",
    "不得添加原始请求和无效响应中不存在的新 GUI 动作、目标、坐标、元素、观察、结果或结论。",
    "不得重新解释任务，也不得借修复之机重新规划；只进行生成合法响应所需的最小修改。",
    "如果某个必填值无法在不虚构信息的前提下恢复，应在结构允许时使用 null、retry、uncertain、空列表或明确失败值等保守值。",
    "需要纯 JSON 输出时，应删除 Markdown 代码块、对象前后的说明、注释和不受支持的字段。",
    "只返回一个修复后的响应，不得解释修复过程或暴露隐藏思维链。",
)


@dataclass(frozen=True, slots=True)
class RepairPrompt:
    """A repair prompt split into provider-friendly message fields."""

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


def _required_text(name: str, value: Any, max_chars: int) -> str:
    """Normalize a required text field and reject empty values."""

    if isinstance(value, str):
        text = value.strip()
    else:
        text = safe_json_dumps(value, indent=2)
    if not text or text in {'""', "null"}:
        raise ValueError(f"{name} must not be empty")
    return truncate_text(text, max_chars)


def _invalid_response_text(value: Any, config: PromptConfig) -> str:
    """Preserve a raw model response or serialize a parsed response safely."""

    if isinstance(value, str):
        text = value.strip()
    else:
        text = safe_json_dumps(
            value,
            indent=config.json_indent,
            ensure_ascii=config.ensure_ascii,
            sort_keys=config.sort_json_keys,
        )
    if not text:
        raise ValueError("invalid_response must not be empty")
    return text


def _validation_error_text(value: Any, config: PromptConfig) -> str:
    """Convert exceptions, mappings, error objects, or strings to compact text."""

    if value is None:
        raise ValueError("validation_error must not be empty")
    if isinstance(value, str):
        text = compact_whitespace(value, preserve_newlines=True)
    else:
        text = format_error(value, max_chars=config.max_error_chars)
    if not text:
        raise ValueError("validation_error must not be empty")
    return truncate_text(text, config.max_error_chars)


def _schema_text(value: Mapping[str, Any], config: PromptConfig) -> str:
    return safe_json_dumps(
        value,
        indent=config.json_indent,
        ensure_ascii=config.ensure_ascii,
        sort_keys=config.sort_json_keys,
    )


def build_repair_rules(
    config: PromptConfig | None = None,
    *,
    language: PromptLanguage | str | None = None,
) -> str:
    """Build numbered, language-aware response-repair rules."""

    cfg = config or PromptConfig()
    hint = language.value if isinstance(language, PromptLanguage) else language
    resolved = cfg.resolve_language(str(hint) if hint is not None else None)
    rules = list(REPAIR_RULES_ZH if resolved is PromptLanguage.ZH else REPAIR_RULES_EN)
    if cfg.strict_schema:
        rules.append(
            "修复结果必须通过目标 Schema 校验，不得包含 additionalProperties 禁止的字段。"
            if resolved is PromptLanguage.ZH
            else "The repaired response must pass the target schema and must not contain fields forbidden by additionalProperties."
        )
    return "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, start=1))


def _generation_notice(config: PromptConfig, language: PromptLanguage) -> str:
    if not config.add_generation_notice:
        return ""
    output_format = config.response_format_for(PromptKind.REPAIR)
    if output_format is PromptFormat.JSON:
        return (
            "只输出一个符合上述 Schema 的 JSON 对象；不要使用 Markdown 代码块，也不要添加解释。"
            if language is PromptLanguage.ZH
            else "Return exactly one JSON object matching the schema above. Do not use Markdown fences or add an explanation."
        )
    if output_format is PromptFormat.MARKDOWN:
        return "只输出修复后的简洁 Markdown。" if language is PromptLanguage.ZH else "Return only the corrected concise Markdown."
    return "只输出修复后的结果。" if language is PromptLanguage.ZH else "Return only the corrected result."


def build_repair_messages(
    original_request: Any,
    invalid_response: Any,
    validation_error: Any,
    config: PromptConfig | Any | None = None,
    *,
    response_schema: Mapping[str, Any] | None = None,
    target_kind: PromptKind | str = PromptKind.PLANNER,
    task_language: str | None = None,
) -> RepairPrompt:
    """Build separate system/user messages for one repair attempt.

    ``target_kind`` chooses the built-in schema when ``response_schema`` is not
    supplied.  REPAIR itself maps to the Planner schema for backward
    compatibility, so Planner is also the explicit default here.
    """

    cfg = prompt_config_from_planner_config(config)
    language_hint = task_language or _read(original_request, "language")
    language = cfg.resolve_language(language_hint)
    resolved_kind = PromptKind.coerce(target_kind)
    schema = dict(response_schema) if response_schema is not None else get_response_schema(resolved_kind)

    request_text = _required_text(
        "original_request", original_request, cfg.max_prompt_chars
    )
    invalid_text = _invalid_response_text(invalid_response, cfg)
    error_text = _validation_error_text(validation_error, cfg)

    rules_omitted = "（规则已省略）" if language is PromptLanguage.ZH else "(rules omitted)"
    schema_omitted = "（响应结构已省略）" if language is PromptLanguage.ZH else "(schema omitted)"
    values = {
        "rules": build_repair_rules(cfg, language=language) if cfg.include_rules else rules_omitted,
        "original_request": request_text,
        "invalid_response": invalid_text,
        "validation_error": error_text,
        "response_schema": _schema_text(schema, cfg) if cfg.include_schema else schema_omitted,
        "generation_notice": _generation_notice(cfg, language),
    }

    system = ""
    if cfg.include_system_prompt:
        system = cfg.system_override_for(PromptKind.REPAIR) or REPAIR_TEMPLATE.system(language)
    prompt = RepairPrompt(
        system=system,
        user=REPAIR_TEMPLATE.render(values, language=language, include_system=False),
    )

    if len(prompt.text) > cfg.max_prompt_chars:
        fixed = dict(values)
        fixed["original_request"] = ""
        fixed["invalid_response"] = ""
        fixed_user = REPAIR_TEMPLATE.render(fixed, language=language, include_system=False)
        available = cfg.max_prompt_chars - len(system) - len(fixed_user) - 4
        if available < 512:
            raise ValueError("max_prompt_chars is too small for repair rules, error, and response schema")

        error_budget = min(len(error_text), max(128, available // 5))
        remaining = available - error_budget
        request_budget = max(128, remaining // 3)
        invalid_budget = remaining - request_budget
        values["validation_error"] = truncate_text(error_text, error_budget)
        values["original_request"] = truncate_text(request_text, request_budget)
        values["invalid_response"] = truncate_text(invalid_text, invalid_budget)
        prompt = RepairPrompt(
            system=system,
            user=REPAIR_TEMPLATE.render(values, language=language, include_system=False),
        )
        if len(prompt.text) > cfg.max_prompt_chars:
            raise ValueError("max_prompt_chars is too small for the repair prompt")
    return prompt


def build_repair_prompt(
    original_request: Any,
    invalid_response: Any,
    validation_error: Any,
    config: PromptConfig | Any | None = None,
    *,
    response_schema: Mapping[str, Any] | None = None,
    target_kind: PromptKind | str = PromptKind.PLANNER,
    task_language: str | None = None,
    separate_messages: bool = False,
) -> str | RepairPrompt:
    """Build a complete repair prompt or return separated message fields."""

    prompt = build_repair_messages(
        original_request,
        invalid_response,
        validation_error,
        config,
        response_schema=response_schema,
        target_kind=target_kind,
        task_language=task_language,
    )
    return prompt if separate_messages else prompt.text


__all__ = [
    "REPAIR_RULES_EN",
    "REPAIR_RULES_ZH",
    "RepairPrompt",
    "build_repair_messages",
    "build_repair_prompt",
    "build_repair_rules",
]