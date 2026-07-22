"""
prompt/memory_prompt
Build evidence-based memory and observation-summary prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import PromptConfig, PromptFormat, PromptKind, PromptLanguage
from .context_builder import ContextBuilder, compact_history
from .formatters import compact_whitespace, safe_json_dumps, truncate_text
from .planner_prompt import prompt_config_from_planner_config
from .schemas import MEMORY_SUMMARY_SCHEMA
from .templates import MEMORY_SUMMARY_TEMPLATE


MEMORY_RULES_EN: tuple[str, ...] = (
    "Preserve the task goal and only information useful to future planning.",
    "Put directly observed information in verified_facts; do not present inference as fact.",
    "A completed step requires visible or explicit verification, not executor success alone.",
    "Merge valid existing memory with new evidence and remove superseded or contradicted details.",
    "Record successful methods with the visible outcome that showed they worked.",
    "Record failed attempts compactly, including the observed effect and avoiding duplicate entries.",
    "Keep unresolved uncertainty in open_issues instead of guessing a hidden GUI state.",
    "Use next_focus for planning priorities, not executable actions, coordinates or private reasoning.",
    "Retain important elements only when their identity, text, role, location or state helps later steps.",
    "Return concise standalone memory that remains understandable after the raw history is discarded.",
)


MEMORY_RULES_ZH: tuple[str, ...] = (
    "保留任务目标以及对后续规划有用的信息。",
    "verified_facts 只能填写直接观察到的事实，不得把推断写成事实。",
    "只有存在可见或明确验证证据时才能把步骤记为已完成，执行器成功本身不等于完成。",
    "将仍然有效的已有记忆与新证据合并，并删除已过时或与当前证据冲突的信息。",
    "记录成功方法时，应同时保留证明其生效的可见结果。",
    "简洁记录失败尝试及其观察结果，并合并重复失败记录。",
    "无法确认的信息应放入 open_issues，不得猜测隐藏界面状态。",
    "next_focus 只描述后续规划重点，不得输出可执行动作、坐标或隐藏思维过程。",
    "仅在元素的身份、文本、类型、位置或状态有助于后续操作时保留该元素。",
    "摘要应简洁且可独立理解，即使原始历史被删除也能继续支持任务。",
)


OBSERVATION_RULES_EN: tuple[str, ...] = (
    "Summarise only the supplied observation and task; do not infer unavailable history.",
    "Describe visible text, controls, window state and spatial relationships that matter to the task.",
    "Use verified_facts only for directly visible or explicitly supplied evidence.",
    "Do not claim a task step is completed unless the current observation proves it.",
    "Use open_issues for ambiguity, occlusion, missing content or uncertain element identity.",
    "Do not invent coordinates, controls, application state, errors or prior actions.",
    "Important elements must correspond to supplied elements or clearly visible observation content.",
    "Keep the summary compact and useful to the next planning cycle.",
)


OBSERVATION_RULES_ZH: tuple[str, ...] = (
    "只总结提供的当前观察和任务，不得推断未提供的执行历史。",
    "描述与任务有关的可见文本、控件、窗口状态及空间关系。",
    "verified_facts 只能包含直接可见或明确提供的证据。",
    "除非当前观察能够证明，否则不得声称某个任务步骤已经完成。",
    "界面含糊、被遮挡、内容缺失或元素身份不确定时，应写入 open_issues。",
    "不得虚构坐标、控件、应用状态、错误或此前执行的动作。",
    "important_elements 必须对应已提供的元素或观察中清晰可见的内容。",
    "摘要应紧凑，并能直接支持下一轮规划。",
)


@dataclass(frozen=True, slots=True)
class MemoryPrompt:
    """A memory prompt split into provider-friendly message fields."""

    system: str
    user: str
    kind: PromptKind = PromptKind.MEMORY_SUMMARY

    @property
    def text(self) -> str:
        return "\n\n".join(part.strip() for part in (self.system, self.user) if part.strip())

    def as_messages(self) -> list[dict[str, str]]:
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


def _task_text(state: Any, explicit_task: Any = None) -> str:
    task = explicit_task
    if task is None:
        task = _read(state, "instruction", "user_instruction", "goal")
    if task is None:
        task_object = _read(state, "task")
        task = _read(task_object, "instruction", "task", "goal", default=task_object)
    text = compact_whitespace(task)
    if not text:
        raise ValueError("Memory task instruction must not be empty")
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


def build_memory_rules(
    config: PromptConfig | None = None,
    *,
    language: PromptLanguage | str | None = None,
    observation_only: bool = False,
) -> str:
    """Build numbered memory or observation-summary rules."""

    cfg = config or PromptConfig()
    hint = language.value if isinstance(language, PromptLanguage) else language
    resolved = cfg.resolve_language(str(hint) if hint is not None else None)
    if observation_only:
        rules = OBSERVATION_RULES_ZH if resolved is PromptLanguage.ZH else OBSERVATION_RULES_EN
    else:
        rules = MEMORY_RULES_ZH if resolved is PromptLanguage.ZH else MEMORY_RULES_EN
    result = list(rules)
    if cfg.strict_schema:
        result.append(
            "不得返回响应 Schema 未定义的字段。"
            if resolved is PromptLanguage.ZH
            else "Do not return fields that are not defined by the response schema."
        )
    return "\n".join(f"{index}. {rule}" for index, rule in enumerate(result, start=1))


def _generation_notice(config: PromptConfig, language: PromptLanguage, kind: PromptKind) -> str:
    if not config.add_generation_notice:
        return ""
    output_format = config.response_format_for(kind)
    if output_format is PromptFormat.JSON:
        return (
            "只输出一个符合上述 Schema 的 JSON 对象；不要使用 Markdown 代码块，也不要添加解释。"
            if language is PromptLanguage.ZH
            else "Return exactly one JSON object matching the schema above. Do not use Markdown fences or add commentary."
        )
    if output_format is PromptFormat.MARKDOWN:
        return "使用简洁的 Markdown 输出。" if language is PromptLanguage.ZH else "Return concise Markdown."
    return "只输出请求的摘要。" if language is PromptLanguage.ZH else "Return only the requested summary."


def _resolve_existing_memory(state: Any, memory: Any) -> Any:
    if memory is not None:
        return memory
    resolved = _read(state, "memory")
    if resolved is None:
        resolved = _read(_read(state, "metadata", default={}), "memory")
    return resolved


def _fit_prompt(
    values: dict[str, str],
    *,
    system: str,
    config: PromptConfig,
    language: PromptLanguage,
    kind: PromptKind,
) -> MemoryPrompt:
    def render(current: Mapping[str, str]) -> MemoryPrompt:
        return MemoryPrompt(
            system=system,
            user=MEMORY_SUMMARY_TEMPLATE.render(current, language=language, include_system=False),
            kind=kind,
        )

    prompt = render(values)
    if len(prompt.text) <= config.max_prompt_chars:
        return prompt

    fixed = dict(values)
    for name in ("existing_memory", "history", "current_observation"):
        fixed[name] = ""
    fixed_prompt = render(fixed)
    budget = config.max_prompt_chars - len(fixed_prompt.text) - 4
    if budget < 768:
        raise ValueError("max_prompt_chars is too small for memory rules and response schema")

    existing_budget = max(128, budget // 5)
    history_budget = max(256, budget * 2 // 5)
    observation_budget = budget - existing_budget - history_budget
    values["existing_memory"] = truncate_text(values["existing_memory"], existing_budget)
    values["history"] = truncate_text(values["history"], history_budget)
    values["current_observation"] = truncate_text(
        values["current_observation"], observation_budget
    )
    prompt = render(values)
    if len(prompt.text) > config.max_prompt_chars:
        raise ValueError("max_prompt_chars is too small for the memory prompt")
    return prompt


def build_memory_messages(
    state: Any,
    config: PromptConfig | Any | None = None,
    *,
    task: Any = None,
    existing_memory: Any = None,
    history: Any = None,
    current_observation: Any = None,
    task_language: str | None = None,
    observation_only: bool = False,
) -> MemoryPrompt:
    """Build separate system/user messages for memory or observation summary."""

    cfg = prompt_config_from_planner_config(config)
    kind = PromptKind.OBSERVATION_SUMMARY if observation_only else PromptKind.MEMORY_SUMMARY
    task_text = _task_text(state, task)
    language_hint = task_language or _read(_read(state, "task"), "language")
    language = cfg.resolve_language(language_hint)
    builder = ContextBuilder(cfg)

    observation = current_observation
    if observation is None:
        observation = _read(state, "observation", "current_observation")
    observation_context = builder.observation(observation, label="current")

    raw_history = history if history is not None else _read(state, "history", "completed_steps")
    history_text = "" if observation_only else compact_history(raw_history, config=cfg)
    memory = None if observation_only else _resolve_existing_memory(state, existing_memory)

    none_label = "（未提供）" if language is PromptLanguage.ZH else "(not provided)"
    omitted_label = "（不适用：仅总结当前观察）" if language is PromptLanguage.ZH else "(not applicable: current observation only)"
    values = {
        "rules": build_memory_rules(cfg, language=language, observation_only=observation_only)
        if cfg.include_rules
        else ("（规则已省略）" if language is PromptLanguage.ZH else "(rules omitted)"),
        "task": task_text,
        "existing_memory": omitted_label if observation_only else _json(memory, cfg, empty_label=none_label),
        "history": omitted_label if observation_only else (history_text or none_label),
        "current_observation": _json(observation_context, cfg, empty_label=none_label),
        "response_schema": _json(MEMORY_SUMMARY_SCHEMA, cfg, empty_label="null")
        if cfg.include_schema
        else ("（响应结构已省略）" if language is PromptLanguage.ZH else "(schema omitted)"),
        "generation_notice": _generation_notice(cfg, language, kind),
    }

    system = ""
    if cfg.include_system_prompt:
        system = cfg.system_override_for(kind) or MEMORY_SUMMARY_TEMPLATE.system(language)
    return _fit_prompt(values, system=system, config=cfg, language=language, kind=kind)


def build_memory_prompt(
    state: Any,
    config: PromptConfig | Any | None = None,
    *,
    task: Any = None,
    existing_memory: Any = None,
    history: Any = None,
    current_observation: Any = None,
    task_language: str | None = None,
    separate_messages: bool = False,
) -> str | MemoryPrompt:
    """Build a complete trajectory-memory prompt or separated messages."""

    prompt = build_memory_messages(
        state,
        config,
        task=task,
        existing_memory=existing_memory,
        history=history,
        current_observation=current_observation,
        task_language=task_language,
    )
    return prompt if separate_messages else prompt.text


def build_observation_summary_messages(
    observation: Any,
    task: Any,
    config: PromptConfig | Any | None = None,
    *,
    task_language: str | None = None,
) -> MemoryPrompt:
    """Build messages that compress one observation without trajectory claims."""

    state = {"task": task, "observation": observation}
    return build_memory_messages(
        state,
        config,
        task=task,
        current_observation=observation,
        task_language=task_language,
        observation_only=True,
    )


def build_observation_summary_prompt(
    observation: Any,
    task: Any,
    config: PromptConfig | Any | None = None,
    *,
    task_language: str | None = None,
    separate_messages: bool = False,
) -> str | MemoryPrompt:
    """Build a current-observation summary prompt or separated messages."""

    prompt = build_observation_summary_messages(
        observation,
        task,
        config,
        task_language=task_language,
    )
    return prompt if separate_messages else prompt.text


__all__ = [
    "MEMORY_RULES_EN",
    "MEMORY_RULES_ZH",
    "OBSERVATION_RULES_EN",
    "OBSERVATION_RULES_ZH",
    "MemoryPrompt",
    "build_memory_messages",
    "build_memory_prompt",
    "build_memory_rules",
    "build_observation_summary_messages",
    "build_observation_summary_prompt",
]