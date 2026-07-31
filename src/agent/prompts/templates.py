"""
prompts/templates
Reusable prompt templates for the GUI Agent prompt package.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from types import MappingProxyType
from typing import Any, Mapping

from .config import PromptKind, PromptLanguage


class PromptTemplateError(ValueError):
    """Raised when a prompt template cannot be rendered safely."""


def _clean_block(value: Any) -> str:
    """Convert a template value to a clean text block."""

    if value is None:
        return ""
    return str(value).strip()


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A bilingual, strictly renderable prompt template.

    Parameters
    ----------
    name:
        Stable template identifier used in logs and tests.
    kind:
        Prompt workflow this template belongs to.
    system_en, system_zh:
        System instructions.  They contain no runtime placeholders.
    body_en, body_zh:
        User-prompt bodies using ``$variable`` placeholders.
    required_variables:
        Variables that callers must supply to :meth:`render`.
    description:
        Short developer-facing purpose statement.
    """

    name: str
    kind: PromptKind
    system_en: str
    system_zh: str
    body_en: str
    body_zh: str
    required_variables: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise PromptTemplateError("PromptTemplate.name must not be empty")

        kind = PromptKind.coerce(self.kind)
        variables = tuple(dict.fromkeys(self.required_variables))
        if any(not item or not item.isidentifier() for item in variables):
            raise PromptTemplateError(
                "required_variables must contain valid Python identifiers"
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "required_variables", variables)

        for field_name in ("system_en", "system_zh", "body_en", "body_zh"):
            text = getattr(self, field_name).strip()
            if not text:
                raise PromptTemplateError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, text)

        declared = set(variables)
        for language, body in (("en", self.body_en), ("zh", self.body_zh)):
            referenced = {
                match.group("named") or match.group("braced")
                for match in Template.pattern.finditer(body)
                if match.group("named") or match.group("braced")
            }
            undeclared = referenced - declared
            missing = declared - referenced
            if undeclared or missing:
                raise PromptTemplateError(
                    f"{self.name}.{language} variable mismatch: "
                    f"undeclared={sorted(undeclared)}, missing={sorted(missing)}"
                )

    def system(self, language: PromptLanguage | str = PromptLanguage.EN) -> str:
        """Return the system instruction in the requested language."""

        resolved = PromptLanguage.coerce(language)
        if resolved is PromptLanguage.AUTO:
            resolved = PromptLanguage.EN
        return self.system_zh if resolved is PromptLanguage.ZH else self.system_en

    def body(self, language: PromptLanguage | str = PromptLanguage.EN) -> str:
        """Return the unresolved user-prompt body."""

        resolved = PromptLanguage.coerce(language)
        if resolved is PromptLanguage.AUTO:
            resolved = PromptLanguage.EN
        return self.body_zh if resolved is PromptLanguage.ZH else self.body_en

    def render(
        self,
        values: Mapping[str, Any] | None = None,
        *,
        language: PromptLanguage | str = PromptLanguage.EN,
        include_system: bool = True,
        system_override: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Render a complete prompt and reject missing required variables.

        ``values`` and keyword arguments may both be used.  Keyword arguments
        take precedence.  Extra values are accepted so a shared context mapping
        can be passed to different templates.
        """

        context = dict(values or {})
        context.update(kwargs)
        missing = [name for name in self.required_variables if name not in context]
        if missing:
            raise PromptTemplateError(
                f"Missing variables for {self.name}: {', '.join(missing)}"
            )

        substitutions = {
            name: _clean_block(context[name]) for name in self.required_variables
        }
        try:
            rendered_body = Template(self.body(language)).substitute(substitutions)
        except (KeyError, ValueError) as error:
            raise PromptTemplateError(
                f"Failed to render template {self.name}: {error}"
            ) from error

        parts: list[str] = []
        if include_system:
            selected_system = (
                system_override.strip()
                if system_override and system_override.strip()
                else self.system(language)
            )
            parts.append(selected_system)
        parts.append(rendered_body.strip())
        return "\n\n".join(part for part in parts if part).strip()

    def variables(self) -> tuple[str, ...]:
        """Return the template's declared runtime variables."""

        return self.required_variables


PLANNER_TEMPLATE = PromptTemplate(
    name="planner",
    kind=PromptKind.PLANNER,
    description="Select exactly one next GUI action or terminal decision.",
    required_variables=(
        "rules",
        "task",
        "search_stage",
        "context",
        "allowed_actions",
        "response_schema",
        "generation_notice",
    ),
    system_en="""
You are the Planner of a desktop GUI agent. Inspect the task, the latest GUI
observation, and recent execution history, then choose exactly one safe next
action. Ground every decision in visible evidence. Never claim that an action
succeeded merely because it was requested. Do not reveal private chain-of-
thought; provide only the concise reason requested by the response schema.
""",
    system_zh="""
你是桌面 GUI Agent 的规划器。请根据任务、最新 GUI 观察结果和近期执行历史，选择且仅
选择一个安全的下一步动作。所有判断必须以可见证据为依据，不能因为某动作已被请求就
假定它已经成功。不要输出隐藏的思维链，只按响应结构给出简短理由。
""",
    body_en="""
## Planner rules
$rules

## Current task
$task

$search_stage

## Current agent context
$context

## Allowed actions
$allowed_actions

## Required response schema
$response_schema

$generation_notice
""",
    body_zh="""
## 规划规则
$rules

## 当前任务
$task

$search_stage

## 当前 Agent 上下文
$context

## 允许的动作
$allowed_actions

## 必须遵循的响应结构
$response_schema

$generation_notice
""",
)


VERIFY_TEMPLATE = PromptTemplate(
    name="verify",
    kind=PromptKind.VERIFY,
    description="Verify whether an executed action advanced or completed the task.",
    required_variables=(
        "rules",
        "task",
        "planned_action",
        "execution_result",
        "before_observation",
        "after_observation",
        "response_schema",
        "generation_notice",
    ),
    system_en="""
You are the Verifier of a desktop GUI agent. Determine what actually changed
after an action and whether the intended result is visibly supported. Be
conservative: absence of an executor error is not proof of success. For focus
clicks on search boxes or inputs, judge success by combined visual signals such
as a still-present box, an attached history dropdown underneath it, and caret or
highlight changes—not by a caret alone. Treat the Google homepage as reached—and
the website-open task as complete—when both the Google logo and the central Google
search box are visible. A Bing/Edge keyword results page for "Google" is not the
Google homepage. Do not propose or
execute a new action. Return only the structured verification result.
""",
    system_zh="""
你是桌面 GUI Agent 的验证器。请判断动作执行后实际发生了什么变化，以及界面证据是否
支持预期结果。应采用保守判断：执行器没有报错并不等于动作成功。对搜索框或输入框的
点击聚焦，应依据搜索框仍在、下方关联的历史下拉列表、光标或边框高亮等组合视觉信号
判断，不能只依赖能否看到光标。若同时看到 Google logo 与中央 Google 搜索框，即认定
已打开 Google 官网且网站打开类任务完成。Bing/Edge 上搜索关键词 Google 的结果页
不算 Google 首页。不要提出或执行新的动作，只返回结构化验证结果。
""",
    body_en="""
## Verification rules
$rules

## Task
$task

## Planned action
$planned_action

## Executor result
$execution_result

## Observation before the action
$before_observation

## Observation after the action
$after_observation

## Required response schema
$response_schema

$generation_notice
""",
    body_zh="""
## 验证规则
$rules

## 任务
$task

## 计划动作
$planned_action

## 执行器结果
$execution_result

## 动作执行前的观察
$before_observation

## 动作执行后的观察
$after_observation

## 必须遵循的响应结构
$response_schema

$generation_notice
""",
)


REPAIR_TEMPLATE = PromptTemplate(
    name="repair",
    kind=PromptKind.REPAIR,
    description="Repair a malformed or schema-invalid model response.",
    required_variables=(
        "rules",
        "original_request",
        "invalid_response",
        "validation_error",
        "response_schema",
        "generation_notice",
    ),
    system_en="""
You repair invalid structured responses for a desktop GUI agent. Preserve the
original intent, change only what is necessary to satisfy the schema and the
validation error, and never invent missing GUI evidence. Return the corrected
response only; do not discuss the repair.
""",
    system_zh="""
你负责修复桌面 GUI Agent 的无效结构化响应。请保留原始意图，只修改违反响应结构或
校验规则的内容，不得虚构缺失的 GUI 证据。只返回修复后的响应，不要解释修复过程。
""",
    body_en="""
## Repair rules
$rules

## Original request
$original_request

## Invalid response
$invalid_response

## Validation error
$validation_error

## Required response schema
$response_schema

$generation_notice
""",
    body_zh="""
## 修复规则
$rules

## 原始请求
$original_request

## 无效响应
$invalid_response

## 校验错误
$validation_error

## 必须遵循的响应结构
$response_schema

$generation_notice
""",
)


REFLECTION_TEMPLATE = PromptTemplate(
    name="reflection",
    kind=PromptKind.REFLECTION,
    description="Diagnose progress, repeated failures, and the next strategy.",
    required_variables=(
        "rules",
        "task",
        "current_context",
        "history",
        "failure_context",
        "response_schema",
        "generation_notice",
    ),
    system_en="""
You are the Reflection module of a desktop GUI agent. Review outcomes rather
than intentions, identify the most likely cause of stalled or failed progress,
and recommend a concise strategy adjustment. Do not issue executable GUI
actions and do not reveal private chain-of-thought.
""",
    system_zh="""
你是桌面 GUI Agent 的反思模块。请依据实际结果而非原始意图，识别任务停滞或失败的
最可能原因，并给出简洁的策略调整建议。不要直接输出可执行 GUI 动作，也不要暴露隐藏
的思维链。
""",
    body_en="""
## Reflection rules
$rules

## Task
$task

## Current context
$current_context

## Recent history
$history

## Failure or stall context
$failure_context

## Required response schema
$response_schema

$generation_notice
""",
    body_zh="""
## 反思规则
$rules

## 任务
$task

## 当前上下文
$current_context

## 近期历史
$history

## 失败或停滞信息
$failure_context

## 必须遵循的响应结构
$response_schema

$generation_notice
""",
)


MEMORY_SUMMARY_TEMPLATE = PromptTemplate(
    name="memory_summary",
    kind=PromptKind.MEMORY_SUMMARY,
    description="Compress an agent trajectory into durable, evidence-based memory.",
    required_variables=(
        "rules",
        "task",
        "existing_memory",
        "history",
        "current_observation",
        "response_schema",
        "generation_notice",
    ),
    system_en="""
You summarise GUI-agent trajectories into compact working memory. Retain only
facts that help future decisions: the goal, verified progress, important GUI
state, successful and failed approaches, unresolved blockers, and the next
useful focus. Distinguish observations from inferences and never invent facts.
""",
    system_zh="""
你负责把 GUI Agent 的执行轨迹压缩为简洁的工作记忆。仅保留有助于后续决策的信息：
任务目标、已验证进展、重要界面状态、成功与失败的方法、未解决阻碍及下一步重点。
明确区分观察事实与推断，不得虚构信息。
""",
    body_en="""
## Memory rules
$rules

## Task
$task

## Existing memory
$existing_memory

## Recent history
$history

## Current observation
$current_observation

## Required response schema
$response_schema

$generation_notice
""",
    body_zh="""
## 记忆规则
$rules

## 任务
$task

## 现有记忆
$existing_memory

## 近期历史
$history

## 当前观察
$current_observation

## 必须遵循的响应结构
$response_schema

$generation_notice
""",
)


TEMPLATES: Mapping[PromptKind, PromptTemplate] = MappingProxyType(
    {
        PromptKind.PLANNER: PLANNER_TEMPLATE,
        PromptKind.VERIFY: VERIFY_TEMPLATE,
        PromptKind.REPAIR: REPAIR_TEMPLATE,
        PromptKind.REFLECTION: REFLECTION_TEMPLATE,
        PromptKind.MEMORY_SUMMARY: MEMORY_SUMMARY_TEMPLATE,
        # Observation summaries use the same evidence-preserving compression
        # contract; memory_prompt.py supplies an observation-specific context.
        PromptKind.OBSERVATION_SUMMARY: MEMORY_SUMMARY_TEMPLATE,
    }
)


def get_template(kind: PromptKind | str) -> PromptTemplate:
    """Return the registered template for one prompt kind."""

    return TEMPLATES[PromptKind.coerce(kind)]


__all__ = [
    "MEMORY_SUMMARY_TEMPLATE",
    "PLANNER_TEMPLATE",
    "REFLECTION_TEMPLATE",
    "REPAIR_TEMPLATE",
    "TEMPLATES",
    "VERIFY_TEMPLATE",
    "PromptTemplate",
    "PromptTemplateError",
    "get_template",
]