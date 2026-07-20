"""
schema.py

Convert all datasets (ScreenAgent, Mind2Web, WebArena) into schema format.
Task step: step_id, screenshot_path, instruction, action, llm_response, corrected_response, language, metadata
Task sample: task_id, source, instruction, steps, language, metadata
Planning sample: task_id, instruction, sub_tasks, planner_prompt, planner_answer, screenshot_path, language, metadata
Data statistics: source, num_tasks, num_steps, avg_steps_per_task, action_distribution, language_distribution, metadata
"""


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Reuse executor Action
from ..executor.action import Action, ActionSequence


# ============================================================
# GUI Task Step
# ============================================================

@dataclass
class GUITaskStep:

    step_id: int

    screenshot_path: Path | None

    instruction: str

    action: Action | SemanticAction

    llm_response: str | None = None

    corrected_response: str | None = None

    language: str = "en"

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def screenshot_exists(self) -> bool:
        return (
            self.screenshot_path is not None
            and self.screenshot_path.exists()
            )

    @property
    def is_executable(self) -> bool:
        return isinstance(self.action, Action)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "screenshot_path": (
                str(self.screenshot_path)
                if self.screenshot_path is not None
                else None
            ),
            "instruction": self.instruction,
            "action": self.action.to_dict(),
            "action_kind": (
                "executable"
                if isinstance(self.action, Action)
                else "semantic"
            ),
            "llm_response": self.llm_response,
            "corrected_response": self.corrected_response,
            "language": self.language,
            "metadata": self.metadata,
        }


# ============================================================
# GUI Task Sample
# ============================================================

@dataclass
class GUITaskSample:

    task_id: str

    source: str

    instruction: str

    steps: list[GUITaskStep] = field(default_factory=list) 

    language: str = "en"

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:

        return len(self.steps)
    

    @property
    def has_demonstration(self) -> bool:
        return bool(self.steps)

    @property
    def executable_step_count(self) -> int:
        return sum(
            isinstance(step.action, Action)
            for step in self.steps
            )

    def to_action_sequence(self) -> ActionSequence:

        executable_actions = [
            step.action
            for step in self.steps
            if isinstance(step.action, Action)
        ]

        return ActionSequence(executable_actions)
    
    def to_dict(self):

        return {

            "task_id": self.task_id,

            "source": self.source,

            "instruction": self.instruction,

            "language": self.language,

            "steps": [
                step.to_dict()
                for step in self.steps
            ],

            "metadata": self.metadata,
        }


# ============================================================
# WebArena evaluation configuration
# ============================================================

@dataclass
class WebArenaEvaluation:
    """
    WebArena task evaluation specification.
    """

    eval_types: list[str] = field(
        default_factory=list
    )

    reference_answers: dict[str, Any] = field(
        default_factory=dict
    )

    reference_url: str = ""

    program_html: list[Any] = field(
        default_factory=list
    )

    string_note: str = ""

    reference_answer_raw_annotation: Any = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_types": self.eval_types,
            "reference_answers": self.reference_answers,
            "reference_url": self.reference_url,
            "program_html": self.program_html,
            "string_note": self.string_note,
            "reference_answer_raw_annotation": (
                self.reference_answer_raw_annotation
            ),
            "metadata": self.metadata,
        }


# ============================================================
# WebArena task configuration
# ============================================================

@dataclass
class WebArenaTaskConfig:
    """
    Structured representation of one WebArena benchmark task.

    This is the source-level WebArena schema before conversion into
    GUITaskSample.
    """

    task_id: str

    sites: list[str]

    require_login: bool

    storage_state: str | None

    start_url: str

    geolocation: dict[str, Any] | None

    intent_template: str

    instantiation_dict: dict[str, Any]

    intent: str

    require_reset: bool

    evaluation: WebArenaEvaluation

    intent_template_id: int | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_gui_task_sample(self) -> GUITaskSample:
        """
        Convert WebArena configuration into unified GUI Agent format.

        WebArena does not provide a demonstration trajectory, so
        steps remains empty.
        """
        return GUITaskSample(
            task_id=self.task_id,
            source="webarena",
            instruction=self.intent,
            steps=[],
            language="en",
            metadata={
                "sites": self.sites,
                "require_login": self.require_login,
                "storage_state": self.storage_state,
                "start_url": self.start_url,
                "geolocation": self.geolocation,
                "intent_template": self.intent_template,
                "instantiation_dict": (
                    self.instantiation_dict
                ),
                "require_reset": self.require_reset,
                "evaluation": self.evaluation.to_dict(),
                "intent_template_id": (
                    self.intent_template_id
                ),
                "has_demonstration_trajectory": False,
                **self.metadata,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sites": self.sites,
            "require_login": self.require_login,
            "storage_state": self.storage_state,
            "start_url": self.start_url,
            "geolocation": self.geolocation,
            "intent_template": self.intent_template,
            "instantiation_dict": self.instantiation_dict,
            "intent": self.intent,
            "require_reset": self.require_reset,
            "eval": self.evaluation.to_dict(),
            "intent_template_id": self.intent_template_id,
            "metadata": self.metadata,
        }
    

# ============================================================
# Planning Sample
# ============================================================

@dataclass
class PlanningSample:

    task_id: str

    instruction: str

    sub_tasks: list[str]

    planner_prompt: str | None = None

    planner_answer: str | None = None

    screenshot_path: Path | None = None

    language: str = "en"

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):

        return {

            "task_id": self.task_id,

            "instruction": self.instruction,

            "sub_tasks": self.sub_tasks,

            "planner_prompt": self.planner_prompt,

            "planner_answer": self.planner_answer,

            "screenshot_path":
                str(self.screenshot_path)
                if self.screenshot_path
                else None,

            "language": self.language,

            "metadata": self.metadata,
        }


# ============================================================
# Dataset Statistics
# ============================================================

@dataclass
class DatasetStatistics:

    source: str

    num_tasks: int = 0

    num_steps: int = 0

    avg_steps_per_task: float = 0.0

    action_distribution: dict[str, int] = field(
        default_factory=dict
    )

    language_distribution: dict[str, int] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):

        return {

            "source": self.source,

            "num_tasks": self.num_tasks,

            "num_steps": self.num_steps,

            "avg_steps_per_task":
                self.avg_steps_per_task,

            "action_distribution":
                self.action_distribution,

            "language_distribution":
                self.language_distribution,

            "metadata":
                self.metadata,
        }


# ============================================================
# Dataset Split
# ============================================================

@dataclass
class DatasetSplit:

    train: list[GUITaskSample] = field(
        default_factory=list
    )

    validation: list[GUITaskSample] = field(
        default_factory=list
    )

    test: list[GUITaskSample] = field(
        default_factory=list
    )

    @property
    def train_size(self):

        return len(self.train)

    @property
    def validation_size(self):

        return len(self.validation)

    @property
    def test_size(self):

        return len(self.test)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
        }
    

@dataclass
class SemanticAction:
    """
    没有屏幕坐标的语义 GUI 动作。

    主要用于 Mind2Web 等基于 DOM 的网页操作数据。
    """

    action_type: str

    value: str | None = None

    target_tag: str | None = None

    target_attributes: dict[str, Any] = field(default_factory=dict)

    backend_node_id: str | None = None

    action_repr: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "value": self.value,
            "target_tag": self.target_tag,
            "target_attributes": self.target_attributes,
            "backend_node_id": self.backend_node_id,
            "action_repr": self.action_repr,
            "metadata": self.metadata,
        }