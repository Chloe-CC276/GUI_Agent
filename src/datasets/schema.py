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


    def to_dict(self) -> dict:

        return {
            "step_id": self.step_id,
            "screenshot_path": str(self.screenshot_path),
            "instruction": self.instruction,
            "action": self.action.to_dict(),
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

    steps: list[GUITaskStep]

    language: str = "en"

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:

        return len(self.steps)

    def to_action_sequence(self) -> ActionSequence:

        return ActionSequence(
            [step.action for step in self.steps]
        )

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

    avg_steps_per_task: float = 0

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

    train: list[GUITaskSample]

    validation: list[GUITaskSample]

    test: list[GUITaskSample]

    @property
    def train_size(self):

        return len(self.train)

    @property
    def validation_size(self):

        return len(self.validation)

    @property
    def test_size(self):

        return len(self.test)
    

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