"""Load evaluation YAML and resolve paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class VariantConfig:
    name: str
    model_type: str  # original | local_adapter
    prompt_version: str  # original | optimized
    adapter_path: str | None = None
    model_name: str | None = None


@dataclass
class ExperimentConfig:
    name: str = "gui_agent_prompt_adapter_comparison"
    repeat_per_task: int = 3
    random_seed: int = 42
    max_steps: int = 20
    task_timeout_seconds: int = 180
    output_dir: Path = field(default_factory=lambda: ROOT / "outputs" / "gui_agent_evaluation")
    countdown_seconds: int = 3
    language: str = "zh"
    region: str = "frankfurt"
    cloud_model: str = "qwen3-vl-plus"
    local_base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    allow_local_adapter: bool = False  # must be explicitly enabled (GPU risk)
    allow_real_actions: bool = False
    allow_cloud_vlm: bool = False  # must be explicitly enabled (API cost)
    variants: list[VariantConfig] = field(default_factory=list)
    tasks_path: Path | None = None
    offline_cases_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _as_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    cfg_path = _as_path(path, ROOT / "configs" / "gui_agent_evaluation.yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    exp = data.get("experiment") or {}
    variants_raw = data.get("variants") or {}
    variants: list[VariantConfig] = []
    for name, body in variants_raw.items():
        body = body or {}
        variants.append(
            VariantConfig(
                name=str(name),
                model_type=str(body.get("model_type") or "original"),
                prompt_version=str(body.get("prompt_version") or "original"),
                adapter_path=body.get("adapter_path"),
                model_name=body.get("model_name"),
            )
        )
    return ExperimentConfig(
        name=str(exp.get("name") or "gui_agent_prompt_adapter_comparison"),
        repeat_per_task=int(exp.get("repeat_per_task") or 3),
        random_seed=int(exp.get("random_seed") or 42),
        max_steps=int(exp.get("max_steps") or 20),
        task_timeout_seconds=int(exp.get("task_timeout_seconds") or 180),
        output_dir=_as_path(exp.get("output_dir"), ROOT / "outputs" / "gui_agent_evaluation"),
        countdown_seconds=int(exp.get("countdown_seconds") or 3),
        language=str(exp.get("language") or "zh"),
        region=str(exp.get("region") or "frankfurt"),
        cloud_model=str(exp.get("cloud_model") or "qwen3-vl-plus"),
        local_base_model=str(exp.get("local_base_model") or "Qwen/Qwen2.5-VL-7B-Instruct"),
        allow_local_adapter=bool(exp.get("allow_local_adapter", False)),
        allow_real_actions=bool(exp.get("allow_real_actions", False)),
        allow_cloud_vlm=bool(exp.get("allow_cloud_vlm", False)),
        variants=variants,
        tasks_path=_as_path(
            exp.get("tasks_path") or data.get("tasks_path"),
            ROOT / "data" / "gui_eval" / "tasks.yaml",
        ),
        offline_cases_path=_as_path(
            exp.get("offline_cases_path") or data.get("offline_cases_path"),
            ROOT / "data" / "gui_eval" / "offline_cases.jsonl",
        ),
        raw=data,
    )
