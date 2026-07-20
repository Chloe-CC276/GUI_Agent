from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.datasets.screenagent_loader import ScreenAgentLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCREENAGENT_ROOT = (
    PROJECT_ROOT
    / "external"
    / "ScreenAgent"
    / "data"
    / "ScreenAgent"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "screenagent"
)


@pytest.fixture(scope="module")
def train_loader() -> ScreenAgentLoader:
    return ScreenAgentLoader(
        data_root=SCREENAGENT_ROOT / "train",
        language="zh",
        strict=False,
        require_images=True,
    )


def test_train_loader_not_empty(
    train_loader: ScreenAgentLoader,
) -> None:
    assert len(train_loader) > 0


def test_first_sample_structure(
    train_loader: ScreenAgentLoader,
) -> None:
    sample = train_loader[0]

    assert sample.source == "screenagent"
    assert sample.task_id
    assert sample.instruction
    assert sample.num_steps > 0

    for step in sample.steps:
        assert step.step_id >= 0
        assert step.action is not None
        assert step.screenshot_path is not None


def test_screenagent_statistics(
    train_loader: ScreenAgentLoader,
) -> None:
    statistics = train_loader.statistics(limit=10)

    assert statistics.source == "screenagent"
    assert statistics.num_tasks > 0
    assert statistics.num_steps > 0


def test_export_train_jsonl(
    train_loader: ScreenAgentLoader,
) -> None:
    output_path = train_loader.export_jsonl(
        PROCESSED_ROOT / "train.jsonl"
    )

    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        first_line = file.readline().strip()

    assert first_line

    record = json.loads(first_line)

    assert record["source"] == "screenagent"
    assert record["task_id"]
    assert isinstance(record["steps"], list)


def test_export_test_jsonl() -> None:
    test_root = SCREENAGENT_ROOT / "test"

    if not test_root.exists():
        pytest.skip("ScreenAgent test directory does not exist.")

    loader = ScreenAgentLoader(
        data_root=test_root,
        language="zh",
        strict=False,
        require_images=True,
    )

    if len(loader) == 0:
        pytest.skip("ScreenAgent test directory is empty.")

    output_path = loader.export_jsonl(
        PROCESSED_ROOT / "test.jsonl"
    )

    assert output_path.exists()