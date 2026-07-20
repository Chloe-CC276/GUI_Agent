from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.datasets.webarena_loader import WebArenaLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEBARENA_ROOT = (
    PROJECT_ROOT
    / "external"
    / "WebArena"
    /"config_files"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "webarena"
)


@pytest.fixture(scope="module")
def webarena_loader() -> WebArenaLoader:
    return WebArenaLoader(
        source=WEBARENA_ROOT,
        strict=False,
        resolve_start_url=False,
        include_raw_record=False,
    )


def test_loader_not_empty(
    webarena_loader: WebArenaLoader,
) -> None:
    assert len(webarena_loader) > 0


def test_first_task_structure(
    webarena_loader: WebArenaLoader,
) -> None:
    sample = webarena_loader[0]

    assert sample.source == "webarena"
    assert sample.task_id
    assert sample.instruction

    # WebArena 没有标准动作轨迹
    assert sample.num_steps == 0

    assert isinstance(
        sample.metadata["sites"],
        list,
    )

    assert isinstance(
        sample.metadata["require_login"],
        bool,
    )

    assert isinstance(
        sample.metadata["require_reset"],
        bool,
    )

    assert isinstance(
        sample.metadata["evaluation"],
        dict,
    )


def test_task_791(
    webarena_loader: WebArenaLoader,
) -> None:
    try:
        sample = webarena_loader.get_by_task_id(791)
    except KeyError:
        pytest.skip("Task 791 is not available.")

    assert sample.task_id == "791"
    assert "gitlab" in sample.metadata["sites"]
    assert "reddit" in sample.metadata["sites"]
    assert sample.metadata["require_login"] is True
    assert sample.metadata["start_url"] == "__GITLAB__"
    assert sample.num_steps == 0


def test_webarena_statistics(
    webarena_loader: WebArenaLoader,
) -> None:
    statistics = webarena_loader.statistics()

    assert statistics.source == "webarena"
    assert statistics.num_tasks > 0
    assert statistics.num_steps == 0


def test_export_tasks_jsonl(
    webarena_loader: WebArenaLoader,
) -> None:
    output_path = webarena_loader.export_jsonl(
        PROCESSED_ROOT / "tasks.jsonl"
    )

    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        first_line = file.readline().strip()

    assert first_line

    record = json.loads(first_line)

    assert record["source"] == "webarena"
    assert record["task_id"]
    assert record["steps"] == []
    assert "evaluation" in record["metadata"]


def test_export_configs_jsonl(
    webarena_loader: WebArenaLoader,
) -> None:
    output_path = (
        webarena_loader.export_configs_jsonl(
            PROCESSED_ROOT
            / "task_configs.jsonl"
        )
    )

    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        first_line = file.readline().strip()

    record = json.loads(first_line)

    assert "task_id" in record
    assert "intent" in record
    assert "eval" in record