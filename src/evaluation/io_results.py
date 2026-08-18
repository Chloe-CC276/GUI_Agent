"""JSONL / CSV writers for evaluation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics import format_metric


def ensure_layout(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "raw": output_dir / "raw",
        "screenshots": output_dir / "screenshots",
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "report": output_dir / "report",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_rows = [{k: format_metric(row.get(k)) for k in fieldnames} for row in rows]

    def _write(target: Path) -> None:
        with target.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in payload_rows:
                writer.writerow(row)

    try:
        _write(path)
    except PermissionError:
        # Common when Excel/WPS has the CSV open.
        fallback = path.with_name(f"{path.stem}_locked{path.suffix}")
        _write(fallback)
        print(
            f"WARNING: cannot write {path} (file locked). Wrote {fallback} instead. "
            "Close Excel/WPS and re-run generate_report if needed."
        )


OVERALL_FIELDS = [
    "test_layer",
    "variant",
    "sample_count",
    "json_valid_rate",
    "schema_valid_rate",
    "action_accuracy",
    "target_accuracy",
    "click_hit_rate",
    "task_success_rate",
    "avg_success_steps",
    "misoperation_rate",
    "retry_rate",
    "repeated_action_rate",
    "timeout_rate",
    "avg_latency_seconds",
    "avg_total_tokens",
    "auto_recovery_rate",
    "notes",
]
