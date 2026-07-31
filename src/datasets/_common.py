"""
_common.py

Shared private helpers used by the dataset loaders.

These functions were previously duplicated inside each loader class.
Loaders re-bind them as ``staticmethod`` class attributes so existing
``self._helper(...)`` call sites keep working unchanged.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence


def _csv_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _write_csv(destination: Path, rows: Sequence[dict[str, Any]], *, encoding: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["task_id"]
    with destination.open("w", encoding=encoding, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in ratios):
        raise TypeError("Split ratios must be numeric.")
    if any(v < 0 for v in ratios):
        raise ValueError("Split ratios must not be negative.")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(
            "train_ratio + validation_ratio + test_ratio must equal 1.0."
        )


def _natural_sort_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)
