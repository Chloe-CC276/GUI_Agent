"""Metric helpers — use None / N/A for missing data, never fake zeros."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


NA = "N/A"


def point_in_bbox(x: float, y: float, bbox: Sequence[float] | None) -> bool | None:
    """Return True/False if bbox is valid; None when bbox missing (caller → N/A)."""

    if bbox is None or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if x2 < x1 or y2 < y1:
        return None
    return x1 <= float(x) <= x2 and y1 <= float(y) <= y2


def safe_rate(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return numer / denom


def relative_lift(baseline: float | None, optimized: float | None) -> float | None:
    if baseline is None or optimized is None:
        return None
    if baseline == 0:
        return None
    return (optimized - baseline) / baseline


def decrease_rate(baseline: float | None, optimized: float | None) -> float | None:
    """For lower-is-better metrics."""

    if baseline is None or optimized is None or baseline == 0:
        return None
    return (baseline - optimized) / baseline


def _mean(values: Iterable[float]) -> float | None:
    data = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not data:
        return None
    return sum(data) / len(data)


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one test layer × variant cohort."""

    n = len(records)
    if n == 0:
        return {
            "sample_count": 0,
            "json_valid_rate": None,
            "schema_valid_rate": None,
            "action_accuracy": None,
            "target_accuracy": None,
            "click_hit_rate": None,
            "task_success_rate": None,
            "avg_success_steps": None,
            "avg_steps_all": None,
            "misoperation_rate": None,
            "retry_rate": None,
            "repeated_action_rate": None,
            "timeout_rate": None,
            "avg_latency_seconds": None,
            "avg_total_tokens": None,
            "status_counts": {},
            "error_type_counts": {},
        }

    json_ok = sum(1 for r in records if r.get("json_valid") is True)
    json_n = sum(1 for r in records if r.get("json_valid") is not None)
    schema_ok = sum(1 for r in records if r.get("schema_valid") is True)
    schema_n = sum(1 for r in records if r.get("schema_valid") is not None)
    action_ok = sum(1 for r in records if r.get("action_correct") is True)
    action_n = sum(1 for r in records if r.get("action_correct") is not None)
    target_ok = sum(1 for r in records if r.get("target_correct") is True)
    target_n = sum(1 for r in records if r.get("target_correct") is not None)
    click_ok = sum(1 for r in records if r.get("click_hit") is True)
    click_n = sum(1 for r in records if r.get("click_hit") is not None)
    success = sum(1 for r in records if r.get("task_success") is True)
    success_n = sum(1 for r in records if r.get("task_success") is not None)
    misop = sum(1 for r in records if r.get("misoperation") is True)
    misop_n = sum(1 for r in records if r.get("misoperation") is not None)
    retries = sum(int(r.get("planner_retry_count") or 0) for r in records)
    repeated = sum(int(r.get("repeated_action_count") or 0) for r in records)
    timeouts = sum(1 for r in records if r.get("timeout") is True or r.get("status") == "timeout")

    success_steps = [
        float(r["total_steps"])
        for r in records
        if r.get("task_success") is True and r.get("total_steps") is not None
    ]
    all_steps = [float(r["total_steps"]) for r in records if r.get("total_steps") is not None]
    latencies = [float(r["latency_seconds"]) for r in records if r.get("latency_seconds") is not None]
    tokens = [float(r["total_tokens"]) for r in records if r.get("total_tokens") is not None]

    status_counts = Counter(str(r.get("status") or "unknown") for r in records)
    error_counts = Counter(
        str(r.get("error_type")) for r in records if r.get("error_type")
    )

    return {
        "sample_count": n,
        "json_valid_rate": safe_rate(json_ok, json_n),
        "schema_valid_rate": safe_rate(schema_ok, schema_n),
        "action_accuracy": safe_rate(action_ok, action_n),
        "target_accuracy": safe_rate(target_ok, target_n),
        "click_hit_rate": safe_rate(click_ok, click_n),
        "task_success_rate": safe_rate(success, success_n),
        "avg_success_steps": _mean(success_steps),
        "avg_steps_all": _mean(all_steps),
        "misoperation_rate": safe_rate(misop, misop_n),
        "retry_rate": safe_rate(retries, n) if n else None,
        "repeated_action_rate": safe_rate(repeated, n) if n else None,
        "timeout_rate": safe_rate(timeouts, n) if n else None,
        "avg_latency_seconds": _mean(latencies),
        "avg_total_tokens": _mean(tokens),
        "status_counts": dict(status_counts),
        "error_type_counts": dict(error_counts),
    }


def confusion_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    gold_key: str = "expected_action_type",
    pred_key: str = "predicted_action_type",
) -> dict[str, Any]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        gold = row.get(gold_key)
        pred = row.get(pred_key)
        if gold is None or pred is None:
            continue
        matrix[str(gold)][str(pred)] += 1
    return {g: dict(preds) for g, preds in matrix.items()}


def format_metric(value: Any) -> Any:
    if value is None:
        return NA
    if isinstance(value, float):
        return round(value, 6)
    return value
