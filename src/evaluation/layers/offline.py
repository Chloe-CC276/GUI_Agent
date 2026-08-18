"""Layer 1: offline screenshot planning (no Executor)."""

from __future__ import annotations

import time
import uuid
from typing import Any

from src.evaluation.fixtures import state_from_offline_case
from src.evaluation.metrics import point_in_bbox
from src.evaluation.variants import VariantRuntime


def _action_type(action: Any) -> str | None:
    if action is None:
        return None
    if isinstance(action, dict):
        return str(action.get("type") or action.get("action_type") or "").lower() or None
    for attr in ("type", "action_type", "name"):
        val = getattr(action, attr, None)
        if val is not None:
            raw = getattr(val, "value", val)
            return str(raw).lower()
    return None


def _action_params(action: Any) -> dict[str, Any]:
    if action is None:
        return {}
    if isinstance(action, dict):
        return dict(action.get("parameters") or action)
    params = getattr(action, "parameters", None)
    if isinstance(params, dict):
        return dict(params)
    return {}


def _xy_from_action(action: Any) -> tuple[float, float] | None:
    params = _action_params(action)
    meta = getattr(action, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    for source in (params, meta, meta.get("target_validation") or {}):
        if not isinstance(source, dict):
            continue
        if source.get("x") is not None and source.get("y") is not None:
            try:
                return float(source["x"]), float(source["y"])
            except (TypeError, ValueError):
                pass
        center = source.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            try:
                return float(center[0]), float(center[1])
            except (TypeError, ValueError):
                pass
    return None


def run_offline_case(
    *,
    case: dict[str, Any],
    runtime: VariantRuntime,
    experiment_id: str,
    repeat_index: int = 0,
    return_action: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], Any]:
    """Plan one offline fixture case.

    When ``return_action=True``, also return the live Action object so
    single-step can execute without calling the VLM a second time.
    """

    base = {
        "experiment_id": experiment_id,
        "test_layer": "offline",
        "variant": runtime.variant.name,
        "prompt_version": runtime.prompt_version,
        "base_model": runtime.base_model,
        "adapter_path": runtime.adapter_path,
        "adapter_loaded": runtime.adapter_loaded,
        "task_id": case.get("task_id"),
        "case_id": case.get("case_id"),
        "repeat_index": repeat_index,
        "record_id": uuid.uuid4().hex,
    }
    if runtime.status != "ready" or runtime.planner is None:
        blocked = {
            **base,
            "status": "blocked",
            "error_type": "variant_blocked",
            "error_message": runtime.blocked_reason,
            "json_valid": None,
            "schema_valid": None,
            "action_correct": None,
            "target_correct": None,
            "click_hit": None,
            "task_success": None,
            "latency_seconds": None,
            "total_tokens": None,
        }
        return (blocked, None) if return_action else blocked

    state = state_from_offline_case(case)
    t0 = time.perf_counter()
    try:
        result = runtime.planner.plan(state)
        latency = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001 — record real failures
        failed = {
            **base,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "json_valid": False,
            "schema_valid": False,
            "action_correct": None,
            "target_correct": None,
            "click_hit": None,
            "task_success": None,
            "latency_seconds": time.perf_counter() - t0,
            "total_tokens": None,
        }
        return (failed, None) if return_action else failed

    usage = getattr(result, "usage", None)
    total_tokens = None
    if usage is not None:
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None and isinstance(usage, dict):
            total_tokens = usage.get("total_tokens")

    pred_type = _action_type(result.action)
    if result.decision and str(result.decision.value if hasattr(result.decision, "value") else result.decision).lower() in {
        "finish",
        "retry",
        "fail",
    }:
        pred_type = str(
            result.decision.value if hasattr(result.decision, "value") else result.decision
        ).lower()

    params = _action_params(result.action)
    pred_target = params.get("target_text") or (result.metadata or {}).get("target_text")
    expected_types = [str(x).lower() for x in (case.get("expected_action_types") or [])]
    expected_target = case.get("expected_target_text")
    should_finish = bool(case.get("should_finish"))

    json_valid = result.parsed_output is not None or result.raw_output is not None
    schema_valid = result.status.value == "success" if hasattr(result.status, "value") else result.error is None

    action_correct = None
    if expected_types:
        action_correct = pred_type in expected_types if pred_type else False

    target_correct = None
    if expected_target is not None:
        target_correct = (
            str(pred_target or "").strip().lower() == str(expected_target).strip().lower()
        )

    click_hit = None
    xy = _xy_from_action(result.action)
    bbox = case.get("target_bbox")
    if bbox is None:
        click_hit = None  # N/A — no annotation
    elif xy is not None:
        hit = point_in_bbox(xy[0], xy[1], bbox)
        click_hit = hit if hit is not None else None
    elif pred_type in {"click", "double_click", "right_click", "middle_click"}:
        click_hit = False

    finish_correct = None
    if "should_finish" in case:
        is_finish = str(getattr(result.decision, "value", result.decision)).lower() == "finish"
        finish_correct = is_finish == should_finish

    record = {
        **base,
        "status": "success" if schema_valid else "failed",
        "json_valid": bool(json_valid),
        "schema_valid": bool(schema_valid),
        "predicted_action_type": pred_type,
        "expected_action_type": expected_types[0] if len(expected_types) == 1 else None,
        "expected_action_types": expected_types,
        "action_correct": action_correct,
        "predicted_target_text": pred_target,
        "expected_target_text": expected_target,
        "target_correct": target_correct,
        "click_hit": click_hit,
        "finish_correct": finish_correct,
        "decision": str(getattr(result.decision, "value", result.decision)),
        "reason": result.reason,
        "latency_seconds": latency,
        "total_tokens": total_tokens,
        "raw_output": result.raw_output,
        "parsed_output": result.parsed_output,
        "error_type": None if schema_valid else "planner_error",
        "error_message": None if result.error is None else str(result.error),
        "error_class": (
            getattr(result.error, "error_class", None)
            if result.error is not None
            else ("empty_ocr" if str(getattr(result.decision, "value", result.decision)).lower() == "retry" and not (case.get("ocr_text") or case.get("gui_elements")) else None)
        ),
        "planner_retry_count": (
            1 if str(getattr(result.decision, "value", result.decision)).lower() == "retry" else 0
        ),
        "repeated_action_count": 0,
        "recoverable_failure": str(getattr(result.decision, "value", result.decision)).lower() == "retry",
        "task_success": None,
        "notes": "click_hit=N/A when target_bbox missing",
    }
    if return_action:
        return record, getattr(result, "action", None)
    return record
