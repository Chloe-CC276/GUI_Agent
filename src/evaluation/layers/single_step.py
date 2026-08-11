"""Layer 2: plan once → execute exactly one action → stop."""

from __future__ import annotations

import time
import uuid
from typing import Any

from src.evaluation.layers.offline import _action_type, run_offline_case
from src.evaluation.variants import VariantRuntime
from src.executor.executor import Executor


def run_single_step_case(
    *,
    case: dict[str, Any],
    runtime: VariantRuntime,
    experiment_id: str,
    dry_run: bool = True,
    countdown_seconds: int = 0,
    execute_real_actions: bool = False,
    repeat_index: int = 0,
) -> dict[str, Any]:
    # Plan exactly once — a second VLM call is non-deterministic and used to
    # wipe a correct offline plan (e.g. open_file element_id-only).
    plan_record, action = run_offline_case(
        case=case,
        runtime=runtime,
        experiment_id=experiment_id,
        repeat_index=repeat_index,
        return_action=True,
    )
    plan_record["test_layer"] = "single_step"
    plan_record["dry_run"] = dry_run
    plan_record["execute_real_actions"] = bool(execute_real_actions and not dry_run)

    if plan_record.get("status") == "blocked":
        return plan_record

    if runtime.planner is None:
        return plan_record

    if plan_record.get("decision") != "act":
        plan_record["status"] = plan_record.get("status") or "success"
        plan_record["executed"] = False
        plan_record["notes"] = "no act decision — execution skipped"
        return plan_record

    if action is None:
        plan_record["executed"] = False
        plan_record["error_type"] = "no_action"
        plan_record["status"] = "failed"
        return plan_record

    if execute_real_actions and not dry_run:
        if countdown_seconds > 0:
            time.sleep(float(countdown_seconds))
    else:
        dry_run = True

    executor = Executor(dry_run=dry_run, raise_on_error=False)
    t0 = time.perf_counter()
    try:
        exec_result = executor.execute(action)
        exec_latency = time.perf_counter() - t0
        ok = bool(
            getattr(exec_result, "success", False)
            or getattr(exec_result, "ok", False)
        )
        if hasattr(exec_result, "status"):
            status_val = getattr(exec_result.status, "value", exec_result.status)
            ok = ok or str(status_val).lower() == "success"
        plan_record["executed"] = True
        plan_record["exec_ok"] = ok
        plan_record["exec_message"] = str(
            getattr(exec_result, "message", "")
            or getattr(exec_result, "error", "")
        )
        plan_record["exec_latency_seconds"] = exec_latency
        plan_record["ui_changed"] = None  # no second observation in fixture mode
        plan_record["verify_success"] = None
        plan_record["misoperation"] = None if ok else True
        # Keep planner outcome as primary status; exec failure marks misoperation.
        if plan_record.get("status") == "success" and not ok:
            plan_record["status"] = "failed"
        plan_record["single_step_only"] = True
        plan_record["predicted_action_type"] = _action_type(action) or plan_record.get(
            "predicted_action_type"
        )
    except Exception as exc:  # noqa: BLE001
        plan_record["executed"] = True
        plan_record["exec_ok"] = False
        plan_record["error_type"] = type(exc).__name__
        plan_record["error_message"] = str(exc)
        plan_record["status"] = "failed"
        plan_record["misoperation"] = True

    plan_record["record_id"] = uuid.uuid4().hex
    return plan_record
