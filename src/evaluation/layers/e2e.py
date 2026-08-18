"""Layer 3: end-to-end AgentChain (live desktop requires explicit approval)."""

from __future__ import annotations

import time
import uuid
from typing import Any

from src.evaluation.fixtures import state_from_offline_case
from src.evaluation.variants import VariantRuntime


def run_e2e_task(
    *,
    task: dict[str, Any],
    runtime: VariantRuntime,
    experiment_id: str,
    max_steps: int,
    timeout_seconds: int,
    dry_run: bool = True,
    execute_real_actions: bool = False,
    allow_live_desktop: bool = False,
    fixture_case: dict[str, Any] | None = None,
    repeat_index: int = 0,
) -> dict[str, Any]:
    base = {
        "experiment_id": experiment_id,
        "test_layer": "e2e",
        "variant": runtime.variant.name,
        "prompt_version": runtime.prompt_version,
        "base_model": runtime.base_model,
        "adapter_path": runtime.adapter_path,
        "adapter_loaded": runtime.adapter_loaded,
        "task_id": task.get("task_id"),
        "case_id": task.get("task_id"),
        "repeat_index": repeat_index,
        "record_id": uuid.uuid4().hex,
        "dry_run": dry_run,
        "instruction": task.get("instruction"),
        "max_steps": max_steps,
    }

    if runtime.status != "ready" or runtime.planner is None:
        return {
            **base,
            "status": "blocked",
            "error_type": "variant_blocked",
            "error_message": runtime.blocked_reason,
            "task_success": None,
            "total_steps": None,
            "json_valid": None,
            "schema_valid": None,
        }

    if not allow_live_desktop:
        # Safe default: fixture simulation loop (same observation each step) OR block.
        if fixture_case is None:
            return {
                **base,
                "status": "blocked",
                "error_type": "live_desktop_not_approved",
                "error_message": (
                    "e2e live desktop not approved. "
                    "Re-run with --allow-live-desktop after consent, "
                    "or provide offline fixture_case for simulated e2e."
                ),
                "task_success": None,
                "total_steps": None,
                "json_valid": None,
                "schema_valid": None,
            }
        return _fixture_e2e_loop(
            base=base,
            fixture_case=fixture_case,
            runtime=runtime,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )

    if execute_real_actions and not dry_run:
        # Caller must have already obtained operator consent.
        pass

    # Live AgentChain path (operator-approved).
    try:
        from src.agent.agent_chain import AgentChainConfig, create_agent_chain
        from src.agent.memory import AgentMemory
        from src.agent.tools import AgentTools
        from src.executor.executor import Executor
        from src.perception.perception_pipeline import PerceptionPipeline

        perception = PerceptionPipeline()
        executor = Executor(
            dry_run=dry_run or not execute_real_actions,
            raise_on_error=False,
        )
        tools = AgentTools(perception=perception, executor=executor)
        memory = AgentMemory()
        chain = create_agent_chain(
            planner=runtime.planner,
            tools=tools,
            vlm=runtime.vlm,
            memory=memory,
            config=AgentChainConfig(
                max_chain_iterations=max_steps,
                max_planner_retries=0,
                max_model_repairs=1,
                max_reflections=1,
                synthetic_verify_on_dry_run=bool(dry_run or not execute_real_actions),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "status": "blocked",
            "error_type": "chain_setup_failed",
            "error_message": str(exc),
            "task_success": None,
        }

    from src.agent.state import AgentState

    state = AgentState.create(
        task=str(task.get("instruction") or task.get("task_id")),
        max_steps=max_steps,
        metadata={
            "dry_run": bool(dry_run or not execute_real_actions),
            "evaluation": True,
        },
    )
    t0 = time.perf_counter()
    try:
        final = chain.invoke(state)
        latency = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "task_success": False,
            "latency_seconds": time.perf_counter() - t0,
            "total_steps": getattr(state, "step_index", None),
        }

    # AgentChain.invoke returns AgentRunResult (not AgentState).
    run_result = final
    if hasattr(final, "to_run_result"):
        run_result = final.to_run_result()

    if hasattr(run_result, "succeeded"):
        success = bool(run_result.succeeded)
    elif hasattr(final, "is_finished"):
        success = bool(final.is_finished)
    else:
        success = str(getattr(run_result, "status", "")).lower() in {
            "success",
            "ResultStatus.SUCCESS",
        }

    if hasattr(run_result, "step_count"):
        total_steps = int(run_result.step_count)
    elif hasattr(final, "step_index"):
        total_steps = int(final.step_index)
    else:
        total_steps = len(getattr(run_result, "steps", None) or [])

    err = getattr(run_result, "error", None) or getattr(final, "error", None)
    err_type = getattr(err, "error_type", None) if err is not None else None
    err_msg = getattr(err, "message", None) if err is not None else None
    if err_msg is None:
        err_msg = getattr(run_result, "final_message", None)

    # Dry-run cannot visually complete desktop tasks; report plan/exec progress honestly.
    notes = ["live AgentChain run"]
    if dry_run or not execute_real_actions:
        notes.append("dry_run_synthetic_verify")
        if not success and total_steps:
            notes.append("task_success=false expected unless planner emits finish")

    from src.agent.robustness import robustness_fields_from_run

    robustness = robustness_fields_from_run(run_result)
    error_class = robustness.get("error_class") or getattr(err, "error_class", None)

    return {
        **base,
        "status": "success" if success else "failed",
        "task_success": success,
        "total_steps": total_steps,
        "latency_seconds": latency,
        "timeout": latency >= timeout_seconds,
        "json_valid": None,
        "schema_valid": None,
        "notes": ";".join(notes),
        "error_type": None if success else err_type,
        "error_message": None if success else err_msg,
        "error_class": error_class,
        **robustness,
    }


def _fixture_e2e_loop(
    *,
    base: dict[str, Any],
    fixture_case: dict[str, Any],
    runtime: VariantRuntime,
    max_steps: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Simulated e2e: reuse one fixture observation; no live screen capture.

    Also records single-step-comparable fields from the first planner step:
    schema_valid / action_correct / misoperation (N/A — no real Executor).
    """

    from src.evaluation.layers.offline import _action_type

    assert runtime.planner is not None
    t0 = time.perf_counter()
    steps = 0
    invalid_json = 0
    decisions: list[str] = []
    finished = False
    error_type = None
    error_message = None
    first_schema_valid: bool | None = None
    first_json_valid: bool | None = None
    first_action_type: str | None = None
    first_action_correct: bool | None = None
    expected_types = [
        str(x).lower() for x in (fixture_case.get("expected_action_types") or [])
    ]

    while steps < max_steps and (time.perf_counter() - t0) < timeout_seconds:
        state = state_from_offline_case(fixture_case)
        try:
            result = runtime.planner.plan(state)
        except Exception as exc:  # noqa: BLE001
            invalid_json += 1
            error_type = type(exc).__name__
            error_message = str(exc)
            if first_schema_valid is None:
                first_schema_valid = False
                first_json_valid = False
                first_action_correct = False
            break
        steps += 1
        decision = str(getattr(result.decision, "value", result.decision)).lower()
        decisions.append(decision)
        schema_ok = (
            result.status.value == "success"
            if hasattr(result.status, "value")
            else result.error is None
        )
        json_ok = result.parsed_output is not None or result.raw_output is not None
        if not json_ok:
            invalid_json += 1
        if first_schema_valid is None:
            first_schema_valid = bool(schema_ok)
            first_json_valid = bool(json_ok)
            pred = _action_type(result.action)
            if decision in {"finish", "fail", "retry"}:
                pred = decision
            first_action_type = pred
            if expected_types:
                first_action_correct = bool(pred and pred in expected_types)
        if decision in {"finish", "fail"}:
            finished = decision == "finish"
            break
        # Fixture observation never changes — one correct act is enough progress.
        if decision == "act" and first_action_correct is True:
            finished = True
            break

    latency = time.perf_counter() - t0
    # Prefer first-step schema; fall back to loop-level invalid_json.
    schema_valid = (
        first_schema_valid
        if first_schema_valid is not None
        else (invalid_json == 0 if steps else None)
    )
    return {
        **base,
        "status": "success" if finished else "failed",
        "task_success": finished,
        "total_steps": steps,
        "latency_seconds": latency,
        "timeout": latency >= timeout_seconds,
        "invalid_json_count": invalid_json,
        "decisions": decisions,
        "error_type": error_type,
        "error_message": error_message,
        "json_valid": first_json_valid if first_json_valid is not None else (
            invalid_json == 0 if steps else None
        ),
        "schema_valid": schema_valid,
        "predicted_action_type": first_action_type,
        "expected_action_types": expected_types,
        "action_correct": first_action_correct,
        # No real OS clicks in fixture-sim → misoperation is N/A
        "misoperation": None,
        "planner_retry_count": sum(1 for item in decisions if item == "retry"),
        "repeated_action_count": max(0, len(decisions) - len(set(decisions))),
        "error_class": error_type,
        "recoverable_failure": "retry" in decisions,
        "notes": (
            "fixture_sim_e2e (not live desktop); "
            "task_success if finish OR first act matches expected_action_types"
        ),
    }
