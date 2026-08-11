"""CLI entry for GUI Agent 3-layer evaluation.

Sensitive flags (require explicit operator approval):
  --allow-cloud-vlm
  --allow-local-adapter
  --allow-live-desktop
  --execute-real-actions
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.adapter_check import check_adapter
from src.evaluation.config import load_experiment_config
from src.evaluation.fixtures import load_offline_cases, load_tasks
from src.evaluation.io_results import append_jsonl, ensure_layout, write_jsonl
from src.evaluation.layers.offline import run_offline_case
from src.evaluation.report import write_report
from src.evaluation.variants import assert_original_rules_intact, build_variant_runtime

# single_step / e2e pull Executor → pyautogui; import lazily so Colab offline works.


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--layer",
        choices=["offline", "single-step", "e2e", "all"],
        default="offline",
    )
    p.add_argument(
        "--variants",
        type=str,
        default=(
            "baseline,optimized_prompt,adapter_v1_original,adapter_v1_optimized"
        ),
        help="Comma-separated variant names from config",
    )
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--no-dry-run", action="store_true", help="Disable executor dry-run")
    p.add_argument("--execute-real-actions", action="store_true")
    p.add_argument("--allow-cloud-vlm", action="store_true")
    p.add_argument("--allow-local-adapter", action="store_true")
    p.add_argument("--allow-live-desktop", action="store_true")
    p.add_argument(
        "--scripted",
        action="store_true",
        help="Use ScriptedVLM (no API). For pipeline/metrics checks only.",
    )
    p.add_argument("--repeat", type=int, default=None)
    p.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Run only one task_id (recommended for live desktop). "
        "Example: open_browser | browser_search | open_file | close_word_document | send_message",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append JSONL records instead of overwriting layer files (use when running tasks one-by-one).",
    )
    p.add_argument("--write-report", action="store_true", default=True)
    return p.parse_args(argv)


def validate(exp) -> dict:
    assert_original_rules_intact()
    adapter = check_adapter(
        next(
            (v.adapter_path for v in exp.variants if v.model_type == "local_adapter"),
            None,
        )
    )
    tasks = load_tasks(exp.tasks_path) if exp.tasks_path and exp.tasks_path.is_file() else []
    cases = (
        load_offline_cases(exp.offline_cases_path)
        if exp.offline_cases_path and exp.offline_cases_path.is_file()
        else []
    )
    return {
        "config": exp.name,
        "tasks": len(tasks),
        "offline_cases": len(cases),
        "variants": [v.name for v in exp.variants],
        "adapter": adapter.to_dict(),
        "original_prompt_intact": True,
        "output_dir": str(exp.output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    exp = load_experiment_config(args.config)
    if args.allow_cloud_vlm:
        exp.allow_cloud_vlm = True
    if args.allow_local_adapter:
        exp.allow_local_adapter = True
    if args.repeat is not None:
        exp.repeat_per_task = args.repeat

    dry_run = not args.no_dry_run
    if args.execute_real_actions and not args.allow_live_desktop:
        print(
            "REFUSED: --execute-real-actions requires --allow-live-desktop "
            "and explicit operator approval."
        )
        return 2
    if args.execute_real_actions:
        dry_run = False
        print(
            "WARNING: real mouse/keyboard execution enabled. "
            f"Countdown={exp.countdown_seconds}s per single-step action."
        )

    info = validate(exp)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if args.validate_only:
        return 0

    wanted = {x.strip() for x in args.variants.split(",") if x.strip()}
    variants = [v for v in exp.variants if v.name in wanted]
    if not variants:
        print(f"No variants matched: {wanted}")
        return 1

    paths = ensure_layout(exp.output_dir)
    experiment_id = uuid.uuid4().hex[:12]
    tasks = load_tasks(exp.tasks_path)
    cases = load_offline_cases(exp.offline_cases_path)
    if args.task_id:
        tid = args.task_id.strip()
        tasks = [t for t in tasks if str(t.get("task_id")) == tid]
        cases = [c for c in cases if str(c.get("task_id")) == tid]
        if not tasks and not cases:
            print(f"No task/case matched --task-id={tid!r}")
            return 1
        print(f"Single-task mode: task_id={tid} (tasks={len(tasks)} cases={len(cases)})")
    # Map task_id -> first fixture case for fixture e2e
    case_by_task = {}
    for case in cases:
        case_by_task.setdefault(str(case.get("task_id")), case)

    layers = (
        ["offline", "single-step", "e2e"] if args.layer == "all" else [args.layer]
    )
    blocked_notes: list[str] = []
    all_offline: list[dict] = []
    all_single: list[dict] = []
    all_e2e: list[dict] = []

    for variant in variants:
        runtime = build_variant_runtime(exp, variant, scripted=args.scripted)
        print(
            f"[variant {variant.name}] status={runtime.status} "
            f"prompt={runtime.prompt_version} adapter_loaded={runtime.adapter_loaded}"
        )
        if runtime.status != "ready":
            blocked_notes.append(f"{variant.name}: {runtime.blocked_reason}")

        if "offline" in layers:
            for case in cases:
                for rep in range(exp.repeat_per_task):
                    rec = run_offline_case(
                        case=case,
                        runtime=runtime,
                        experiment_id=experiment_id,
                        repeat_index=rep,
                    )
                    all_offline.append(rec)

        if "single-step" in layers:
            from src.evaluation.layers.single_step import run_single_step_case

            for case in cases:
                for rep in range(exp.repeat_per_task):
                    rec = run_single_step_case(
                        case=case,
                        runtime=runtime,
                        experiment_id=experiment_id,
                        dry_run=dry_run,
                        countdown_seconds=exp.countdown_seconds if not dry_run else 0,
                        execute_real_actions=bool(args.execute_real_actions),
                        repeat_index=rep,
                    )
                    all_single.append(rec)

        if "e2e" in layers:
            from src.evaluation.layers.e2e import run_e2e_task

            for task in tasks:
                fixture = case_by_task.get(str(task.get("task_id")))
                for rep in range(exp.repeat_per_task):
                    rec = run_e2e_task(
                        task=task,
                        runtime=runtime,
                        experiment_id=experiment_id,
                        max_steps=exp.max_steps,
                        timeout_seconds=exp.task_timeout_seconds,
                        dry_run=dry_run,
                        execute_real_actions=bool(args.execute_real_actions),
                        allow_live_desktop=bool(args.allow_live_desktop),
                        fixture_case=None if args.allow_live_desktop else fixture,
                        repeat_index=rep,
                    )
                    all_e2e.append(rec)

    if args.append:
        if all_offline:
            append_jsonl(paths["raw"] / "offline_results.jsonl", all_offline)
        if all_single:
            append_jsonl(paths["raw"] / "single_step_results.jsonl", all_single)
        if all_e2e:
            append_jsonl(paths["raw"] / "e2e_results.jsonl", all_e2e)
    else:
        # In single-task mode without --append, still write only this run's rows
        # into the layer files (overwrites). Use --append to accumulate tasks.
        write_jsonl(paths["raw"] / "offline_results.jsonl", all_offline)
        write_jsonl(paths["raw"] / "single_step_results.jsonl", all_single)
        write_jsonl(paths["raw"] / "e2e_results.jsonl", all_e2e)

    meta = {
        "experiment_id": experiment_id,
        "layers": layers,
        "scripted": args.scripted,
        "dry_run": dry_run,
        "allow_cloud_vlm": exp.allow_cloud_vlm,
        "allow_local_adapter": exp.allow_local_adapter,
        "allow_live_desktop": args.allow_live_desktop,
        "blocked_notes": blocked_notes,
        "adapter": info["adapter"],
    }
    (paths["root"] / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.write_report:
        report_path = write_report(
            exp.output_dir,
            adapter_status=info["adapter"],
            blocked=blocked_notes,
        )
        print(f"Wrote report {report_path}")

    print(f"experiment_id={experiment_id}")
    print(f"offline={len(all_offline)} single_step={len(all_single)} e2e={len(all_e2e)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
