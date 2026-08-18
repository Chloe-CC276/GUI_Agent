"""Build evaluation_report.md and summary JSON from recorded results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_results import OVERALL_FIELDS, write_csv
from .metrics import (
    aggregate_records,
    decrease_rate,
    format_metric,
    relative_lift,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def collect_records(output_dir: Path) -> list[dict[str, Any]]:
    raw = output_dir / "raw"
    records: list[dict[str, Any]] = []
    for name in ("offline_results.jsonl", "single_step_results.jsonl", "e2e_results.jsonl"):
        records.extend(load_jsonl(raw / name))
    return records


def build_tables(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall: list[dict[str, Any]] = []
    by_task: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    task_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        layer = str(row.get("test_layer") or "unknown")
        variant = str(row.get("variant") or "unknown")
        groups[(layer, variant)].append(row)
        task_groups[(layer, variant, str(row.get("task_id") or "unknown"))].append(row)

    for (layer, variant), rows in sorted(groups.items()):
        metrics = aggregate_records(rows)
        notes = []
        if any(r.get("status") == "blocked" for r in rows):
            notes.append("contains_blocked")
        n_fixture = sum(1 for r in rows if "fixture_sim" in str(r.get("notes") or ""))
        n_live = sum(1 for r in rows if "live AgentChain" in str(r.get("notes") or ""))
        if n_fixture:
            notes.append(f"fixture_sim={n_fixture}")
        if n_live:
            notes.append(f"live={n_live}")
        if any(r.get("adapter_loaded") is False and "adapter" in variant for r in rows):
            notes.append("adapter_not_loaded")
        overall.append(
            {
                "test_layer": layer,
                "variant": variant,
                **metrics,
                "notes": ",".join(notes) if notes else "",
            }
        )

    for (layer, variant, task_id), rows in sorted(task_groups.items()):
        metrics = aggregate_records(rows)
        by_task.append(
            {
                "test_layer": layer,
                "variant": variant,
                "task_id": task_id,
                **metrics,
            }
        )
    return overall, by_task


def _find(overall: Sequence[Mapping[str, Any]], layer: str, variant: str) -> Mapping[str, Any] | None:
    for row in overall:
        if row.get("test_layer") == layer and row.get("variant") == variant:
            return row
    return None


def lift_table(overall: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layers = sorted({str(r.get("test_layer")) for r in overall})
    for layer in layers:
        base = _find(overall, layer, "baseline")
        opt = _find(overall, layer, "optimized_prompt")
        lora = _find(overall, layer, "adapter_v1_original")
        ada = _find(overall, layer, "adapter_v1_optimized")
        for metric, lower_better in (
            ("task_success_rate", False),
            ("action_accuracy", False),
            ("json_valid_rate", False),
            ("misoperation_rate", True),
            ("retry_rate", True),
            ("timeout_rate", True),
            ("auto_recovery_rate", False),
            ("avg_latency_seconds", True),
        ):
            b = None if base is None else base.get(metric)
            o = None if opt is None else opt.get(metric)
            l = None if lora is None else lora.get(metric)
            a = None if ada is None else ada.get(metric)
            if lower_better:
                opt_delta = decrease_rate(b if isinstance(b, (int, float)) else None, o if isinstance(o, (int, float)) else None)
                lora_vs_base = decrease_rate(b if isinstance(b, (int, float)) else None, l if isinstance(l, (int, float)) else None)
                ada_vs_base = decrease_rate(b if isinstance(b, (int, float)) else None, a if isinstance(a, (int, float)) else None)
                ada_vs_opt = decrease_rate(o if isinstance(o, (int, float)) else None, a if isinstance(a, (int, float)) else None)
                ada_vs_lora = decrease_rate(l if isinstance(l, (int, float)) else None, a if isinstance(a, (int, float)) else None)
            else:
                opt_delta = relative_lift(b if isinstance(b, (int, float)) else None, o if isinstance(o, (int, float)) else None)
                lora_vs_base = relative_lift(b if isinstance(b, (int, float)) else None, l if isinstance(l, (int, float)) else None)
                ada_vs_base = relative_lift(b if isinstance(b, (int, float)) else None, a if isinstance(a, (int, float)) else None)
                ada_vs_opt = relative_lift(o if isinstance(o, (int, float)) else None, a if isinstance(a, (int, float)) else None)
                ada_vs_lora = relative_lift(l if isinstance(l, (int, float)) else None, a if isinstance(a, (int, float)) else None)
            rows.append(
                {
                    "test_layer": layer,
                    "metric": metric,
                    "baseline": format_metric(b),
                    "optimized_prompt": format_metric(o),
                    "adapter_v1_original": format_metric(l),
                    "adapter_v1_optimized": format_metric(a),
                    "opt_vs_base_relative": format_metric(opt_delta),
                    "lora_vs_base_relative": format_metric(lora_vs_base),
                    "adapter_vs_base_relative": format_metric(ada_vs_base),
                    "adapter_vs_opt_relative": format_metric(ada_vs_opt),
                    "adapter_vs_lora_relative": format_metric(ada_vs_lora),
                }
            )
    return rows


def write_report(
    output_dir: Path,
    *,
    adapter_status: Mapping[str, Any] | None = None,
    blocked: Sequence[str] | None = None,
) -> Path:
    records = collect_records(output_dir)
    overall, by_task = build_tables(records)
    lifts = lift_table(overall)

    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "overall_comparison.csv", overall, OVERALL_FIELDS)
    task_fields = ["test_layer", "variant", "task_id"] + [
        f for f in OVERALL_FIELDS if f not in {"test_layer", "variant", "notes"}
    ]
    write_csv(tables / "task_comparison.csv", by_task, task_fields)
    write_csv(
        tables / "detailed_results.csv",
        list(records),
        sorted({k for row in records for k in row.keys()}),
    )
    write_csv(
        tables / "lift_comparison.csv",
        lifts,
        [
            "test_layer",
            "metric",
            "baseline",
            "optimized_prompt",
            "adapter_v1_original",
            "adapter_v1_optimized",
            "opt_vs_base_relative",
            "lora_vs_base_relative",
            "adapter_vs_base_relative",
            "adapter_vs_opt_relative",
            "adapter_vs_lora_relative",
        ],
    )

    errors: dict[str, int] = defaultdict(int)
    for row in records:
        if row.get("error_type"):
            errors[str(row["error_type"])] += 1
    write_csv(
        tables / "error_analysis.csv",
        [{"error_type": k, "count": v} for k, v in sorted(errors.items())],
        ["error_type", "count"],
    )

    summary = {
        "num_records": len(records),
        "overall": overall,
        "lifts": lifts,
        "adapter_status": adapter_status,
        "blocked_notes": list(blocked or []),
        "disclaimer": (
            "Metrics are computed only from recorded runs. "
            "None/N/A means missing data, not zero. "
            "Blocked variants are not treated as Adapter successes."
        ),
    }
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    md = _render_markdown(summary, records)
    md_path = report_dir / "evaluation_report.md"
    md_path.write_text(md, encoding="utf-8")
    return md_path


def _render_markdown(summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# GUI Agent Evaluation Report",
        "",
        "## 1. Test goal",
        "Compare baseline / optimized prompt / LoRA-only / LoRA+optimized across five GUI tasks and three layers.",
        "",
        "## 2. Disclaimer",
        str(summary.get("disclaimer")),
        "",
        "## 3. Adapter status",
        "```json",
        json.dumps(summary.get("adapter_status"), ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## 4. Blocked / not run",
    ]
    blocked = summary.get("blocked_notes") or []
    if blocked:
        lines.extend(f"- {item}" for item in blocked)
    else:
        lines.append("- (none recorded)")
    lines.extend(["", "## 5. Overall comparison", ""])
    lines.append("| layer | variant | n | json_ok | action_acc | click_hit | task_success | notes |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in summary.get("overall") or []:
        lines.append(
            "| {test_layer} | {variant} | {sample_count} | {json_valid_rate} | {action_accuracy} | "
            "{click_hit_rate} | {task_success_rate} | {notes} |".format(
                test_layer=row.get("test_layer"),
                variant=row.get("variant"),
                sample_count=row.get("sample_count"),
                json_valid_rate=format_metric(row.get("json_valid_rate")),
                action_accuracy=format_metric(row.get("action_accuracy")),
                click_hit_rate=format_metric(row.get("click_hit_rate")),
                task_success_rate=format_metric(row.get("task_success_rate")),
                notes=row.get("notes") or "",
            )
        )
    lines.extend(["", "## 5.1 Robustness", ""])
    lines.append(
        "| layer | variant | n | misop | retry | repeated | timeout | auto_recovery | error_classes |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in summary.get("overall") or []:
        classes = row.get("error_class_counts") or {}
        class_text = ",".join(f"{k}:{v}" for k, v in classes.items()) if classes else ""
        lines.append(
            "| {test_layer} | {variant} | {sample_count} | {misoperation_rate} | "
            "{retry_rate} | {repeated_action_rate} | {timeout_rate} | "
            "{auto_recovery_rate} | {error_classes} |".format(
                test_layer=row.get("test_layer"),
                variant=row.get("variant"),
                sample_count=row.get("sample_count"),
                misoperation_rate=format_metric(row.get("misoperation_rate")),
                retry_rate=format_metric(row.get("retry_rate")),
                repeated_action_rate=format_metric(row.get("repeated_action_rate")),
                timeout_rate=format_metric(row.get("timeout_rate")),
                auto_recovery_rate=format_metric(row.get("auto_recovery_rate")),
                error_classes=class_text,
            )
        )
    lines.extend(["", "## 6. Relative lifts", ""])
    lines.append(
        "| layer | metric | baseline | optimized | lora_only | lora+opt | "
        "opt/base | lora/base | ada/base | ada/opt | ada/lora |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary.get("lifts") or []:
        lines.append(
            "| {test_layer} | {metric} | {baseline} | {optimized_prompt} | {adapter_v1_original} | "
            "{adapter_v1_optimized} | {opt_vs_base_relative} | {lora_vs_base_relative} | "
            "{adapter_vs_base_relative} | {adapter_vs_opt_relative} | {adapter_vs_lora_relative} |".format(
                test_layer=row.get("test_layer"),
                metric=row.get("metric"),
                baseline=format_metric(row.get("baseline")),
                optimized_prompt=format_metric(row.get("optimized_prompt")),
                adapter_v1_original=format_metric(row.get("adapter_v1_original")),
                adapter_v1_optimized=format_metric(row.get("adapter_v1_optimized")),
                opt_vs_base_relative=format_metric(row.get("opt_vs_base_relative")),
                lora_vs_base_relative=format_metric(row.get("lora_vs_base_relative")),
                adapter_vs_base_relative=format_metric(row.get("adapter_vs_base_relative")),
                adapter_vs_opt_relative=format_metric(row.get("adapter_vs_opt_relative")),
                adapter_vs_lora_relative=format_metric(row.get("adapter_vs_lora_relative")),
            )
        )

    # Typical cases
    blocked_recs = [r for r in records if r.get("status") == "blocked"][:5]
    failed = [r for r in records if r.get("status") == "failed"][:5]
    success = [r for r in records if r.get("status") == "success"][:5]
    lines.extend(["", "## 7. Example successes", ""])
    for row in success:
        lines.append(
            f"- {row.get('test_layer')}/{row.get('variant')}/{row.get('task_id')}: "
            f"{row.get('predicted_action_type') or row.get('decision')}"
        )
    if not success:
        lines.append("- (none)")
    lines.extend(["", "## 8. Example failures", ""])
    for row in failed:
        lines.append(
            f"- {row.get('test_layer')}/{row.get('variant')}/{row.get('case_id')}: "
            f"{row.get('error_type')} — {row.get('error_message')}"
        )
    if not failed:
        lines.append("- (none)")
    lines.extend(["", "## 9. Blocked examples", ""])
    for row in blocked_recs:
        lines.append(
            f"- {row.get('variant')}: {row.get('error_message')}"
        )
    if not blocked_recs:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## 10. Limitations",
            "- Cloud VLM and live desktop runs require explicit flags.",
            "- Local adapters_v1 is not loaded without approval (laptop GPU risk).",
            "- click_hit is N/A when fixtures lack target_bbox.",
            "- fixture_sim_e2e is not equivalent to live Observe→Plan→Execute.",
            "",
            "## 11. Next steps",
            "- Approve cloud VLM for real offline/single-step numbers.",
            "- Approve live desktop dry-run e2e on a prepared test desktop.",
            "- Approve CSF3/GPU host for adapter_v1_optimized.",
            "",
        ]
    )
    return "\n".join(lines)
