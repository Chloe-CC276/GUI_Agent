"""
Stratified ScreenAgent split (v1).

- Labels: app, length, interact (monitor only)
- Stratum key: app#length
- Split: task-level 70% / 15% / 15% within each stratum
- Keeps existing random CSV as v0 baseline (does not overwrite)

Outputs under:
  data/processed/screenagent/split_stratified_v1/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_DIR = PROJECT_ROOT / "data" / "processed" / "screenagent"
DEFAULT_OUT_DIR = DEFAULT_CSV_DIR / "split_stratified_v1"
V0_FILES = (
    "screenagent_processed_train.csv",
    "screenagent_processed_validation.csv",
    "screenagent_processed_test.csv",
)

APP_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("gimp", re.compile(r"\bGIMP\b|gimp|图层|选区|画笔|涂鸦|图像大小|打印大小|油漆", re.I)),
    (
        "office_calc",
        re.compile(
            r"\bCalc\b|电子表格|单元格|公式|spreadsheet|"
            r"创建\s*\d+\s*\*\s*\d+.*表|表格的第|在表格",
            re.I,
        ),
    ),
    (
        "office_writer",
        re.compile(
            r"\bWriter\b|\bWord\b|文档中的|段落|居中对齐|Liberation|"
            r"文本颜色|字体大小|打开的word|文本文档",
            re.I,
        ),
    ),
    (
        "ide",
        re.compile(
            r"VS Code|Visual Studio Code|vscode|FastAPI|"
            r"根据.*报错|修复代码|python脚本|编写.*代码",
            re.I,
        ),
    ),
    (
        "terminal",
        re.compile(r"终端|Terminal|命令行|\bshell\b|\bbash\b", re.I),
    ),
    (
        "file_manager",
        re.compile(r"文件管理|文件夹|Nautilus|复制文件|移动文件|创建文件夹", re.I),
    ),
    (
        "settings",
        re.compile(r"系统设置|设置中心|\bSettings\b|外观设置|网络设置", re.I),
    ),
    (
        "browser",
        re.compile(
            r"浏览器|\bChrome\b|\bFirefox\b|网页|https?://|\bbing\b|百度|"
            r"购物车|ChatGPT|YouTube|维基百科|\bwiki\b|打开的网页",
            re.I,
        ),
    ),
]


CLICK_ACTIONS = {"click", "double_click", "right_click"}
TYPE_ACTIONS = {"type_text", "press", "hotkey"}
COMPLEX_ACTIONS = {"drag_to", "scroll", "move_to"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def infer_app(text: str) -> str:
    for name, pattern in APP_RULES:
        if pattern.search(text):
            return name
    return "other"


def infer_length(n_steps: int) -> str:
    if n_steps <= 4:
        return "short"
    if n_steps <= 9:
        return "mid"
    return "long"


def infer_interact(action_counts: Counter[str]) -> str:
    total = sum(action_counts.values()) or 1
    click_ratio = sum(action_counts[a] for a in CLICK_ACTIONS) / total
    type_ratio = sum(action_counts[a] for a in TYPE_ACTIONS) / total
    complex_n = sum(action_counts[a] for a in COMPLEX_ACTIONS)
    if complex_n > 0 and (action_counts["drag_to"] + action_counts["scroll"]) > 0:
        return "pointing_complex"
    if action_counts["move_to"] / total >= 0.25:
        return "pointing_complex"
    if type_ratio >= 0.40:
        return "type_heavy"
    if click_ratio >= 0.60:
        return "click_heavy"
    return "mixed"


def load_all_rows(csv_dir: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Load v0 CSVs; return rows and task_id -> old_split."""
    all_rows: list[dict[str, str]] = []
    old_split: dict[str, str] = {}
    for split, filename in zip(
        ("train", "validation", "test"),
        V0_FILES,
    ):
        path = csv_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_csv_rows(path)
        for row in rows:
            tid = row["task_id"]
            old_split[tid] = split
            all_rows.append(row)
    return all_rows, old_split


def build_task_table(
    rows: list[dict[str, str]],
    old_split: dict[str, str],
) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for row in rows:
        tid = row["task_id"]
        if tid not in tasks:
            meta_raw = row.get("task_metadata_json") or ""
            try:
                meta = json.loads(meta_raw) if meta_raw else {}
            except json.JSONDecodeError:
                meta = {}
            prompt_en = str(meta.get("task_prompt_en") or "")
            prompt_zh = str(meta.get("task_prompt_zh") or row.get("task_instruction") or "")
            text = " ".join(
                [
                    row.get("task_instruction") or "",
                    prompt_zh,
                    prompt_en,
                    (row.get("step_instruction") or "")[:300],
                ]
            )
            tasks[tid] = {
                "task_id": tid,
                "instruction": (row.get("task_instruction") or "").strip(),
                "prompt_en": prompt_en,
                "prompt_zh": prompt_zh,
                "n_steps": 0,
                "actions": Counter(),
                "old_split": old_split.get(tid, ""),
                "text_for_app": text,
                "rows": [],
            }
        tasks[tid]["n_steps"] += 1
        tasks[tid]["actions"][row.get("action_type") or "?"] += 1
        tasks[tid]["rows"].append(row)

    for task in tasks.values():
        app = infer_app(task["text_for_app"])
        length = infer_length(int(task["n_steps"]))
        interact = infer_interact(task["actions"])
        task["app"] = app
        task["length"] = length
        task["interact"] = interact
        task["stratum"] = f"{app}#{length}"
    return tasks


def quota_for_n(n: int) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) summing to n under ~70/15/15."""
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    if n == 3:
        return 2, 1, 0
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    # Prefer giving remainder to test/val when int() under-allocates.
    if n_test < 0:
        n_test = 0
    # Ensure for reasonably large strata both val and test get >=1 when possible.
    if n >= 7:
        if n_val == 0:
            n_val = 1
            n_train -= 1
        if n_test == 0:
            n_test = 1
            n_train -= 1
    return n_train, n_val, n_test


def stratified_split(
    tasks: dict[str, dict[str, Any]],
    seed: int = 42,
) -> dict[str, str]:
    """Return task_id -> split."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for tid, task in tasks.items():
        by_stratum[task["stratum"]].append(tid)

    assignment: dict[str, str] = {}
    for stratum, tids in sorted(by_stratum.items()):
        tids = list(tids)
        rng.shuffle(tids)
        n_train, n_val, n_test = quota_for_n(len(tids))
        parts = (
            ("train", tids[:n_train]),
            ("validation", tids[n_train : n_train + n_val]),
            ("test", tids[n_train + n_val :]),
        )
        assert n_train + n_val + n_test == len(tids)
        for split_name, chunk in parts:
            for tid in chunk:
                assignment[tid] = split_name
    return assignment


def distribution(
    tasks: dict[str, dict[str, Any]],
    assignment: dict[str, str],
    key: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "validation", "test", "all"):
        counter: Counter[str] = Counter()
        for tid, task in tasks.items():
            if split != "all" and assignment[tid] != split:
                continue
            counter[str(task[key])] += 1
        total = sum(counter.values()) or 1
        out[split] = {
            "counts": dict(sorted(counter.items())),
            "ratios": {k: round(v / total, 4) for k, v in sorted(counter.items())},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, old_split = load_all_rows(args.csv_dir)
    tasks = build_task_table(rows, old_split)
    assignment = stratified_split(tasks, seed=args.seed)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) task labels
    label_path = out_dir / "task_labels.csv"
    label_fields = [
        "task_id",
        "app",
        "length",
        "interact",
        "stratum",
        "n_steps",
        "old_split_v0",
        "new_split_v1",
        "instruction",
        "prompt_en",
    ]
    label_rows = []
    for tid, task in sorted(tasks.items()):
        label_rows.append(
            {
                "task_id": tid,
                "app": task["app"],
                "length": task["length"],
                "interact": task["interact"],
                "stratum": task["stratum"],
                "n_steps": task["n_steps"],
                "old_split_v0": task["old_split"],
                "new_split_v1": assignment[tid],
                "instruction": task["instruction"],
                "prompt_en": task["prompt_en"],
            }
        )
    write_csv_rows(label_path, label_rows, label_fields)

    # 2) task id lists
    for split in ("train", "validation", "test"):
        ids = sorted(tid for tid, s in assignment.items() if s == split)
        (out_dir / f"{split}_task_ids.txt").write_text(
            "\n".join(ids) + ("\n" if ids else ""),
            encoding="utf-8",
        )

    # 3) step-level CSVs
    fieldnames = list(rows[0].keys())
    split_rows: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for tid, task in tasks.items():
        split_rows[assignment[tid]].extend(task["rows"])

    for split, split_row_list in split_rows.items():
        # stable order by task_id then step_id
        split_row_list.sort(
            key=lambda r: (r.get("task_id", ""), int(r.get("step_id") or 0))
        )
        write_csv_rows(
            out_dir / f"screenagent_stratified_{split}.csv",
            split_row_list,
            fieldnames,
        )

    # 4) report
    split_counts = Counter(assignment.values())
    stratum_counts = Counter(t["stratum"] for t in tasks.values())
    moved = sum(
        1
        for tid, task in tasks.items()
        if task["old_split"] and task["old_split"] != assignment[tid]
    )
    report = {
        "version": "split_stratified_v1",
        "seed": args.seed,
        "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "stratum_key": "app#length",
        "num_tasks": len(tasks),
        "num_steps": len(rows),
        "split_task_counts": dict(split_counts),
        "split_step_counts": {
            s: len(split_rows[s]) for s in ("train", "validation", "test")
        },
        "tasks_moved_vs_v0": moved,
        "stratum_sizes": dict(sorted(stratum_counts.items())),
        "app_distribution": distribution(tasks, assignment, "app"),
        "length_distribution": distribution(tasks, assignment, "length"),
        "interact_distribution_monitor": distribution(
            tasks, assignment, "interact"
        ),
        "notes": [
            "v0 random CSVs under data/processed/screenagent/screenagent_processed_*.csv are unchanged.",
            "interact is for monitoring only and was not used in stratum sampling.",
            "Tiny strata (n<=3) follow quota_for_n special-cases; see stratum_sizes.",
        ],
    }
    report_path = out_dir / "split_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 5) README pointer
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# ScreenAgent stratified split v1",
                "",
                "- Stratum: `app#length`",
                "- Ratios: train/val/test = 70/15/15 (task-level)",
                "- `interact`: monitoring only",
                "- v0 baseline CSVs left untouched in parent folder",
                "",
                "Files:",
                "- `task_labels.csv`",
                "- `*_task_ids.txt`",
                "- `screenagent_stratified_{train,validation,test}.csv`",
                "- `split_report.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote labels -> {label_path}")
    print(f"Wrote report -> {report_path}")
    print("split_task_counts:", dict(split_counts))
    print("split_step_counts:", {s: len(split_rows[s]) for s in split_rows})
    print("tasks_moved_vs_v0:", moved)
    print("app counts:", dict(Counter(t["app"] for t in tasks.values())))
    print("length counts:", dict(Counter(t["length"] for t in tasks.values())))


if __name__ == "__main__":
    main()
