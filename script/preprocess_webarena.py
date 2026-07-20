from __future__ import annotations

from pathlib import Path

from src.datasets.webarena_loader import WebArenaLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "external"
    / "WebArena"
    /"config_files"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "webarena"
)


def preprocess_webarena() -> dict[str, Path]:
    loader = WebArenaLoader(
        source=SOURCE_ROOT,
        strict=False,
        resolve_start_url=False,
        include_raw_record=False,
    )

    tasks_path = loader.export_jsonl(
        OUTPUT_ROOT / "tasks.jsonl"
    )

    configs_path = loader.export_configs_jsonl(
        OUTPUT_ROOT / "task_configs.jsonl"
    )

    print(
        f"[WebArena] {len(loader)} 个任务 "
        f"-> {tasks_path}"
    )

    return {
        "tasks": tasks_path,
        "configs": configs_path,
    }


def main() -> None:
    preprocess_webarena()


if __name__ == "__main__":
    main()