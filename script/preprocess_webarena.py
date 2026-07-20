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


def main() -> None:
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

    print("任务数量：", len(loader))
    print("统一任务格式：", tasks_path)
    print("WebArena配置格式：", configs_path)
    print("统计结果：", loader.statistics().to_dict())


if __name__ == "__main__":
    main()