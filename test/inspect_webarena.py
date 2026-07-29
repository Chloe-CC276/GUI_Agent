from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBARENA_ROOT = PROJECT_ROOT / "external" / "WebArena"


def print_json_structure(value: Any, indent: int = 0) -> None:
    prefix = " " * indent

    if isinstance(value, dict):
        for key, item in value.items():
            print(f"{prefix}{key}: {type(item).__name__}")

            if isinstance(item, (dict, list)):
                print_json_structure(item, indent + 4)

    elif isinstance(value, list):
        print(f"{prefix}list length: {len(value)}")

        if value:
            print_json_structure(value[0], indent + 4)


def main() -> None:
    config_dir = WEBARENA_ROOT / "config_files"

    config_files = sorted(
        config_dir.glob("*.json"),
        key=lambda path: (
            int(path.stem)
            if path.stem.isdigit()
            else path.stem
        ),
    )

    if config_files:
        selected_path = config_files[0]

        with selected_path.open("r", encoding="utf-8") as file:
            sample = json.load(file)

        print("配置文件：", selected_path)
        print("=" * 80)
        print_json_structure(sample)
        print("=" * 80)
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        return

    raw_path = WEBARENA_ROOT / "test.raw.json"

    if not raw_path.exists():
        raise FileNotFoundError(
            "没有找到 config_files/*.json 或 test.raw.json"
        )

    with raw_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError("test.raw.json 应当是非空列表")

    print("原始任务文件：", raw_path)
    print("任务数量：", len(raw_data))
    print("=" * 80)
    print_json_structure(raw_data[0])
    print("=" * 80)
    print(json.dumps(raw_data[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()