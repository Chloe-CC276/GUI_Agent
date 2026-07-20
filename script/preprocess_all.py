from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from .preprocess_mind2web import (
    preprocess_mind2web,
)
from .preprocess_screenagent import (
    preprocess_screenagent,
)
from .preprocess_webarena import (
    preprocess_webarena,
)


LOGGER = logging.getLogger(__name__)


def run_preprocessor(
    name: str,
    function: Callable[[], dict[str, Path]],
) -> dict[str, Path]:
    print()
    print("=" * 80)
    print(f"开始处理：{name}")
    print("=" * 80)

    try:
        outputs = function()
    except Exception:
        LOGGER.exception("%s 预处理失败", name)
        return {}

    print(f"{name} 预处理完成")

    for output_name, output_path in outputs.items():
        print(f"  {output_name}: {output_path}")

    return outputs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    all_outputs = {
        "screenagent": run_preprocessor(
            "ScreenAgent",
            preprocess_screenagent,
        ),
        "mind2web": run_preprocessor(
            "Mind2Web",
            preprocess_mind2web,
        ),
        "webarena": run_preprocessor(
            "WebArena",
            preprocess_webarena,
        ),
    }

    print()
    print("=" * 80)
    print("全部数据集处理结果")
    print("=" * 80)

    for dataset_name, outputs in all_outputs.items():
        if not outputs:
            print(f"{dataset_name}: 失败或无输出")
            continue

        print(f"{dataset_name}:")

        for output_name, output_path in outputs.items():
            print(f"  {output_name}: {output_path}")


if __name__ == "__main__":
    main()