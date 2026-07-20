from __future__ import annotations

from pathlib import Path

from src.datasets.screenagent_loader import ScreenAgentLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "external"
    / "ScreenAgent"
    / "data"
    / "ScreenAgent"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "screenagent"
)


def preprocess_screenagent() -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    for split in ("train", "test"):
        source_path = SOURCE_ROOT / split

        if not source_path.exists():
            print(f"[ScreenAgent] 跳过不存在的目录：{source_path}")
            continue

        loader = ScreenAgentLoader(
            data_root=source_path,
            language="zh",
            strict=False,
            require_images=True,
        )

        output_path = loader.export_jsonl(
            OUTPUT_ROOT / f"{split}.jsonl"
        )

        outputs[split] = output_path

        print(
            f"[ScreenAgent] {split}: "
            f"{len(loader)} 个任务 -> {output_path}"
        )

    return outputs


def main() -> None:
    preprocess_screenagent()


if __name__ == "__main__":
    main()