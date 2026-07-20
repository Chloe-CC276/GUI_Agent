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


def export_split(split: str) -> None:
    source_path = SOURCE_ROOT / split

    if not source_path.exists():
        print(f"跳过，不存在：{source_path}")
        return

    loader = ScreenAgentLoader(
        data_root=source_path,
        language="zh",
        strict=False,
        require_images=True,
    )

    output_path = loader.export_jsonl(
        OUTPUT_ROOT / f"{split}.jsonl"
    )

    print(f"{split} Session 数量：{len(loader)}")
    print(f"{split} 导出路径：{output_path}")


def main() -> None:
    export_split("train")
    export_split("test")


if __name__ == "__main__":
    main()