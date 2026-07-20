from __future__ import annotations

from pathlib import Path

from src.datasets.mind2web_loader import Mind2WebLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    PROJECT_ROOT
    / "external"
    / "Mind2Web"
    / "dataset"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mind2web"
)


def preprocess_mind2web() -> dict[str, Path]:
    loader = Mind2WebLoader(
        dataset_path=SOURCE_PATH,
        split="train",
        strict=False,
    )

    output_path = loader.export_jsonl(
        OUTPUT_ROOT / "train.jsonl"
    )

    print(
        f"[Mind2Web] train: "
        f"{len(loader)} 个任务 -> {output_path}"
    )

    return {
        "train": output_path
    }


def main() -> None:
    preprocess_mind2web()


if __name__ == "__main__":
    main()