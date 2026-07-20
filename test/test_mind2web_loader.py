from pathlib import Path

from src.datasets.mind2web_loader import Mind2WebLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "external"
    / "Mind2Web"
    / "dataset"
)


def main() -> None:
    loader = Mind2WebLoader(
        dataset_path=DATASET_PATH,
        split="train",
        strict=False,
    )

    print(loader)
    print("任务数量:", len(loader))

    sample = loader[0]

    print("\n任务基本信息")
    print("-" * 80)
    print("task_id:", sample.task_id)
    print("instruction:", sample.instruction)
    print("website:", sample.metadata["website"])
    print("domain:", sample.metadata["domain"])
    print("subdomain:", sample.metadata["subdomain"])
    print("steps:", sample.num_steps)

    print("\n动作")
    print("-" * 80)

    for step in sample.steps:
        print("step_id:", step.step_id)
        print("instruction:", step.instruction)
        print("action:", step.action.to_dict())
        print("target:", step.metadata["selected_target_candidate"])
        print()

    stats = loader.statistics(limit=20)

    print("\n统计信息")
    print("-" * 80)
    print(stats.to_dict())

    output_path = loader.export_jsonl(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "mind2web"
        / "train.jsonl"
    )

    print("\n导出路径:", output_path)


if __name__ == "__main__":
    main()