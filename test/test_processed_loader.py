from pathlib import Path
from src.datasets.processed_loader import ProcessedDatasetLoader


def main() -> None:
    print("当前工作目录：", Path.cwd())
    print("预期输出目录：", Path("data/processed").resolve())

    loader = ProcessedDatasetLoader(
        source="data/processed",
        allowed_sources={"mind2web"},
        recursive=True,
        strict=False,
    )

    paths = loader.export_split_csv(
        output_dir="data/processed/mind2web",
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )

    print("导出结果：")
    for split_name, path in paths.items():
        print(split_name, path, path.exists())


if __name__ == "__main__":
    main()


# loader = ProcessedDatasetLoader(
#     source=[
#         "data/processed/screenagent/train.jsonl",
#         "data/processed/mind2web/train.jsonl",
#         "data/processed/webarena/tasks.jsonl",
#     ]
# )

# print(len(loader))
# print(loader.source_counts())

# sample = loader[0]

# print(sample.task_id)
# print(sample.source)
# print(sample.instruction)
# print(sample.num_steps)

# # Statistics
# stats = loader.statistics()
# print(stats.to_dict())

# screenagent_stats = loader.statistics(
#     source="screenagent"
# )

# # Split datasets
# split = loader.split_dataset(
#     train_ratio=0.8,
#     validation_ratio=0.1,
#     test_ratio=0.1,
#     seed=42,
#     source="mind2web",
# )