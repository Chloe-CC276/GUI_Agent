from src.datasets.processed_loader import ProcessedDatasetLoader


loader = ProcessedDatasetLoader(
    source="data/processed",
    allowed_sources={"screenagent"},
    recursive=True,
    strict=False,
)

# loader = ProcessedDatasetLoader(
#     source=[
#         "data/processed/screenagent/train.jsonl",
#         "data/processed/mind2web/train.jsonl",
#         "data/processed/webarena/tasks.jsonl",
#     ]
# )

print(len(loader))
print(loader.source_counts())

sample = loader[0]

print(sample.task_id)
print(sample.source)
print(sample.instruction)
print(sample.num_steps)

# Statistics
stats = loader.statistics()
print(stats.to_dict())

screenagent_stats = loader.statistics(
    source="screenagent"
)

# Split datasets
split = loader.split_dataset(
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    source="mind2web",
)