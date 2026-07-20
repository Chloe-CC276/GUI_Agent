# from datasets import load_dataset

# dataset = load_dataset("osunlp/Mind2Web")
# dataset.save_to_disk("external/Mind2Web/dataset")

from datasets import load_from_disk
import json

dataset = load_from_disk(
    "D:/GUIAgent_project/GUI_Agent/external/Mind2Web/dataset"
)

print(dataset)

train = dataset["train"]

print("=" * 80)
print(train.column_names)

print("=" * 80)
print(train.features)

import json

sample = train[0]

with open(
    "sample.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        sample,
        f,
        ensure_ascii=False,
        indent=4
    )