"""
processed_loader.py

Read one or more processed GUI Agent JSONL files and restore them into
GUITaskSample objects.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

from .schema import (
    DatasetSplit,
    DatasetStatistics,
    GUITaskSample,
    GUITaskStep,
    SemanticAction,
)
from ..executor.action import Action, ActionSequence

LOGGER = logging.getLogger(__name__)


class ProcessedDatasetError(ValueError):
    """Processed JSONL data is invalid."""


class DuplicateTaskIDError(ProcessedDatasetError):
    """Duplicate (source, task_id) records were found."""


class ProcessedDatasetLoader:
    """
    Read unified processed JSONL files.

    Parameters
    ----------
    source:
        One JSONL file, one directory, or a sequence of files/directories.
    recursive:
        Recursively discover JSONL files in directories.
    strict:
        Raise immediately for malformed records when True; otherwise skip.
    cache_samples:
        Cache restored samples in memory.
    screenshot_root:
        Optional root for resolving relative screenshot paths.
    allowed_sources:
        Optional source filter, e.g. {"screenagent", "mind2web"}.
    verify_screenshots:
        Require restored screenshot paths to exist.
    allow_duplicate_task_ids:
        Allow duplicate (source, task_id) pairs.
    encoding:
        JSONL file encoding.
    """

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        recursive: bool = True,
        strict: bool = False,
        cache_samples: bool = True,
        screenshot_root: str | Path | None = None,
        allowed_sources: set[str] | Sequence[str] | None = None,
        verify_screenshots: bool = False,
        allow_duplicate_task_ids: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        self.recursive = recursive
        self.strict = strict
        self.cache_samples = cache_samples
        self.verify_screenshots = verify_screenshots
        self.allow_duplicate_task_ids = allow_duplicate_task_ids
        self.encoding = encoding

        self.screenshot_root = (
            Path(screenshot_root).expanduser().resolve()
            if screenshot_root is not None
            else None
        )

        self.allowed_sources = (
            {str(item).strip().lower() for item in allowed_sources if str(item).strip()}
            if allowed_sources is not None
            else None
        )

        self.jsonl_files = self._discover_jsonl_files(source)
        self._record_index: list[tuple[Path, int, int]] = []
        self._sample_cache: dict[int, GUITaskSample] = {}
        self._task_lookup: dict[tuple[str, str], list[int]] = {}
        self._build_index()

    def __len__(self) -> int:
        return len(self._record_index)

    def __getitem__(self, index: int | slice) -> GUITaskSample | list[GUITaskSample]:
        if isinstance(index, slice):
            return [self.load_sample(i) for i in range(*index.indices(len(self)))]
        return self.load_sample(index)

    def __iter__(self) -> Iterator[GUITaskSample]:
        for index in range(len(self)):
            try:
                yield self.load_sample(index)
            except Exception as error:
                if self.strict:
                    raise
                LOGGER.warning("Skipping processed sample %d: %s", index, error)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(files={len(self.jsonl_files)}, "
            f"samples={len(self)}, strict={self.strict}, "
            f"cache_samples={self.cache_samples})"
        )

    def _discover_jsonl_files(
        self,
        source: str | Path | Sequence[str | Path],
    ) -> list[Path]:
        if isinstance(source, (str, Path)):
            sources = [source]
        elif isinstance(source, Sequence):
            sources = list(source)
        else:
            raise TypeError("source must be a path or a sequence of paths.")

        if not sources:
            raise ValueError("source must not be empty.")

        discovered: set[Path] = set()

        for raw_source in sources:
            path = Path(raw_source).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Processed dataset source not found: {path}")

            if path.is_file():
                if path.suffix.lower() != ".jsonl":
                    raise ProcessedDatasetError(f"Expected JSONL file: {path}")
                discovered.add(path)
                continue

            iterator = path.rglob("*.jsonl") if self.recursive else path.glob("*.jsonl")
            discovered.update(item.resolve() for item in iterator if item.is_file())

        files = sorted(discovered, key=lambda item: str(item).casefold())
        if not files:
            raise FileNotFoundError("No processed JSONL files were found.")
        return files

    def _build_index(self) -> None:
        seen: set[tuple[str, str]] = set()

        for jsonl_path in self.jsonl_files:
            with jsonl_path.open("r", encoding=self.encoding) as file:
                line_number = 0
                while True:
                    offset = file.tell()
                    line = file.readline()
                    if line == "":
                        break
                    line_number += 1
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ProcessedDatasetError("Each line must be a JSON object.")

                        source = self._normalise_source(record.get("source"))
                        if self.allowed_sources is not None and source not in self.allowed_sources:
                            continue

                        task_id = self._required_string(record.get("task_id"), "task_id")
                        key = (source, task_id)
                        if not self.allow_duplicate_task_ids and key in seen:
                            raise DuplicateTaskIDError(f"Duplicate task: {key}")
                        seen.add(key)
                    except Exception as error:
                        if self.strict:
                            raise ProcessedDatasetError(
                                f"Invalid record in {jsonl_path}, line {line_number}: {error}"
                            ) from error
                        LOGGER.warning(
                            "Skipping invalid record in %s line %d: %s",
                            jsonl_path,
                            line_number,
                            error,
                        )
                        continue

                    index = len(self._record_index)
                    self._record_index.append((jsonl_path, offset, line_number))
                    self._task_lookup.setdefault(key, []).append(index)

        if not self._record_index:
            raise ProcessedDatasetError("No valid processed samples were indexed.")

    def load_sample(self, index: int, use_cache: bool = True) -> GUITaskSample:
        index = self._resolve_index(index)
        if use_cache and self.cache_samples and index in self._sample_cache:
            return self._sample_cache[index]

        jsonl_path, offset, line_number = self._record_index[index]
        with jsonl_path.open("r", encoding=self.encoding) as file:
            file.seek(offset)
            line = file.readline()

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProcessedDatasetError(
                f"Invalid JSON in {jsonl_path}, line {line_number}: {error}"
            ) from error

        sample = self._record_to_sample(
            record=record,
            jsonl_path=jsonl_path,
            line_number=line_number,
            dataset_index=index,
        )

        if use_cache and self.cache_samples:
            self._sample_cache[index] = sample
        return sample

    def load_all(self, limit: int | None = None) -> list[GUITaskSample]:
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool):
                raise TypeError("limit must be an integer or None.")
            if limit <= 0:
                raise ValueError("limit must be greater than zero.")

        total = len(self) if limit is None else min(limit, len(self))
        samples: list[GUITaskSample] = []
        for index in range(total):
            try:
                samples.append(self.load_sample(index))
            except Exception as error:
                if self.strict:
                    raise
                LOGGER.warning("Skipping sample %d: %s", index, error)
        return samples

    def _record_to_sample(
        self,
        record: dict[str, Any],
        jsonl_path: Path,
        line_number: int,
        dataset_index: int,
    ) -> GUITaskSample:
        task_id = self._required_string(record.get("task_id"), "task_id")
        source = self._normalise_source(record.get("source"))
        instruction = self._required_string(record.get("instruction"), "instruction")
        language = self._optional_string(record.get("language")) or "en"

        metadata = record.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            raise ProcessedDatasetError("sample metadata must be a dictionary.")

        raw_steps = record.get("steps", []) or []
        if not isinstance(raw_steps, list):
            raise ProcessedDatasetError("sample steps must be a list.")

        steps: list[GUITaskStep] = []
        for step_index, raw_step in enumerate(raw_steps):
            try:
                steps.append(
                    self._record_to_step(
                        raw_step=raw_step,
                        default_step_id=step_index,
                        sample_source=source,
                        jsonl_path=jsonl_path,
                    )
                )
            except Exception as error:
                if self.strict:
                    raise ProcessedDatasetError(
                        f"Task {task_id}, step {step_index}: {error}"
                    ) from error
                LOGGER.warning(
                    "Skipping invalid step %d in task %s: %s",
                    step_index,
                    task_id,
                    error,
                )

        restored_metadata = dict(metadata)
        restored_metadata.setdefault("processed_jsonl_path", str(jsonl_path))
        restored_metadata.setdefault("processed_jsonl_line", line_number)
        restored_metadata.setdefault("processed_dataset_index", dataset_index)

        return GUITaskSample(
            task_id=task_id,
            source=source,
            instruction=instruction,
            steps=steps,
            language=language,
            metadata=restored_metadata,
        )

    def _record_to_step(
        self,
        raw_step: Any,
        default_step_id: int,
        sample_source: str,
        jsonl_path: Path,
    ) -> GUITaskStep:
        if not isinstance(raw_step, dict):
            raise ProcessedDatasetError("step must be a dictionary.")

        step_id = raw_step.get("step_id", default_step_id)
        if not isinstance(step_id, int) or isinstance(step_id, bool):
            raise ProcessedDatasetError("step_id must be an integer.")

        instruction = self._required_string(
            raw_step.get("instruction"),
            "step instruction",
        )
        language = self._optional_string(raw_step.get("language")) or "en"
        screenshot_path = self._resolve_screenshot_path(
            raw_step.get("screenshot_path"),
            jsonl_path,
        )

        raw_action = raw_step.get("action")
        if not isinstance(raw_action, dict):
            raise ProcessedDatasetError("step action must be a dictionary.")

        action_kind = self._resolve_action_kind(raw_step, sample_source, raw_action)
        action: Action | SemanticAction
        if action_kind == "executable":
            action = Action.from_dict(raw_action)
        else:
            action = self._build_semantic_action(raw_action)

        metadata = raw_step.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            raise ProcessedDatasetError("step metadata must be a dictionary.")

        return GUITaskStep(
            step_id=step_id,
            screenshot_path=screenshot_path,
            instruction=instruction,
            action=action,
            llm_response=self._optional_string(raw_step.get("llm_response")),
            corrected_response=self._optional_string(raw_step.get("corrected_response")),
            language=language,
            metadata=dict(metadata),
        )

    @staticmethod
    def _resolve_action_kind(
        raw_step: dict[str, Any],
        sample_source: str,
        raw_action: dict[str, Any],
    ) -> str:
        explicit = raw_step.get("action_kind")
        if isinstance(explicit, str):
            value = explicit.strip().lower()
            if value in {"executable", "semantic"}:
                return value
            raise ProcessedDatasetError(f"Unsupported action_kind: {explicit!r}")

        if sample_source == "mind2web":
            return "semantic"
        if "action_type" in raw_action and "type" not in raw_action:
            return "semantic"
        return "executable"

    @staticmethod
    def _build_semantic_action(raw_action: dict[str, Any]) -> SemanticAction:
        action_type = raw_action.get("action_type")
        if not isinstance(action_type, str) or not action_type.strip():
            raise ProcessedDatasetError(
                "SemanticAction action_type must be a non-empty string."
            )

        target_attributes = raw_action.get("target_attributes", {}) or {}
        metadata = raw_action.get("metadata", {}) or {}
        if not isinstance(target_attributes, dict):
            raise ProcessedDatasetError("target_attributes must be a dictionary.")
        if not isinstance(metadata, dict):
            raise ProcessedDatasetError("SemanticAction metadata must be a dictionary.")

        return SemanticAction(
            action_type=action_type.strip(),
            value=ProcessedDatasetLoader._optional_string(raw_action.get("value")),
            target_tag=ProcessedDatasetLoader._optional_string(raw_action.get("target_tag")),
            target_attributes=dict(target_attributes),
            backend_node_id=ProcessedDatasetLoader._optional_string(
                raw_action.get("backend_node_id")
            ),
            action_repr=ProcessedDatasetLoader._optional_string(
                raw_action.get("action_repr")
            ),
            metadata=dict(metadata),
        )

    def _resolve_screenshot_path(self, value: Any, jsonl_path: Path) -> Path | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ProcessedDatasetError("screenshot_path must be a string or None.")

        text = value.strip()
        if not text:
            return None

        path = Path(text).expanduser()
        if not path.is_absolute():
            base = self.screenshot_root or jsonl_path.parent
            path = base / path
        path = path.resolve()

        if self.verify_screenshots and not path.is_file():
            raise FileNotFoundError(f"Screenshot does not exist: {path}")
        return path

    def get_by_task_id(
        self,
        task_id: str | int,
        source: str | None = None,
    ) -> GUITaskSample:
        target = str(task_id).strip()
        if not target:
            raise ValueError("task_id must not be empty.")

        if source is not None:
            matches = self._task_lookup.get((self._normalise_source(source), target), [])
        else:
            matches = [
                index
                for (stored_source, stored_id), indices in self._task_lookup.items()
                if stored_id == target
                for index in indices
            ]

        if not matches:
            raise KeyError(f"Processed task not found: {target!r}")
        if len(matches) > 1:
            raise DuplicateTaskIDError(
                "Multiple tasks match; specify source or use find_by_task_id()."
            )
        return self.load_sample(matches[0])

    def find_by_task_id(
        self,
        task_id: str | int,
        source: str | None = None,
    ) -> list[GUITaskSample]:
        target = str(task_id).strip()
        source_name = self._normalise_source(source) if source is not None else None
        indices = [
            index
            for (stored_source, stored_id), values in self._task_lookup.items()
            if stored_id == target and (source_name is None or stored_source == source_name)
            for index in values
        ]
        return [self.load_sample(index) for index in indices]

    def find(
        self,
        keyword: str,
        case_sensitive: bool = False,
        source: str | None = None,
        limit: int | None = None,
    ) -> list[GUITaskSample]:
        if not isinstance(keyword, str):
            raise TypeError("keyword must be a string.")
        query = keyword.strip()
        if not query:
            raise ValueError("keyword must not be empty.")

        source_name = self._normalise_source(source) if source is not None else None
        if not case_sensitive:
            query = query.casefold()

        results: list[GUITaskSample] = []
        for sample in self:
            if source_name is not None and sample.source != source_name:
                continue

            parts = [sample.task_id, sample.source, sample.instruction]
            for step in sample.steps:
                parts.extend(
                    [
                        step.instruction,
                        step.llm_response or "",
                        step.corrected_response or "",
                    ]
                )

            text = "\n".join(parts)
            if not case_sensitive:
                text = text.casefold()

            if query in text:
                results.append(sample)
                if limit is not None and len(results) >= limit:
                    break
        return results

    def filter_by_source(self, source: str) -> list[GUITaskSample]:
        source_name = self._normalise_source(source)
        return [sample for sample in self if sample.source == source_name]

    def get_action_sequence(self, index: int) -> ActionSequence:
        return self.load_sample(index).to_action_sequence()

    def statistics(
        self,
        limit: int | None = None,
        source: str | None = None,
    ) -> DatasetStatistics:
        source_name = self._normalise_source(source) if source is not None else None
        samples: list[GUITaskSample] = []

        for sample in self:
            if source_name is None or sample.source == source_name:
                samples.append(sample)
                if limit is not None and len(samples) >= limit:
                    break

        action_counter: Counter[str] = Counter()
        language_counter: Counter[str] = Counter()
        source_counter: Counter[str] = Counter()
        total_steps = 0
        executable_steps = 0
        semantic_steps = 0
        tasks_with_demonstrations = 0

        for sample in samples:
            source_counter[sample.source] += 1
            language_counter[sample.language] += 1
            total_steps += sample.num_steps
            if sample.has_demonstration:
                tasks_with_demonstrations += 1

            for step in sample.steps:
                if isinstance(step.action, Action):
                    executable_steps += 1
                    action_name = getattr(step.action.type, "value", str(step.action.type))
                else:
                    semantic_steps += 1
                    action_name = step.action.action_type
                action_counter[str(action_name)] += 1

        num_tasks = len(samples)
        return DatasetStatistics(
            source=source_name or "processed",
            num_tasks=num_tasks,
            num_steps=total_steps,
            avg_steps_per_task=(total_steps / num_tasks if num_tasks else 0.0),
            action_distribution=dict(action_counter),
            language_distribution=dict(language_counter),
            metadata={
                "jsonl_files": [str(path) for path in self.jsonl_files],
                "source_distribution": dict(source_counter),
                "tasks_with_demonstrations": tasks_with_demonstrations,
                "tasks_without_demonstrations": num_tasks - tasks_with_demonstrations,
                "executable_step_count": executable_steps,
                "semantic_step_count": semantic_steps,
            },
        )

    def split_dataset(
        self,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        shuffle: bool = True,
        source: str | None = None,
    ) -> DatasetSplit:
        self._validate_split_ratios(train_ratio, validation_ratio, test_ratio)
        samples = self.filter_by_source(source) if source is not None else self.load_all()
        samples = list(samples)
        if shuffle:
            random.Random(seed).shuffle(samples)

        train_end = int(len(samples) * train_ratio)
        validation_end = train_end + int(len(samples) * validation_ratio)
        return DatasetSplit(
            train=samples[:train_end],
            validation=samples[train_end:validation_end],
            test=samples[validation_end:],
        )

    def export_jsonl(
        self,
        output_path: str | Path,
        samples: Sequence[GUITaskSample] | None = None,
        ensure_ascii: bool = False,
    ) -> Path:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        output_samples = list(samples) if samples is not None else self.load_all()

        with destination.open("w", encoding="utf-8") as file:
            for sample in output_samples:
                file.write(json.dumps(sample.to_dict(), ensure_ascii=ensure_ascii))
                file.write("\n")
        return destination

    def source_counts(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for (source_name, _task_id), indices in self._task_lookup.items():
            counter[source_name] += len(indices)
        return dict(counter)

    def clear_cache(self) -> None:
        self._sample_cache.clear()

    def _resolve_index(self, index: int) -> int:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("index must be an integer.")
        resolved = index + len(self) if index < 0 else index
        if resolved < 0 or resolved >= len(self):
            raise IndexError(f"Processed dataset index out of range: {index}")
        return resolved

    @staticmethod
    def _normalise_source(value: Any) -> str:
        if not isinstance(value, str):
            raise ProcessedDatasetError("source must be a string.")
        source = value.strip().lower()
        if not source:
            raise ProcessedDatasetError("source must not be empty.")
        return source

    @staticmethod
    def _required_string(value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise ProcessedDatasetError(f"{field_name} must be a string.")
        text = value.strip()
        if not text:
            raise ProcessedDatasetError(f"{field_name} must not be empty.")
        return text

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return str(value)

    @staticmethod
    def _validate_split_ratios(
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> None:
        ratios = (train_ratio, validation_ratio, test_ratio)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in ratios):
            raise TypeError("Split ratios must be numeric.")
        if any(v < 0 for v in ratios):
            raise ValueError("Split ratios must not be negative.")
        if abs(sum(ratios) - 1.0) > 1e-8:
            raise ValueError(
                "train_ratio + validation_ratio + test_ratio must equal 1.0."
            )