"""
mind2web_loader

Load Hugging Face Mind2Web data and convert it into the
unified GUI Agent dataset schema.

Mind2Web Hugging Face structure:

DatasetDict({
    train: Dataset({
        features: [
            "website",
            "domain",
            "subdomain",
            "annotation_id",
            "confirmed_task",
            "action_reprs",
            "actions",
        ]
    })
})

Important
---------
The Hugging Face version of Mind2Web mainly contains DOM-based
annotations and does not contain executable screen coordinates.

Therefore:

    Mind2Web operation
        -> SemanticAction
"""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

from datasets import Dataset, DatasetDict, load_from_disk

from .schema import (
    DatasetSplit,
    DatasetStatistics,
    GUITaskSample,
    GUITaskStep,
    SemanticAction,
)


LOGGER = logging.getLogger(__name__)


class Mind2WebDataError(ValueError):
    """Raised when Mind2Web data has an invalid structure."""


class UnsupportedMind2WebOperationError(Mind2WebDataError):
    """Raised when an unsupported operation is encountered."""


class Mind2WebLoader:
    """
    Mind2Web dataset loader.

    Parameters
    ----------
    dataset_path:
        Directory created by Hugging Face ``save_to_disk``.

        Example:

        external/Mind2Web/hf_dataset

    split:
        Dataset split to load. The downloaded dataset currently
        usually contains only ``train``.

    strict:
        When True, invalid samples or actions raise exceptions.

        When False, invalid samples or actions are skipped and a
        warning is logged.

    screenshot_root:
        Optional root directory of Mind2Web screenshots.

        The current Hugging Face dataset does not directly contain
        screenshot filenames. This field is reserved for later
        raw_dump integration.

    include_negative_candidates:
        Whether to preserve negative element candidates in metadata.

    cache_samples:
        Whether converted GUITaskSample objects should be cached.

    language:
        Language label stored in the unified schema. Mind2Web tasks
        are normally English.
    """

    SUPPORTED_OPERATIONS = {
        "CLICK",
        "TYPE",
        "SELECT",
        "HOVER",
        "PRESS_ENTER",
        "SCROLL",
        "STOP",
    }

    OPERATION_ALIASES = {
        "CLICK": "CLICK",
        "MOUSE_CLICK": "CLICK",

        "TYPE": "TYPE",
        "TEXT": "TYPE",
        "INPUT": "TYPE",
        "ENTER_TEXT": "TYPE",

        "SELECT": "SELECT",
        "SELECT_OPTION": "SELECT",

        "HOVER": "HOVER",
        "MOUSEOVER": "HOVER",

        "PRESS_ENTER": "PRESS_ENTER",
        "ENTER": "PRESS_ENTER",

        "SCROLL": "SCROLL",

        "STOP": "STOP",
        "FINISH": "STOP",
    }

    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        strict: bool = False,
        screenshot_root: str | Path | None = None,
        include_negative_candidates: bool = True,
        cache_samples: bool = True,
        language: str = "en",
    ) -> None:
        self.dataset_path = Path(
            dataset_path
        ).expanduser().resolve()

        self.split = split
        self.strict = strict
        self.include_negative_candidates = (
            include_negative_candidates
        )
        self.cache_samples = cache_samples
        self.language = language

        self.screenshot_root = (
            Path(screenshot_root).expanduser().resolve()
            if screenshot_root is not None
            else None
        )

        self._validate_configuration()

        loaded_dataset = load_from_disk(
            str(self.dataset_path)
        )

        self.dataset = self._select_split(
            loaded_dataset
        )

        self._sample_cache: dict[int, GUITaskSample] = {}

        LOGGER.info(
            "Mind2WebLoader initialized: path=%s, split=%s, rows=%d",
            self.dataset_path,
            self.split,
            len(self.dataset),
        )

    # ============================================================
    # Basic sequence interface
    # ============================================================

    def __len__(self) -> int:
        """Return the number of Mind2Web tasks."""
        return len(self.dataset)

    def __getitem__(
        self,
        index: int | slice,
    ) -> GUITaskSample | list[GUITaskSample]:
        """
        Load one task or a slice of tasks.

        Examples
        --------
        sample = loader[0]

        samples = loader[:10]
        """
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))

            return [
                self.load_sample(item_index)
                for item_index in indices
            ]

        return self.load_sample(index)

    def __iter__(self) -> Iterator[GUITaskSample]:
        """Iterate over valid Mind2Web samples."""
        for index in range(len(self)):
            try:
                yield self.load_sample(index)
            except Exception as error:
                if self.strict:
                    raise

                LOGGER.warning(
                    "Skipping Mind2Web sample %d: %s",
                    index,
                    error,
                )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset_path={str(self.dataset_path)!r}, "
            f"split={self.split!r}, "
            f"rows={len(self)}, "
            f"strict={self.strict}"
            f")"
        )

    # ============================================================
    # Dataset loading
    # ============================================================

    def _validate_configuration(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                "Mind2Web dataset directory not found: "
                f"{self.dataset_path}"
            )

        if not self.dataset_path.is_dir():
            raise NotADirectoryError(
                "Mind2Web dataset path is not a directory: "
                f"{self.dataset_path}"
            )

        if self.screenshot_root is not None:
            if not self.screenshot_root.exists():
                raise FileNotFoundError(
                    "Mind2Web screenshot root not found: "
                    f"{self.screenshot_root}"
                )

    def _select_split(
        self,
        loaded_dataset: Dataset | DatasetDict,
    ) -> Dataset:
        if isinstance(loaded_dataset, Dataset):
            return loaded_dataset

        if not isinstance(loaded_dataset, DatasetDict):
            raise Mind2WebDataError(
                "Expected Dataset or DatasetDict, got "
                f"{type(loaded_dataset).__name__}."
            )

        if self.split not in loaded_dataset:
            available = list(loaded_dataset.keys())

            raise KeyError(
                f"Split {self.split!r} does not exist. "
                f"Available splits: {available}"
            )

        return loaded_dataset[self.split]

    # ============================================================
    # Sample conversion
    # ============================================================

    def load_sample(
        self,
        index: int,
        use_cache: bool = True,
    ) -> GUITaskSample:
        """
        Convert one Mind2Web dataset row into GUITaskSample.

        One dataset row corresponds to one complete task.
        Each element in ``actions`` becomes one GUITaskStep.
        """
        self._validate_index(index)

        if (
            use_cache
            and self.cache_samples
            and index in self._sample_cache
        ):
            return self._sample_cache[index]

        record = self.dataset[index]

        if not isinstance(record, dict):
            raise Mind2WebDataError(
                f"Sample {index} is not a dictionary."
            )

        sample = self._convert_record(
            record=record,
            dataset_index=index,
        )

        if use_cache and self.cache_samples:
            self._sample_cache[index] = sample

        return sample

    def _convert_record(
        self,
        record: dict[str, Any],
        dataset_index: int,
    ) -> GUITaskSample:
        annotation_id = self._required_string(
            record,
            "annotation_id",
            sample_index=dataset_index,
        )

        confirmed_task = self._required_string(
            record,
            "confirmed_task",
            sample_index=dataset_index,
        )

        raw_actions = record.get("actions")
        action_reprs = record.get("action_reprs", [])

        if not isinstance(raw_actions, list):
            raise Mind2WebDataError(
                f"Sample {dataset_index}: actions must be a list."
            )

        if not isinstance(action_reprs, list):
            if self.strict:
                raise Mind2WebDataError(
                    f"Sample {dataset_index}: "
                    "action_reprs must be a list."
                )

            LOGGER.warning(
                "Sample %d has invalid action_reprs; using empty list.",
                dataset_index,
            )

            action_reprs = []

        steps: list[GUITaskStep] = []

        for action_index, raw_action in enumerate(raw_actions):
            try:
                action_repr = (
                    action_reprs[action_index]
                    if action_index < len(action_reprs)
                    else None
                )

                step = self._convert_action_to_step(
                    raw_action=raw_action,
                    action_repr=action_repr,
                    task_instruction=confirmed_task,
                    annotation_id=annotation_id,
                    dataset_index=dataset_index,
                    action_index=action_index,
                )

            except Exception as error:
                if self.strict:
                    raise

                LOGGER.warning(
                    "Skipping action %d in sample %d: %s",
                    action_index,
                    dataset_index,
                    error,
                )

                continue

            steps.append(step)

        if not steps and raw_actions:
            raise Mind2WebDataError(
                f"Sample {dataset_index} contains no valid actions."
            )

        return GUITaskSample(
            task_id=annotation_id,
            source="mind2web",
            instruction=confirmed_task,
            steps=steps,
            language=self.language,
            metadata={
                "dataset_index": dataset_index,
                "split": self.split,
                "website": record.get("website"),
                "domain": record.get("domain"),
                "subdomain": record.get("subdomain"),
                "annotation_id": annotation_id,
                "raw_action_count": len(raw_actions),
                "valid_action_count": len(steps),
                "has_screenshots": False,
            },
        )

    def _convert_action_to_step(
        self,
        raw_action: dict[str, Any],
        action_repr: Any,
        task_instruction: str,
        annotation_id: str,
        dataset_index: int,
        action_index: int,
    ) -> GUITaskStep:
        if not isinstance(raw_action, dict):
            raise Mind2WebDataError(
                f"Action {action_index} must be a dictionary."
            )

        action_uid = self._optional_string(
            raw_action.get("action_uid")
        )

        operation = raw_action.get("operation")

        if not isinstance(operation, dict):
            raise Mind2WebDataError(
                f"Action {action_index}: operation must be a dictionary."
            )

        operation_type = self._normalise_operation(
            operation.get("op")
            or operation.get("original_op")
        )

        operation_value = self._optional_string(
            operation.get("value")
        )

        pos_candidates = raw_action.get(
            "pos_candidates",
            [],
        )

        neg_candidates = raw_action.get(
            "neg_candidates",
            [],
        )

        if not isinstance(pos_candidates, list):
            raise Mind2WebDataError(
                f"Action {action_index}: "
                "pos_candidates must be a list."
            )

        if not isinstance(neg_candidates, list):
            raise Mind2WebDataError(
                f"Action {action_index}: "
                "neg_candidates must be a list."
            )

        target_candidate = self._select_target_candidate(
            pos_candidates
        )

        semantic_action = self._build_semantic_action(
            operation_type=operation_type,
            operation_value=operation_value,
            action_repr=self._optional_string(
                action_repr
            ),
            target_candidate=target_candidate,
            raw_action=raw_action,
            action_uid=action_uid,
        )

        step_instruction = self._build_step_instruction(
            task_instruction=task_instruction,
            action_repr=self._optional_string(
                action_repr
            ),
            operation_type=operation_type,
            operation_value=operation_value,
        )

        metadata: dict[str, Any] = {
            "source": "mind2web",
            "dataset_index": dataset_index,
            "annotation_id": annotation_id,
            "action_index": action_index,
            "action_uid": action_uid,
            "operation": operation,
            "action_repr": action_repr,
            "raw_html": raw_action.get("raw_html"),
            "cleaned_html": raw_action.get("cleaned_html"),
            "positive_candidates": pos_candidates,
            "selected_target_candidate": target_candidate,
            "screenshot_available": False,
        }

        if self.include_negative_candidates:
            metadata["negative_candidates"] = (
                neg_candidates
            )

        return GUITaskStep(
            step_id=action_index,
            screenshot_path=None,
            instruction=step_instruction,
            action=semantic_action,
            llm_response=None,
            corrected_response=None,
            language=self.language,
            metadata=metadata,
        )

    # ============================================================
    # Semantic action conversion
    # ============================================================

    def _build_semantic_action(
        self,
        operation_type: str,
        operation_value: str | None,
        action_repr: str | None,
        target_candidate: dict[str, Any] | None,
        raw_action: dict[str, Any],
        action_uid: str | None,
    ) -> SemanticAction:
        target_candidate = target_candidate or {}

        target_attributes = self._parse_attributes(
            target_candidate.get("attributes")
        )

        target_tag = self._optional_string(
            target_candidate.get("tag")
        )

        backend_node_id = self._optional_string(
            target_candidate.get("backend_node_id")
        )

        return SemanticAction(
            action_type=operation_type,
            value=operation_value,
            target_tag=target_tag,
            target_attributes=target_attributes,
            backend_node_id=backend_node_id,
            action_repr=action_repr,
            metadata={
                "dataset": "mind2web",
                "action_uid": action_uid,
                "original_operation": raw_action.get(
                    "operation"
                ),
                "is_original_target": target_candidate.get(
                    "is_original_target"
                ),
                "is_top_level_target": target_candidate.get(
                    "is_top_level_target"
                ),
            },
        )

    def _normalise_operation(
        self,
        value: Any,
    ) -> str:
        if value is None:
            raise Mind2WebDataError(
                "Mind2Web operation type is missing."
            )

        operation = (
            str(value)
            .strip()
            .upper()
            .replace("-", "_")
            .replace(" ", "_")
        )

        operation = self.OPERATION_ALIASES.get(
            operation,
            operation,
        )

        if operation not in self.SUPPORTED_OPERATIONS:
            if self.strict:
                raise UnsupportedMind2WebOperationError(
                    "Unsupported Mind2Web operation: "
                    f"{value!r}"
                )

            LOGGER.warning(
                "Unknown Mind2Web operation %r; "
                "preserving its normalized value.",
                value,
            )

        return operation

    @staticmethod
    def _select_target_candidate(
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Select the most suitable positive candidate.

        Priority:

        1. is_original_target == True
        2. is_top_level_target == True
        3. first valid candidate
        """
        valid_candidates = [
            candidate
            for candidate in candidates
            if isinstance(candidate, dict)
        ]

        if not valid_candidates:
            return None

        for candidate in valid_candidates:
            if candidate.get("is_original_target") is True:
                return candidate

        for candidate in valid_candidates:
            if candidate.get("is_top_level_target") is True:
                return candidate

        return valid_candidates[0]

    @staticmethod
    def _parse_attributes(
        value: Any,
    ) -> dict[str, Any]:
        """
        Convert the candidate attributes field into a dictionary.

        In the Hugging Face schema, attributes is stored as a string.
        Depending on the source data it may contain:

        - JSON text
        - Python-like dictionary text
        - ordinary raw text

        This loader safely parses valid JSON and otherwise preserves
        the original string.
        """
        if value is None:
            return {}

        if isinstance(value, dict):
            return dict(value)

        if not isinstance(value, str):
            return {
                "raw_attributes": value
            }

        text = value.strip()

        if not text:
            return {}

        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

            return {
                "parsed_attributes": parsed
            }

        except json.JSONDecodeError:
            return {
                "raw_attributes": text
            }

    @staticmethod
    def _build_step_instruction(
        task_instruction: str,
        action_repr: str | None,
        operation_type: str,
        operation_value: str | None,
    ) -> str:
        """
        Build the instruction associated with one action step.

        action_reprs usually provides the clearest step-level
        description, so it is preferred when available.
        """
        if action_repr:
            return action_repr

        if operation_type == "TYPE" and operation_value:
            return f"Type text: {operation_value}"

        if operation_type == "SELECT" and operation_value:
            return f"Select option: {operation_value}"

        if operation_type == "CLICK":
            return "Click the target webpage element"

        if operation_type == "HOVER":
            return "Hover over the target webpage element"

        if operation_type == "PRESS_ENTER":
            return "Press Enter"

        if operation_type == "SCROLL":
            return "Scroll the webpage"

        if operation_type == "STOP":
            return "Finish the task"

        return task_instruction

    # ============================================================
    # Public batch methods
    # ============================================================

    def load_all(
        self,
        limit: int | None = None,
    ) -> list[GUITaskSample]:
        """Load all or the first ``limit`` valid samples."""
        if limit is not None:
            if not isinstance(limit, int):
                raise TypeError(
                    "limit must be an integer or None."
                )

            if limit <= 0:
                raise ValueError(
                    "limit must be greater than zero."
                )

        total = (
            len(self)
            if limit is None
            else min(limit, len(self))
        )

        samples: list[GUITaskSample] = []

        for index in range(total):
            try:
                samples.append(
                    self.load_sample(index)
                )
            except Exception as error:
                if self.strict:
                    raise

                LOGGER.warning(
                    "Skipping sample %d: %s",
                    index,
                    error,
                )

        return samples

    def find(
        self,
        keyword: str,
        case_sensitive: bool = False,
        limit: int | None = None,
    ) -> list[GUITaskSample]:
        """
        Search tasks by task instruction, website, domain or step text.
        """
        if not isinstance(keyword, str):
            raise TypeError(
                "keyword must be a string."
            )

        query = keyword.strip()

        if not query:
            raise ValueError(
                "keyword must not be empty."
            )

        if not case_sensitive:
            query = query.casefold()

        results: list[GUITaskSample] = []

        for sample in self:
            searchable_parts = [
                sample.instruction,
                str(sample.metadata.get("website", "")),
                str(sample.metadata.get("domain", "")),
                str(sample.metadata.get("subdomain", "")),
            ]

            searchable_parts.extend(
                step.instruction
                for step in sample.steps
            )

            text = "\n".join(searchable_parts)

            if not case_sensitive:
                text = text.casefold()

            if query in text:
                results.append(sample)

                if (
                    limit is not None
                    and len(results) >= limit
                ):
                    break

        return results

    # ============================================================
    # Statistics
    # ============================================================

    def statistics(
        self,
        limit: int | None = None,
    ) -> DatasetStatistics:
        """Calculate task and semantic action statistics."""
        samples = self.load_all(limit=limit)

        action_counter: Counter[str] = Counter()
        language_counter: Counter[str] = Counter()
        domain_counter: Counter[str] = Counter()
        website_counter: Counter[str] = Counter()

        total_steps = 0

        for sample in samples:
            total_steps += sample.num_steps
            language_counter[sample.language] += 1

            domain = sample.metadata.get("domain")
            website = sample.metadata.get("website")

            if domain:
                domain_counter[str(domain)] += 1

            if website:
                website_counter[str(website)] += 1

            for step in sample.steps:
                action_counter[
                    step.action.action_type
                ] += 1

        num_tasks = len(samples)

        return DatasetStatistics(
            source="mind2web",
            num_tasks=num_tasks,
            num_steps=total_steps,
            avg_steps_per_task=(
                total_steps / num_tasks
                if num_tasks
                else 0.0
            ),
            action_distribution=dict(
                action_counter
            ),
            language_distribution=dict(
                language_counter
            ),
            metadata={
                "dataset_path": str(
                    self.dataset_path
                ),
                "split": self.split,
                "domain_distribution": dict(
                    domain_counter
                ),
                "website_distribution": dict(
                    website_counter
                ),
                "screenshots_available": False,
            },
        )

    # ============================================================
    # Export
    # ============================================================

    def export_jsonl(
        self,
        output_path: str | Path,
        samples: Sequence[GUITaskSample] | None = None,
        ensure_ascii: bool = False,
    ) -> Path:
        """
        Export converted samples to JSONL.

        Each line contains one complete GUITaskSample.
        """
        destination = Path(
            output_path
        ).expanduser().resolve()

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_samples = (
            list(samples)
            if samples is not None
            else self.load_all()
        )

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:
            for sample in output_samples:
                record = sample.to_dict()

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=ensure_ascii,
                    )
                )

                file.write("\n")

        return destination

    # ============================================================
    # Train / validation / test split
    # ============================================================

    def export_csv(
        self,
        output_path: str | Path,
        samples: Sequence[GUITaskSample] | None = None,
        *,
        encoding: str = "utf-8-sig",
        include_action_json: bool = True,
        include_metadata_json: bool = True,
    ) -> Path:
        """Export Mind2Web data to CSV, one semantic action per row."""
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".csv":
            destination = destination.with_suffix(".csv")
        destination.parent.mkdir(parents=True, exist_ok=True)

        output_samples = list(samples) if samples is not None else self.load_all()
        rows: list[dict[str, Any]] = []

        for sample in output_samples:
            for step in sample.steps:
                action = step.action
                action_dict = action.to_dict()
                row: dict[str, Any] = {
                    "task_id": sample.task_id,
                    "source": sample.source,
                    "task_instruction": sample.instruction,
                    "task_language": sample.language,
                    "task_num_steps": sample.num_steps,
                    "dataset_index": sample.metadata.get("dataset_index", ""),
                    "split": sample.metadata.get("split", self.split),
                    "website": sample.metadata.get("website", ""),
                    "domain": sample.metadata.get("domain", ""),
                    "subdomain": sample.metadata.get("subdomain", ""),
                    "step_id": step.step_id,
                    "step_instruction": step.instruction,
                    "step_language": step.language,
                    "screenshot_path": str(step.screenshot_path) if step.screenshot_path else "",
                    "llm_response": step.llm_response or "",
                    "corrected_response": step.corrected_response or "",
                    "action_kind": "semantic",
                    "action_type": action_dict.get("action_type", ""),
                    "action_value": action_dict.get("value", ""),
                    "target_tag": action_dict.get("target_tag", ""),
                    "backend_node_id": action_dict.get("backend_node_id", ""),
                    "action_repr": action_dict.get("action_repr", ""),
                    "target_attributes_json": self._csv_json(action_dict.get("target_attributes", {})),
                }
                if include_action_json:
                    row["action_json"] = self._csv_json(action_dict)
                if include_metadata_json:
                    row["task_metadata_json"] = self._csv_json(sample.metadata)
                    row["step_metadata_json"] = self._csv_json(step.metadata)
                    row["action_metadata_json"] = self._csv_json(action_dict.get("metadata", {}))
                rows.append(row)

        self._write_csv(destination, rows, encoding=encoding)
        LOGGER.info("Exported Mind2Web CSV: path=%s, tasks=%d, rows=%d", destination, len(output_samples), len(rows))
        return destination

    def export_split_csv(
        self,
        output_dir: str | Path,
        *,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        shuffle: bool = True,
        group_by_domain: bool = False,
        prefix: str = "mind2web",
        encoding: str = "utf-8-sig",
    ) -> dict[str, Path]:
        """Split Mind2Web tasks and export train/validation/test CSV files."""
        split = self.split_dataset(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
            shuffle=shuffle,
            group_by_domain=group_by_domain,
        )
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        return {
            "train": self.export_csv(destination / f"{prefix}_train.csv", split.train, encoding=encoding),
            "validation": self.export_csv(destination / f"{prefix}_validation.csv", split.validation, encoding=encoding),
            "test": self.export_csv(destination / f"{prefix}_test.csv", split.test, encoding=encoding),
        }

    def export_statistics_csv(
        self,
        output_path: str | Path,
        *,
        limit: int | None = None,
        encoding: str = "utf-8-sig",
    ) -> Path:
        """Export the dataset statistics summary as a single-row CSV."""
        statistics = self.statistics(limit=limit).to_dict()
        row = {
            key: self._csv_json(value) if isinstance(value, (dict, list, tuple)) else value
            for key, value in statistics.items()
        }
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".csv":
            destination = destination.with_suffix(".csv")
        self._write_csv(destination, [row], encoding=encoding)
        return destination

    @staticmethod
    def _csv_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))

    @staticmethod
    def _write_csv(destination: Path, rows: Sequence[dict[str, Any]], *, encoding: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        if not fieldnames:
            fieldnames = ["task_id"]
        with destination.open("w", encoding=encoding, newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def split_dataset(
        self,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        shuffle: bool = True,
        group_by_domain: bool = False,
    ) -> DatasetSplit:
        """
        Split converted Mind2Web tasks.

        Parameters
        ----------
        group_by_domain:
            When False, split individual tasks randomly.

            When True, keep all tasks from the same domain in one
            split. This reduces website-domain leakage.
        """
        self._validate_split_ratios(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )

        samples = self.load_all()

        if group_by_domain:
            return self._grouped_domain_split(
                samples=samples,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                seed=seed,
                shuffle=shuffle,
            )

        samples = list(samples)

        if shuffle:
            random.Random(seed).shuffle(samples)

        total = len(samples)

        train_end = int(
            total * train_ratio
        )

        validation_end = (
            train_end
            + int(total * validation_ratio)
        )

        return DatasetSplit(
            train=samples[:train_end],
            validation=samples[
                train_end:validation_end
            ],
            test=samples[validation_end:],
        )

    @staticmethod
    def _grouped_domain_split(
        samples: list[GUITaskSample],
        train_ratio: float,
        validation_ratio: float,
        seed: int,
        shuffle: bool,
    ) -> DatasetSplit:
        groups: dict[str, list[GUITaskSample]] = {}

        for sample in samples:
            domain = str(
                sample.metadata.get(
                    "domain",
                    "__unknown_domain__",
                )
            )

            groups.setdefault(
                domain,
                [],
            ).append(sample)

        domains = list(groups.keys())

        if shuffle:
            random.Random(seed).shuffle(domains)
        else:
            domains.sort()

        total_samples = len(samples)
        train_target = int(
            total_samples * train_ratio
        )
        validation_target = int(
            total_samples * validation_ratio
        )

        train_samples: list[GUITaskSample] = []
        validation_samples: list[GUITaskSample] = []
        test_samples: list[GUITaskSample] = []

        for domain in domains:
            domain_samples = groups[domain]

            if len(train_samples) < train_target:
                train_samples.extend(
                    domain_samples
                )

            elif (
                len(validation_samples)
                < validation_target
            ):
                validation_samples.extend(
                    domain_samples
                )

            else:
                test_samples.extend(
                    domain_samples
                )

        return DatasetSplit(
            train=train_samples,
            validation=validation_samples,
            test=test_samples,
        )

    @staticmethod
    def _validate_split_ratios(
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> None:
        ratios = [
            train_ratio,
            validation_ratio,
            test_ratio,
        ]

        if any(
            not isinstance(value, (int, float))
            for value in ratios
        ):
            raise TypeError(
                "Split ratios must be numeric."
            )

        if any(value < 0 for value in ratios):
            raise ValueError(
                "Split ratios must not be negative."
            )

        if abs(sum(ratios) - 1.0) > 1e-8:
            raise ValueError(
                "train_ratio + validation_ratio + "
                "test_ratio must equal 1.0."
            )

    # ============================================================
    # Helpers
    # ============================================================

    def _validate_index(
        self,
        index: int,
    ) -> None:
        if not isinstance(index, int):
            raise TypeError(
                "index must be an integer."
            )

        if index < 0:
            index += len(self.dataset)

        if index < 0 or index >= len(self.dataset):
            raise IndexError(
                f"Mind2Web index out of range: {index}"
            )

    @staticmethod
    def _required_string(
        record: dict[str, Any],
        field_name: str,
        sample_index: int,
    ) -> str:
        value = record.get(field_name)

        if not isinstance(value, str):
            raise Mind2WebDataError(
                f"Sample {sample_index}: "
                f"{field_name} must be a string."
            )

        value = value.strip()

        if not value:
            raise Mind2WebDataError(
                f"Sample {sample_index}: "
                f"{field_name} must not be empty."
            )

        return value

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        if isinstance(value, str):
            text = value.strip()
            return text or None

        return str(value)

    def clear_cache(self) -> None:
        """Clear converted sample cache."""
        self._sample_cache.clear()