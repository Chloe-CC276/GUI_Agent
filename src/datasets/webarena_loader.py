"""
webarena_loader.py

Load WebArena benchmark task configuration and convert it into the
unified GUI Agent schema.

Supported source formats
------------------------
1. test.raw.json containing a list of task dictionaries.
2. A JSON file containing one task dictionary.
3. A directory containing one JSON file per task.
4. The WebArena repository root.

WebArena tasks do not contain reference action trajectories.
Therefore, each task becomes a GUITaskSample with steps=[].
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import urlparse

from ._common import (
    _csv_json,
    _natural_sort_key,
    _optional_string,
    _validate_split_ratios,
    _write_csv,
)
from .schema import (
    DatasetSplit,
    DatasetStatistics,
    GUITaskSample,
    WebArenaEvaluation,
    WebArenaTaskConfig,
)


LOGGER = logging.getLogger(__name__)


class WebArenaDataError(ValueError):
    """WebArena task data is invalid."""


class WebArenaLoader:
    """
    Load WebArena task definitions.

    Parameters
    ----------
    source:
        WebArena repository root, test.raw.json, config_files
        directory, or a single task JSON file.

    strict:
        True:
            Invalid data immediately raises an exception.

        False:
            Invalid records are logged and skipped.

    include_raw_record:
        Preserve the complete original task record in metadata.

    resolve_start_url:
        Resolve placeholders such as ``__GITLAB__`` from environment
        variables.

    environment:
        Optional mapping used to resolve URL placeholders.

        Example:

        {
            "GITLAB": "http://localhost:8023",
            "REDDIT": "http://localhost:9999"
        }

        When omitted, operating-system environment variables are used.

    encoding:
        JSON text encoding.

    language:
        Language label for GUITaskSample.
    """

    REQUIRED_FIELDS = {
        "sites",
        "task_id",
        "require_login",
        "start_url",
        "intent_template",
        "instantiation_dict",
        "intent",
        "require_reset",
        "eval",
    }

    PLACEHOLDER_PATTERN = re.compile(
        r"^__([A-Z0-9_]+)__$"
    )

    EXCLUDED_FILE_NAMES = {
        "dataset_info.json",
        "state.json",
        "package.json",
        "package-lock.json",
        "tsconfig.json",
    }

    def __init__(
        self,
        source: str | Path,
        strict: bool = False,
        include_raw_record: bool = False,
        resolve_start_url: bool = False,
        environment: dict[str, str] | None = None,
        encoding: str = "utf-8",
        language: str = "en",
    ) -> None:
        self.source = Path(
            source
        ).expanduser().resolve()

        self.strict = strict
        self.include_raw_record = include_raw_record
        self.resolve_start_url = resolve_start_url
        self.environment = (
            dict(environment)
            if environment is not None
            else dict(os.environ)
        )
        self.encoding = encoding
        self.language = language

        self._validate_source()

        self._task_configs = self._load_task_configs()
        self._sample_cache: dict[int, GUITaskSample] = {}

        LOGGER.info(
            "WebArenaLoader initialized: source=%s, tasks=%d",
            self.source,
            len(self._task_configs),
        )

    # ============================================================
    # Sequence interface
    # ============================================================

    def __len__(self) -> int:
        return len(self._task_configs)

    def __getitem__(
        self,
        index: int | slice,
    ) -> GUITaskSample | list[GUITaskSample]:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))

            return [
                self.load_sample(item_index)
                for item_index in indices
            ]

        return self.load_sample(index)

    def __iter__(self) -> Iterator[GUITaskSample]:
        for index in range(len(self)):
            try:
                yield self.load_sample(index)
            except Exception as error:
                if self.strict:
                    raise

                LOGGER.warning(
                    "Skipping WebArena sample %d: %s",
                    index,
                    error,
                )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"source={str(self.source)!r}, "
            f"tasks={len(self)}, "
            f"strict={self.strict}, "
            f"resolve_start_url={self.resolve_start_url}"
            f")"
        )

    # ============================================================
    # Source discovery
    # ============================================================

    def _validate_source(self) -> None:
        if not self.source.exists():
            raise FileNotFoundError(
                f"WebArena source not found: {self.source}"
            )

        if self.source.is_file():
            if self.source.suffix.lower() != ".json":
                raise WebArenaDataError(
                    "WebArena source file must be JSON: "
                    f"{self.source}"
                )

        elif not self.source.is_dir():
            raise WebArenaDataError(
                "WebArena source must be a JSON file "
                f"or directory: {self.source}"
            )

    def _load_task_configs(
        self,
    ) -> list[WebArenaTaskConfig]:
        raw_records = self._load_raw_records()

        configs: list[WebArenaTaskConfig] = []

        for index, record in enumerate(raw_records):
            try:
                config = self._parse_task_config(
                    record=record,
                    dataset_index=index,
                )
            except Exception as error:
                if self.strict:
                    raise

                LOGGER.warning(
                    "Skipping WebArena record %d: %s",
                    index,
                    error,
                )
                continue

            configs.append(config)

        if not configs:
            raise WebArenaDataError(
                "No valid WebArena task configurations found."
            )

        configs.sort(
            key=lambda config: self._task_sort_key(
                config.task_id
            )
        )

        return configs

    def _load_raw_records(
        self,
    ) -> list[dict[str, Any]]:
        if self.source.is_file():
            return self._read_json_records(
                self.source
            )

        # Repository root: prefer test.raw.json.
        raw_json_candidates = [
            self.source / "test.raw.json",
            self.source / "config_files" / "test.raw.json",
        ]

        for raw_path in raw_json_candidates:
            if raw_path.is_file():
                return self._read_json_records(
                    raw_path
                )

        # Repository root with generated config_files.
        config_dir = self.source / "config_files"

        if config_dir.is_dir():
            records = self._read_config_directory(
                config_dir
            )

            if records:
                return records

        # The supplied source may itself be config_files.
        records = self._read_config_directory(
            self.source
        )

        if records:
            return records

        raise FileNotFoundError(
            "No WebArena task records found under "
            f"{self.source}"
        )

    def _read_json_records(
        self,
        path: Path,
    ) -> list[dict[str, Any]]:
        data = self._read_json(path)

        if isinstance(data, dict):
            record = dict(data)
            record["_source_file"] = str(path)
            record["_source_index"] = 0

            return [record]

        if isinstance(data, list):
            records: list[dict[str, Any]] = []

            for index, item in enumerate(data):
                if not isinstance(item, dict):
                    message = (
                        f"Item {index} in {path} "
                        "is not a JSON object."
                    )

                    if self.strict:
                        raise WebArenaDataError(message)

                    LOGGER.warning(message)
                    continue

                record = dict(item)
                record["_source_file"] = str(path)
                record["_source_index"] = index
                records.append(record)

            return records

        raise WebArenaDataError(
            f"Expected JSON object or list in {path}, "
            f"got {type(data).__name__}."
        )

    def _read_config_directory(
        self,
        directory: Path,
    ) -> list[dict[str, Any]]:
        if not directory.is_dir():
            return []

        candidates = sorted(
            (
                path
                for path in directory.glob("*.json")
                if path.is_file()
                and path.name not in self.EXCLUDED_FILE_NAMES
                and path.name != "test.raw.json"
            ),
            key=lambda path: self._natural_sort_key(
                path.name
            ),
        )

        records: list[dict[str, Any]] = []

        for source_index, path in enumerate(candidates):
            try:
                data = self._read_json(path)
            except Exception as error:
                if self.strict:
                    raise

                LOGGER.warning(
                    "Skipping JSON file %s: %s",
                    path,
                    error,
                )
                continue

            if not isinstance(data, dict):
                continue

            if not self._looks_like_webarena_task(data):
                continue

            record = dict(data)
            record["_source_file"] = str(path)
            record["_source_index"] = source_index
            records.append(record)

        return records

    def _read_json(
        self,
        path: Path,
    ) -> Any:
        try:
            with path.open(
                "r",
                encoding=self.encoding,
            ) as file:
                return json.load(file)

        except UnicodeDecodeError as error:
            raise WebArenaDataError(
                f"Cannot decode {path} using "
                f"{self.encoding}: {error}"
            ) from error

        except json.JSONDecodeError as error:
            raise WebArenaDataError(
                f"Invalid JSON file {path}: {error}"
            ) from error

    @classmethod
    def _looks_like_webarena_task(
        cls,
        record: dict[str, Any],
    ) -> bool:
        return (
            "task_id" in record
            and "intent" in record
            and "sites" in record
            and "eval" in record
        )

    # ============================================================
    # Parsing
    # ============================================================

    def _parse_task_config(
        self,
        record: dict[str, Any],
        dataset_index: int,
    ) -> WebArenaTaskConfig:
        missing_fields = [
            field_name
            for field_name in self.REQUIRED_FIELDS
            if field_name not in record
        ]

        if missing_fields:
            raise WebArenaDataError(
                f"Task {dataset_index} is missing fields: "
                f"{missing_fields}"
            )

        task_id = self._normalise_task_id(
            record.get("task_id"),
            dataset_index,
        )

        sites = self._parse_sites(
            record.get("sites"),
            task_id,
        )

        require_login = self._parse_bool(
            record.get("require_login"),
            field_name="require_login",
            task_id=task_id,
        )

        require_reset = self._parse_bool(
            record.get("require_reset"),
            field_name="require_reset",
            task_id=task_id,
        )

        storage_state = self._optional_string(
            record.get("storage_state")
        )

        start_url_raw = self._required_string(
            record.get("start_url"),
            field_name="start_url",
            task_id=task_id,
        )

        resolved_start_url = self._resolve_url_placeholder(
            start_url_raw
        )

        intent_template = self._required_string(
            record.get("intent_template"),
            field_name="intent_template",
            task_id=task_id,
        )

        intent = self._required_string(
            record.get("intent"),
            field_name="intent",
            task_id=task_id,
        )

        instantiation_dict = record.get(
            "instantiation_dict"
        )

        if not isinstance(instantiation_dict, dict):
            raise WebArenaDataError(
                f"Task {task_id}: instantiation_dict "
                "must be a dictionary."
            )

        geolocation = record.get("geolocation")

        if (
            geolocation is not None
            and not isinstance(geolocation, dict)
        ):
            raise WebArenaDataError(
                f"Task {task_id}: geolocation must be "
                "a dictionary or None."
            )

        evaluation = self._parse_evaluation(
            record.get("eval"),
            task_id,
        )

        intent_template_id = record.get(
            "intent_template_id"
        )

        if intent_template_id is not None:
            if (
                not isinstance(intent_template_id, int)
                or isinstance(intent_template_id, bool)
            ):
                raise WebArenaDataError(
                    f"Task {task_id}: intent_template_id "
                    "must be an integer or None."
                )

        known_fields = {
            *self.REQUIRED_FIELDS,
            "storage_state",
            "geolocation",
            "intent_template_id",
            "_source_file",
            "_source_index",
        }

        extra_fields = {
            key: value
            for key, value in record.items()
            if key not in known_fields
        }

        metadata: dict[str, Any] = {
            "source_file": record.get(
                "_source_file"
            ),
            "source_index": record.get(
                "_source_index"
            ),
            "original_start_url": start_url_raw,
            "start_url_resolved": (
                resolved_start_url != start_url_raw
            ),
        }

        if extra_fields:
            metadata["extra_fields"] = extra_fields

        if self.include_raw_record:
            metadata["raw_record"] = {
                key: value
                for key, value in record.items()
                if not key.startswith("_")
            }

        return WebArenaTaskConfig(
            task_id=task_id,
            sites=sites,
            require_login=require_login,
            storage_state=storage_state,
            start_url=resolved_start_url,
            geolocation=geolocation,
            intent_template=intent_template,
            instantiation_dict=dict(
                instantiation_dict
            ),
            intent=intent,
            require_reset=require_reset,
            evaluation=evaluation,
            intent_template_id=intent_template_id,
            metadata=metadata,
        )

    def _parse_evaluation(
        self,
        value: Any,
        task_id: str,
    ) -> WebArenaEvaluation:
        if not isinstance(value, dict):
            raise WebArenaDataError(
                f"Task {task_id}: eval must be a dictionary."
            )

        eval_types = self._normalise_string_list(
            value.get("eval_types")
        )

        reference_answers = value.get(
            "reference_answers",
            {},
        )

        if not isinstance(reference_answers, dict):
            raise WebArenaDataError(
                f"Task {task_id}: reference_answers "
                "must be a dictionary."
            )

        reference_url = self._optional_string(
            value.get("reference_url")
        ) or ""

        program_html = value.get(
            "program_html",
            [],
        )

        if not isinstance(program_html, list):
            raise WebArenaDataError(
                f"Task {task_id}: program_html "
                "must be a list."
            )

        string_note = self._optional_string(
            value.get("string_note")
        ) or ""

        known_fields = {
            "eval_types",
            "reference_answers",
            "reference_url",
            "program_html",
            "string_note",
            "reference_answer_raw_annotation",
        }

        extra_fields = {
            key: item
            for key, item in value.items()
            if key not in known_fields
        }

        return WebArenaEvaluation(
            eval_types=eval_types,
            reference_answers=dict(
                reference_answers
            ),
            reference_url=reference_url,
            program_html=list(program_html),
            string_note=string_note,
            reference_answer_raw_annotation=value.get(
                "reference_answer_raw_annotation"
            ),
            metadata=extra_fields,
        )

    # ============================================================
    # URL resolution
    # ============================================================

    def _resolve_url_placeholder(
        self,
        start_url: str,
    ) -> str:
        """
        Resolve placeholders such as:

            __GITLAB__
            __REDDIT__
            __SHOPPING__

        using the supplied environment mapping.

        When resolve_start_url=False, the original placeholder is
        preserved.
        """
        if not self.resolve_start_url:
            return start_url

        match = self.PLACEHOLDER_PATTERN.fullmatch(
            start_url
        )

        if match is None:
            return start_url

        environment_key = match.group(1)
        resolved_value = self.environment.get(
            environment_key
        )

        if resolved_value:
            return resolved_value

        message = (
            f"Cannot resolve WebArena URL placeholder "
            f"{start_url!r}: environment variable "
            f"{environment_key!r} is missing."
        )

        if self.strict:
            raise WebArenaDataError(message)

        LOGGER.warning(message)
        return start_url

    # ============================================================
    # Public loading
    # ============================================================

    def load_config(
        self,
        index: int,
    ) -> WebArenaTaskConfig:
        resolved_index = self._resolve_index(index)
        return self._task_configs[resolved_index]

    def load_sample(
        self,
        index: int,
        use_cache: bool = True,
    ) -> GUITaskSample:
        resolved_index = self._resolve_index(index)

        if (
            use_cache
            and resolved_index in self._sample_cache
        ):
            return self._sample_cache[
                resolved_index
            ]

        config = self._task_configs[
            resolved_index
        ]

        sample = config.to_gui_task_sample()

        sample.metadata["dataset_index"] = (
            resolved_index
        )

        sample.metadata["start_url_domain"] = (
            self._extract_domain(
                config.start_url
            )
        )

        if use_cache:
            self._sample_cache[
                resolved_index
            ] = sample

        return sample

    def load_all(
        self,
        limit: int | None = None,
    ) -> list[GUITaskSample]:
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
                    "Skipping WebArena task %d: %s",
                    index,
                    error,
                )

        return samples

    # ============================================================
    # Search
    # ============================================================

    def get_by_task_id(
        self,
        task_id: str | int,
    ) -> GUITaskSample:
        target = str(task_id)

        for index, config in enumerate(
            self._task_configs
        ):
            if config.task_id == target:
                return self.load_sample(index)

        raise KeyError(
            f"WebArena task not found: {task_id}"
        )

    def find(
        self,
        keyword: str,
        case_sensitive: bool = False,
        limit: int | None = None,
    ) -> list[GUITaskSample]:
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
                str(
                    sample.metadata.get(
                        "intent_template",
                        "",
                    )
                ),
                str(
                    sample.metadata.get(
                        "start_url",
                        "",
                    )
                ),
                *sample.metadata.get(
                    "sites",
                    [],
                ),
                *[
                    str(value)
                    for value in sample.metadata.get(
                        "instantiation_dict",
                        {},
                    ).values()
                ],
            ]

            text = "\n".join(
                searchable_parts
            )

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

    def find_by_site(
        self,
        site: str,
    ) -> list[GUITaskSample]:
        if not isinstance(site, str):
            raise TypeError(
                "site must be a string."
            )

        target = site.strip().casefold()

        if not target:
            raise ValueError(
                "site must not be empty."
            )

        return [
            sample
            for sample in self
            if target
            in {
                item.casefold()
                for item in sample.metadata.get(
                    "sites",
                    [],
                )
            }
        ]

    # ============================================================
    # Statistics
    # ============================================================

    def statistics(
        self,
        limit: int | None = None,
    ) -> DatasetStatistics:
        samples = self.load_all(
            limit=limit
        )

        site_counter: Counter[str] = Counter()
        evaluation_counter: Counter[str] = (
            Counter()
        )
        language_counter: Counter[str] = Counter()

        login_required_count = 0
        reset_required_count = 0
        multi_site_count = 0
        unresolved_url_count = 0

        for sample in samples:
            language_counter[
                sample.language
            ] += 1

            sites = sample.metadata.get(
                "sites",
                [],
            )

            if len(sites) > 1:
                multi_site_count += 1

            for site in sites:
                site_counter[site] += 1

            if sample.metadata.get(
                "require_login"
            ):
                login_required_count += 1

            if sample.metadata.get(
                "require_reset"
            ):
                reset_required_count += 1

            start_url = sample.metadata.get(
                "start_url",
                "",
            )

            if (
                isinstance(start_url, str)
                and self.PLACEHOLDER_PATTERN.fullmatch(
                    start_url
                )
            ):
                unresolved_url_count += 1

            evaluation = sample.metadata.get(
                "evaluation",
                {},
            )

            for eval_type in evaluation.get(
                "eval_types",
                [],
            ):
                evaluation_counter[
                    eval_type
                ] += 1

        return DatasetStatistics(
            source="webarena",
            num_tasks=len(samples),
            num_steps=0,
            avg_steps_per_task=0.0,
            action_distribution={},
            language_distribution=dict(
                language_counter
            ),
            metadata={
                "source": str(self.source),
                "site_distribution": dict(
                    site_counter
                ),
                "evaluation_type_distribution": dict(
                    evaluation_counter
                ),
                "login_required_count": (
                    login_required_count
                ),
                "reset_required_count": (
                    reset_required_count
                ),
                "multi_site_task_count": (
                    multi_site_count
                ),
                "unresolved_start_url_count": (
                    unresolved_url_count
                ),
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
                file.write(
                    json.dumps(
                        sample.to_dict(),
                        ensure_ascii=ensure_ascii,
                    )
                )
                file.write("\n")

        return destination

    def export_csv(
        self,
        output_path: str | Path,
        samples: Sequence[GUITaskSample] | None = None,
        *,
        encoding: str = "utf-8-sig",
        include_metadata_json: bool = True,
    ) -> Path:
        """Export WebArena tasks to CSV, one task configuration per row."""
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".csv":
            destination = destination.with_suffix(".csv")
        destination.parent.mkdir(parents=True, exist_ok=True)

        output_samples = list(samples) if samples is not None else self.load_all()
        rows: list[dict[str, Any]] = []
        for sample in output_samples:
            metadata = sample.metadata
            evaluation = metadata.get("evaluation", {}) or {}
            row: dict[str, Any] = {
                "task_id": sample.task_id,
                "source": sample.source,
                "instruction": sample.instruction,
                "language": sample.language,
                "sites": self._csv_json(metadata.get("sites", [])),
                "require_login": metadata.get("require_login", False),
                "storage_state": metadata.get("storage_state", ""),
                "start_url": metadata.get("start_url", ""),
                "start_url_domain": metadata.get("start_url_domain", ""),
                "geolocation_json": self._csv_json(metadata.get("geolocation")),
                "intent_template": metadata.get("intent_template", ""),
                "intent_template_id": metadata.get("intent_template_id", ""),
                "instantiation_dict_json": self._csv_json(metadata.get("instantiation_dict", {})),
                "require_reset": metadata.get("require_reset", False),
                "eval_types": self._csv_json(evaluation.get("eval_types", [])),
                "reference_answers_json": self._csv_json(evaluation.get("reference_answers", {})),
                "reference_url": evaluation.get("reference_url", ""),
                "program_html_json": self._csv_json(evaluation.get("program_html", [])),
                "string_note": evaluation.get("string_note", ""),
                "has_demonstration": sample.has_demonstration,
                "num_steps": sample.num_steps,
            }
            if include_metadata_json:
                row["metadata_json"] = self._csv_json(metadata)
            rows.append(row)

        self._write_csv(destination, rows, encoding=encoding)
        LOGGER.info("Exported WebArena CSV: path=%s, tasks=%d", destination, len(rows))
        return destination

    def export_configs_csv(
        self,
        output_path: str | Path,
        *,
        encoding: str = "utf-8-sig",
        include_metadata_json: bool = True,
    ) -> Path:
        """Export internal WebArenaTaskConfig objects to CSV."""
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".csv":
            destination = destination.with_suffix(".csv")
        rows: list[dict[str, Any]] = []
        for config in self._task_configs:
            data = config.to_dict()
            evaluation = data.get("eval", {}) or {}
            row: dict[str, Any] = {
                "task_id": data.get("task_id", ""),
                "sites": self._csv_json(data.get("sites", [])),
                "require_login": data.get("require_login", False),
                "storage_state": data.get("storage_state", ""),
                "start_url": data.get("start_url", ""),
                "geolocation_json": self._csv_json(data.get("geolocation")),
                "intent_template": data.get("intent_template", ""),
                "intent_template_id": data.get("intent_template_id", ""),
                "instantiation_dict_json": self._csv_json(data.get("instantiation_dict", {})),
                "intent": data.get("intent", ""),
                "require_reset": data.get("require_reset", False),
                "eval_types": self._csv_json(evaluation.get("eval_types", [])),
                "reference_answers_json": self._csv_json(evaluation.get("reference_answers", {})),
                "reference_url": evaluation.get("reference_url", ""),
                "program_html_json": self._csv_json(evaluation.get("program_html", [])),
                "string_note": evaluation.get("string_note", ""),
            }
            if include_metadata_json:
                row["metadata_json"] = self._csv_json(data.get("metadata", {}))
                row["evaluation_metadata_json"] = self._csv_json(evaluation.get("metadata", {}))
            rows.append(row)
        self._write_csv(destination, rows, encoding=encoding)
        return destination

    def export_configs_jsonl(
        self,
        output_path: str | Path,
        ensure_ascii: bool = False,
    ) -> Path:
        """Export internal WebArenaTaskConfig objects to JSONL."""
        destination = Path(
            output_path
        ).expanduser().resolve()

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:
            for config in self._task_configs:
                file.write(
                    json.dumps(
                        config.to_dict(),
                        ensure_ascii=ensure_ascii,
                        default=str,
                    )
                )
                file.write("\n")

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
        prefix: str = "webarena",
        encoding: str = "utf-8-sig",
    ) -> dict[str, Path]:
        split = self.split_dataset(train_ratio, validation_ratio, test_ratio, seed, shuffle)
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        return {
            "train": self.export_csv(destination / f"{prefix}_train.csv", split.train, encoding=encoding),
            "validation": self.export_csv(destination / f"{prefix}_validation.csv", split.validation, encoding=encoding),
            "test": self.export_csv(destination / f"{prefix}_test.csv", split.test, encoding=encoding),
        }

    def export_statistics_csv(self, output_path: str | Path, *, limit: int | None = None, encoding: str = "utf-8-sig") -> Path:
        statistics = self.statistics(limit=limit).to_dict()
        row = {key: self._csv_json(value) if isinstance(value, (dict, list, tuple)) else value for key, value in statistics.items()}
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".csv":
            destination = destination.with_suffix(".csv")
        self._write_csv(destination, [row], encoding=encoding)
        return destination

    _csv_json = staticmethod(_csv_json)

    _write_csv = staticmethod(_write_csv)

    def split_dataset(
        self,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        shuffle: bool = True,
    ) -> DatasetSplit:
        self._validate_split_ratios(
            train_ratio,
            validation_ratio,
            test_ratio,
        )

        samples = self.load_all()

        if shuffle:
            random.Random(seed).shuffle(
                samples
            )

        train_end = int(
            len(samples) * train_ratio
        )

        validation_end = (
            train_end
            + int(
                len(samples)
                * validation_ratio
            )
        )

        return DatasetSplit(
            train=samples[:train_end],
            validation=samples[
                train_end:validation_end
            ],
            test=samples[
                validation_end:
            ],
        )

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _parse_sites(
        value: Any,
        task_id: str,
    ) -> list[str]:
        if not isinstance(value, list):
            raise WebArenaDataError(
                f"Task {task_id}: sites must be a list."
            )

        sites: list[str] = []

        for site in value:
            if not isinstance(site, str):
                raise WebArenaDataError(
                    f"Task {task_id}: every site "
                    "must be a string."
                )

            text = site.strip()

            if text:
                sites.append(text)

        if not sites:
            raise WebArenaDataError(
                f"Task {task_id}: sites must not be empty."
            )

        return sites

    @staticmethod
    def _parse_bool(
        value: Any,
        field_name: str,
        task_id: str,
    ) -> bool:
        if not isinstance(value, bool):
            raise WebArenaDataError(
                f"Task {task_id}: {field_name} "
                "must be bool."
            )

        return value

    @staticmethod
    def _required_string(
        value: Any,
        field_name: str,
        task_id: str,
    ) -> str:
        if not isinstance(value, str):
            raise WebArenaDataError(
                f"Task {task_id}: {field_name} "
                "must be a string."
            )

        text = value.strip()

        if not text:
            raise WebArenaDataError(
                f"Task {task_id}: {field_name} "
                "must not be empty."
            )

        return text

    _optional_string = staticmethod(_optional_string)

    @staticmethod
    def _normalise_string_list(
        value: Any,
    ) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []

        if isinstance(value, list):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        return [str(value)]

    @staticmethod
    def _normalise_task_id(
        value: Any,
        dataset_index: int,
    ) -> str:
        if value is None:
            return str(dataset_index)

        task_id = str(value).strip()

        return (
            task_id
            if task_id
            else str(dataset_index)
        )

    def _resolve_index(
        self,
        index: int,
    ) -> int:
        if not isinstance(index, int):
            raise TypeError(
                "index must be an integer."
            )

        resolved = index

        if resolved < 0:
            resolved += len(self)

        if (
            resolved < 0
            or resolved >= len(self)
        ):
            raise IndexError(
                f"WebArena index out of range: {index}"
            )

        return resolved

    @staticmethod
    def _extract_domain(
        value: str,
    ) -> str:
        if not isinstance(value, str):
            return ""

        if value.startswith("__"):
            return ""

        try:
            return urlparse(value).netloc
        except Exception:
            return ""

    _natural_sort_key = staticmethod(_natural_sort_key)

    @staticmethod
    def _task_sort_key(
        task_id: str,
    ) -> tuple[int, int | str]:
        if task_id.isdigit():
            return 0, int(task_id)

        return 1, task_id

    _validate_split_ratios = staticmethod(_validate_split_ratios)

    def clear_cache(self) -> None:
        self._sample_cache.clear()