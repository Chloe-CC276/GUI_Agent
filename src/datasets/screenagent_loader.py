"""
screenagent_loader

将ScreenAgent数据集转换为GUI Agent数据结构

数据层级：
    ScreenAgent train/test
        └── session 目录
            ├── images
            │   └── <timestamp>.jpg
            └── <timestamp>_translate.json

转换结果：
    一个 session -> 一个 GUITaskSample
    一个原始动作 -> 一个 GUITaskStep
    ScreenAgent action -> executor.action.Action

支持功能：
    1. 扫描 Session
    2. 读取并验证 JSON
    3. 解析中英文任务字段
    4. 转换鼠标、键盘和等待动作
    5. 按时间排序
    6. 数据统计
    7. 关键字搜索
    8. JSONL 导出
    9. 按 Session 划分训练、验证和测试集
"""


from __future__ import annotations

import csv
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

from .schema import (
    DatasetSplit,
    DatasetStatistics,
    GUITaskSample,
    GUITaskStep,
)

from ..executor.action import (
    Action,
    ActionType,
    MouseButton,
)


LOGGER = logging.getLogger(__name__)


class ScreenAgentDataError(ValueError):
    """ScreenAgent 数据内容不合法。"""


class UnsupportedScreenAgentActionError(ScreenAgentDataError):
    """ScreenAgent 动作类型暂不受支持。"""


class ScreenAgentLoader:
    """
    ScreenAgent 数据加载器。

    Parameters
    ----------
    data_root:
        ScreenAgent train 或 test 目录。

        示例：
        external/ScreenAgent/data/ScreenAgent/train

    language:
        读取语言，可选：
        - "zh"
        - "en"
        - "auto"

    strict:
        True：
            遇到损坏 JSON、缺失截图或未知动作时直接抛出异常。

        False：
            记录警告并跳过有问题的文件或动作。

    require_images:
        是否要求 saved_image_name 对应的截图真实存在。

    recursive_sessions:
        是否递归寻找包含 JSON 文件的目录。

    encoding:
        JSON 文件编码，ScreenAgent 通常为 UTF-8。
    """

    SUPPORTED_LANGUAGES = {"zh", "en", "auto"}

    JSON_PATTERNS = (
        "*.json",
        "*_translate.json",
    )

    KEY_ALIASES = {
        # 常用控制键
        "return": "enter",
        "enter": "enter",
        "escape": "esc",
        "esc": "esc",
        "backspace": "backspace",
        "delete": "delete",
        "tab": "tab",
        "space": "space",
        "spacebar": "space",

        # 方向键
        "left": "left",
        "right": "right",
        "up": "up",
        "down": "down",

        # 修饰键
        "control_l": "ctrl",
        "control_r": "ctrl",
        "ctrl_l": "ctrl",
        "ctrl_r": "ctrl",
        "ctrl": "ctrl",
        "shift_l": "shift",
        "shift_r": "shift",
        "shift": "shift",
        "alt_l": "alt",
        "alt_r": "alt",
        "alt": "alt",
        "super_l": "winleft",
        "super_r": "winright",
        "meta_l": "winleft",
        "meta_r": "winright",

        # 页面按键
        "prior": "pageup",
        "page_up": "pageup",
        "pageup": "pageup",
        "next": "pagedown",
        "page_down": "pagedown",
        "pagedown": "pagedown",
        "home": "home",
        "end": "end",

        # 编辑快捷键中的名称
        "plus": "+",
        "minus": "-",
    }

    def __init__(
        self,
        data_root: str | Path,
        language: str = "zh",
        strict: bool = False,
        require_images: bool = True,
        recursive_sessions: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.language = language.strip().lower()
        self.strict = strict
        self.require_images = require_images
        self.recursive_sessions = recursive_sessions
        self.encoding = encoding

        self._validate_configuration()

        self.session_dirs = self._discover_session_dirs()
        self._sample_cache: dict[Path, GUITaskSample] = {}

        LOGGER.info(
            "ScreenAgentLoader initialized: root=%s, sessions=%d",
            self.data_root,
            len(self.session_dirs),
        )

    # ================================================================
    # Public sequence interface
    # ================================================================

    def __len__(self) -> int:
        """返回 Session 数量。"""
        return len(self.session_dirs)

    def __getitem__(
        self,
        index: int | slice,
    ) -> GUITaskSample | list[GUITaskSample]:
        """
        按索引读取 Session。

        Example
        -------
        sample = loader[0]
        samples = loader[:5]
        """
        if isinstance(index, slice):
            return [
                self.load_session(path)
                for path in self.session_dirs[index]
            ]

        session_path = self.session_dirs[index]
        return self.load_session(session_path)

    def __iter__(self) -> Iterator[GUITaskSample]:
        """逐个产生 GUITaskSample。"""
        for session_path in self.session_dirs:
            try:
                yield self.load_session(session_path)
            except Exception as error:
                if self.strict:
                    raise

                LOGGER.warning(
                    "Skipping session %s: %s",
                    session_path,
                    error,
                )

    # ================================================================
    # Main loading methods
    # ================================================================

    def load_session(
        self,
        session: str | Path,
        use_cache: bool = True,
    ) -> GUITaskSample:
        """
        加载单个 Session。

        Parameters
        ----------
        session:
            Session 目录路径、目录名或 session_id。

        use_cache:
            是否使用内存缓存。
        """
        session_path = self._resolve_session_path(session)

        if use_cache and session_path in self._sample_cache:
            return self._sample_cache[session_path]

        json_files = self._list_session_json_files(session_path)

        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in session: {session_path}"
            )

        all_steps: list[GUITaskStep] = []
        source_files: list[str] = []
        session_records: list[dict[str, Any]] = []

        step_id = 0

        for json_path in json_files:
            try:
                record = self._read_json(json_path)
                record_steps = self._build_steps(
                    record=record,
                    json_path=json_path,
                    session_path=session_path,
                    start_step_id=step_id,
                )
            except Exception as error:
                if self.strict:
                    raise

                LOGGER.warning(
                    "Skipping JSON file %s: %s",
                    json_path,
                    error,
                )
                continue

            if not record_steps:
                continue

            all_steps.extend(record_steps)
            session_records.append(record)
            source_files.append(str(json_path))
            step_id += len(record_steps)

        if not all_steps:
            raise ScreenAgentDataError(
                f"No valid steps were loaded from session: {session_path}"
            )

        first_record = session_records[0]

        session_id = str(
            first_record.get("session_id")
            or session_path.name
        )

        instruction = self._select_language_field(
            first_record,
            base_name="task_prompt",
            default="",
        )

        if not instruction:
            raise ScreenAgentDataError(
                f"Missing task_prompt in session: {session_path}"
            )

        sample = GUITaskSample(
            task_id=session_id,
            source="screenagent",
            instruction=instruction,
            steps=all_steps,
            language=self._resolved_language(first_record),
            metadata={
                "session_path": str(session_path),
                "json_file_count": len(source_files),
                "source_files": source_files,
                "video_width": first_record.get("video_width"),
                "video_height": first_record.get("video_height"),
                "task_prompt_en": first_record.get("task_prompt_en"),
                "task_prompt_zh": first_record.get("task_prompt_zh"),
            },
        )

        if use_cache:
            self._sample_cache[session_path] = sample

        return sample

    def load_all(
        self,
        limit: int | None = None,
    ) -> list[GUITaskSample]:
        """
        加载全部或指定数量的 Session。

        Parameters
        ----------
        limit:
            最大 Session 数；None 表示全部。
        """
        if limit is not None:
            if not isinstance(limit, int):
                raise TypeError("limit must be int or None.")

            if limit <= 0:
                raise ValueError("limit must be greater than zero.")

        paths = (
            self.session_dirs
            if limit is None
            else self.session_dirs[:limit]
        )

        samples: list[GUITaskSample] = []

        for path in paths:
            try:
                samples.append(self.load_session(path))
            except Exception as error:
                if self.strict:
                    raise

                LOGGER.warning(
                    "Skipping session %s: %s",
                    path,
                    error,
                )

        return samples

    def load_json_file(
        self,
        json_path: str | Path,
    ) -> list[GUITaskStep]:
        """
        单独加载一个 ScreenAgent JSON。

        一个 JSON 中可能包含多个原子动作，因此返回 Step 列表。
        """
        path = Path(json_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"JSON file not found: {path}"
            )

        record = self._read_json(path)
        session_path = path.parent

        return self._build_steps(
            record=record,
            json_path=path,
            session_path=session_path,
            start_step_id=0,
        )

    # ================================================================
    # Session discovery
    # ================================================================

    def _validate_configuration(self) -> None:
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"ScreenAgent data root not found: {self.data_root}"
            )

        if not self.data_root.is_dir():
            raise NotADirectoryError(
                f"ScreenAgent data root is not a directory: "
                f"{self.data_root}"
            )

        if self.language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {self.language!r}. "
                f"Expected one of {sorted(self.SUPPORTED_LANGUAGES)}."
            )

    def _discover_session_dirs(self) -> list[Path]:
        """
        查找所有包含 JSON 文件的 Session 目录。

        同时兼容：

        train/
            <session_id>/
                images/
                *.json

        以及：

        train/
            *.json
            images/
        """
        candidates: set[Path] = set()

        if any(self.data_root.glob("*.json")):
            candidates.add(self.data_root)

        iterator = (
            self.data_root.rglob("*.json")
            if self.recursive_sessions
            else self.data_root.glob("*/*.json")
        )

        for json_path in iterator:
            candidates.add(json_path.parent)

        session_dirs = sorted(
            candidates,
            key=lambda path: self._natural_sort_key(
                str(path.relative_to(self.data_root))
            ),
        )

        if not session_dirs:
            raise FileNotFoundError(
                "No ScreenAgent JSON files were found under: "
                f"{self.data_root}"
            )

        return session_dirs

    def _resolve_session_path(
        self,
        session: str | Path,
    ) -> Path:
        path = Path(session)

        if path.exists():
            resolved = path.expanduser().resolve()

            if not resolved.is_dir():
                raise NotADirectoryError(
                    f"Session path is not a directory: {resolved}"
                )

            return resolved

        session_text = str(session)

        exact_matches = [
            path
            for path in self.session_dirs
            if path.name == session_text
        ]

        if len(exact_matches) == 1:
            return exact_matches[0]

        if len(exact_matches) > 1:
            raise ScreenAgentDataError(
                f"Multiple session directories match {session_text!r}."
            )

        raise FileNotFoundError(
            f"ScreenAgent session not found: {session}"
        )

    def _list_session_json_files(
        self,
        session_path: Path,
    ) -> list[Path]:
        files = [
            path
            for path in session_path.glob("*.json")
            if path.is_file()
        ]

        return sorted(
            files,
            key=lambda path: self._natural_sort_key(path.name),
        )

    # ================================================================
    # JSON and step parsing
    # ================================================================

    def _read_json(
        self,
        json_path: Path,
    ) -> dict[str, Any]:
        try:
            with json_path.open(
                "r",
                encoding=self.encoding,
            ) as file:
                data = json.load(file)
        except UnicodeDecodeError as error:
            raise ScreenAgentDataError(
                f"Cannot decode {json_path} using "
                f"{self.encoding}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise ScreenAgentDataError(
                f"Invalid JSON file {json_path}: {error}"
            ) from error

        if not isinstance(data, dict):
            raise ScreenAgentDataError(
                f"Expected JSON object in {json_path}, "
                f"got {type(data).__name__}."
            )

        return data

    def _build_steps(
        self,
        record: dict[str, Any],
        json_path: Path,
        session_path: Path,
        start_step_id: int,
    ) -> list[GUITaskStep]:
        raw_actions = record.get("actions", [])

        if raw_actions is None:
            raw_actions = []

        if not isinstance(raw_actions, list):
            raise ScreenAgentDataError(
                f"'actions' must be a list in {json_path}."
            )

        if not raw_actions:
            message = f"No actions found in {json_path}"

            if self.strict:
                raise ScreenAgentDataError(message)

            LOGGER.warning(message)
            return []

        screenshot_path = self._resolve_screenshot_path(
            saved_image_name=record.get("saved_image_name"),
            session_path=session_path,
            json_path=json_path,
        )

        instruction = self._select_language_field(
            record,
            base_name="send_prompt",
            default=self._select_language_field(
                record,
                base_name="task_prompt",
                default="",
            ),
        )

        task_prompt = self._select_language_field(
            record,
            base_name="task_prompt",
            default="",
        )

        llm_response = record.get("LLM_response")

        corrected_response = self._select_language_field(
            record,
            base_name="LLM_response_editer",
            default=record.get("LLM_response_editer"),
        )

        steps: list[GUITaskStep] = []

        for action_index, raw_action in enumerate(raw_actions):
            if not isinstance(raw_action, dict):
                error = ScreenAgentDataError(
                    f"Action at index {action_index} in "
                    f"{json_path} is not an object."
                )

                if self.strict:
                    raise error

                LOGGER.warning("%s", error)
                continue

            try:
                action = self._convert_action(
                    raw_action=raw_action,
                    record=record,
                    json_path=json_path,
                    action_index=action_index,
                )
            except Exception as error:
                if self.strict:
                    raise

                LOGGER.warning(
                    "Skipping action %d in %s: %s",
                    action_index,
                    json_path,
                    error,
                )
                continue

            step = GUITaskStep(
                step_id=start_step_id + len(steps),
                screenshot_path=screenshot_path,
                instruction=instruction,
                action=action,
                llm_response=llm_response,
                corrected_response=corrected_response,
                language=self._resolved_language(record),
                metadata={
                    "source": "screenagent",
                    "session_id": record.get("session_id"),
                    "task_prompt": task_prompt,
                    "task_prompt_en": record.get("task_prompt_en"),
                    "task_prompt_zh": record.get("task_prompt_zh"),
                    "json_path": str(json_path),
                    "saved_image_name": record.get("saved_image_name"),
                    "video_width": record.get("video_width"),
                    "video_height": record.get("video_height"),
                    "action_index_in_json": action_index,
                    "action_count_in_json": len(raw_actions),
                    "raw_action": raw_action,
                },
            )

            steps.append(step)

        return steps

    def _resolve_screenshot_path(
        self,
        saved_image_name: Any,
        session_path: Path,
        json_path: Path,
    ) -> Path:
        if not isinstance(saved_image_name, str):
            raise ScreenAgentDataError(
                f"Missing or invalid saved_image_name in {json_path}."
            )

        image_name = Path(saved_image_name).name

        candidates = [
            session_path / "images" / image_name,
            session_path / image_name,
            json_path.parent / "images" / image_name,
            self.data_root / "images" / image_name,
        ]

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        unresolved = candidates[0].resolve()

        if self.require_images:
            raise FileNotFoundError(
                f"Screenshot not found for {json_path}. "
                f"Expected: {unresolved}"
            )

        LOGGER.warning(
            "Screenshot does not exist: %s",
            unresolved,
        )

        return unresolved

    # ================================================================
    # Action conversion
    # ================================================================

    def _convert_action(
        self,
        raw_action: dict[str, Any],
        record: dict[str, Any],
        json_path: Path,
        action_index: int,
    ) -> Action:
        action_type = self._normalise_name(
            raw_action.get("action_type")
        )

        if action_type == "mouseaction":
            action = self._convert_mouse_action(raw_action)

        elif action_type == "keyboardaction":
            action = self._convert_keyboard_action(raw_action)

        elif action_type == "waitaction":
            action = self._convert_wait_action(raw_action)

        else:
            raise UnsupportedScreenAgentActionError(
                f"Unsupported ScreenAgent action_type "
                f"{raw_action.get('action_type')!r} "
                f"in {json_path}."
            )

        action.metadata.update(
            {
                "dataset": "screenagent",
                "session_id": record.get("session_id"),
                "source_json": str(json_path),
                "source_action_index": action_index,
                "raw_action": raw_action,
            }
        )

        return action

    def _convert_mouse_action(
        self,
        raw_action: dict[str, Any],
    ) -> Action:
        mouse_action_type = self._normalise_name(
            raw_action.get("mouse_action_type")
        )

        button = self._normalise_mouse_button(
            raw_action.get("mouse_button", "left")
        )

        x, y = self._extract_mouse_position(
            raw_action,
            required=mouse_action_type
            in {
                "click",
                "double_click",
                "doubleclick",
                "move",
                "move_to",
                "drag",
                "drag_to",
            },
        )

        if mouse_action_type == "click":
            if button == MouseButton.RIGHT:
                return Action.right_click(
                    x=x,
                    y=y,
                    description="ScreenAgent right click",
                )

            if button == MouseButton.MIDDLE:
                return Action(
                    type=ActionType.MIDDLE_CLICK,
                    x=x,
                    y=y,
                    button=button,
                    description="ScreenAgent middle click",
                )

            return Action.click(
                x=x,
                y=y,
                button=button,
                description="ScreenAgent click",
            )

        if mouse_action_type in {
            "double_click",
            "doubleclick",
        }:
            return Action.double_click(
                x=x,
                y=y,
                description="ScreenAgent double click",
            )

        if mouse_action_type in {
            "move",
            "move_to",
        }:
            return Action(
                type=ActionType.MOVE_TO,
                x=x,
                y=y,
                description="ScreenAgent move mouse",
            )

        if mouse_action_type in {
            "drag",
            "drag_to",
        }:
            return Action.drag_to(
                x=self._require_int(x, "mouse_position.width"),
                y=self._require_int(y, "mouse_position.height"),
                button=button,
                description="ScreenAgent drag to",
            )

        if mouse_action_type == "scroll_up":
            repeat = self._extract_scroll_repeat(raw_action)

            return Action.scroll(
                amount=repeat,
                x=x,
                y=y,
                description="ScreenAgent scroll up",
            )

        if mouse_action_type == "scroll_down":
            repeat = self._extract_scroll_repeat(raw_action)

            return Action.scroll(
                amount=-repeat,
                x=x,
                y=y,
                description="ScreenAgent scroll down",
            )

        if mouse_action_type == "mouse_down":
            return Action(
                type=ActionType.MOUSE_DOWN,
                x=x,
                y=y,
                button=button,
                description="ScreenAgent mouse down",
            )

        if mouse_action_type == "mouse_up":
            return Action(
                type=ActionType.MOUSE_UP,
                x=x,
                y=y,
                button=button,
                description="ScreenAgent mouse up",
            )

        raise UnsupportedScreenAgentActionError(
            "Unsupported mouse_action_type: "
            f"{raw_action.get('mouse_action_type')!r}"
        )

    def _convert_keyboard_action(
        self,
        raw_action: dict[str, Any],
    ) -> Action:
        keyboard_action_type = self._normalise_name(
            raw_action.get("keyboard_action_type")
        )

        # 部分 ScreenAgent 数据可能没有 keyboard_action_type，
        # 但存在 keyboard_text。
        if not keyboard_action_type:
            if "keyboard_text" in raw_action:
                keyboard_action_type = "text"
            elif "keyboard_key" in raw_action:
                keyboard_action_type = "press"

        if keyboard_action_type in {
            "text",
            "type",
            "type_text",
            "write",
        }:
            text = raw_action.get("keyboard_text")

            if not isinstance(text, str):
                raise ScreenAgentDataError(
                    "Keyboard text action requires keyboard_text."
                )

            return Action.type_text(
                text=text,
                description="ScreenAgent type text",
            )

        if keyboard_action_type in {
            "press",
            "key",
            "keypress",
        }:
            raw_key = raw_action.get("keyboard_key")

            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ScreenAgentDataError(
                    "Keyboard press action requires keyboard_key."
                )

            keys = self._parse_key_combination(raw_key)

            if len(keys) == 1:
                return Action.press_key(
                    key=keys[0],
                    description=f"ScreenAgent press {keys[0]}",
                )

            return Action.hotkey_action(
                *keys,
                description=(
                    "ScreenAgent hotkey "
                    + "+".join(keys)
                ),
            )

        if keyboard_action_type in {
            "hotkey",
            "shortcut",
        }:
            raw_keys = (
                raw_action.get("keyboard_keys")
                or raw_action.get("keyboard_key")
            )

            keys = self._parse_keys_value(raw_keys)

            if len(keys) < 2:
                raise ScreenAgentDataError(
                    "Hotkey action requires at least two keys."
                )

            return Action.hotkey_action(
                *keys,
                description=(
                    "ScreenAgent hotkey "
                    + "+".join(keys)
                ),
            )

        if keyboard_action_type == "key_down":
            key = self._single_normalised_key(
                raw_action.get("keyboard_key")
            )

            return Action(
                type=ActionType.KEY_DOWN,
                key=key,
                description=f"ScreenAgent key down {key}",
            )

        if keyboard_action_type == "key_up":
            key = self._single_normalised_key(
                raw_action.get("keyboard_key")
            )

            return Action(
                type=ActionType.KEY_UP,
                key=key,
                description=f"ScreenAgent key up {key}",
            )

        raise UnsupportedScreenAgentActionError(
            "Unsupported keyboard_action_type: "
            f"{raw_action.get('keyboard_action_type')!r}"
        )

    @staticmethod
    def _convert_wait_action(
        raw_action: dict[str, Any],
    ) -> Action:
        seconds = raw_action.get(
            "wait_time",
            raw_action.get("seconds"),
        )

        if not isinstance(seconds, (int, float)):
            raise ScreenAgentDataError(
                "WaitAction requires numeric wait_time."
            )

        if seconds < 0:
            raise ScreenAgentDataError(
                "wait_time must not be negative."
            )

        return Action.wait(
            seconds=float(seconds),
            description="ScreenAgent wait",
        )

    # ================================================================
    # Action helper methods
    # ================================================================

    @classmethod
    def _extract_mouse_position(
        cls,
        raw_action: dict[str, Any],
        required: bool,
    ) -> tuple[int | None, int | None]:
        position = raw_action.get("mouse_position")

        if position is None:
            if required:
                raise ScreenAgentDataError(
                    "Mouse action requires mouse_position."
                )

            return None, None

        if not isinstance(position, dict):
            raise ScreenAgentDataError(
                "mouse_position must be an object."
            )

        x = position.get("width", position.get("x"))
        y = position.get("height", position.get("y"))

        if x is None and y is None and not required:
            return None, None

        return (
            cls._require_int(x, "mouse_position.width"),
            cls._require_int(y, "mouse_position.height"),
        )

    @staticmethod
    def _extract_scroll_repeat(
        raw_action: dict[str, Any],
    ) -> int:
        repeat = raw_action.get(
            "scroll_repeat",
            raw_action.get("amount", 1),
        )

        if not isinstance(repeat, int) or isinstance(repeat, bool):
            raise ScreenAgentDataError(
                "scroll_repeat must be an integer."
            )

        if repeat <= 0:
            raise ScreenAgentDataError(
                "scroll_repeat must be greater than zero."
            )

        return repeat

    @staticmethod
    def _normalise_mouse_button(
        value: Any,
    ) -> MouseButton:
        normalised = str(value).strip().lower()

        aliases = {
            "button1": "left",
            "button2": "middle",
            "button3": "right",
            "1": "left",
            "2": "middle",
            "3": "right",
        }

        normalised = aliases.get(
            normalised,
            normalised,
        )

        try:
            return MouseButton(normalised)
        except ValueError as error:
            raise ScreenAgentDataError(
                f"Unsupported mouse button: {value!r}"
            ) from error

    @classmethod
    def _parse_key_combination(
        cls,
        raw_key: str,
    ) -> tuple[str, ...]:
        text = raw_key.strip()

        if not text:
            raise ScreenAgentDataError(
                "keyboard_key must not be empty."
            )

        # 保留单独的加号键。
        if text == "+":
            return ("+",)

        parts = [
            part.strip()
            for part in re.split(r"\s*\+\s*", text)
            if part.strip()
        ]

        if not parts:
            raise ScreenAgentDataError(
                f"Invalid keyboard key: {raw_key!r}"
            )

        return tuple(
            cls._normalise_keyboard_key(part)
            for part in parts
        )

    @classmethod
    def _parse_keys_value(
        cls,
        value: Any,
    ) -> tuple[str, ...]:
        if isinstance(value, str):
            return cls._parse_key_combination(value)

        if isinstance(value, Sequence):
            keys = tuple(
                cls._normalise_keyboard_key(str(key))
                for key in value
            )

            if not keys:
                raise ScreenAgentDataError(
                    "keyboard_keys must not be empty."
                )

            return keys

        raise ScreenAgentDataError(
            "keyboard_keys must be a string or sequence."
        )

    @classmethod
    def _single_normalised_key(
        cls,
        value: Any,
    ) -> str:
        if not isinstance(value, str):
            raise ScreenAgentDataError(
                "keyboard_key must be a string."
            )

        keys = cls._parse_key_combination(value)

        if len(keys) != 1:
            raise ScreenAgentDataError(
                "Expected one keyboard key."
            )

        return keys[0]

    @classmethod
    def _normalise_keyboard_key(
        cls,
        key: str,
    ) -> str:
        text = key.strip()

        if not text:
            raise ScreenAgentDataError(
                "Keyboard key must not be empty."
            )

        normalised = text.lower().replace("-", "_")

        if normalised in cls.KEY_ALIASES:
            return cls.KEY_ALIASES[normalised]

        # X11 Function keys：F1、F2 等
        if re.fullmatch(r"f\d{1,2}", normalised):
            return normalised

        # 单个字母、数字或符号。
        if len(text) == 1:
            return text.lower()

        return normalised

    # ================================================================
    # Language handling
    # ================================================================

    def _select_language_field(
        self,
        record: dict[str, Any],
        base_name: str,
        default: Any = None,
    ) -> Any:
        """
        按 language 选择字段。

        例如 base_name="task_prompt"：
        zh   -> task_prompt_zh
        en   -> task_prompt_en
        auto -> 优先原字段，然后 zh，再 en
        """
        if self.language == "zh":
            candidates = (
                f"{base_name}_zh",
                base_name,
                f"{base_name}_en",
            )
        elif self.language == "en":
            candidates = (
                f"{base_name}_en",
                base_name,
                f"{base_name}_zh",
            )
        else:
            candidates = (
                base_name,
                f"{base_name}_zh",
                f"{base_name}_en",
            )

        for field_name in candidates:
            value = record.get(field_name)

            if value is not None and value != "":
                return value

        return default

    def _resolved_language(
        self,
        record: dict[str, Any],
    ) -> str:
        if self.language in {"zh", "en"}:
            return self.language

        task_prompt = record.get("task_prompt", "")

        if isinstance(task_prompt, str) and re.search(
            r"[\u4e00-\u9fff]",
            task_prompt,
        ):
            return "zh"

        return "en"

    # ================================================================
    # Statistics, search and export
    # ================================================================

    def statistics(
        self,
        limit: int | None = None,
    ) -> DatasetStatistics:
        """统计任务数量、Step 数量和动作分布。"""
        samples = self.load_all(limit=limit)

        action_counter: Counter[str] = Counter()
        language_counter: Counter[str] = Counter()

        num_steps = 0

        for sample in samples:
            language_counter[sample.language] += 1
            num_steps += sample.num_steps

            for step in sample.steps:
                action_type = step.action.type

                if isinstance(action_type, ActionType):
                    action_name = action_type.value
                else:
                    action_name = str(action_type)

                action_counter[action_name] += 1

        num_tasks = len(samples)

        return DatasetStatistics(
            source="screenagent",
            num_tasks=num_tasks,
            num_steps=num_steps,
            avg_steps_per_task=(
                num_steps / num_tasks
                if num_tasks
                else 0.0
            ),
            action_distribution=dict(action_counter),
            language_distribution=dict(language_counter),
            metadata={
                "data_root": str(self.data_root),
                "strict": self.strict,
                "require_images": self.require_images,
            },
        )

    def find(
        self,
        keyword: str,
        case_sensitive: bool = False,
        limit: int | None = None,
    ) -> list[GUITaskSample]:
        """
        根据任务、指令或模型回复搜索 Session。
        """
        if not isinstance(keyword, str):
            raise TypeError("keyword must be a string.")

        query = keyword.strip()

        if not query:
            raise ValueError("keyword must not be empty.")

        if not case_sensitive:
            query = query.casefold()

        results: list[GUITaskSample] = []

        for sample in self:
            searchable_parts = [
                sample.instruction,
                *[
                    step.instruction
                    for step in sample.steps
                ],
                *[
                    step.llm_response or ""
                    for step in sample.steps
                ],
                *[
                    step.corrected_response or ""
                    for step in sample.steps
                ],
            ]

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

    def export_csv(
        self,
        output_path: str | Path,
        samples: Sequence[GUITaskSample] | None = None,
        *,
        encoding: str = "utf-8-sig",
        include_action_json: bool = True,
        include_metadata_json: bool = True,
    ) -> Path:
        """
        将 ScreenAgent 数据导出为 CSV。

        导出粒度
        --------
        每个 ``GUITaskStep`` 对应 CSV 中的一行，因此一个 Session 中的
        多个动作会被展开成多行。任务级字段（task_id、task_instruction
        等）会在每个 Step 行中重复，便于直接使用 pandas 读取、筛选和
        训练。

        复杂字段处理
        ------------
        - ``action`` 的常用字段会被展开为独立列；
        - 完整 action 可选保存到 ``action_json``；
        - task/step metadata 可选保存为 JSON 字符串；
        - 截图仅保存文件路径，不把图片二进制写入 CSV。

        Parameters
        ----------
        output_path:
            CSV 输出路径。

        samples:
            指定导出的样本；None 时导出全部 Session。

        encoding:
            默认 ``utf-8-sig``，便于 Windows Excel 正确显示中文。

        include_action_json:
            是否保留完整 Action JSON 字符串。

        include_metadata_json:
            是否保留 task_metadata 和 step_metadata JSON 字符串。
        """
        destination = Path(
            output_path
        ).expanduser().resolve()

        if destination.suffix.lower() != ".csv":
            destination = destination.with_suffix(".csv")

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = (
            list(samples)
            if samples is not None
            else self.load_all()
        )

        rows: list[dict[str, Any]] = []

        for sample in records:
            for step in sample.steps:
                rows.append(
                    self._step_to_csv_row(
                        sample=sample,
                        step=step,
                        include_action_json=include_action_json,
                        include_metadata_json=include_metadata_json,
                    )
                )

        fieldnames = self._csv_fieldnames(rows)

        with destination.open(
            "w",
            encoding=encoding,
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        field_name: row.get(field_name, "")
                        for field_name in fieldnames
                    }
                )

        LOGGER.info(
            "Exported ScreenAgent CSV: path=%s, tasks=%d, rows=%d",
            destination,
            len(records),
            len(rows),
        )

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
        file_prefix: str = "screenagent",
        encoding: str = "utf-8-sig",
    ) -> dict[str, Path]:
        """
        按 Session 划分数据并分别导出 train/validation/test CSV。

        同一 Session 的所有 Step 始终位于同一数据集，避免轨迹泄漏。
        """
        destination = Path(
            output_dir
        ).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)

        dataset_split = self.split(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
            shuffle=shuffle,
        )

        paths = {
            "train": self.export_csv(
                destination / f"{file_prefix}_train.csv",
                samples=dataset_split.train,
                encoding=encoding,
            ),
            "validation": self.export_csv(
                destination / f"{file_prefix}_validation.csv",
                samples=dataset_split.validation,
                encoding=encoding,
            ),
            "test": self.export_csv(
                destination / f"{file_prefix}_test.csv",
                samples=dataset_split.test,
                encoding=encoding,
            ),
        }

        return paths

    def export_statistics_csv(
        self,
        output_path: str | Path,
        *,
        limit: int | None = None,
        encoding: str = "utf-8-sig",
    ) -> Path:
        """将数据集总体统计和动作/语言分布导出为 CSV。"""
        destination = Path(
            output_path
        ).expanduser().resolve()

        if destination.suffix.lower() != ".csv":
            destination = destination.with_suffix(".csv")

        destination.parent.mkdir(parents=True, exist_ok=True)
        stats = self.statistics(limit=limit)

        rows: list[dict[str, Any]] = [
            {
                "category": "summary",
                "name": "num_tasks",
                "value": stats.num_tasks,
            },
            {
                "category": "summary",
                "name": "num_steps",
                "value": stats.num_steps,
            },
            {
                "category": "summary",
                "name": "avg_steps_per_task",
                "value": stats.avg_steps_per_task,
            },
        ]

        rows.extend(
            {
                "category": "action_distribution",
                "name": action_name,
                "value": count,
            }
            for action_name, count in sorted(
                stats.action_distribution.items()
            )
        )

        rows.extend(
            {
                "category": "language_distribution",
                "name": language_name,
                "value": count,
            }
            for language_name, count in sorted(
                stats.language_distribution.items()
            )
        )

        with destination.open(
            "w",
            encoding=encoding,
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["category", "name", "value"],
            )
            writer.writeheader()
            writer.writerows(rows)

        return destination

    @classmethod
    def _step_to_csv_row(
        cls,
        *,
        sample: GUITaskSample,
        step: GUITaskStep,
        include_action_json: bool,
        include_metadata_json: bool,
    ) -> dict[str, Any]:
        """将一个 GUITaskStep 展平为单行 CSV 记录。"""
        action_dict = cls._action_to_dict(step.action)

        action_type = action_dict.get("type")

        if isinstance(action_type, ActionType):
            action_type = action_type.value

        row: dict[str, Any] = {
            # Task-level fields
            "task_id": sample.task_id,
            "source": sample.source,
            "task_instruction": sample.instruction,
            "task_language": sample.language,
            "task_num_steps": sample.num_steps,

            # Step-level fields
            "step_id": step.step_id,
            "step_instruction": step.instruction,
            "screenshot_path": (
                str(step.screenshot_path)
                if step.screenshot_path is not None
                else ""
            ),
            "screenshot_exists": step.screenshot_exists,
            "step_language": step.language,
            "llm_response": step.llm_response or "",
            "corrected_response": step.corrected_response or "",

            # Common action fields
            "action_type": action_type or "",
            "action_status": cls._enum_value(
                action_dict.get("status")
            ),
            "action_description": action_dict.get(
                "description",
                "",
            ),
            "x": action_dict.get("x", ""),
            "y": action_dict.get("y", ""),
            "offset_x": action_dict.get("offset_x", ""),
            "offset_y": action_dict.get("offset_y", ""),
            "button": cls._enum_value(
                action_dict.get("button")
            ),
            "clicks": action_dict.get("clicks", ""),
            "interval": action_dict.get("interval", ""),
            "duration": action_dict.get("duration", ""),
            "amount": action_dict.get("amount", ""),
            "key": action_dict.get("key", ""),
            "keys": cls._json_cell(
                action_dict.get("keys")
            ),
            "text": action_dict.get("text", ""),
            "seconds": action_dict.get("seconds", ""),
        }

        # Preserve additional Action fields without losing information.
        common_action_fields = {
            "type",
            "status",
            "description",
            "x",
            "y",
            "offset_x",
            "offset_y",
            "button",
            "clicks",
            "interval",
            "duration",
            "amount",
            "key",
            "keys",
            "text",
            "seconds",
            "metadata",
        }

        for key, value in action_dict.items():
            if key in common_action_fields:
                continue

            column_name = f"action_{key}"
            row[column_name] = cls._csv_scalar(value)

        if include_action_json:
            row["action_json"] = cls._json_cell(
                action_dict
            )

        if include_metadata_json:
            row["task_metadata_json"] = cls._json_cell(
                sample.metadata
            )
            row["step_metadata_json"] = cls._json_cell(
                step.metadata
            )
            row["action_metadata_json"] = cls._json_cell(
                action_dict.get("metadata", {})
            )

        return row

    @staticmethod
    def _action_to_dict(action: Any) -> dict[str, Any]:
        if isinstance(action, dict):
            return dict(action)

        to_dict = getattr(action, "to_dict", None)

        if callable(to_dict):
            converted = to_dict()

            if not isinstance(converted, dict):
                raise ScreenAgentDataError(
                    "Action.to_dict() must return a dictionary."
                )

            return converted

        raise ScreenAgentDataError(
            "CSV export requires Action or dictionary action data."
        )

    @staticmethod
    def _enum_value(value: Any) -> Any:
        enum_value = getattr(value, "value", None)
        return enum_value if enum_value is not None else value

    @classmethod
    def _csv_scalar(cls, value: Any) -> Any:
        if value is None:
            return ""

        value = cls._enum_value(value)

        if isinstance(value, (str, int, float, bool)):
            return value

        return cls._json_cell(value)

    @classmethod
    def _json_cell(cls, value: Any) -> str:
        if value is None:
            return ""

        def default(item: Any) -> Any:
            enum_value = getattr(item, "value", None)

            if enum_value is not None:
                return enum_value

            to_dict = getattr(item, "to_dict", None)

            if callable(to_dict):
                return to_dict()

            if isinstance(item, Path):
                return str(item)

            return str(item)

        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=default,
        )

    @staticmethod
    def _csv_fieldnames(
        rows: Sequence[dict[str, Any]],
    ) -> list[str]:
        """生成稳定列顺序，同时保留动态 Action 字段。"""
        preferred = [
            "task_id",
            "source",
            "task_instruction",
            "task_language",
            "task_num_steps",
            "step_id",
            "step_instruction",
            "screenshot_path",
            "screenshot_exists",
            "step_language",
            "llm_response",
            "corrected_response",
            "action_type",
            "action_status",
            "action_description",
            "x",
            "y",
            "offset_x",
            "offset_y",
            "button",
            "clicks",
            "interval",
            "duration",
            "amount",
            "key",
            "keys",
            "text",
            "seconds",
            "action_json",
            "task_metadata_json",
            "step_metadata_json",
            "action_metadata_json",
        ]

        discovered = {
            key
            for row in rows
            for key in row
        }

        ordered = [
            field_name
            for field_name in preferred
            if field_name in discovered
        ]

        ordered.extend(
            sorted(discovered.difference(ordered))
        )

        # Empty datasets still need a valid, predictable CSV header.
        return ordered or preferred

    def split(
        self,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        shuffle: bool = True,
    ) -> DatasetSplit:
        """
        按 Session 划分数据，避免同一轨迹泄漏。

        注意：
            这是本地划分，不等于 ScreenAgent 官方测试集。
        """
        ratios = (
            train_ratio,
            validation_ratio,
            test_ratio,
        )

        if any(
            not isinstance(value, (int, float))
            for value in ratios
        ):
            raise TypeError("Split ratios must be numeric.")

        if any(value < 0 for value in ratios):
            raise ValueError(
                "Split ratios must not be negative."
            )

        if abs(sum(ratios) - 1.0) > 1e-8:
            raise ValueError(
                "train_ratio + validation_ratio + "
                "test_ratio must equal 1.0."
            )

        samples = self.load_all()

        if shuffle:
            random.Random(seed).shuffle(samples)

        total = len(samples)
        train_end = int(total * train_ratio)
        validation_end = train_end + int(
            total * validation_ratio
        )

        return DatasetSplit(
            train=samples[:train_end],
            validation=samples[
                train_end:validation_end
            ],
            test=samples[validation_end:],
        )

    def clear_cache(self) -> None:
        """清除 Session 内存缓存。"""
        self._sample_cache.clear()

    # ================================================================
    # Generic helpers
    # ================================================================

    @staticmethod
    def _normalise_name(value: Any) -> str:
        if value is None:
            return ""

        return (
            str(value)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _require_int(
        value: Any,
        name: str,
    ) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise ScreenAgentDataError(
                f"{name} must be an integer, "
                f"got {value!r}."
            )

        return value

    @staticmethod
    def _natural_sort_key(
        value: str,
    ) -> tuple[Any, ...]:
        """
        自然排序：

        2.json 在 10.json 前；
        时间戳文件也可正确按名称顺序排列。
        """
        return tuple(
            int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", value)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"data_root={str(self.data_root)!r}, "
            f"language={self.language!r}, "
            f"sessions={len(self.session_dirs)}, "
            f"strict={self.strict}"
            f")"
        )