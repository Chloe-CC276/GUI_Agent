"""
action
1.定义所有鼠标和键盘操作类型
2.将JSON输出转换为结构化的Action对象
3.执行前验证参数
4.将操作序列转换为字典或JSON
5.实现多步骤任务
"""


from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence


class ActionType(str, Enum):
    """
    Supported GUI action types.
    """

    # Mouse movement
    MOVE_TO = "move_to"
    MOVE_BY = "move_by"

    # Mouse clicks
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"

    # Mouse button actions
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"

    # Mouse drag
    DRAG_TO = "drag_to"
    DRAG_BY = "drag_by"

    # Mouse scroll
    SCROLL = "scroll"
    HORIZONTAL_SCROLL = "horizontal_scroll"

    # Keyboard
    PRESS = "press"
    HOTKEY = "hotkey"
    TYPE_TEXT = "type_text"
    PASTE_TEXT = "paste_text"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"

    # Common keyboard actions
    ENTER = "enter"
    ESCAPE = "escape"
    TAB = "tab"
    BACKSPACE = "backspace"
    DELETE = "delete"
    COPY = "copy"
    PASTE = "paste"
    CUT = "cut"
    SELECT_ALL = "select_all"
    SAVE = "save"
    UNDO = "undo"
    REDO = "redo"
    FIND = "find"
    CLOSE_WINDOW = "close_window"
    SWITCH_WINDOW = "switch_window"

    # Flow control
    WAIT = "wait"
    FINISH = "finish"
    FAIL = "fail"


class MouseButton(str, Enum):
    """
    Supported mouse buttons.
    """

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class ActionStatus(str, Enum):
    """
    Action lifecycle status.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Action:

    type: ActionType | str

    action_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    description: str = ""

    # Absolute mouse coordinates
    x: Optional[int] = None
    y: Optional[int] = None

    # Relative mouse offsets
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None

    # Normalised coordinates
    normalised_x: Optional[float] = None
    normalised_y: Optional[float] = None

    # Mouse properties
    button: MouseButton | str = MouseButton.LEFT
    clicks: int = 1
    amount: Optional[int] = None

    # Keyboard properties
    text: Optional[str] = None
    key: Optional[str] = None
    keys: tuple[str, ...] = field(default_factory=tuple)
    presses: int = 1

    # Timing
    interval: float = 0.0
    duration: Optional[float] = None
    seconds: Optional[float] = None

    # Clipboard / text options
    restore_clipboard: bool = True
    use_clipboard_for_unicode: bool = True

    # Execution information
    status: ActionStatus | str = ActionStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalise enums and validate the action.
        """

        self.type = self._normalise_action_type(self.type)
        self.button = self._normalise_mouse_button(self.button)
        self.status = self._normalise_status(self.status)

        if isinstance(self.keys, list):
            self.keys = tuple(self.keys)

        self.validate()


    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate parameters according to the action type.
        """

        self._validate_common_fields()

        validators = {
            ActionType.MOVE_TO: self._validate_absolute_position,
            ActionType.MOVE_BY: self._validate_relative_position,
            ActionType.CLICK: self._validate_click,
            ActionType.DOUBLE_CLICK: self._validate_optional_position,
            ActionType.RIGHT_CLICK: self._validate_optional_position,
            ActionType.MIDDLE_CLICK: self._validate_optional_position,
            ActionType.MOUSE_DOWN: self._validate_optional_position,
            ActionType.MOUSE_UP: self._validate_optional_position,
            ActionType.DRAG_TO: self._validate_absolute_position,
            ActionType.DRAG_BY: self._validate_relative_position,
            ActionType.SCROLL: self._validate_scroll,
            ActionType.HORIZONTAL_SCROLL: self._validate_scroll,
            ActionType.PRESS: self._validate_press,
            ActionType.HOTKEY: self._validate_hotkey,
            ActionType.TYPE_TEXT: self._validate_text,
            ActionType.PASTE_TEXT: self._validate_text,
            ActionType.KEY_DOWN: self._validate_single_key,
            ActionType.KEY_UP: self._validate_single_key,
            ActionType.TAB: self._validate_presses,
            ActionType.BACKSPACE: self._validate_presses,
            ActionType.DELETE: self._validate_presses,
            ActionType.WAIT: self._validate_wait,
        }

        validator = validators.get(self.type)

        if validator is not None:
            validator()

    def _validate_common_fields(self) -> None:
        if not isinstance(self.action_id, str):
            raise TypeError("action_id must be a string.")

        if not self.action_id.strip():
            raise ValueError("action_id must not be empty.")

        if not isinstance(self.description, str):
            raise TypeError("description must be a string.")

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        self._validate_non_negative_number(
            self.interval,
            "interval",
        )

        if self.duration is not None:
            self._validate_non_negative_number(
                self.duration,
                "duration",
            )

        if not isinstance(self.restore_clipboard, bool):
            raise TypeError(
                "restore_clipboard must be a bool."
            )

        if not isinstance(
            self.use_clipboard_for_unicode,
            bool,
        ):
            raise TypeError(
                "use_clipboard_for_unicode must be a bool."
            )

    # 校验相对坐标，必须非空
    def _validate_absolute_position(self) -> None:
        if self._has_normalised_position():
            self._validate_normalised_position()
            return

        if self.x is None or self.y is None:
            raise ValueError(
                f"{self.type.value} requires x and y, "
                "or normalised_x and normalised_y."
            )

        self._validate_integer(self.x, "x")
        self._validate_integer(self.y, "y")

    # 校验可选坐标，有x,就必须有y
    def _validate_optional_position(self) -> None:
        if self.x is None and self.y is None:
            if self._has_normalised_position():
                self._validate_normalised_position()
            return

        if self.x is None or self.y is None:
            raise ValueError(
                "x and y must both be provided or both omitted."
            )

        self._validate_integer(self.x, "x")
        self._validate_integer(self.y, "y")

    # 校验绝对坐标
    def _validate_relative_position(self) -> None:
        if self.offset_x is None or self.offset_y is None:
            raise ValueError(
                f"{self.type.value} requires "
                "offset_x and offset_y."
            )

        self._validate_integer(
            self.offset_x,
            "offset_x",
        )
        self._validate_integer(
            self.offset_y,
            "offset_y",
        )

    # 校验点击动作必须是正整数
    def _validate_click(self) -> None:
        self._validate_optional_position()

        self._validate_positive_integer(
            self.clicks,
            "clicks",
        )

    # amount必须存在且是整数
    def _validate_scroll(self) -> None:
        if self.amount is None:
            raise ValueError(
                f"{self.type.value} requires amount."
            )

        self._validate_integer(
            self.amount,
            "amount",
        )

        if self.amount == 0:
            raise ValueError(
                "scroll amount must not be zero."
            )

        self._validate_optional_position()

    def _validate_press(self) -> None:
        self._validate_single_key()
        self._validate_presses()

    def _validate_single_key(self) -> None:
        if not isinstance(self.key, str):
            raise TypeError(
                f"{self.type.value} requires key as string."
            )

        if not self.key.strip():
            raise ValueError("key must not be empty.")

    def _validate_hotkey(self) -> None:
        if not isinstance(self.keys, tuple):
            raise TypeError(
                "keys must be a tuple or list of strings."
            )

        if len(self.keys) < 2:
            raise ValueError(
                "hotkey requires at least two keys."
            )

        for index, key in enumerate(self.keys):
            if not isinstance(key, str):
                raise TypeError(
                    f"keys[{index}] must be a string."
                )

            if not key.strip():
                raise ValueError(
                    f"keys[{index}] must not be empty."
                )

    def _validate_text(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(
                f"{self.type.value} requires text as string."
            )

        if self.text == "":
            raise ValueError("text must not be empty.")

    def _validate_presses(self) -> None:
        self._validate_positive_integer(
            self.presses,
            "presses",
        )

    def _validate_wait(self) -> None:
        if self.seconds is None:
            raise ValueError(
                "wait action requires seconds."
            )

        self._validate_non_negative_number(
            self.seconds,
            "seconds",
        )

    def _validate_normalised_position(self) -> None:
        if (
            self.normalised_x is None
            or self.normalised_y is None
        ):
            raise ValueError(
                "normalised_x and normalised_y "
                "must both be provided."
            )

        self._validate_normalised_coordinate(
            self.normalised_x,
            "normalised_x",
        )

        self._validate_normalised_coordinate(
            self.normalised_y,
            "normalised_y",
        )

    # 是否传入归一化坐标
    def _has_normalised_position(self) -> bool:
        return (
            self.normalised_x is not None
            or self.normalised_y is not None
        )
    

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    # Action转换成dict
    def to_dict(
        self,
        exclude_none: bool = True,
    ) -> dict[str, Any]:
        """
        Convert the action to a JSON-compatible dictionary.
        """

        result = asdict(self)

        result["type"] = self.type.value
        result["button"] = self.button.value
        result["status"] = self.status.value
        result["keys"] = list(self.keys)

        if exclude_none:
            result = {
                key: value
                for key, value in result.items()
                if value is not None
            }

        return result

    def to_json(
        self,
        exclude_none: bool = True,
        indent: Optional[int] = 2,
        ensure_ascii: bool = False,
    ) -> str:
        """
        Serialize the action as JSON.
        """

        return json.dumps(
            self.to_dict(
                exclude_none=exclude_none,
            ),
            indent=indent,
            ensure_ascii=ensure_ascii,
        )

    # 从字典创建Action
    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "Action":
        """
        Create an Action from a dictionary.

        Unknown fields are stored inside metadata.
        """

        if not isinstance(data, dict):
            raise TypeError(
                "data must be a dictionary."
            )

        if "type" not in data:
            raise ValueError(
                "Action dictionary must contain 'type'."
            )

        known_fields = {
            "type",
            "action_id",
            "description",
            "x",
            "y",
            "offset_x",
            "offset_y",
            "normalised_x",
            "normalised_y",
            "button",
            "clicks",
            "amount",
            "text",
            "key",
            "keys",
            "presses",
            "interval",
            "duration",
            "seconds",
            "restore_clipboard",
            "use_clipboard_for_unicode",
            "status",
            "metadata",
        }

        action_data = {
            key: value
            for key, value in data.items()
            if key in known_fields
        }

        unknown_fields = {
            key: value
            for key, value in data.items()
            if key not in known_fields
        }

        metadata = dict(
            action_data.get("metadata") or {}
        )

        if unknown_fields:
            metadata["extra_fields"] = unknown_fields

        action_data["metadata"] = metadata

        if isinstance(action_data.get("keys"), list):
            action_data["keys"] = tuple(
                action_data["keys"]
            )

        return cls(**action_data)

    # 从JSON创建Action
    @classmethod
    def from_json(
        cls,
        json_text: str,
    ) -> "Action":
        """
        Create an Action from a JSON string.
        """

        if not isinstance(json_text, str):
            raise TypeError(
                "json_text must be a string."
            )

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid action JSON: {error}"
            ) from error

        if not isinstance(data, dict):
            raise ValueError(
                "Action JSON must represent an object."
            )

        return cls.from_dict(data)
    

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def mark_running(self) -> None:

        self.status = ActionStatus.RUNNING

    def mark_success(self) -> None:

        self.status = ActionStatus.SUCCESS

    def mark_failed(
        self,
        error: Optional[str] = None,
    ) -> None:

        self.status = ActionStatus.FAILED

        if error is not None:
            self.metadata["error"] = error

    def mark_skipped(
        self,
        reason: Optional[str] = None,
    ) -> None:

        self.status = ActionStatus.SKIPPED

        if reason is not None:
            self.metadata["skip_reason"] = reason


    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def move_to(
        cls,
        x: int,
        y: int,
        duration: Optional[float] = None,
        description: str = "",
    ) -> "Action":
        return cls(
            type=ActionType.MOVE_TO,
            x=x,
            y=y,
            duration=duration,
            description=description,
        )

    @classmethod
    def click(
        cls,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: MouseButton | str = MouseButton.LEFT,
        clicks: int = 1,
        interval: float = 0.0,
        duration: Optional[float] = None,
        description: str = "",
    ) -> "Action":
        return cls(
            type=ActionType.CLICK,
            x=x,
            y=y,
            button=button,
            clicks=clicks,
            interval=interval,
            duration=duration,
            description=description,
        )

    @classmethod
    def double_click(
        cls,
        x: Optional[int] = None,
        y: Optional[int] = None,
        interval: float = 0.10,
        duration: Optional[float] = None,
        description: str = "",
    ) -> "Action":
        return cls(
            type=ActionType.DOUBLE_CLICK,
            x=x,
            y=y,
            interval=interval,
            duration=duration,
            description=description,
        )

    @classmethod
    def right_click(
        cls,
        x: Optional[int] = None,
        y: Optional[int] = None,
        duration: Optional[float] = None,
        description: str = "",
    ) -> "Action":
        return cls(
            type=ActionType.RIGHT_CLICK,
            x=x,
            y=y,
            button=MouseButton.RIGHT,
            duration=duration,
            description=description,
        )

    @classmethod
    def drag_to(
        cls,
        x: int,
        y: int,
        duration: Optional[float] = None,
        button: MouseButton | str = MouseButton.LEFT,
        description: str = "",
    ) -> "Action":
        return cls(
            type=ActionType.DRAG_TO,
            x=x,
            y=y,
            duration=duration,
            button=button,
            description=description,
        )

    @classmethod
    def scroll(
        cls,
        amount: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
        description: str = "",
    ) -> "Action":
        return cls(
            type=ActionType.SCROLL,
            amount=amount,
            x=x,
            y=y,
            description=description,
        )

    @classmethod
    def press_key(
        cls,
        key: str,
        presses: int = 1,
        interval: float = 0.0,
        description: str = "",
    ) -> "Action":
        return cls(
            type=ActionType.PRESS,
            key=key,
            presses=presses,
            interval=interval,
            description=description,
        )

    @classmethod
    def hotkey_action(
        cls,
        *keys: str,
        interval: float = 0.0,
        description: str = "",
    ) -> "Action":
        return cls(
            type=ActionType.HOTKEY,
            keys=tuple(keys),
            interval=interval,
            description=description,
        )

    @classmethod
    def type_text(
        cls,
        text: str,
        interval: float = 0.0,
        description: str = "",
    ) -> "Action":
        return cls(
            type=ActionType.TYPE_TEXT,
            text=text,
            interval=interval,
            description=description,
        )

    @classmethod
    def paste_text(
        cls,
        text: str,
        restore_clipboard: bool = True,
        description: str = "",
    ) -> "Action":
        return cls(
            type=ActionType.PASTE_TEXT,
            text=text,
            restore_clipboard=restore_clipboard,
            description=description,
        )

    @classmethod
    def wait(
        cls,
        seconds: float,
        description: str = "",
    ) -> "Action":
        return cls(
            type=ActionType.WAIT,
            seconds=seconds,
            description=description,
        )

    @classmethod
    def finish(
        cls,
        description: str = "Task completed.",
    ) -> "Action":
        return cls(
            type=ActionType.FINISH,
            description=description,
        )

    @classmethod
    def fail(
        cls,
        description: str,
    ) -> "Action":
        return cls(
            type=ActionType.FAIL,
            description=description,
        )
    

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_action_type(
        value: ActionType | str,
    ) -> ActionType:
        if isinstance(value, ActionType):
            return value

        if not isinstance(value, str):
            raise TypeError(
                "type must be ActionType or string."
            )

        normalised = value.strip().lower()

        aliases = {
            "doubleclick": "double_click",
            "rightclick": "right_click",
            "middleclick": "middle_click",
            "moveto": "move_to",
            "moveby": "move_by",
            "dragto": "drag_to",
            "dragby": "drag_by",
            "type": "type_text",
            "write": "type_text",
            "input": "type_text",
            "paste": "paste_text",
            "key": "press",
            "shortcut": "hotkey",
            "done": "finish",
            "complete": "finish",
        }

        normalised = aliases.get(
            normalised,
            normalised,
        )

        try:
            return ActionType(normalised)
        except ValueError as error:
            valid_types = [
                action_type.value
                for action_type in ActionType
            ]

            raise ValueError(
                f"Unsupported action type: {value!r}. "
                f"Valid types: {valid_types}"
            ) from error

    @staticmethod
    def _normalise_mouse_button(
        value: MouseButton | str,
    ) -> MouseButton:
        if isinstance(value, MouseButton):
            return value

        if not isinstance(value, str):
            raise TypeError(
                "button must be MouseButton or string."
            )

        normalised = value.strip().lower()

        try:
            return MouseButton(normalised)
        except ValueError as error:
            raise ValueError(
                f"Unsupported mouse button: {value!r}. "
                "Valid buttons are left, right and middle."
            ) from error

    @staticmethod
    def _normalise_status(
        value: ActionStatus | str,
    ) -> ActionStatus:
        if isinstance(value, ActionStatus):
            return value

        if not isinstance(value, str):
            raise TypeError(
                "status must be ActionStatus or string."
            )

        try:
            return ActionStatus(
                value.strip().lower()
            )
        except ValueError as error:
            raise ValueError(
                f"Unsupported action status: {value!r}."
            ) from error
        

    # ------------------------------------------------------------------
    # Primitive validators
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_integer(
        value: Any,
        name: str,
    ) -> None:
        if not isinstance(value, int) or isinstance(
            value,
            bool,
        ):
            raise TypeError(
                f"{name} must be an integer."
            )

    @classmethod
    def _validate_positive_integer(
        cls,
        value: Any,
        name: str,
    ) -> None:
        cls._validate_integer(value, name)

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

    @staticmethod
    def _validate_non_negative_number(
        value: Any,
        name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be an int or float."
            )

        if value < 0:
            raise ValueError(
                f"{name} must be non-negative."
            )

    @staticmethod
    def _validate_normalised_coordinate(
        value: Any,
        name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be numeric."
            )

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{name} must be in [0, 1]."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"type={self.type.value!r}, "
            f"action_id={self.action_id!r}, "
            f"status={self.status.value!r}, "
            f"description={self.description!r}"
            f")"
        )


@dataclass
class ActionSequence:
    """
    Ordered collection of GUI actions.

    This structure can represent a plan generated by an LLM.
    """

    actions: list[Action] = field(
        default_factory=list
    )

    sequence_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    description: str = ""
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.actions, list):
            self.actions = list(self.actions)

        for index, action in enumerate(self.actions):
            if not isinstance(action, Action):
                raise TypeError(
                    f"actions[{index}] must be Action."
                )

        if not isinstance(self.description, str):
            raise TypeError(
                "description must be a string."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

    def add(
        self,
        action: Action,
    ) -> None:
        """
        Append an action.
        """

        if not isinstance(action, Action):
            raise TypeError(
                "action must be an Action instance."
            )

        self.actions.append(action)

    def extend(
        self,
        actions: Iterable[Action],
    ) -> None:
        """
        Append multiple actions.
        """

        for action in actions:
            self.add(action)

    def insert(
        self,
        index: int,
        action: Action,
    ) -> None:
        """
        Insert an action at a selected position.
        """

        if not isinstance(index, int):
            raise TypeError(
                "index must be an integer."
            )

        if not isinstance(action, Action):
            raise TypeError(
                "action must be an Action instance."
            )

        self.actions.insert(index, action)

    def remove(
        self,
        action_id: str,
    ) -> Action:
        """
        Remove and return an action by ID.
        """

        for index, action in enumerate(self.actions):
            if action.action_id == action_id:
                return self.actions.pop(index)

        raise KeyError(
            f"Action not found: {action_id}"
        )

    def get(
        self,
        action_id: str,
    ) -> Optional[Action]:
        """
        Return an action by ID.
        """

        for action in self.actions:
            if action.action_id == action_id:
                return action

        return None

    def pending_actions(self) -> list[Action]:
        """
        Return all pending actions.
        """

        return [
            action
            for action in self.actions
            if action.status == ActionStatus.PENDING
        ]

    def failed_actions(self) -> list[Action]:
        """
        Return all failed actions.
        """

        return [
            action
            for action in self.actions
            if action.status == ActionStatus.FAILED
        ]

    def successful_actions(self) -> list[Action]:
        """
        Return all successful actions.
        """

        return [
            action
            for action in self.actions
            if action.status == ActionStatus.SUCCESS
        ]

    def to_dict(
        self,
        exclude_none: bool = True,
    ) -> dict[str, Any]:
        """
        Convert the sequence to dictionary.
        """

        return {
            "sequence_id": self.sequence_id,
            "description": self.description,
            "metadata": self.metadata,
            "actions": [
                action.to_dict(
                    exclude_none=exclude_none
                )
                for action in self.actions
            ],
        }

    def to_json(
        self,
        exclude_none: bool = True,
        indent: Optional[int] = 2,
        ensure_ascii: bool = False,
    ) -> str:
        """
        Serialize the sequence as JSON.
        """

        return json.dumps(
            self.to_dict(
                exclude_none=exclude_none,
            ),
            indent=indent,
            ensure_ascii=ensure_ascii,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "ActionSequence":
        """
        Create a sequence from a dictionary.
        """

        if not isinstance(data, dict):
            raise TypeError(
                "data must be a dictionary."
            )

        raw_actions = data.get("actions", [])

        if not isinstance(raw_actions, list):
            raise TypeError(
                "actions must be a list."
            )

        actions = [
            Action.from_dict(action_data)
            for action_data in raw_actions
        ]

        return cls(
            actions=actions,
            sequence_id=data.get(
                "sequence_id",
                str(uuid.uuid4()),
            ),
            description=data.get(
                "description",
                "",
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )

    @classmethod
    def from_json(
        cls,
        json_text: str,
    ) -> "ActionSequence":
        """
        Create a sequence from JSON.
        """

        if not isinstance(json_text, str):
            raise TypeError(
                "json_text must be a string."
            )

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid action-sequence JSON: {error}"
            ) from error

        return cls.from_dict(data)

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def __getitem__(
        self,
        index: int,
    ) -> Action:
        return self.actions[index]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sequence_id={self.sequence_id!r}, "
            f"action_count={len(self.actions)}, "
            f"description={self.description!r}"
            f")"
        )