"""
mouse
1.识别鼠标当前位置
2.移动鼠标到绝对坐标
3.移动鼠标到相对偏量
4.执行左键、右键和中键单击
5.执行双击和多击
6.点击拖动到绝对坐标
7.点击拖动到相对偏量
8.垂直滚动和水平滑动
9.验证屏幕坐标
10.输出标准化坐标
11.返回结构化执行结果
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from ._optional_pyautogui import pyautogui


logger = logging.getLogger(__name__)


MouseButton = Literal["left", "right", "middle"]


@dataclass(frozen=True)
class MousePosition:

    x: int
    y: int

    def as_tuple(self) -> tuple[int, int]:

        return self.x, self.y


@dataclass
class MouseActionResult:

    action: str
    success: bool
    start_position: MousePosition
    end_position: MousePosition
    elapsed_time: float
    dry_run: bool = False
    message: str = ""
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class MouseController:

    VALID_BUTTONS: set[str] = {
        "left",
        "right",
        "middle",
    }

    def __init__(
        self,
        pause: float = 0.10,    # 操作后秒级自动延迟
        fail_safe: bool = True,     #移动到左上角出发执行中断
        default_duration: float = 0.20,     # 默认移动/拖动持续时间
        default_button: MouseButton = "left",   # 默认点击按钮
        dry_run: bool = False,      #验证和记录操作
        raise_on_error: bool = True,    # True:记录后引发执行错误 False:操作失败，返回MouseActionResult
    ) -> None:
        self._validate_non_negative_number(
            pause,
            "pause",
        )
        self._validate_non_negative_number(
            default_duration,
            "default_duration",
        )
        self._validate_button(default_button)

        if not isinstance(fail_safe, bool):
            raise TypeError("fail_safe must be a bool.")

        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a bool.")

        if not isinstance(raise_on_error, bool):
            raise TypeError("raise_on_error must be a bool.")

        self.pause = float(pause)
        self.fail_safe = fail_safe
        self.default_duration = float(default_duration)
        self.default_button = default_button
        self.dry_run = dry_run
        self.raise_on_error = raise_on_error

        # Protect mouse actions from concurrent execution.
        self._lock = threading.RLock()

        pyautogui.PAUSE = self.pause
        pyautogui.FAILSAFE = self.fail_safe
    

    # ------------------------------------------------------------------
    # Screen and position information
    # ------------------------------------------------------------------

    def get_position(self) -> MousePosition:

        position = pyautogui.position()

        return MousePosition(
            x=int(position.x),
            y=int(position.y),
        )

    def get_screen_size(self) -> tuple[int, int]:
        """
        Return the primary screen size as ``(width, height)``.
        """

        size = pyautogui.size()

        return int(size.width), int(size.height)

    def is_on_screen(
        self,
        x: int,
        y: int,
    ) -> bool:

        self._validate_coordinate_type(x, y)

        return bool(pyautogui.onScreen(x, y))

    # 验证绝对坐标是否在屏幕内
    def validate_position(
        self,
        x: int,
        y: int,
    ) -> tuple[int, int]:

        self._validate_coordinate_type(x, y)

        if not self.is_on_screen(x, y):
            width, height = self.get_screen_size()

            raise ValueError(
                "Mouse coordinate lies outside the primary screen: "
                f"(x={x}, y={y}), screen_size=({width}, {height})."
            )

        return x, y

    # 屏幕坐标标准化
    def normalised_to_screen(
        self,
        x: float,
        y: float,
    ) -> tuple[int, int]:

        self._validate_normalised_coordinate(x, "x")
        self._validate_normalised_coordinate(y, "y")

        width, height = self.get_screen_size()

        # width - 1 and height - 1 ensure the result remains on screen.
        pixel_x = int(round(x * max(width - 1, 0)))
        pixel_y = int(round(y * max(height - 1, 0)))

        self.validate_position(pixel_x, pixel_y)

        return pixel_x, pixel_y
    # ------------------------------------------------------------------
    # Mouse movement
    # ------------------------------------------------------------------

    # 移动到绝对位置
    def move_to(
        self,
        x: int,
        y: int,
        duration: Optional[float] = None,
        tween: Optional[Callable[[float], float]] = None,
    ) -> MouseActionResult:
        
        self.validate_position(x, y)
        resolved_duration = self._resolve_duration(duration)

        return self._execute_action(
            action_name="move_to",
            operation=lambda: pyautogui.moveTo(
                x=x,
                y=y,
                duration=resolved_duration,
                tween=tween or pyautogui.linear,
            ),
            message=f"Move mouse to ({x}, {y}).",
            metadata={
                "x": x,
                "y": y,
                "duration": resolved_duration,
            },
            dry_run_end_position=MousePosition(x, y),
        )
    

    # 移动到相对偏量
    def move_by(
        self,
        offset_x: int,
        offset_y: int,
        duration: Optional[float] = None,
        tween: Optional[Callable[[float], float]] = None,
    ) -> MouseActionResult:
        
        self._validate_coordinate_type(
            offset_x,
            offset_y,
            names=("offset_x", "offset_y"),
        )

        start = self.get_position()
        target_x = start.x + offset_x
        target_y = start.y + offset_y

        self.validate_position(target_x, target_y)
        resolved_duration = self._resolve_duration(duration)

        return self._execute_action(
            action_name="move_by",
            operation=lambda: pyautogui.moveRel(
                xOffset=offset_x,
                yOffset=offset_y,
                duration=resolved_duration,
                tween=tween or pyautogui.linear,
            ),
            message=(
                f"Move mouse by ({offset_x}, {offset_y}) "
                f"to ({target_x}, {target_y})."
            ),
            metadata={
                "offset_x": offset_x,
                "offset_y": offset_y,
                "target_x": target_x,
                "target_y": target_y,
                "duration": resolved_duration,
            },
            dry_run_end_position=MousePosition(
                target_x,
                target_y,
            ),
        )
    

    # 移动到标准化坐标
    def move_to_normalised(
        self,
        x: float,
        y: float,
        duration: Optional[float] = None,
    ) -> MouseActionResult:
        
        pixel_x, pixel_y = self.normalised_to_screen(x, y)

        result = self.move_to(
            x=pixel_x,
            y=pixel_y,
            duration=duration,
        )

        if result.metadata is None:
            result.metadata = {}

        result.metadata.update(
            {
                "normalised_x": x,
                "normalised_y": y,
            }
        )

        return result
    

    # ------------------------------------------------------------------
    # Clicking
    # ------------------------------------------------------------------

    def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: Optional[MouseButton] = None,
        clicks: int = 1,
        interval: float = 0.0,
        duration: Optional[float] = None,
    ) -> MouseActionResult:
        
        resolved_button = button or self.default_button
        self._validate_button(resolved_button)
        self._validate_click_count(clicks)
        self._validate_non_negative_number(
            interval,
            "interval",
        )

        target = self._resolve_optional_position(x, y)
        resolved_duration = self._resolve_duration(duration)

        def operation() -> None:
            if target is not None:
                pyautogui.moveTo(
                    target.x,
                    target.y,
                    duration=resolved_duration,
                    tween=pyautogui.linear,
                )

            pyautogui.click(
                clicks=clicks,
                interval=float(interval),
                button=resolved_button,
            )

        target_description = (
            f" at ({target.x}, {target.y})"
            if target is not None
            else " at current position"
        )

        return self._execute_action(
            action_name="click",
            operation=operation,
            message=(
                f"Perform {clicks} {resolved_button} click(s)"
                f"{target_description}."
            ),
            metadata={
                "x": target.x if target else None,
                "y": target.y if target else None,
                "button": resolved_button,
                "clicks": clicks,
                "interval": float(interval),
                "duration": resolved_duration,
            },
            dry_run_end_position=target,
        )
    

    def left_click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> MouseActionResult:
        
        return self.click(
            x=x,
            y=y,
            button="left",
            clicks=1,
            duration=duration,
        )
    

    def right_click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> MouseActionResult:
        
         return self.click(
            x=x,
            y=y,
            button="right",
            clicks=1,
            duration=duration,
        )
    

    def middle_click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> MouseActionResult:

        return self.click(
            x=x,
            y=y,
            button="middle",
            clicks=1,
            duration=duration,
        )

    def double_click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: Optional[MouseButton] = None,
        interval: float = 0.10,
        duration: Optional[float] = None,
    ) -> MouseActionResult:

        return self.click(
            x=x,
            y=y,
            button=button or self.default_button,
            clicks=2,
            interval=interval,
            duration=duration,
        )
    
    def click_normalised(
        self,
        x: float,
        y: float,
        button: Optional[MouseButton] = None,
        clicks: int = 1,
        interval: float = 0.0,
        duration: Optional[float] = None,
    ) -> MouseActionResult:

        pixel_x, pixel_y = self.normalised_to_screen(x, y)

        result = self.click(
            x=pixel_x,
            y=pixel_y,
            button=button,
            clicks=clicks,
            interval=interval,
            duration=duration,
        )

        if result.metadata is None:
            result.metadata = {}

        result.metadata.update(
            {
                "normalised_x": x,
                "normalised_y": y,
            }
        )

        return result


    # ------------------------------------------------------------------
    # Mouse button press and release 长按和释放
    # ------------------------------------------------------------------

    def mouse_down(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: Optional[MouseButton] = None,
        duration: Optional[float] = None,
    ) -> MouseActionResult:

        resolved_button = button or self.default_button
        self._validate_button(resolved_button)

        target = self._resolve_optional_position(x, y)
        resolved_duration = self._resolve_duration(duration)

        def operation() -> None:
            if target is not None:
                pyautogui.moveTo(
                    target.x,
                    target.y,
                    duration=resolved_duration,
                    tween=pyautogui.linear,
                )

            pyautogui.mouseDown(button=resolved_button)

        return self._execute_action(
            action_name="mouse_down",
            operation=operation,
            message=f"Press and hold the {resolved_button} mouse button.",
            metadata={
                "button": resolved_button,
                "x": target.x if target else None,
                "y": target.y if target else None,
            },
            dry_run_end_position=target,
        )


    def mouse_up(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: Optional[MouseButton] = None,
        duration: Optional[float] = None,
    ) -> MouseActionResult:
        """
        Release a mouse button.
        """

        resolved_button = button or self.default_button
        self._validate_button(resolved_button)

        target = self._resolve_optional_position(x, y)
        resolved_duration = self._resolve_duration(duration)

        def operation() -> None:
            if target is not None:
                pyautogui.moveTo(
                    target.x,
                    target.y,
                    duration=resolved_duration,
                    tween=pyautogui.linear,
                )

            pyautogui.mouseUp(button=resolved_button)

        return self._execute_action(
            action_name="mouse_up",
            operation=operation,
            message=f"Release the {resolved_button} mouse button.",
            metadata={
                "button": resolved_button,
                "x": target.x if target else None,
                "y": target.y if target else None,
            },
            dry_run_end_position=target,
        )


    # ------------------------------------------------------------------
    # Dragging
    # ------------------------------------------------------------------

    # 拖拽到绝对位置
    def drag_to(
        self,
        x: int,
        y: int,
        duration: Optional[float] = None,
        button: Optional[MouseButton] = None,
        tween: Optional[Callable[[float], float]] = None,
    ) -> MouseActionResult:

        self.validate_position(x, y)

        resolved_duration = self._resolve_duration(duration)
        resolved_button = button or self.default_button
        self._validate_button(resolved_button)

        return self._execute_action(
            action_name="drag_to",
            operation=lambda: pyautogui.dragTo(
                x=x,
                y=y,
                duration=resolved_duration,
                button=resolved_button,
                tween=tween or pyautogui.linear,
            ),
            message=(
                f"Drag the {resolved_button} mouse button "
                f"to ({x}, {y})."
            ),
            metadata={
                "x": x,
                "y": y,
                "duration": resolved_duration,
                "button": resolved_button,
            },
            dry_run_end_position=MousePosition(x, y),
        )
    

    def drag_by(
        self,
        offset_x: int,
        offset_y: int,
        duration: Optional[float] = None,
        button: Optional[MouseButton] = None,
        tween: Optional[Callable[[float], float]] = None,
    ) -> MouseActionResult:
        """
        Drag the mouse by a relative offset.
        """

        self._validate_coordinate_type(
            offset_x,
            offset_y,
            names=("offset_x", "offset_y"),
        )

        start = self.get_position()
        target_x = start.x + offset_x
        target_y = start.y + offset_y

        self.validate_position(target_x, target_y)

        resolved_duration = self._resolve_duration(duration)
        resolved_button = button or self.default_button
        self._validate_button(resolved_button)

        return self._execute_action(
            action_name="drag_by",
            operation=lambda: pyautogui.dragRel(
                xOffset=offset_x,
                yOffset=offset_y,
                duration=resolved_duration,
                button=resolved_button,
                tween=tween or pyautogui.linear,
            ),
            message=(
                f"Drag the {resolved_button} mouse button by "
                f"({offset_x}, {offset_y})."
            ),
            metadata={
                "offset_x": offset_x,
                "offset_y": offset_y,
                "target_x": target_x,
                "target_y": target_y,
                "duration": resolved_duration,
                "button": resolved_button,
            },
            dry_run_end_position=MousePosition(
                target_x,
                target_y,
            ),
        )
    

    # ------------------------------------------------------------------
    # Scrolling
    # ------------------------------------------------------------------

    # 垂直滚动，正-up,负-down
    def scroll(
        self,
        amount: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> MouseActionResult:

        if not isinstance(amount, int) or isinstance(amount, bool):
            raise TypeError("amount must be an integer.")

        if amount == 0:
            raise ValueError("amount must not be zero.")

        target = self._resolve_optional_position(x, y)

        def operation() -> None:
            if target is not None:
                pyautogui.moveTo(
                    target.x,
                    target.y,
                    duration=self.default_duration,
                    tween=pyautogui.linear,
                )

            pyautogui.scroll(amount)

        return self._execute_action(
            action_name="scroll",
            operation=operation,
            message=f"Scroll vertically by {amount}.",
            metadata={
                "amount": amount,
                "x": target.x if target else None,
                "y": target.y if target else None,
            },
            dry_run_end_position=target,
        )

    # 水平滚动
    def horizontal_scroll(
        self,
        amount: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> MouseActionResult:

        if not isinstance(amount, int) or isinstance(amount, bool):
            raise TypeError("amount must be an integer.")

        if amount == 0:
            raise ValueError("amount must not be zero.")

        if not hasattr(pyautogui, "hscroll"):
            raise NotImplementedError(
                "Horizontal scrolling is not supported by this "
                "PyAutoGUI installation."
            )

        target = self._resolve_optional_position(x, y)

        def operation() -> None:
            if target is not None:
                pyautogui.moveTo(
                    target.x,
                    target.y,
                    duration=self.default_duration,
                    tween=pyautogui.linear,
                )

            pyautogui.hscroll(amount)

        return self._execute_action(
            action_name="horizontal_scroll",
            operation=operation,
            message=f"Scroll horizontally by {amount}.",
            metadata={
                "amount": amount,
                "x": target.x if target else None,
                "y": target.y if target else None,
            },
            dry_run_end_position=target,
        )
    

    # ------------------------------------------------------------------
    # Timing and safety
    # ------------------------------------------------------------------

    @staticmethod
    def wait(seconds: float) -> None:
        """
        Pause execution for the specified number of seconds.
        """

        MouseController._validate_non_negative_number(
            seconds,
            "seconds",
        )

        time.sleep(float(seconds))

    def set_dry_run(self, enabled: bool) -> None:
        """
        Enable or disable dry-run mode.
        """

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool.")

        self.dry_run = enabled

    def set_fail_safe(self, enabled: bool) -> None:
        """
        Enable or disable PyAutoGUI fail-safe behaviour.
        """

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool.")

        self.fail_safe = enabled
        pyautogui.FAILSAFE = enabled


    # ------------------------------------------------------------------
    # Internal action execution
    # ------------------------------------------------------------------

    def _execute_action(
        self,
        action_name: str,
        operation: Callable[[], None],
        message: str,
        metadata: Optional[dict[str, Any]] = None,
        dry_run_end_position: Optional[MousePosition] = None,
    ) -> MouseActionResult:
        """
        Execute one mouse operation and return a structured result.
        """

        with self._lock:
            start_time = time.perf_counter()
            start_position = self.get_position()

            try:
                if self.dry_run:
                    logger.info(
                        "[DRY RUN] %s",
                        message,
                    )

                    end_position = (
                        dry_run_end_position
                        if dry_run_end_position is not None
                        else start_position
                    )
                else:
                    logger.info(message)
                    operation()
                    end_position = self.get_position()

                elapsed_time = time.perf_counter() - start_time

                return MouseActionResult(
                    action=action_name,
                    success=True,
                    start_position=start_position,
                    end_position=end_position,
                    elapsed_time=elapsed_time,
                    dry_run=self.dry_run,
                    message=message,
                    metadata=metadata or {},
                )

            except pyautogui.FailSafeException as error:
                elapsed_time = time.perf_counter() - start_time
                end_position = self.get_position()

                logger.warning(
                    "Mouse fail-safe interrupted action %s: %s",
                    action_name,
                    error,
                )

                result = MouseActionResult(
                    action=action_name,
                    success=False,
                    start_position=start_position,
                    end_position=end_position,
                    elapsed_time=elapsed_time,
                    dry_run=self.dry_run,
                    message=message,
                    error=str(error),
                    metadata=metadata or {},
                )

                if self.raise_on_error:
                    raise

                return result

            except Exception as error:
                elapsed_time = time.perf_counter() - start_time

                try:
                    end_position = self.get_position()
                except Exception:
                    end_position = start_position

                logger.exception(
                    "Mouse action %s failed.",
                    action_name,
                )

                result = MouseActionResult(
                    action=action_name,
                    success=False,
                    start_position=start_position,
                    end_position=end_position,
                    elapsed_time=elapsed_time,
                    dry_run=self.dry_run,
                    message=message,
                    error=str(error),
                    metadata=metadata or {},
                )

                if self.raise_on_error:
                    raise

                return result
    

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _resolve_optional_position(
        self,
        x: Optional[int],
        y: Optional[int],
    ) -> Optional[MousePosition]:
        """
        Resolve optional x/y values into MousePosition.
        """

        if x is None and y is None:
            return None

        if x is None or y is None:
            raise ValueError(
                "x and y must either both be provided or both be omitted."
            )

        self.validate_position(x, y)

        return MousePosition(x=x, y=y)

    def _resolve_duration(
        self,
        duration: Optional[float],
    ) -> float:
        """
        Resolve an optional duration value.
        """

        if duration is None:
            return self.default_duration

        self._validate_non_negative_number(
            duration,
            "duration",
        )

        return float(duration)

    @classmethod
    def _validate_button(
        cls,
        button: str,
    ) -> None:
        """
        Validate a mouse button name.
        """

        if not isinstance(button, str):
            raise TypeError("button must be a string.")

        if button not in cls.VALID_BUTTONS:
            raise ValueError(
                f"Unsupported mouse button: {button!r}. "
                f"Expected one of {sorted(cls.VALID_BUTTONS)}."
            )

    @staticmethod
    def _validate_click_count(clicks: int) -> None:
        """
        Validate a click count.
        """

        if not isinstance(clicks, int) or isinstance(clicks, bool):
            raise TypeError("clicks must be an integer.")

        if clicks <= 0:
            raise ValueError("clicks must be greater than zero.")

    @staticmethod
    def _validate_coordinate_type(
        x: int,
        y: int,
        names: tuple[str, str] = ("x", "y"),
    ) -> None:
        """
        Validate coordinate data types.
        """

        for name, value in zip(names, (x, y)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name} must be an integer, "
                    f"received {type(value).__name__}."
                )

    @staticmethod
    def _validate_non_negative_number(
        value: float,
        name: str,
    ) -> None:
        """
        Validate a non-negative numeric value.
        """

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
        value: float,
        name: str,
    ) -> None:
        """
        Validate a normalised coordinate in [0, 1].
        """

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be a numeric value."
            )

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{name} must be in the range [0, 1]."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"pause={self.pause}, "
            f"fail_safe={self.fail_safe}, "
            f"default_duration={self.default_duration}, "
            f"default_button={self.default_button!r}, "
            f"dry_run={self.dry_run}, "
            f"raise_on_error={self.raise_on_error}"
            f")"
        )    