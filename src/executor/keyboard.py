"""
keyboard
1.按单一键
2.重复按键
3.热键组合
4.输入文本
5.通过剪贴板粘贴文本
6.支持快捷键
7.验证键名和参数
8.返回结构化执行结果
"""


from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from ._optional_pyautogui import pyautogui

try:
    import pyperclip
except ImportError:
    pyperclip = None


logger = logging.getLogger(__name__)


@dataclass
class KeyboardActionResult:

    action: str
    success: bool
    elapsed_time: float
    dry_run: bool = False
    message: str = ""
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class KeyboardController:

    def __init__(
        self,
        pause: float = 0.10,
        fail_safe: bool = True,
        default_interval: float = 0.03,
        dry_run: bool = False,
        raise_on_error: bool = True,
    ) -> None:
        self._validate_non_negative_number(
            pause,
            "pause",
        )
        self._validate_non_negative_number(
            default_interval,
            "default_interval",
        )

        if not isinstance(fail_safe, bool):
            raise TypeError("fail_safe must be a bool.")

        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a bool.")

        if not isinstance(raise_on_error, bool):
            raise TypeError("raise_on_error must be a bool.")

        self.pause = float(pause)
        self.fail_safe = fail_safe
        self.default_interval = float(default_interval)
        self.dry_run = dry_run
        self.raise_on_error = raise_on_error

        self._lock = threading.RLock()

        pyautogui.PAUSE = self.pause
        pyautogui.FAILSAFE = self.fail_safe

        self.system_name = platform.system().lower()


    # ------------------------------------------------------------------
    # Basic key information
    # ------------------------------------------------------------------

    # 查看支持哪些按键
    @staticmethod
    def available_keys() -> tuple[str, ...]:
        """
        Return all key names supported by PyAutoGUI.
        """

        return tuple(pyautogui.KEYBOARD_KEYS)

    # 判断按键是否有效
    @staticmethod
    def is_valid_key(key: str) -> bool:
        """
        Return whether a key name is supported by PyAutoGUI.
        """

        if not isinstance(key, str):
            return False

        return key.lower() in pyautogui.KEYBOARD_KEYS

    # 检查并返回标准化按键名称
    def validate_key(self, key: str) -> str:
        """
        Validate and normalise a key name.
        """

        if not isinstance(key, str):
            raise TypeError("key must be a string.")

        normalised = key.strip().lower()

        if not normalised:
            raise ValueError("key must not be empty.")

        if normalised not in pyautogui.KEYBOARD_KEYS:
            raise ValueError(
                f"Unsupported key: {key!r}. "
                "Use KeyboardController.available_keys() "
                "to inspect valid key names."
            )

        return normalised
    

    # ------------------------------------------------------------------
    # Single-key operations
    # ------------------------------------------------------------------

    def press(
        self,
        key: str,
        presses: int = 1,
        interval: Optional[float] = None,
    ) -> KeyboardActionResult:

        resolved_key = self.validate_key(key)
        self._validate_press_count(presses)

        resolved_interval = self._resolve_interval(interval)

        return self._execute_action(
            action_name="press",
            operation=lambda: pyautogui.press(
                resolved_key,
                presses=presses,
                interval=resolved_interval,
            ),
            message=(
                f"Press key {resolved_key!r} "
                f"{presses} time(s)."
            ),
            metadata={
                "key": resolved_key,
                "presses": presses,
                "interval": resolved_interval,
            },
        )

    def key_down(
        self,
        key: str,
    ) -> KeyboardActionResult:
        """
        Press and hold a key.
        """

        resolved_key = self.validate_key(key)

        return self._execute_action(
            action_name="key_down",
            operation=lambda: pyautogui.keyDown(
                resolved_key
            ),
            message=f"Hold key {resolved_key!r}.",
            metadata={
                "key": resolved_key,
            },
        )

    def key_up(
        self,
        key: str,
    ) -> KeyboardActionResult:
        """
        Release a held key.
        """

        resolved_key = self.validate_key(key)

        return self._execute_action(
            action_name="key_up",
            operation=lambda: pyautogui.keyUp(
                resolved_key
            ),
            message=f"Release key {resolved_key!r}.",
            metadata={
                "key": resolved_key,
            },
        )


    # ------------------------------------------------------------------
    # Hotkeys
    # ------------------------------------------------------------------

    def hotkey(
        self,
        *keys: str,
        interval: Optional[float] = None,
    ) -> KeyboardActionResult:

        if len(keys) < 2:
            raise ValueError(
                "hotkey requires at least two keys."
            )

        resolved_keys = tuple(
            self.validate_key(key)
            for key in keys
        )

        resolved_interval = self._resolve_interval(
            interval
        )

        return self._execute_action(
            action_name="hotkey",
            operation=lambda: pyautogui.hotkey(
                *resolved_keys,
                interval=resolved_interval,
            ),
            message=(
                "Press hotkey "
                + " + ".join(resolved_keys)
                + "."
            ),
            metadata={
                "keys": resolved_keys,
                "interval": resolved_interval,
            },
        )

    def shortcut(
        self,
        keys: Sequence[str],
        interval: Optional[float] = None,
    ) -> KeyboardActionResult:
        """
        Press a keyboard shortcut from a sequence.
        """

        if not isinstance(keys, Sequence):
            raise TypeError(
                "keys must be a sequence of strings."
            )

        return self.hotkey(
            *keys,
            interval=interval,
        )
    

    # ------------------------------------------------------------------
    # Text input
    # ------------------------------------------------------------------

    def type_text(
        self,
        text: str,
        interval: Optional[float] = None,
    ) -> KeyboardActionResult:
        """
        Type text using PyAutoGUI.

        Important
        ---------
        PyAutoGUI.write() works best for ASCII text. For Chinese or other
        Unicode text, use paste_text().
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if text == "":
            raise ValueError("text must not be empty.")

        resolved_interval = self._resolve_interval(
            interval
        )

        return self._execute_action(
            action_name="type_text",
            operation=lambda: pyautogui.write(
                text,
                interval=resolved_interval,
            ),
            message=f"Type text: {text!r}.",
            metadata={
                "text": text,
                "length": len(text),
                "interval": resolved_interval,
            },
        )

    def paste_text(
        self,
        text: str,
        restore_clipboard: bool = True,
    ) -> KeyboardActionResult:
        """
        Paste Unicode text through the system clipboard.

        This method is recommended for Chinese text and other Unicode
        characters.

        Parameters
        ----------
        text:
            Text to paste.

        restore_clipboard:
            Whether to restore the previous clipboard content.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if text == "":
            raise ValueError("text must not be empty.")

        if not isinstance(restore_clipboard, bool):
            raise TypeError(
                "restore_clipboard must be a bool."
            )

        if pyperclip is None:
            raise RuntimeError(
                "pyperclip is not installed. "
                "Run: uv pip install pyperclip"
            )

        def operation() -> None:
            previous_text: Optional[str] = None

            if restore_clipboard:
                try:
                    previous_text = pyperclip.paste()
                except Exception:
                    previous_text = None

            pyperclip.copy(text)

            if self.system_name == "darwin":
                pyautogui.hotkey("command", "v")
            else:
                pyautogui.hotkey("ctrl", "v")

            if restore_clipboard:
                time.sleep(0.05)

                if previous_text is not None:
                    pyperclip.copy(previous_text)

        return self._execute_action(
            action_name="paste_text",
            operation=operation,
            message=f"Paste text: {text!r}.",
            metadata={
                "text": text,
                "length": len(text),
                "restore_clipboard": restore_clipboard,
            },
        )

    def write(
        self,
        text: str,
        interval: Optional[float] = None,
        use_clipboard_for_unicode: bool = True,
    ) -> KeyboardActionResult:
        """
        Write text using the most appropriate method.

        ASCII-only text uses PyAutoGUI.write(). Unicode text can
        automatically use clipboard pasting.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if text == "":
            raise ValueError("text must not be empty.")

        if not isinstance(
            use_clipboard_for_unicode,
            bool,
        ):
            raise TypeError(
                "use_clipboard_for_unicode must be a bool."
            )

        contains_non_ascii = any(
            ord(character) > 127
            for character in text
        )

        if (
            contains_non_ascii
            and use_clipboard_for_unicode
        ):
            return self.paste_text(text)

        return self.type_text(
            text=text,
            interval=interval,
        )


    # ------------------------------------------------------------------
    # Common navigation actions
    # ------------------------------------------------------------------

    def enter(self) -> KeyboardActionResult:

        return self.press("enter")

    def escape(self) -> KeyboardActionResult:

        return self.press("esc")

    def tab(
        self,
        presses: int = 1,
    ) -> KeyboardActionResult:

        return self.press(
            "tab",
            presses=presses,
        )

    def backspace(
        self,
        presses: int = 1,
    ) -> KeyboardActionResult:

        return self.press(
            "backspace",
            presses=presses,
        )

    def delete(
        self,
        presses: int = 1,
    ) -> KeyboardActionResult:

        return self.press(
            "delete",
            presses=presses,
        )

    def home(self) -> KeyboardActionResult:

        return self.press("home")

    def end(self) -> KeyboardActionResult:

        return self.press("end")

    def page_up(
        self,
        presses: int = 1,
    ) -> KeyboardActionResult:

        return self.press(
            "pageup",
            presses=presses,
        )

    def page_down(
        self,
        presses: int = 1,
    ) -> KeyboardActionResult:

        return self.press(
            "pagedown",
            presses=presses,
        )

    def arrow_up(
        self,
        presses: int = 1,
    ) -> KeyboardActionResult:

        return self.press(
            "up",
            presses=presses,
        )

    def arrow_down(
        self,
        presses: int = 1,
    ) -> KeyboardActionResult:

        return self.press(
            "down",
            presses=presses,
        )

    def arrow_left(
        self,
        presses: int = 1,
    ) -> KeyboardActionResult:

        return self.press(
            "left",
            presses=presses,
        )

    def arrow_right(
        self,
        presses: int = 1,
    ) -> KeyboardActionResult:

        return self.press(
            "right",
            presses=presses,
        )
    

     # ------------------------------------------------------------------
    # Common editing shortcuts
    # ------------------------------------------------------------------

    def copy(self) -> KeyboardActionResult:

        return self._platform_hotkey("c")

    def paste(self) -> KeyboardActionResult:

        return self._platform_hotkey("v")

    def cut(self) -> KeyboardActionResult:

        return self._platform_hotkey("x")

    def select_all(self) -> KeyboardActionResult:

        return self._platform_hotkey("a")

    def save(self) -> KeyboardActionResult:

        return self._platform_hotkey("s")

    def undo(self) -> KeyboardActionResult:

        return self._platform_hotkey("z")

    def redo(self) -> KeyboardActionResult:

        if self.system_name == "darwin":
            return self.hotkey(
                "command",
                "shift",
                "z",
            )

        return self.hotkey(
            "ctrl",
            "y",
        )

    def find(self) -> KeyboardActionResult:

        return self._platform_hotkey("f")

    def new(self) -> KeyboardActionResult:

        return self._platform_hotkey("n")

    def open(self) -> KeyboardActionResult:

        return self._platform_hotkey("o")

    def close_window(self) -> KeyboardActionResult:
        """
        Close the active window.

        Windows/Linux:
            Alt + F4

        macOS:
            Command + W
        """

        if self.system_name == "darwin":
            return self.hotkey(
                "command",
                "w",
            )

        return self.hotkey(
            "alt",
            "f4",
        )

    def switch_window(self) -> KeyboardActionResult:
        """
        Switch to another application window.

        Windows/Linux:
            Alt + Tab

        macOS:
            Command + Tab
        """

        if self.system_name == "darwin":
            return self.hotkey(
                "command",
                "tab",
            )

        return self.hotkey(
            "alt",
            "tab",
        )

    def open_task_manager(self) -> KeyboardActionResult:
        """
        Open Task Manager on Windows.

        Raises NotImplementedError on non-Windows systems.
        """

        if self.system_name != "windows":
            raise NotImplementedError(
                "open_task_manager is only supported on Windows."
            )

        return self.hotkey(
            "ctrl",
            "shift",
            "esc",
        )
    

    # ------------------------------------------------------------------
    # Key sequence operations
    # ------------------------------------------------------------------

    def press_sequence(
        self,
        keys: Sequence[str],
        interval: Optional[float] = None,
    ) -> KeyboardActionResult:
        """
        Press a sequence of individual keys in order.

        Example
        -------
        press_sequence(["down", "down", "enter"])
        """

        if not isinstance(keys, Sequence):
            raise TypeError(
                "keys must be a sequence."
            )

        if not keys:
            raise ValueError(
                "keys must not be empty."
            )

        resolved_keys = [
            self.validate_key(key)
            for key in keys
        ]

        resolved_interval = self._resolve_interval(
            interval
        )

        def operation() -> None:
            for key in resolved_keys:
                pyautogui.press(key)

                if resolved_interval > 0:
                    time.sleep(resolved_interval)

        return self._execute_action(
            action_name="press_sequence",
            operation=operation,
            message=(
                "Press key sequence: "
                + " -> ".join(resolved_keys)
                + "."
            ),
            metadata={
                "keys": tuple(resolved_keys),
                "interval": resolved_interval,
            },
        )

    def type_and_enter(
        self,
        text: str,
        interval: Optional[float] = None,
        use_clipboard_for_unicode: bool = True,
    ) -> list[KeyboardActionResult]:
        """
        Type text and press Enter.

        Returns two action results.
        """

        type_result = self.write(
            text=text,
            interval=interval,
            use_clipboard_for_unicode=(
                use_clipboard_for_unicode
            ),
        )

        enter_result = self.enter()

        return [
            type_result,
            enter_result,
        ]

    def clear_field(self) -> list[KeyboardActionResult]:
        """
        Select all content and delete it.
        """

        select_result = self.select_all()
        delete_result = self.delete()

        return [
            select_result,
            delete_result,
        ]
    

    # ------------------------------------------------------------------
    # Safety and configuration
    # ------------------------------------------------------------------

    @staticmethod
    def wait(seconds: float) -> None:
        """
        Pause execution.
        """

        KeyboardController._validate_non_negative_number(
            seconds,
            "seconds",
        )

        time.sleep(float(seconds))

    def set_dry_run(
        self,
        enabled: bool,
    ) -> None:
        """
        Enable or disable dry-run mode.
        """

        if not isinstance(enabled, bool):
            raise TypeError(
                "enabled must be a bool."
            )

        self.dry_run = enabled

    def set_fail_safe(
        self,
        enabled: bool,
    ) -> None:
        """
        Enable or disable PyAutoGUI fail-safe.
        """

        if not isinstance(enabled, bool):
            raise TypeError(
                "enabled must be a bool."
            )

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
    ) -> KeyboardActionResult:
        """
        Execute one keyboard action and return a structured result.
        """

        with self._lock:
            start_time = time.perf_counter()

            try:
                if self.dry_run:
                    logger.info(
                        "[DRY RUN] %s",
                        message,
                    )
                else:
                    logger.info(message)
                    operation()

                elapsed_time = (
                    time.perf_counter()
                    - start_time
                )

                return KeyboardActionResult(
                    action=action_name,
                    success=True,
                    elapsed_time=elapsed_time,
                    dry_run=self.dry_run,
                    message=message,
                    metadata=metadata or {},
                )

            except pyautogui.FailSafeException as error:
                elapsed_time = (
                    time.perf_counter()
                    - start_time
                )

                logger.warning(
                    "Keyboard fail-safe interrupted "
                    "action %s: %s",
                    action_name,
                    error,
                )

                result = KeyboardActionResult(
                    action=action_name,
                    success=False,
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
                elapsed_time = (
                    time.perf_counter()
                    - start_time
                )

                logger.exception(
                    "Keyboard action %s failed.",
                    action_name,
                )

                result = KeyboardActionResult(
                    action=action_name,
                    success=False,
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
    # Helper methods
    # ------------------------------------------------------------------

    def _platform_hotkey(
        self,
        key: str,
    ) -> KeyboardActionResult:
        """
        Press Command+key on macOS or Ctrl+key elsewhere.
        """

        modifier = (
            "command"
            if self.system_name == "darwin"
            else "ctrl"
        )

        return self.hotkey(
            modifier,
            key,
        )

    def _resolve_interval(
        self,
        interval: Optional[float],
    ) -> float:
        """
        Resolve an optional interval.
        """

        if interval is None:
            return self.default_interval

        self._validate_non_negative_number(
            interval,
            "interval",
        )

        return float(interval)

    @staticmethod
    def _validate_press_count(
        presses: int,
    ) -> None:
        """
        Validate repeated key-press count.
        """

        if not isinstance(
            presses,
            int,
        ) or isinstance(
            presses,
            bool,
        ):
            raise TypeError(
                "presses must be an integer."
            )

        if presses <= 0:
            raise ValueError(
                "presses must be greater than zero."
            )

    @staticmethod
    def _validate_non_negative_number(
        value: float,
        name: str,
    ) -> None:
        """
        Validate a non-negative number.
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"pause={self.pause}, "
            f"fail_safe={self.fail_safe}, "
            f"default_interval={self.default_interval}, "
            f"dry_run={self.dry_run}, "
            f"raise_on_error={self.raise_on_error}, "
            f"system_name={self.system_name!r}"
            f")"
        )