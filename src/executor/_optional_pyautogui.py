"""Optional pyautogui import for Colab / CI dry-run (no desktop)."""

from __future__ import annotations

from typing import Any


class FailSafeException(Exception):
    """Stand-in when real pyautogui is unavailable."""


class _Pos:
    x = 0
    y = 0


def _build_stub() -> Any:
    common_keys = (
        list("abcdefghijklmnopqrstuvwxyz0123456789")
        + [
            "enter",
            "esc",
            "escape",
            "tab",
            "space",
            "backspace",
            "delete",
            "shift",
            "ctrl",
            "alt",
            "win",
            "command",
            "up",
            "down",
            "left",
            "right",
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "home",
            "end",
            "pageup",
            "pagedown",
            "l",
            "w",
            "c",
            "v",
            "a",
        ]
    )

    class _Stub:
        PAUSE = 0.0
        FAILSAFE = False
        KEYBOARD_KEYS = tuple(dict.fromkeys(common_keys))
        FailSafeException = FailSafeException
        linear = None

        @staticmethod
        def position() -> _Pos:
            return _Pos()

        @staticmethod
        def size() -> tuple[int, int]:
            return (1920, 1080)

        @staticmethod
        def onScreen(x: int, y: int) -> bool:  # noqa: N802 — match pyautogui API
            return True

        def __getattr__(self, name: str) -> Any:
            def _missing(*_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError(
                    f"pyautogui is not installed (missing call: {name}). "
                    "Install with: pip install pyautogui "
                    "Or keep Executor(dry_run=True) so real OS calls are skipped."
                )

            return _missing

    return _Stub()


try:
    import pyautogui as pyautogui
except ImportError:  # pragma: no cover
    pyautogui = _build_stub()
