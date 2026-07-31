"""
win32_windows

Top-level window enumeration through the Win32 API.

Used by close-document workflows so the agent can identify, activate and
detect disappearance of a Word window without another full OCR pass.
"""

from __future__ import annotations

import ctypes
import platform
from ctypes import wintypes
from dataclasses import dataclass
from typing import Iterable, Sequence


SM_CXSCREEN = 0
SM_CYSCREEN = 1
SW_RESTORE = 9
GW_OWNER = 4

# Microsoft Office main windows.
_WORD_CLASS_NAMES: frozenset[str] = frozenset({"opusapp"})
_WORD_TITLE_HINTS: tuple[str, ...] = ("word", "microsoft word")


@dataclass(frozen=True, slots=True)
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


class Win32WindowError(RuntimeError):
    """Raised when the Win32 window backend cannot run."""


def is_supported() -> bool:
    return platform.system().lower() == "windows"


def _user32() -> ctypes.WinDLL:
    if not is_supported():
        raise Win32WindowError("Win32 window helpers are only available on Windows.")
    return ctypes.WinDLL("user32", use_last_error=True)


def screen_size() -> tuple[int, int]:
    user32 = _user32()
    width = int(user32.GetSystemMetrics(SM_CXSCREEN))
    height = int(user32.GetSystemMetrics(SM_CYSCREEN))
    return max(width, 1), max(height, 1)


def is_window(hwnd: int | None) -> bool:
    if hwnd is None:
        return False
    try:
        value = int(hwnd)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    return bool(_user32().IsWindow(wintypes.HWND(value)))


def foreground_hwnd() -> int | None:
    hwnd = int(_user32().GetForegroundWindow() or 0)
    return hwnd or None


def _window_text(user32: ctypes.WinDLL, hwnd: wintypes.HWND) -> str:
    length = int(user32.GetWindowTextLengthW(hwnd) or 0)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return str(buffer.value or "").strip()


def _class_name(user32: ctypes.WinDLL, hwnd: wintypes.HWND) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, 256)
    return str(buffer.value or "").strip()


def _rect(user32: ctypes.WinDLL, hwnd: wintypes.HWND) -> tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    left, top, right, bottom = int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def list_top_level_windows(*, visible_only: bool = True) -> list[WindowInfo]:
    """Return visible top-level windows with a non-empty title."""

    user32 = _user32()
    windows: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _callback(hwnd: wintypes.HWND, _lparam: wintypes.LPARAM) -> bool:
        if visible_only and not user32.IsWindowVisible(hwnd):
            return True
        if user32.GetWindow(hwnd, GW_OWNER):
            return True
        title = _window_text(user32, hwnd)
        if not title:
            return True
        box = _rect(user32, hwnd)
        if box is None:
            return True
        windows.append(
            WindowInfo(
                hwnd=int(hwnd),
                title=title,
                class_name=_class_name(user32, hwnd),
                left=box[0],
                top=box[1],
                right=box[2],
                bottom=box[3],
            )
        )
        return True

    if not user32.EnumWindows(_callback, 0):
        raise Win32WindowError(
            f"EnumWindows failed: error={ctypes.get_last_error()}."
        )
    return windows


def looks_like_word_window(window: WindowInfo) -> bool:
    class_name = window.class_name.casefold()
    if class_name in _WORD_CLASS_NAMES:
        return True
    title = window.title.casefold()
    return any(hint in title for hint in _WORD_TITLE_HINTS)


def list_word_windows() -> list[WindowInfo]:
    return [window for window in list_top_level_windows() if looks_like_word_window(window)]


def filter_windows_by_side(
    windows: Sequence[WindowInfo],
    side: str | None,
    *,
    screen_width: int | None = None,
) -> list[WindowInfo]:
    if not side:
        return list(windows)
    width = int(screen_width or screen_size()[0])
    if side == "left":
        return [window for window in windows if window.center_x <= width * 0.55]
    if side == "right":
        return [window for window in windows if window.center_x >= width * 0.45]
    return list(windows)


def find_word_window(
    *,
    side: str | None = None,
    title_tokens: Iterable[str] | None = None,
    hwnd: int | None = None,
) -> WindowInfo | None:
    """Locate the best Word window match for a close/open task."""

    if hwnd is not None and is_window(hwnd):
        matches = [window for window in list_word_windows() if window.hwnd == int(hwnd)]
        if matches:
            return matches[0]

    candidates = filter_windows_by_side(list_word_windows(), side)
    if not candidates:
        candidates = list_word_windows()
    if not candidates:
        return None

    tokens = [
        str(token).strip().casefold()
        for token in (title_tokens or ())
        if str(token).strip()
    ]
    if tokens:
        ranked = sorted(
            candidates,
            key=lambda window: (
                sum(token in window.title.casefold() for token in tokens),
                -abs(window.center_x),
            ),
            reverse=True,
        )
        best = ranked[0]
        if any(token in best.title.casefold() for token in tokens):
            return best

    # Prefer the left-most / right-most window when a side was requested.
    if side == "left":
        return min(candidates, key=lambda window: window.center_x)
    if side == "right":
        return max(candidates, key=lambda window: window.center_x)
    return candidates[0]


def activate_window(hwnd: int) -> bool:
    """Bring a top-level window to the foreground."""

    if not is_window(hwnd):
        return False
    user32 = _user32()
    handle = wintypes.HWND(int(hwnd))
    user32.ShowWindow(handle, SW_RESTORE)
    return bool(user32.SetForegroundWindow(handle))


def window_titles() -> list[str]:
    return [window.title for window in list_top_level_windows()]


def word_window_titles(*, side: str | None = None) -> list[str]:
    windows = filter_windows_by_side(list_word_windows(), side)
    return [window.title for window in windows]


__all__ = [
    "WindowInfo",
    "Win32WindowError",
    "activate_window",
    "filter_windows_by_side",
    "find_word_window",
    "foreground_hwnd",
    "is_supported",
    "is_window",
    "list_top_level_windows",
    "list_word_windows",
    "looks_like_word_window",
    "screen_size",
    "window_titles",
    "word_window_titles",
]
