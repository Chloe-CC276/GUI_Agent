"""
clipboard

Windows clipboard access through the Win32 API.

``pyperclip`` is optional: on Windows the native clipboard is reachable with
``ctypes`` alone, so Unicode pasting works without installing extra packages.
"""

from __future__ import annotations

import ctypes
import platform
import time
from contextlib import contextmanager
from ctypes import wintypes
from typing import Iterator, Optional

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

_OPEN_RETRIES = 10
_OPEN_RETRY_DELAY = 0.02


class ClipboardError(RuntimeError):
    """Raised when the platform clipboard cannot be read or written."""


def is_supported() -> bool:
    """Return whether the native Win32 clipboard backend can be used."""

    return platform.system().lower() == "windows"


def _bind() -> tuple[ctypes.WinDLL, ctypes.WinDLL]:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE

    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype = wintypes.HGLOBAL

    return user32, kernel32


@contextmanager
def _clipboard() -> Iterator[tuple[ctypes.WinDLL, ctypes.WinDLL]]:
    if not is_supported():
        raise ClipboardError(
            "The Win32 clipboard backend is only available on Windows."
        )

    user32, kernel32 = _bind()

    # Another process may briefly own the clipboard; retry before failing.
    for attempt in range(_OPEN_RETRIES):
        if user32.OpenClipboard(None):
            break
        if attempt == _OPEN_RETRIES - 1:
            raise ClipboardError(
                "Failed to open the Windows clipboard: "
                f"error={ctypes.get_last_error()}."
            )
        time.sleep(_OPEN_RETRY_DELAY)

    try:
        yield user32, kernel32
    finally:
        user32.CloseClipboard()


def copy_text(text: str) -> None:
    """Place Unicode *text* on the Windows clipboard."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    with _clipboard() as (user32, kernel32):
        if not user32.EmptyClipboard():
            raise ClipboardError(
                "Failed to clear the Windows clipboard: "
                f"error={ctypes.get_last_error()}."
            )

        buffer = ctypes.create_unicode_buffer(text)
        size = ctypes.sizeof(buffer)

        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            raise ClipboardError(
                "Failed to allocate clipboard memory: "
                f"error={ctypes.get_last_error()}."
            )

        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            kernel32.GlobalFree(handle)
            raise ClipboardError(
                "Failed to lock clipboard memory: "
                f"error={ctypes.get_last_error()}."
            )

        try:
            ctypes.memmove(pointer, buffer, size)
        finally:
            kernel32.GlobalUnlock(handle)

        # Ownership of the handle transfers to the system only on success.
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
            raise ClipboardError(
                "Failed to write clipboard text: "
                f"error={ctypes.get_last_error()}."
            )


def get_text() -> Optional[str]:
    """Return the clipboard's Unicode text, or ``None`` when unavailable."""

    with _clipboard() as (user32, kernel32):
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return None

        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return None

        try:
            return ctypes.c_wchar_p(pointer).value
        finally:
            kernel32.GlobalUnlock(handle)


__all__ = [
    "CF_UNICODETEXT",
    "ClipboardError",
    "copy_text",
    "get_text",
    "is_supported",
]
