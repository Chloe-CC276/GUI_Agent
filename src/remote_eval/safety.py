"""Safety guards for real Windows execution in dual-mode eval."""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass
class SafetyGuard:
    """Countdown + emergency stop for live mouse/keyboard actions."""

    countdown_seconds: float = 3.0
    require_confirm: bool = False
    abort_event: threading.Event | None = None

    def __post_init__(self) -> None:
        if self.abort_event is None:
            self.abort_event = threading.Event()

    def request_abort(self) -> None:
        assert self.abort_event is not None
        self.abort_event.set()
        LOGGER.warning("SafetyGuard abort requested")

    def reset(self) -> None:
        assert self.abort_event is not None
        self.abort_event.clear()

    def aborted(self) -> bool:
        assert self.abort_event is not None
        return self.abort_event.is_set()

    def before_action(self, action_summary: str) -> bool:
        """Return False if operator aborted / declined."""

        print(f"\n[Safety] next action: {action_summary}", flush=True)
        if self.require_confirm:
            ans = input("Execute this action? [y/N/abort]: ").strip().lower()
            if ans in {"a", "abort", "q", "quit"}:
                self.request_abort()
                return False
            if ans not in {"y", "yes"}:
                print("[Safety] skipped by operator", flush=True)
                return False

        remaining = float(self.countdown_seconds)
        print(
            f"[Safety] countdown {remaining:.0f}s — move mouse to screen corner "
            "to trigger PyAutoGUI FAILSAFE, or Ctrl+C to abort",
            flush=True,
        )
        while remaining > 0:
            if self.aborted():
                return False
            time.sleep(min(0.5, remaining))
            remaining -= 0.5
        return not self.aborted()


def install_sigint_abort(guard: SafetyGuard) -> None:
    """Best-effort: Ctrl+C sets abort (does not replace default KeyboardInterrupt)."""

    if sys.platform.startswith("win"):
        try:
            import signal

            def _handler(signum, frame):  # noqa: ANN001
                guard.request_abort()
                raise KeyboardInterrupt

            signal.signal(signal.SIGINT, _handler)
        except Exception:  # noqa: BLE001
            pass
