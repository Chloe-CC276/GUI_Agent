from importlib import import_module
from types import ModuleType
from typing import Final


__all__ = [
    "agent",
    "common",
    "executor",
    "model",
    "perception",
]

_SUBPACKAGES: Final[frozenset[str]] = frozenset(__all__)


def __getattr__(name: str) -> ModuleType:
    """Load a public subpackage on first access."""
    if name not in _SUBPACKAGES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Return module attributes together with lazily exposed subpackages."""
    return sorted(set(globals()) | _SUBPACKAGES)