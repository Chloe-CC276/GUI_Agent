"""Validate adapters_v1 without silently falling back to the cloud model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import ROOT


@dataclass
class AdapterStatus:
    path: str
    exists: bool
    has_config: bool
    has_weights: bool
    base_model_name: str | None
    peft_type: str | None
    ready: bool
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_adapter_path() -> Path:
    return ROOT / "lora_tuning" / "adapters" / "adapters_v1"


def check_adapter(path: str | Path | None = None) -> AdapterStatus:
    adapter = Path(path) if path else default_adapter_path()
    if not adapter.is_absolute():
        adapter = (ROOT / adapter).resolve()

    has_config = (adapter / "adapter_config.json").is_file()
    has_weights = (adapter / "adapter_model.safetensors").is_file() or (
        adapter / "adapter_model.bin"
    ).is_file()
    base = None
    peft_type = None
    if has_config:
        try:
            cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
            base = cfg.get("base_model_name_or_path")
            peft_type = cfg.get("peft_type")
        except (OSError, json.JSONDecodeError):
            has_config = False

    ready = adapter.is_dir() and has_config and has_weights and bool(base)
    blocked = None
    if not adapter.is_dir():
        blocked = f"adapter directory missing: {adapter}"
    elif not has_config:
        blocked = "adapter_config.json missing or unreadable"
    elif not has_weights:
        blocked = "adapter weights missing"
    elif not base:
        blocked = "base_model_name_or_path missing in adapter_config.json"

    return AdapterStatus(
        path=str(adapter),
        exists=adapter.is_dir(),
        has_config=has_config,
        has_weights=has_weights,
        base_model_name=base,
        peft_type=peft_type,
        ready=ready,
        blocked_reason=blocked,
    )
