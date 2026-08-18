"""Lazy singleton runtimes for Colab inference (shared LoRA weights)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from src.evaluation.config import ExperimentConfig, VariantConfig, load_experiment_config
from src.evaluation.variants import build_variant_runtime
from src.model.local_peft_vlm import LocalPeftVLM

LOGGER = logging.getLogger(__name__)

# Must be RLock: get_runtime holds the lock and calls get_shared_vlm().
_LOCK = threading.RLock()
_SHARED_VLM: LocalPeftVLM | None = None
_RUNTIMES: dict[str, Any] = {}
_EXP: ExperimentConfig | None = None


def _ensure_exp(config_path: str | Path | None = None) -> ExperimentConfig:
    global _EXP
    if _EXP is None:
        _EXP = load_experiment_config(config_path)
        _EXP.allow_local_adapter = True
    return _EXP


def get_shared_vlm(
    *,
    adapter_path: str | Path | None = None,
    base_model: str | None = None,
) -> LocalPeftVLM:
    global _SHARED_VLM
    with _LOCK:
        if _SHARED_VLM is not None:
            return _SHARED_VLM
        exp = _ensure_exp()
        path = Path(
            adapter_path
            or next(
                (
                    v.adapter_path
                    for v in exp.variants
                    if v.model_type == "local_adapter" and v.adapter_path
                ),
                "lora_tuning/adapters/adapters_v1",
            )
        )
        if not path.is_absolute():
            path = Path.cwd() / path
        print(
            f"Loading shared LocalPeftVLM adapter={path}",
            flush=True,
        )
        _SHARED_VLM = LocalPeftVLM(
            base_model=base_model or exp.local_base_model,
            adapter_path=path,
            load_in_4bit=True,
        )
        print("Shared LocalPeftVLM object created (weights may load on first use)", flush=True)
        return _SHARED_VLM


def get_runtime(variant_name: str):
    """Return VariantRuntime for adapter_v1_original / adapter_v1_optimized."""

    with _LOCK:
        if variant_name in _RUNTIMES:
            return _RUNTIMES[variant_name]
        exp = _ensure_exp()
        variant = next((v for v in exp.variants if v.name == variant_name), None)
        if variant is None:
            prompt = "optimized" if "optimized" in variant_name else "original"
            variant = VariantConfig(
                name=variant_name,
                model_type="local_adapter",
                prompt_version=prompt,
                adapter_path="lora_tuning/adapters/adapters_v1",
            )
        print(f"Building runtime for {variant_name}...", flush=True)
        vlm = get_shared_vlm()
        runtime = build_variant_runtime(exp, variant, vlm_factory=lambda: vlm)
        if runtime.status != "ready":
            raise RuntimeError(
                f"variant {variant_name} not ready: {runtime.blocked_reason}"
            )
        _RUNTIMES[variant_name] = runtime
        print(f"Runtime ready: {variant_name}", flush=True)
        return runtime
