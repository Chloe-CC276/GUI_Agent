"""
GUI Agent model-tuning package.

Subpackages
-----------
- ``data``: convert processed GUI datasets into SFT / DPO records
- ``train``: LoRA / QLoRA training entrypoints
- ``eval``: offline and online evaluation harnesses
- ``configs``: default YAML configs

See ``docs/MODEL_TUNING_FRAMEWORK.md`` for the full framework.
"""

from __future__ import annotations

__all__ = ["TUNING_FRAMEWORK_DOC"]

TUNING_FRAMEWORK_DOC = "docs/MODEL_TUNING_FRAMEWORK.md"
