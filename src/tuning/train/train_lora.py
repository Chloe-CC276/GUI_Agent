"""
QLoRA / LoRA training entrypoint for GUI Agent VLMs.

Status
------
Framework stub. Implementation lands in Phase 3
(see ``docs/MODEL_TUNING_FRAMEWORK.md``).

Expected CLI (planned)::

    python -m src.tuning.train.train_lora \\
        --config src/tuning/configs/default_qlora.yaml
"""

from __future__ import annotations


def train_from_config(config_path: str) -> None:
    """
    Run LoRA/QLoRA SFT from a YAML config.

    Raises
    ------
    NotImplementedError
        Until Phase 3 training is implemented.
    """
    raise NotImplementedError(
        "LoRA training is not implemented yet. "
        "Follow docs/MODEL_TUNING_FRAMEWORK.md Phase 3."
    )


def main() -> None:
    raise SystemExit(
        "src.tuning.train.train_lora is a framework stub. "
        "See docs/MODEL_TUNING_FRAMEWORK.md."
    )


if __name__ == "__main__":
    main()
