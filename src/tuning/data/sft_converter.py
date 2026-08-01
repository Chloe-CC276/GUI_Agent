"""
Convert processed ``GUITaskSample`` JSONL into multimodal SFT chat records.

Status
------
Framework stub. Implementation lands in Phase 1
(see ``docs/MODEL_TUNING_FRAMEWORK.md``).

Expected CLI (planned)::

    python -m src.tuning.data.sft_converter \\
        --input data/processed/screenagent/train.jsonl \\
        --output data/sft/train.jsonl
"""

from __future__ import annotations


def convert_processed_jsonl_to_sft(
    input_path: str,
    output_path: str,
    *,
    prefer_corrected_response: bool = True,
    prompt_version: str = "planner_v0",
) -> int:
    """
    Convert processed GUI JSONL into SFT JSONL.

    Returns the number of written samples.

    Raises
    ------
    NotImplementedError
        Until Phase 1 conversion is implemented.
    """
    raise NotImplementedError(
        "SFT conversion is not implemented yet. "
        "Follow docs/MODEL_TUNING_FRAMEWORK.md Phase 1."
    )


def main() -> None:
    raise SystemExit(
        "src.tuning.data.sft_converter is a framework stub. "
        "See docs/MODEL_TUNING_FRAMEWORK.md."
    )


if __name__ == "__main__":
    main()
