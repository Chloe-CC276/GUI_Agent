"""
Offline metrics for tuned GUI planners.

Status
------
Framework stub. Implementation lands in Phase 4
(see ``docs/MODEL_TUNING_FRAMEWORK.md``).

Planned metrics: ``json_ok``, ``schema_ok``, ``action_type_acc``, ``coord_mae``.
"""

from __future__ import annotations


def evaluate_offline(
    split_path: str,
    *,
    adapter_path: str | None = None,
    backend: str = "local",
) -> dict[str, float]:
    """
    Evaluate a model on a fixed SFT / processed split.

    Raises
    ------
    NotImplementedError
        Until Phase 4 evaluation is implemented.
    """
    raise NotImplementedError(
        "Offline metrics are not implemented yet. "
        "Follow docs/MODEL_TUNING_FRAMEWORK.md Phase 4."
    )


def main() -> None:
    raise SystemExit(
        "src.tuning.eval.offline_metrics is a framework stub. "
        "See docs/MODEL_TUNING_FRAMEWORK.md."
    )


if __name__ == "__main__":
    main()
