"""Aggregate existing evaluation outputs into CSV + markdown report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.adapter_check import check_adapter
from src.evaluation.report import write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "outputs" / "gui_agent_evaluation",
    )
    args = parser.parse_args(argv)
    path = write_report(args.input_dir, adapter_status=check_adapter().to_dict())
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
