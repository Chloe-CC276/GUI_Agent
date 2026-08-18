from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.db import Database  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the additive S0 SQLite migration")
    parser.add_argument(
        "--database",
        type=Path,
        default=BACKEND / "procurement_demo.db",
        help="SQLite database path",
    )
    args = parser.parse_args()
    database_path = args.database.resolve()
    if database_path.exists():
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup = database_path.with_name(f"{database_path.name}.bak-s0-{stamp}")
        shutil.copy2(database_path, backup)
        print(f"Backup: {backup}")
    database = Database(f"sqlite:///{database_path.as_posix()}")
    database.create_all()
    print(f"S0 migration complete: {database_path}")


if __name__ == "__main__":
    main()
