"""Drop and recreate the procurement MVP database with fixed seed data."""

from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.db import Database  # noqa: E402
from app.seed import reset_database  # noqa: E402


def main() -> None:
    database_path = BACKEND_DIR / "procurement_demo.db"
    database = Database(f"sqlite:///{database_path.as_posix()}")
    reset_database(database)
    print(f"Demo database reset: {database_path}")


if __name__ == "__main__":
    main()
