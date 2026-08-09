"""Initialize the procurement MVP database without changing existing demo data."""

from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.db import Database  # noqa: E402
from app.seed import init_database  # noqa: E402


def main() -> None:
    database_path = BACKEND_DIR / "procurement_demo.db"
    database = Database(f"sqlite:///{database_path.as_posix()}")
    init_database(database)
    print(f"Demo database initialized: {database_path}")


if __name__ == "__main__":
    main()
