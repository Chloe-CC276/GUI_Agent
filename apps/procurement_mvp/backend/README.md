# Procurement MVP backend

Python 3.13 compatible FastAPI service for the phase-one procurement workflow.

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

The default demo database is `procurement_demo.db`. Set `DATABASE_URL` to use a
different SQLite database. The API is under `/api/v1`, health is at `/health`,
and OpenAPI documentation is at `/docs`.

Seed helpers can be imported directly:

```python
from app.db import Database
from app.seed import init, reset

database = Database("sqlite:///./procurement_demo.db")
init(database)
reset(database)
```
