from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router, s0_router, v21_router
from .db import Database
from .errors import install_error_handlers
from .seed import init_database


def create_app(database_url: str | None = None) -> FastAPI:
    database = Database(database_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        init_database(database)
        yield

    app = FastAPI(
        title="国企智慧采购智能助手 API",
        version="0.2.1",
        lifespan=lifespan,
    )
    app.state.database = database
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "data": {"status": "healthy", "service": "procurement-mvp"},
            "task_id": None,
            "business_key": None,
            "idempotent_replay": False,
        }

    # v21 static paths (export-candidates, batch-*) must register before
    # parameterized /procurement/requests/{request_id} on s0_router.
    app.include_router(router, prefix="/api/v1")
    app.include_router(v21_router, prefix="/api/v1")
    app.include_router(s0_router, prefix="/api/v1")
    app.include_router(v21_router, prefix="/api", include_in_schema=False)
    app.include_router(s0_router, prefix="/api", include_in_schema=False)
    return app


app = create_app()
