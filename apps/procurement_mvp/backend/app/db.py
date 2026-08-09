from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def build_engine(database_url: str) -> Engine:
    options: dict = {}
    if database_url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
        if database_url in {"sqlite://", "sqlite:///:memory:"}:
            options["poolclass"] = StaticPool
    return create_engine(database_url, **options)


class Database:
    def __init__(self, database_url: str | None = None):
        self.url = database_url or os.getenv(
            "DATABASE_URL", "sqlite:///./procurement_demo.db"
        )
        self.engine = build_engine(self.url)
        self.session_factory = sessionmaker(
            bind=self.engine, autoflush=False, expire_on_commit=False
        )

    def create_all(self) -> None:
        from .migrations import (
            backfill_s0,
            migrate_oa_closure,
            migrate_procurement_cloud,
            migrate_s0,
            migrate_supplier_award_sources,
            migrate_v21,
        )

        migrate_s0(self.engine)
        Base.metadata.create_all(self.engine)
        migrate_v21(self.engine)
        migrate_oa_closure(self.engine)
        migrate_procurement_cloud(self.engine)
        migrate_supplier_award_sources(self.engine)
        backfill_s0(self.engine)

    def session(self) -> Generator[Session, None, None]:
        with self.session_factory() as session:
            yield session
