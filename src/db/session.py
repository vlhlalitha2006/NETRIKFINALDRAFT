from __future__ import annotations

from contextlib import contextmanager
from os import getenv
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


_DATABASE_URL = getenv("DATABASE_URL")
_ENGINE = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


def _get_session_factory() -> sessionmaker[Session]:
    global _ENGINE
    global _SESSION_FACTORY

    if _SESSION_FACTORY is not None:
        return _SESSION_FACTORY
    if not _DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set.")

    _ENGINE = create_engine(_DATABASE_URL, pool_pre_ping=True, future=True)
    _SESSION_FACTORY = sessionmaker(
        bind=_ENGINE,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )
    return _SESSION_FACTORY


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(base_metadata) -> None:
    factory = _get_session_factory()
    engine = factory.kw["bind"]
    base_metadata.create_all(bind=engine)
