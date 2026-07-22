# ============================================================
# Exact file location: app/database/session.py
# Horseshoe Tavern AI
# Phase 1 Part 1.8
# SQLAlchemy engine, sessions, transactions, health checks,
# SQLite/PostgreSQL compatibility, and FastAPI dependencies
# ============================================================

"""
Database engine and session management for Horseshoe Tavern AI.

This module provides:

- SQLite support for local development
- PostgreSQL support for production
- SQLAlchemy engine creation
- Connection-pool configuration
- SQLite foreign-key enforcement
- Thread-safe session factory creation
- FastAPI database-session dependency
- Managed transaction context
- Safe commit and rollback helpers
- Read-only session helper
- Database connectivity checks
- Database health reporting
- Table existence and row-count helpers
- Schema creation and disposal helpers
- Testing engine overrides
- Clear logging around database failures

Durable chatbot conversations, messages, training examples, leads,
knowledge facts, widget state, and analytics will use this session layer.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from sqlalchemy import (
    Engine,
    event,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Connection
from sqlalchemy.exc import (
    DBAPIError,
    IntegrityError,
    OperationalError,
    SQLAlchemyError,
)
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)

from app.config import Settings, get_settings
from app.database.base import Base, utc_now
from app.logging_config import (
    get_logger,
    log_exception,
    measure_operation,
)


# ============================================================
# SECTION 01 - LOGGER AND CONSTANTS
# ============================================================

logger = get_logger(__name__)

DEFAULT_HEALTH_TIMEOUT_SECONDS: Final[float] = 5.0
SQLITE_CONNECT_TIMEOUT_SECONDS: Final[int] = 30
SQLITE_BUSY_TIMEOUT_MILLISECONDS: Final[int] = 30000


# ============================================================
# SECTION 02 - DATABASE HEALTH REPORT
# ============================================================

@dataclass(frozen=True, slots=True)
class DatabaseHealthReport:
    """
    Safe database connectivity and inventory report.
    """

    status: str
    database_connected: bool
    database_backend: str
    database_url_type: str
    response_time_ms: float | None
    server_time: datetime
    table_count: int
    tables: tuple[str, ...]
    pool_status: str | None
    error_type: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "database_connected": self.database_connected,
            "database_backend": self.database_backend,
            "database_url_type": self.database_url_type,
            "response_time_ms": self.response_time_ms,
            "server_time": self.server_time.isoformat(),
            "table_count": self.table_count,
            "tables": list(self.tables),
            "pool_status": self.pool_status,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


# ============================================================
# SECTION 03 - MODULE STATE
# ============================================================

_engine_lock = threading.RLock()
_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_engine_settings_signature: tuple[Any, ...] | None = None


# ============================================================
# SECTION 04 - DATABASE URL AND BACKEND HELPERS
# ============================================================

def database_backend_from_url(database_url: str) -> str:
    """
    Return a normalized backend name.
    """

    normalized = database_url.strip().lower()

    if normalized.startswith("sqlite"):
        return "sqlite"

    if normalized.startswith(
        (
            "postgresql://",
            "postgresql+",
            "postgres://",
        )
    ):
        return "postgresql"

    return "other"


def database_url_type(database_url: str) -> str:
    """
    Return a safe URL classification without exposing credentials.
    """

    backend = database_backend_from_url(database_url)

    if backend == "sqlite":
        if ":memory:" in database_url:
            return "sqlite_memory"

        return "sqlite_file"

    if backend == "postgresql":
        return "postgresql"

    return "unknown"


def sanitize_database_error(
    exception: BaseException,
) -> str:
    """
    Return a compact error message without database credentials.
    """

    message = str(exception)

    if "://" in message and "@" in message:
        return "Database connection failed. Credentials were redacted."

    return message[:500]


# ============================================================
# SECTION 05 - ENGINE CONFIGURATION
# ============================================================

def _settings_signature(
    settings: Settings,
) -> tuple[Any, ...]:
    return (
        settings.database_url,
        settings.database_echo,
        settings.database_pool_size,
        settings.database_max_overflow,
        settings.database_pool_timeout_seconds,
        settings.database_pool_recycle_seconds,
        settings.environment,
    )


def _build_engine_options(
    settings: Settings,
) -> dict[str, Any]:
    """
    Build backend-specific SQLAlchemy engine options.
    """

    backend = database_backend_from_url(
        settings.database_url
    )

    common_options: dict[str, Any] = {
        "echo": settings.database_echo,
        "future": True,
        "pool_pre_ping": True,
    }

    if backend == "sqlite":
        common_options["connect_args"] = {
            "check_same_thread": False,
            "timeout": SQLITE_CONNECT_TIMEOUT_SECONDS,
        }

        return common_options

    if backend == "postgresql":
        common_options.update(
            {
                "pool_size": settings.database_pool_size,
                "max_overflow": settings.database_max_overflow,
                "pool_timeout": (
                    settings.database_pool_timeout_seconds
                ),
                "pool_recycle": (
                    settings.database_pool_recycle_seconds
                ),
            }
        )

    return common_options


def _register_sqlite_events(
    engine: Engine,
) -> None:
    """
    Enable important SQLite connection settings.
    """

    @event.listens_for(engine, "connect")
    def _configure_sqlite_connection(
        dbapi_connection: Any,
        connection_record: Any,
    ) -> None:
        del connection_record

        cursor = dbapi_connection.cursor()

        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(
                f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MILLISECONDS}"
            )
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()


def create_database_engine(
    settings: Settings | None = None,
) -> Engine:
    """
    Create a SQLAlchemy engine for the configured database.
    """

    active_settings = settings or get_settings()

    from sqlalchemy import create_engine

    engine = create_engine(
        active_settings.database_url,
        **_build_engine_options(active_settings),
    )

    if database_backend_from_url(
        active_settings.database_url
    ) == "sqlite":
        _register_sqlite_events(engine)

    logger.info(
        "database_engine_created",
        backend=database_backend_from_url(
            active_settings.database_url
        ),
        url_type=database_url_type(
            active_settings.database_url
        ),
        echo=active_settings.database_echo,
    )

    return engine


def get_engine(
    settings: Settings | None = None,
) -> Engine:
    """
    Return the process-wide SQLAlchemy engine.
    """

    global _engine
    global _engine_settings_signature

    active_settings = settings or get_settings()
    signature = _settings_signature(active_settings)

    with _engine_lock:
        if (
            _engine is None
            or _engine_settings_signature != signature
        ):
            if _engine is not None:
                _engine.dispose()

            _engine = create_database_engine(
                active_settings
            )

            _engine_settings_signature = signature

        return _engine


# ============================================================
# SECTION 06 - SESSION FACTORY
# ============================================================

def create_session_factory(
    engine: Engine | None = None,
) -> sessionmaker[Session]:
    """
    Create the SQLAlchemy session factory.
    """

    active_engine = engine or get_engine()

    return sessionmaker(
        bind=active_engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
        autocommit=False,
        future=True,
    )


def get_session_factory() -> sessionmaker[Session]:
    """
    Return the process-wide session factory.
    """

    global _session_factory

    with _engine_lock:
        if _session_factory is None:
            _session_factory = create_session_factory(
                get_engine()
            )

        return _session_factory


def create_session() -> Session:
    """
    Create one unmanaged SQLAlchemy session.

    Prefer managed_database_session() or get_database_session()
    unless explicit lifecycle control is required.
    """

    return get_session_factory()()


# ============================================================
# SECTION 07 - TRANSACTION HELPERS
# ============================================================

def safe_commit(
    session: Session,
    *,
    operation: str = "database_commit",
) -> None:
    """
    Commit a session and rollback if the commit fails.
    """

    try:
        session.commit()

        logger.debug(
            "database_commit_completed",
            operation=operation,
        )
    except IntegrityError as exc:
        session.rollback()

        logger.warning(
            "database_integrity_error",
            operation=operation,
            error_type=type(exc).__name__,
        )

        raise
    except SQLAlchemyError as exc:
        session.rollback()

        log_exception(
            logger,
            "database_commit_failed",
            exc,
            operation=operation,
        )

        raise


def safe_rollback(
    session: Session,
    *,
    operation: str = "database_rollback",
) -> None:
    """
    Roll back a session without masking the original failure.
    """

    try:
        session.rollback()

        logger.debug(
            "database_rollback_completed",
            operation=operation,
        )
    except SQLAlchemyError as exc:
        log_exception(
            logger,
            "database_rollback_failed",
            exc,
            operation=operation,
        )


@contextlib.contextmanager
def managed_database_session(
    *,
    commit: bool = True,
    operation: str = "managed_database_session",
) -> Iterator[Session]:
    """
    Provide a managed transactional database session.

    Commits automatically when commit=True. Rolls back on failure.
    """

    session = create_session()

    try:
        with measure_operation(
            operation,
            logger=logger,
            warning_threshold_ms=1000.0,
        ):
            yield session

            if commit:
                safe_commit(
                    session,
                    operation=operation,
                )
    except Exception:
        safe_rollback(
            session,
            operation=operation,
        )

        raise
    finally:
        session.close()


@contextlib.contextmanager
def read_only_database_session(
    *,
    operation: str = "read_only_database_session",
) -> Iterator[Session]:
    """
    Provide a session intended for read-only operations.
    """

    session = create_session()

    try:
        with measure_operation(
            operation,
            logger=logger,
            warning_threshold_ms=1000.0,
        ):
            yield session
    finally:
        if session.in_transaction():
            session.rollback()

        session.close()


# ============================================================
# SECTION 08 - FASTAPI SESSION DEPENDENCY
# ============================================================

def get_database_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding one request-scoped session.

    Route handlers explicitly commit when they mutate data.
    Uncommitted work is rolled back when the request ends.
    """

    session = create_session()

    try:
        yield session
    except Exception:
        safe_rollback(
            session,
            operation="fastapi_request_session",
        )

        raise
    finally:
        if session.in_transaction():
            session.rollback()

        session.close()


# ============================================================
# SECTION 09 - SCHEMA MANAGEMENT
# ============================================================

def create_all_tables(
    engine: Engine | None = None,
) -> None:
    """
    Create every table currently registered on Base.metadata.
    """

    active_engine = engine or get_engine()

    with measure_operation(
        "create_all_tables",
        logger=logger,
        warning_threshold_ms=2000.0,
    ):
        Base.metadata.create_all(
            bind=active_engine
        )

    logger.info(
        "database_tables_created",
        table_count=len(Base.metadata.tables),
    )


def drop_all_tables(
    engine: Engine | None = None,
    *,
    allow: bool = False,
) -> None:
    """
    Drop every registered table.

    This requires allow=True to prevent accidental destructive execution.
    """

    if not allow:
        raise PermissionError(
            "drop_all_tables requires allow=True."
        )

    active_engine = engine or get_engine()

    Base.metadata.drop_all(
        bind=active_engine
    )

    logger.warning(
        "database_tables_dropped",
        table_count=len(Base.metadata.tables),
    )


# ============================================================
# SECTION 10 - TABLE INSPECTION
# ============================================================

def table_exists(
    table_name: str,
    engine: Engine | None = None,
) -> bool:
    """
    Return whether the database contains a table.
    """

    active_engine = engine or get_engine()

    return inspect(active_engine).has_table(
        table_name
    )


def collect_database_inventory(
    engine: Engine | None = None,
) -> dict[str, Any]:
    """
    Return database table and view inventory.
    """

    active_engine = engine or get_engine()
    inspector = inspect(active_engine)

    tables = tuple(sorted(inspector.get_table_names()))
    views = tuple(sorted(inspector.get_view_names()))

    return {
        "table_count": len(tables),
        "tables": list(tables),
        "view_count": len(views),
        "views": list(views),
    }


def count_table_rows(
    table_name: str,
    engine: Engine | None = None,
) -> int:
    """
    Count rows using a validated inspected table name.
    """

    active_engine = engine or get_engine()
    inspector = inspect(active_engine)

    valid_tables = set(
        inspector.get_table_names()
    )

    if table_name not in valid_tables:
        raise ValueError(
            f"Unknown table: {table_name}"
        )

    quoted_table = inspector.bind.dialect.identifier_preparer.quote(
        table_name
    )

    with active_engine.connect() as connection:
        result = connection.execute(
            text(
                f"SELECT COUNT(*) FROM {quoted_table}"
            )
        )

        return int(result.scalar_one())


# ============================================================
# SECTION 11 - CONNECTION TESTING
# ============================================================

def test_database_connection(
    engine: Engine | None = None,
) -> float:
    """
    Execute a lightweight database query.

    Returns latency in milliseconds.
    """

    active_engine = engine or get_engine()
    started_at = time.perf_counter()

    with active_engine.connect() as connection:
        connection.execute(
            text("SELECT 1")
        ).scalar_one()

    return round(
        (
            time.perf_counter()
            - started_at
        ) * 1000.0,
        3,
    )


def database_health(
    engine: Engine | None = None,
) -> DatabaseHealthReport:
    """
    Build a safe database health report.
    """

    active_engine = engine or get_engine()
    active_settings = get_settings()

    backend = database_backend_from_url(
        active_settings.database_url
    )

    url_type = database_url_type(
        active_settings.database_url
    )

    try:
        latency_ms = test_database_connection(
            active_engine
        )

        inventory = collect_database_inventory(
            active_engine
        )

        pool_status: str | None

        try:
            pool_status = active_engine.pool.status()
        except Exception:
            pool_status = None

        return DatabaseHealthReport(
            status="ok",
            database_connected=True,
            database_backend=backend,
            database_url_type=url_type,
            response_time_ms=latency_ms,
            server_time=utc_now(),
            table_count=inventory["table_count"],
            tables=tuple(inventory["tables"]),
            pool_status=pool_status,
        )
    except (
        OperationalError,
        DBAPIError,
        SQLAlchemyError,
    ) as exc:
        logger.error(
            "database_health_check_failed",
            error_type=type(exc).__name__,
            database_backend=backend,
            database_url_type=url_type,
        )

        return DatabaseHealthReport(
            status="failed",
            database_connected=False,
            database_backend=backend,
            database_url_type=url_type,
            response_time_ms=None,
            server_time=utc_now(),
            table_count=0,
            tables=(),
            pool_status=None,
            error_type=type(exc).__name__,
            error_message=sanitize_database_error(
                exc
            ),
        )


# ============================================================
# SECTION 12 - BASIC QUERY HELPERS
# ============================================================

def execute_scalar(
    statement: Any,
    *,
    session: Session | None = None,
) -> Any:
    """
    Execute a statement and return scalar_one_or_none().
    """

    if session is not None:
        return session.execute(
            statement
        ).scalar_one_or_none()

    with read_only_database_session(
        operation="execute_scalar"
    ) as local_session:
        return local_session.execute(
            statement
        ).scalar_one_or_none()


def execute_all(
    statement: Any,
    *,
    session: Session | None = None,
) -> list[Any]:
    """
    Execute a statement and return all scalar results.
    """

    if session is not None:
        return list(
            session.execute(
                statement
            ).scalars().all()
        )

    with read_only_database_session(
        operation="execute_all"
    ) as local_session:
        return list(
            local_session.execute(
                statement
            ).scalars().all()
        )


# ============================================================
# SECTION 13 - ENGINE RESET AND TEST SUPPORT
# ============================================================

def dispose_database_engine() -> None:
    """
    Dispose the process-wide engine and clear its session factory.
    """

    global _engine
    global _session_factory
    global _engine_settings_signature

    with _engine_lock:
        if _engine is not None:
            _engine.dispose()

        _engine = None
        _session_factory = None
        _engine_settings_signature = None

    logger.info(
        "database_engine_disposed"
    )


def configure_database_for_testing(
    engine: Engine,
) -> None:
    """
    Replace the global engine and session factory for tests.
    """

    global _engine
    global _session_factory
    global _engine_settings_signature

    with _engine_lock:
        if _engine is not None:
            _engine.dispose()

        _engine = engine
        _session_factory = create_session_factory(
            engine
        )
        _engine_settings_signature = (
            "testing_override",
            id(engine),
        )


# ============================================================
# SECTION 14 - MODULE SELF-TEST
# ============================================================

def validate_database_session_module() -> dict[str, Any]:
    """
    Run deterministic validation against in-memory SQLite.
    """

    from sqlalchemy import (
        String,
        create_engine,
    )
    from sqlalchemy.orm import Mapped, mapped_column

    from app.database.base import (
        StandardRecordMixin,
    )

    class SessionVerificationRecord(
        StandardRecordMixin,
        Base,
    ):
        __tablename__ = (
            "_session_verification_records"
        )

        name: Mapped[str] = mapped_column(
            String(100),
            nullable=False,
        )

    test_engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
        future=True,
        pool_pre_ping=True,
    )

    _register_sqlite_events(test_engine)

    Base.metadata.create_all(
        bind=test_engine,
        tables=[
            SessionVerificationRecord.__table__,
        ],
    )

    test_factory = create_session_factory(
        test_engine
    )

    with test_factory() as session:
        record = SessionVerificationRecord(
            name="Horseshoe Tavern"
        )

        session.add(record)
        session.commit()

        record_id = record.id

    with test_factory() as session:
        loaded = session.get(
            SessionVerificationRecord,
            record_id,
        )

        loaded_name = (
            loaded.name
            if loaded is not None
            else None
        )

    inspector = inspect(test_engine)

    checks = {
        "connection_test": (
            test_database_connection(
                test_engine
            ) >= 0
        ),
        "table_exists": (
            inspector.has_table(
                "_session_verification_records"
            )
        ),
        "record_persisted": (
            loaded_name
            == "Horseshoe Tavern"
        ),
        "inventory_contains_table": (
            "_session_verification_records"
            in collect_database_inventory(
                test_engine
            )["tables"]
        ),
        "row_count": (
            count_table_rows(
                "_session_verification_records",
                test_engine,
            )
            == 1
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    health = database_health(
        test_engine
    )

    test_engine.dispose()

    return {
        "status": (
            "ok"
            if not failed_checks
            else "failed"
        ),
        "checks": checks,
        "failed_checks": failed_checks,
        "health": health.as_dict(),
    }


# ============================================================
# SECTION 15 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import json

    report = validate_database_session_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
