# ============================================================
# Exact file location: app/database/migrations/env.py
# Horseshoe Tavern AI
# Phase 2 Part 2.1
# Alembic online/offline migration environment
# ============================================================

"""
Alembic environment for Horseshoe Tavern AI.

The migration URL is resolved in this order:

1. DATABASE_URL environment variable
2. app.config settings.database_url
3. sqlalchemy.url from alembic.ini

The environment imports every SQLAlchemy model before exposing
Base.metadata to Alembic autogeneration.
"""

from __future__ import annotations

import os
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection

from app.database.base import Base

# Importing models registers mapped tables on Base.metadata.
import app.database.models  # noqa: F401,E402


# ============================================================
# SECTION 01 - ALEMBIC CONFIGURATION
# ============================================================

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ============================================================
# SECTION 02 - DATABASE URL RESOLUTION
# ============================================================

def _normalize_database_url(value: str) -> str:
    """
    Normalize common hosted PostgreSQL URL formats.

    Render and other providers may return postgres:// or
    postgresql:// URLs. The project uses SQLAlchemy with psycopg.
    """

    normalized = value.strip()

    if normalized.startswith("postgres://"):
        return normalized.replace(
            "postgres://",
            "postgresql+psycopg://",
            1,
        )

    if normalized.startswith("postgresql://"):
        return normalized.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )

    return normalized


def _settings_database_url() -> str | None:
    try:
        from app.config import get_settings

        settings = get_settings()
        value = getattr(settings, "database_url", None)

        if value is None:
            return None

        rendered = str(value).strip()
        return rendered or None

    except Exception:
        return None


def resolve_database_url() -> str:
    environment_url = os.getenv("DATABASE_URL", "").strip()

    if environment_url:
        return _normalize_database_url(environment_url)

    settings_url = _settings_database_url()

    if settings_url:
        return _normalize_database_url(settings_url)

    configured_url = config.get_main_option(
        "sqlalchemy.url"
    ).strip()

    if not configured_url:
        raise RuntimeError(
            "No database URL is configured for Alembic."
        )

    return _normalize_database_url(configured_url)


# ============================================================
# SECTION 03 - MIGRATION OPTIONS
# ============================================================

def migration_context_options() -> dict[str, Any]:
    """
    Return shared Alembic migration comparison options.
    """

    return {
        "target_metadata": target_metadata,
        "compare_type": True,
        "compare_server_default": True,
        "include_schemas": False,
        "render_as_batch": resolve_database_url().startswith(
            "sqlite"
        ),
    }


# ============================================================
# SECTION 04 - OFFLINE MIGRATIONS
# ============================================================

def run_migrations_offline() -> None:
    """
    Run migrations without creating a live DBAPI connection.
    """

    url = resolve_database_url()

    context.configure(
        url=url,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
        **migration_context_options(),
    )

    with context.begin_transaction():
        context.run_migrations()


# ============================================================
# SECTION 05 - ONLINE MIGRATIONS
# ============================================================

def _run_migrations_with_connection(
    connection: Connection,
) -> None:
    context.configure(
        connection=connection,
        **migration_context_options(),
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations using a live SQLAlchemy connection.
    """

    configuration = config.get_section(
        config.config_ini_section
    ) or {}

    configuration["sqlalchemy.url"] = (
        resolve_database_url()
    )

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        _run_migrations_with_connection(
            connection
        )

    connectable.dispose()


# ============================================================
# SECTION 06 - EXECUTION
# ============================================================

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
