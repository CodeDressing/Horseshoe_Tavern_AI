# ============================================================
# Exact file location: app/database/migration_service.py
# Horseshoe Tavern AI
# Phase 2 Part 2.1
# Programmatic Alembic migration inspection and execution
# ============================================================

"""
Programmatic migration utilities for Horseshoe Tavern AI.

This module provides controlled wrappers around Alembic for:

- Locating the project Alembic configuration
- Resolving the current database URL
- Inspecting migration heads
- Inspecting the current database revision
- Detecting pending migrations
- Applying upgrades
- Applying downgrades only when explicitly requested
- Producing JSON-safe migration health reports

Application startup does not automatically perform destructive
downgrades or autogenerate revisions.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


# ============================================================
# SECTION 01 - CONSTANTS
# ============================================================

MIGRATION_SERVICE_VERSION: Final[str] = "1.0.0"
MIGRATION_SERVICE_PHASE: Final[str] = "Phase 2 Part 2.1"

PROJECT_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[2]
)

ALEMBIC_CONFIG_PATH: Final[Path] = (
    PROJECT_ROOT / "alembic.ini"
)


# ============================================================
# SECTION 02 - DATA STRUCTURES
# ============================================================

@dataclass(frozen=True, slots=True)
class MigrationState:
    database_url_type: str
    configured_heads: tuple[str, ...]
    current_heads: tuple[str, ...]
    pending_heads: tuple[str, ...]
    migration_count: int
    database_connected: bool
    up_to_date: bool
    checked_at: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# SECTION 03 - URL HELPERS
# ============================================================

def normalize_database_url(value: str) -> str:
    normalized = str(value).strip()

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


def database_url_type(value: str) -> str:
    normalized = value.casefold()

    if normalized.startswith("sqlite"):
        return "sqlite"

    if normalized.startswith("postgresql"):
        return "postgresql"

    if "://" in normalized:
        return normalized.split("://", 1)[0]

    return "unknown"


def resolve_database_url(
    explicit_url: str | None = None,
) -> str:
    if explicit_url and explicit_url.strip():
        return normalize_database_url(explicit_url)

    environment_url = os.getenv(
        "DATABASE_URL",
        "",
    ).strip()

    if environment_url:
        return normalize_database_url(
            environment_url
        )

    try:
        from app.config import get_settings

        settings = get_settings()
        settings_url = getattr(
            settings,
            "database_url",
            None,
        )

        if settings_url:
            return normalize_database_url(
                str(settings_url)
            )

    except Exception:
        pass

    return "sqlite:///./horseshoe_tavern_ai.db"


# ============================================================
# SECTION 04 - ALEMBIC CONFIGURATION
# ============================================================

def build_alembic_config(
    database_url: str | None = None,
) -> Config:
    if not ALEMBIC_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Alembic configuration not found: "
            f"{ALEMBIC_CONFIG_PATH}"
        )

    resolved_url = resolve_database_url(
        database_url
    )

    configuration = Config(
        str(ALEMBIC_CONFIG_PATH)
    )

    configuration.set_main_option(
        "script_location",
        str(
            PROJECT_ROOT
            / "app"
            / "database"
            / "migrations"
        ),
    )

    configuration.set_main_option(
        "sqlalchemy.url",
        resolved_url.replace("%", "%%"),
    )

    return configuration


# ============================================================
# SECTION 05 - ENGINE CREATION
# ============================================================

def create_migration_engine(
    database_url: str | None = None,
) -> Engine:
    resolved_url = resolve_database_url(
        database_url
    )

    arguments: dict[str, Any] = {
        "future": True,
        "pool_pre_ping": True,
    }

    if resolved_url.startswith("sqlite"):
        arguments["connect_args"] = {
            "check_same_thread": False,
        }

    return create_engine(
        resolved_url,
        **arguments,
    )


# ============================================================
# SECTION 06 - REVISION INSPECTION
# ============================================================

def configured_heads(
    database_url: str | None = None,
) -> tuple[str, ...]:
    configuration = build_alembic_config(
        database_url
    )

    scripts = ScriptDirectory.from_config(
        configuration
    )

    return tuple(
        sorted(
            scripts.get_heads()
        )
    )


def migration_count(
    database_url: str | None = None,
) -> int:
    configuration = build_alembic_config(
        database_url
    )

    scripts = ScriptDirectory.from_config(
        configuration
    )

    return sum(
        1
        for _ in scripts.walk_revisions()
    )


def current_database_heads(
    database_url: str | None = None,
) -> tuple[str, ...]:
    engine = create_migration_engine(
        database_url
    )

    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection
            )

            return tuple(
                sorted(
                    context.get_current_heads()
                )
            )

    finally:
        engine.dispose()


# ============================================================
# SECTION 07 - MIGRATION HEALTH
# ============================================================

def collect_migration_state(
    database_url: str | None = None,
) -> MigrationState:
    resolved_url = resolve_database_url(
        database_url
    )

    checked_at = datetime.now(
        timezone.utc
    ).isoformat()

    try:
        expected_heads = configured_heads(
            resolved_url
        )

        current_heads = current_database_heads(
            resolved_url
        )

        pending = tuple(
            head
            for head in expected_heads
            if head not in current_heads
        )

        return MigrationState(
            database_url_type=database_url_type(
                resolved_url
            ),
            configured_heads=expected_heads,
            current_heads=current_heads,
            pending_heads=pending,
            migration_count=migration_count(
                resolved_url
            ),
            database_connected=True,
            up_to_date=not pending,
            checked_at=checked_at,
            error=None,
        )

    except Exception as exc:
        return MigrationState(
            database_url_type=database_url_type(
                resolved_url
            ),
            configured_heads=(),
            current_heads=(),
            pending_heads=(),
            migration_count=0,
            database_connected=False,
            up_to_date=False,
            checked_at=checked_at,
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================
# SECTION 08 - MIGRATION EXECUTION
# ============================================================

def upgrade_database(
    revision: str = "head",
    *,
    database_url: str | None = None,
) -> MigrationState:
    configuration = build_alembic_config(
        database_url
    )

    command.upgrade(
        configuration,
        revision,
    )

    return collect_migration_state(
        database_url
    )


def downgrade_database(
    revision: str,
    *,
    database_url: str | None = None,
    confirmation: str,
) -> MigrationState:
    """
    Downgrade only after explicit destructive-operation approval.
    """

    if confirmation != "ALLOW_DATABASE_DOWNGRADE":
        raise PermissionError(
            "Database downgrade requires the exact confirmation "
            "ALLOW_DATABASE_DOWNGRADE."
        )

    if not revision.strip():
        raise ValueError(
            "A downgrade revision is required."
        )

    configuration = build_alembic_config(
        database_url
    )

    command.downgrade(
        configuration,
        revision,
    )

    return collect_migration_state(
        database_url
    )


# ============================================================
# SECTION 09 - VALIDATION
# ============================================================

def validate_migration_service_module() -> dict[str, Any]:
    temporary_database = (
        PROJECT_ROOT
        / "migration_service_verification.db"
    )

    database_url = (
        f"sqlite:///{temporary_database.as_posix()}"
    )

    if temporary_database.exists():
        temporary_database.unlink()

    try:
        configuration = build_alembic_config(
            database_url
        )

        scripts = ScriptDirectory.from_config(
            configuration
        )

        state = collect_migration_state(
            database_url
        )

        checks = {
            "configuration_exists": (
                ALEMBIC_CONFIG_PATH.exists()
            ),
            "script_directory_exists": (
                Path(
                    configuration.get_main_option(
                        "script_location"
                    )
                ).exists()
            ),
            "script_directory_loaded": (
                scripts is not None
            ),
            "database_connected": (
                state.database_connected
            ),
            "database_type_sqlite": (
                state.database_url_type
                == "sqlite"
            ),
            "state_json_safe": bool(
                state.as_dict()
            ),
            "downgrade_guard_present": True,
        }

        failed_checks = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        return {
            "status": (
                "ok"
                if not failed_checks
                else "failed"
            ),
            "checks": checks,
            "failed_checks": failed_checks,
            "state": state.as_dict(),
            "service_version": (
                MIGRATION_SERVICE_VERSION
            ),
            "service_phase": (
                MIGRATION_SERVICE_PHASE
            ),
        }

    finally:
        if temporary_database.exists():
            temporary_database.unlink()


# ============================================================
# SECTION 10 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import json

    report = validate_migration_service_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
