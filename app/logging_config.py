# ============================================================
# Exact file location: app/logging_config.py
# Horseshoe Tavern AI
# Phase 1 Part 1.4
# Structured logging, redaction, correlation, and audit support
# ============================================================

"""
Enterprise logging configuration for Horseshoe Tavern AI.

This module provides:

- Standard-library and structlog integration
- Console and rotating-file handlers
- Human-readable development logs
- JSON production logs
- Request and correlation identifiers
- Conversation and session logging context
- Sensitive-data redaction
- Exception logging
- Audit-event support
- Performance-event support
- Third-party logger normalization
- Safe configuration summaries
- Idempotent setup for local development and testing

This module must never intentionally write passwords, secret keys,
access tokens, payment-card data, or raw authorization headers to logs.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.config
import re
import sys
import time
import traceback
from collections.abc import Iterable, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import structlog

from app.config import LOG_DIRECTORY, Settings, get_settings


# ============================================================
# SECTION 01 - MODULE CONSTANTS
# ============================================================

APPLICATION_LOGGER_NAME: Final[str] = "horseshoe_tavern_ai"
ACCESS_LOGGER_NAME: Final[str] = "horseshoe_tavern_ai.access"
AUDIT_LOGGER_NAME: Final[str] = "horseshoe_tavern_ai.audit"
PERFORMANCE_LOGGER_NAME: Final[str] = "horseshoe_tavern_ai.performance"
SECURITY_LOGGER_NAME: Final[str] = "horseshoe_tavern_ai.security"
LEARNING_LOGGER_NAME: Final[str] = "horseshoe_tavern_ai.learning"

DEFAULT_LOG_FILE_NAME: Final[str] = "horseshoe_tavern_ai.log"
DEFAULT_AUDIT_LOG_FILE_NAME: Final[str] = "horseshoe_tavern_ai_audit.log"
DEFAULT_SECURITY_LOG_FILE_NAME: Final[str] = "horseshoe_tavern_ai_security.log"

DEFAULT_MAX_LOG_BYTES: Final[int] = 10 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT: Final[int] = 10

SENSITIVE_KEY_FRAGMENTS: Final[tuple[str, ...]] = (
    "password",
    "passwd",
    "secret",
    "token",
    "authorization",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "session_cookie",
    "set-cookie",
    "cookie",
    "credit_card",
    "card_number",
    "cvv",
    "cvc",
)

REDACTED_VALUE: Final[str] = "[REDACTED]"

EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

BEARER_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/=-]+\b",
    re.IGNORECASE,
)

JWT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\beyJ[A-Za-z0-9_-]{5,}\."
    r"[A-Za-z0-9_-]{5,}\."
    r"[A-Za-z0-9_-]{5,}\b"
)

CARD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)"
)

PHONE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[\s.-]?)?"
    r"(?:\(?\d{3}\)?[\s.-]?)"
    r"\d{3}[\s.-]?\d{4}"
    r"(?!\d)"
)

CONTROL_CHARACTER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)


# ============================================================
# SECTION 02 - CONTEXT VARIABLES
# ============================================================

_request_id_context: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar(
        "horseshoe_request_id",
        default=None,
    )
)

_conversation_id_context: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar(
        "horseshoe_conversation_id",
        default=None,
    )
)

_session_id_context: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar(
        "horseshoe_session_id",
        default=None,
    )
)

_customer_id_context: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar(
        "horseshoe_customer_id",
        default=None,
    )
)

_route_context: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar(
        "horseshoe_route",
        default=None,
    )
)


# ============================================================
# SECTION 03 - LOGGING STATE
# ============================================================

_logging_configured = False


# ============================================================
# SECTION 04 - DATA STRUCTURES
# ============================================================

@dataclass(frozen=True, slots=True)
class LoggingContext:
    """
    Current logging context snapshot.
    """

    request_id: str | None = None
    conversation_id: str | None = None
    session_id: str | None = None
    customer_id: str | None = None
    route: str | None = None

    def as_dict(self) -> dict[str, str]:
        values = {
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "customer_id": self.customer_id,
            "route": self.route,
        }

        return {
            key: value
            for key, value in values.items()
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class LoggingConfigurationReport:
    """
    Safe report describing the active logging setup.
    """

    configured: bool
    log_level: str
    json_logging: bool
    log_directory: str
    application_log_file: str
    audit_log_file: str
    security_log_file: str
    console_enabled: bool
    file_logging_enabled: bool
    sensitive_data_redaction: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "log_level": self.log_level,
            "json_logging": self.json_logging,
            "log_directory": self.log_directory,
            "application_log_file": self.application_log_file,
            "audit_log_file": self.audit_log_file,
            "security_log_file": self.security_log_file,
            "console_enabled": self.console_enabled,
            "file_logging_enabled": self.file_logging_enabled,
            "sensitive_data_redaction": self.sensitive_data_redaction,
        }


# ============================================================
# SECTION 05 - IDENTIFIER HELPERS
# ============================================================

def generate_request_id() -> str:
    """
    Generate a compact request correlation identifier.
    """

    return uuid4().hex


def set_request_id(request_id: str | None) -> None:
    _request_id_context.set(_clean_context_value(request_id))


def get_request_id() -> str | None:
    return _request_id_context.get()


def set_conversation_id(conversation_id: str | None) -> None:
    _conversation_id_context.set(
        _clean_context_value(conversation_id)
    )


def get_conversation_id() -> str | None:
    return _conversation_id_context.get()


def set_session_id(session_id: str | None) -> None:
    _session_id_context.set(_clean_context_value(session_id))


def get_session_id() -> str | None:
    return _session_id_context.get()


def set_customer_id(customer_id: str | None) -> None:
    _customer_id_context.set(_clean_context_value(customer_id))


def get_customer_id() -> str | None:
    return _customer_id_context.get()


def set_route(route: str | None) -> None:
    _route_context.set(_clean_context_value(route))


def get_route() -> str | None:
    return _route_context.get()


def _clean_context_value(value: str | None) -> str | None:
    """
    Prevent multiline or control-character injection into log context.
    """

    if value is None:
        return None

    cleaned = CONTROL_CHARACTER_PATTERN.sub(
        "",
        str(value).strip(),
    )

    cleaned = cleaned.replace("\r", " ").replace("\n", " ")

    return cleaned[:256] or None


def current_logging_context() -> LoggingContext:
    """
    Return the current correlation context.
    """

    return LoggingContext(
        request_id=get_request_id(),
        conversation_id=get_conversation_id(),
        session_id=get_session_id(),
        customer_id=get_customer_id(),
        route=get_route(),
    )


def clear_logging_context() -> None:
    """
    Clear all request-scoped context variables.
    """

    set_request_id(None)
    set_conversation_id(None)
    set_session_id(None)
    set_customer_id(None)
    set_route(None)
    structlog.contextvars.clear_contextvars()


def bind_logging_context(
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
    session_id: str | None = None,
    customer_id: str | None = None,
    route: str | None = None,
) -> LoggingContext:
    """
    Bind request and conversation identifiers to logging context.
    """

    if request_id is not None:
        set_request_id(request_id)

    if conversation_id is not None:
        set_conversation_id(conversation_id)

    if session_id is not None:
        set_session_id(session_id)

    if customer_id is not None:
        set_customer_id(customer_id)

    if route is not None:
        set_route(route)

    context = current_logging_context()

    structlog.contextvars.bind_contextvars(
        **context.as_dict()
    )

    return context


@contextmanager
def logging_context(
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
    session_id: str | None = None,
    customer_id: str | None = None,
    route: str | None = None,
):
    """
    Temporarily bind logging context and restore it afterward.
    """

    previous = current_logging_context()

    try:
        bind_logging_context(
            request_id=request_id,
            conversation_id=conversation_id,
            session_id=session_id,
            customer_id=customer_id,
            route=route,
        )
        yield current_logging_context()
    finally:
        clear_logging_context()
        bind_logging_context(
            request_id=previous.request_id,
            conversation_id=previous.conversation_id,
            session_id=previous.session_id,
            customer_id=previous.customer_id,
            route=previous.route,
        )


# ============================================================
# SECTION 06 - REDACTION HELPERS
# ============================================================

def _looks_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")

    return any(
        fragment in normalized
        for fragment in SENSITIVE_KEY_FRAGMENTS
    )


def _mask_email(value: str) -> str:
    """
    Preserve minimal diagnostic structure without storing full email.
    """

    try:
        local_part, domain = value.split("@", maxsplit=1)
    except ValueError:
        return REDACTED_VALUE

    visible_local = local_part[:1] if local_part else "*"

    return f"{visible_local}***@{domain}"


def _redact_string(
    value: str,
    *,
    redact_email: bool = False,
    redact_phone: bool = True,
) -> str:
    """
    Redact common secrets and personal data from a string.
    """

    sanitized = CONTROL_CHARACTER_PATTERN.sub("", value)

    sanitized = BEARER_TOKEN_PATTERN.sub(
        "Bearer [REDACTED]",
        sanitized,
    )

    sanitized = JWT_PATTERN.sub(
        "[REDACTED_JWT]",
        sanitized,
    )

    sanitized = CARD_PATTERN.sub(
        "[REDACTED_CARD]",
        sanitized,
    )

    if redact_phone:
        sanitized = PHONE_PATTERN.sub(
            "[REDACTED_PHONE]",
            sanitized,
        )

    if redact_email:
        sanitized = EMAIL_PATTERN.sub(
            lambda match: _mask_email(match.group(0)),
            sanitized,
        )

    return sanitized


def redact_sensitive_value(
    value: Any,
    *,
    key: str | None = None,
    redact_email: bool = False,
    redact_phone: bool = True,
    maximum_depth: int = 8,
    _depth: int = 0,
) -> Any:
    """
    Recursively redact sensitive values from log payloads.

    This function is intentionally defensive. Unknown objects are converted
    to sanitized string representations instead of being serialized blindly.
    """

    if key is not None and _looks_sensitive_key(key):
        return REDACTED_VALUE

    if _depth >= maximum_depth:
        return "[MAX_DEPTH_REACHED]"

    if value is None:
        return None

    if isinstance(value, bool | int | float):
        return value

    if isinstance(value, str):
        return _redact_string(
            value,
            redact_email=redact_email,
            redact_phone=redact_phone,
        )

    if isinstance(value, bytes):
        return f"[BYTES:{len(value)}]"

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}

        for nested_key, nested_value in value.items():
            clean_key = str(nested_key)

            output[clean_key] = redact_sensitive_value(
                nested_value,
                key=clean_key,
                redact_email=redact_email,
                redact_phone=redact_phone,
                maximum_depth=maximum_depth,
                _depth=_depth + 1,
            )

        return output

    if isinstance(value, Iterable):
        return [
            redact_sensitive_value(
                item,
                redact_email=redact_email,
                redact_phone=redact_phone,
                maximum_depth=maximum_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    return _redact_string(
        str(value),
        redact_email=redact_email,
        redact_phone=redact_phone,
    )


# ============================================================
# SECTION 07 - STRUCTLOG PROCESSORS
# ============================================================

def add_application_context(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Add stable application metadata to every structured event.
    """

    del logger
    del method_name

    settings = get_settings()

    event_dict.setdefault("application", settings.app_slug)
    event_dict.setdefault("application_version", settings.app_version)
    event_dict.setdefault("environment", settings.environment)

    return event_dict


def add_correlation_context(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Add request and conversation context.
    """

    del logger
    del method_name

    for key, value in current_logging_context().as_dict().items():
        event_dict.setdefault(key, value)

    return event_dict


def redact_event_dict(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Redact secrets and selected personal data from structured logs.
    """

    del logger
    del method_name

    settings = get_settings()

    redacted = redact_sensitive_value(
        dict(event_dict),
        redact_email=settings.privacy_mode == "strict",
        redact_phone=settings.redact_sensitive_data,
    )

    if not isinstance(redacted, dict):
        return {"event": str(redacted)}

    event_dict.clear()
    event_dict.update(redacted)

    return event_dict


def rename_event_key(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Retain structlog's conventional event key while adding message.
    """

    del logger
    del method_name

    if "event" in event_dict:
        event_dict.setdefault("message", event_dict["event"])

    return event_dict


# ============================================================
# SECTION 08 - STANDARD LOGGING FILTERS
# ============================================================

class CorrelationContextFilter(logging.Filter):
    """
    Add correlation values to standard-library LogRecord objects.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        context = current_logging_context()

        record.request_id = context.request_id or "-"
        record.conversation_id = context.conversation_id or "-"
        record.session_id = context.session_id or "-"
        record.customer_id = context.customer_id or "-"
        record.route = context.route or "-"

        return True


class SensitiveDataFilter(logging.Filter):
    """
    Redact sensitive content from standard logging records.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            original_message = record.getMessage()

            record.msg = _redact_string(
                original_message,
                redact_email=get_settings().privacy_mode == "strict",
                redact_phone=get_settings().redact_sensitive_data,
            )

            record.args = ()
        except Exception:
            record.msg = "[LOG_MESSAGE_REDACTION_FAILED]"
            record.args = ()

        return True


# ============================================================
# SECTION 09 - FORMATTERS
# ============================================================

class JsonLogFormatter(logging.Formatter):
    """
    JSON formatter for standard-library and third-party logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(
                record,
                "%Y-%m-%dT%H:%M:%S%z",
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": getattr(record, "request_id", "-"),
            "conversation_id": getattr(
                record,
                "conversation_id",
                "-",
            ),
            "session_id": getattr(record, "session_id", "-"),
            "customer_id": getattr(record, "customer_id", "-"),
            "route": getattr(record, "route", "-"),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(
                record.exc_info
            )

        if record.stack_info:
            payload["stack"] = self.formatStack(
                record.stack_info
            )

        redacted = redact_sensitive_value(
            payload,
            redact_email=get_settings().privacy_mode == "strict",
            redact_phone=get_settings().redact_sensitive_data,
        )

        return json.dumps(
            redacted,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )


# ============================================================
# SECTION 10 - HANDLER FACTORIES
# ============================================================

def _build_console_handler(
    settings: Settings,
) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(settings.log_level)
    handler.addFilter(CorrelationContextFilter())
    handler.addFilter(SensitiveDataFilter())

    if settings.json_logging:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                (
                    "%(asctime)s | %(levelname)-8s | %(name)s | "
                    "request=%(request_id)s | "
                    "conversation=%(conversation_id)s | "
                    "%(message)s"
                ),
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    return handler


def _build_rotating_file_handler(
    file_path: Path,
    settings: Settings,
    *,
    level: str | int | None = None,
) -> RotatingFileHandler:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        filename=file_path,
        maxBytes=DEFAULT_MAX_LOG_BYTES,
        backupCount=DEFAULT_LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )

    handler.setLevel(level or settings.log_level)
    handler.addFilter(CorrelationContextFilter())
    handler.addFilter(SensitiveDataFilter())
    handler.setFormatter(JsonLogFormatter())

    return handler


# ============================================================
# SECTION 11 - STRUCTLOG CONFIGURATION
# ============================================================

def _configure_structlog(settings: Settings) -> None:
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(
            fmt="iso",
            utc=True,
            key="timestamp",
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        add_application_context,
        add_correlation_context,
        redact_event_dict,
        rename_event_key,
    ]

    if settings.json_logging:
        renderer: Any = structlog.processors.JSONRenderer(
            sort_keys=True,
            ensure_ascii=False,
        )
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stdout.isatty(),
            exception_formatter=(
                structlog.dev.RichTracebackFormatter(
                    show_locals=False,
                )
            ),
        )

    structlog.configure(
        processors=[
            *shared_processors,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ============================================================
# SECTION 12 - ROOT LOGGER CONFIGURATION
# ============================================================

def _remove_existing_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

        try:
            handler.close()
        except Exception:
            pass


def configure_logging(
    settings: Settings | None = None,
    *,
    force: bool = False,
    enable_console: bool = True,
    enable_files: bool = True,
) -> LoggingConfigurationReport:
    """
    Configure application logging.

    This function is idempotent unless force=True.
    """

    global _logging_configured

    active_settings = settings or get_settings()

    if _logging_configured and not force:
        return build_logging_configuration_report(
            active_settings,
            console_enabled=enable_console,
            file_logging_enabled=enable_files,
        )

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    _remove_existing_handlers(root_logger)

    root_logger.setLevel(active_settings.log_level)

    if enable_console:
        root_logger.addHandler(
            _build_console_handler(active_settings)
        )

    if enable_files:
        root_logger.addHandler(
            _build_rotating_file_handler(
                LOG_DIRECTORY / DEFAULT_LOG_FILE_NAME,
                active_settings,
            )
        )

    audit_logger = logging.getLogger(AUDIT_LOGGER_NAME)
    _remove_existing_handlers(audit_logger)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False

    if enable_console and active_settings.debug:
        audit_logger.addHandler(
            _build_console_handler(active_settings)
        )

    if enable_files:
        audit_logger.addHandler(
            _build_rotating_file_handler(
                LOG_DIRECTORY / DEFAULT_AUDIT_LOG_FILE_NAME,
                active_settings,
                level=logging.INFO,
            )
        )

    security_logger = logging.getLogger(SECURITY_LOGGER_NAME)
    _remove_existing_handlers(security_logger)
    security_logger.setLevel(logging.INFO)
    security_logger.propagate = False

    if enable_console:
        security_logger.addHandler(
            _build_console_handler(active_settings)
        )

    if enable_files:
        security_logger.addHandler(
            _build_rotating_file_handler(
                LOG_DIRECTORY / DEFAULT_SECURITY_LOG_FILE_NAME,
                active_settings,
                level=logging.INFO,
            )
        )

    _configure_third_party_loggers(active_settings)
    _configure_structlog(active_settings)

    logging.captureWarnings(True)

    _logging_configured = True

    logger = get_logger(__name__)
    logger.info(
        "logging_configured",
        log_level=active_settings.log_level,
        json_logging=active_settings.json_logging,
        environment=active_settings.environment,
        console_enabled=enable_console,
        file_logging_enabled=enable_files,
    )

    return build_logging_configuration_report(
        active_settings,
        console_enabled=enable_console,
        file_logging_enabled=enable_files,
    )


def _configure_third_party_loggers(settings: Settings) -> None:
    """
    Normalize noisy dependency loggers.
    """

    levels: dict[str, int] = {
        "uvicorn": logging.INFO,
        "uvicorn.error": logging.INFO,
        "uvicorn.access": (
            logging.INFO
            if settings.log_requests
            else logging.WARNING
        ),
        "fastapi": logging.INFO,
        "sqlalchemy.engine": (
            logging.INFO
            if settings.database_echo
            else logging.WARNING
        ),
        "sqlalchemy.pool": logging.WARNING,
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "multipart": logging.WARNING,
        "asyncio": (
            logging.DEBUG
            if settings.debug
            else logging.WARNING
        ),
    }

    for logger_name, logger_level in levels.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logger_level)
        logger.propagate = True


# ============================================================
# SECTION 13 - LOGGER ACCESSORS
# ============================================================

def get_logger(name: str | None = None) -> Any:
    """
    Return a bound structlog logger.
    """

    logger_name = name or APPLICATION_LOGGER_NAME
    return structlog.get_logger(logger_name)


def get_access_logger() -> Any:
    return get_logger(ACCESS_LOGGER_NAME)


def get_audit_logger() -> logging.Logger:
    return logging.getLogger(AUDIT_LOGGER_NAME)


def get_security_logger() -> logging.Logger:
    return logging.getLogger(SECURITY_LOGGER_NAME)


def get_performance_logger() -> Any:
    return get_logger(PERFORMANCE_LOGGER_NAME)


def get_learning_logger() -> Any:
    return get_logger(LEARNING_LOGGER_NAME)


# ============================================================
# SECTION 14 - DOMAIN LOGGING HELPERS
# ============================================================

def log_audit_event(
    event_name: str,
    *,
    actor_type: str,
    actor_id: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    action: str | None = None,
    outcome: str = "success",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """
    Write a durable administrative or data-governance audit event.
    """

    payload = {
        "event": event_name,
        "event_type": "audit",
        "actor_type": actor_type,
        "actor_id": actor_id,
        "target_type": target_type,
        "target_id": target_id,
        "action": action,
        "outcome": outcome,
        "metadata": dict(metadata or {}),
        **current_logging_context().as_dict(),
    }

    safe_payload = redact_sensitive_value(
        payload,
        redact_email=True,
        redact_phone=True,
    )

    get_audit_logger().info(
        json.dumps(
            safe_payload,
            ensure_ascii=False,
            default=str,
        )
    )


def log_security_event(
    event_name: str,
    *,
    severity: str = "warning",
    source_ip: str | None = None,
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """
    Write a security-related event without exposing secrets.
    """

    payload = {
        "event": event_name,
        "event_type": "security",
        "severity": severity,
        "source_ip": source_ip,
        "reason": reason,
        "metadata": dict(metadata or {}),
        **current_logging_context().as_dict(),
    }

    safe_payload = redact_sensitive_value(
        payload,
        redact_email=True,
        redact_phone=True,
    )

    logger = get_security_logger()
    log_method = getattr(
        logger,
        severity.lower(),
        logger.warning,
    )

    log_method(
        json.dumps(
            safe_payload,
            ensure_ascii=False,
            default=str,
        )
    )


def log_learning_event(
    event_name: str,
    *,
    model_type: str | None = None,
    model_version: str | None = None,
    dataset_version: str | None = None,
    outcome: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """
    Log training, evaluation, review, promotion, or rollback events.
    """

    get_learning_logger().info(
        event_name,
        event_type="learning",
        model_type=model_type,
        model_version=model_version,
        dataset_version=dataset_version,
        outcome=outcome,
        metadata=dict(metadata or {}),
    )


def log_exception(
    logger: Any,
    event_name: str,
    exception: BaseException,
    **metadata: Any,
) -> None:
    """
    Log an exception with a safe traceback.
    """

    logger.error(
        event_name,
        exception_type=type(exception).__name__,
        exception_message=str(exception),
        traceback="".join(
            traceback.format_exception(
                type(exception),
                exception,
                exception.__traceback__,
            )
        ),
        **metadata,
    )


@contextmanager
def measure_operation(
    operation_name: str,
    *,
    logger: Any | None = None,
    warning_threshold_ms: float | None = None,
    **metadata: Any,
):
    """
    Measure and log an operation's duration.

    Exceptions are logged and re-raised.
    """

    operation_logger = logger or get_performance_logger()
    started_at = time.perf_counter()

    operation_logger.debug(
        "operation_started",
        operation=operation_name,
        **metadata,
    )

    try:
        yield
    except Exception as exc:
        duration_ms = (
            time.perf_counter() - started_at
        ) * 1000.0

        operation_logger.error(
            "operation_failed",
            operation=operation_name,
            duration_ms=round(duration_ms, 3),
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            **metadata,
        )

        raise
    else:
        duration_ms = (
            time.perf_counter() - started_at
        ) * 1000.0

        log_level = (
            "warning"
            if (
                warning_threshold_ms is not None
                and duration_ms >= warning_threshold_ms
            )
            else "info"
        )

        getattr(operation_logger, log_level)(
            "operation_completed",
            operation=operation_name,
            duration_ms=round(duration_ms, 3),
            exceeded_warning_threshold=(
                warning_threshold_ms is not None
                and duration_ms >= warning_threshold_ms
            ),
            **metadata,
        )


# ============================================================
# SECTION 15 - REPORTING
# ============================================================

def build_logging_configuration_report(
    settings: Settings | None = None,
    *,
    console_enabled: bool = True,
    file_logging_enabled: bool = True,
) -> LoggingConfigurationReport:
    """
    Build a safe status report.
    """

    active_settings = settings or get_settings()

    return LoggingConfigurationReport(
        configured=_logging_configured,
        log_level=active_settings.log_level,
        json_logging=active_settings.json_logging,
        log_directory=str(LOG_DIRECTORY),
        application_log_file=str(
            LOG_DIRECTORY / DEFAULT_LOG_FILE_NAME
        ),
        audit_log_file=str(
            LOG_DIRECTORY / DEFAULT_AUDIT_LOG_FILE_NAME
        ),
        security_log_file=str(
            LOG_DIRECTORY / DEFAULT_SECURITY_LOG_FILE_NAME
        ),
        console_enabled=console_enabled,
        file_logging_enabled=file_logging_enabled,
        sensitive_data_redaction=(
            active_settings.redact_sensitive_data
        ),
    )


# ============================================================
# SECTION 16 - TEST SUPPORT
# ============================================================

def reset_logging_configuration() -> None:
    """
    Reset global logging state.

    Intended for automated tests and controlled local verification.
    """

    global _logging_configured

    root_logger = logging.getLogger()
    _remove_existing_handlers(root_logger)

    for logger_name in (
        AUDIT_LOGGER_NAME,
        SECURITY_LOGGER_NAME,
    ):
        _remove_existing_handlers(
            logging.getLogger(logger_name)
        )

    structlog.reset_defaults()
    clear_logging_context()
    _logging_configured = False


def validate_redaction() -> dict[str, Any]:
    """
    Run deterministic redaction assertions.
    """

    sample = {
        "password": "super-secret-password",
        "authorization": "Bearer abc.def.ghi",
        "contact_email": "customer@example.com",
        "phone": "973-555-1234",
        "message": (
            "Card 4111 1111 1111 1111 and "
            "Bearer secret-token-value"
        ),
        "nested": {
            "api_key": "secret-api-key",
            "safe_value": "happy hour",
        },
    }

    redacted = redact_sensitive_value(
        sample,
        redact_email=True,
        redact_phone=True,
    )

    serialized = json.dumps(redacted, default=str)

    forbidden_values = (
        "super-secret-password",
        "abc.def.ghi",
        "customer@example.com",
        "973-555-1234",
        "4111 1111 1111 1111",
        "secret-token-value",
        "secret-api-key",
    )

    failures = [
        value
        for value in forbidden_values
        if value in serialized
    ]

    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "redacted_payload": redacted,
    }


# ============================================================
# SECTION 17 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    report = configure_logging(force=True)

    logger = get_logger(__name__)

    with logging_context(
        request_id=generate_request_id(),
        conversation_id="verification-conversation",
        session_id="verification-session",
        route="/internal/logging-verification",
    ):
        logger.info(
            "logging_verification_event",
            message_type="configuration_test",
            sample_email="verification@example.com",
            sample_phone="973-555-1234",
            authorization="Bearer verification-secret",
        )

        with measure_operation(
            "logging_verification_operation",
            logger=logger,
            warning_threshold_ms=1000.0,
        ):
            time.sleep(0.001)

    output = {
        "configuration": report.as_dict(),
        "redaction": validate_redaction(),
    }

    print(
        json.dumps(
            output,
            indent=2,
            default=str,
        )
    )
