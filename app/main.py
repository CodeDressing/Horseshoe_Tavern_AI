# ============================================================
# Exact file location: app/main.py
# Horseshoe Tavern AI
# Phase 1 Part 1.22
# FastAPI application factory, middleware, routers, static widget,
# startup lifecycle, health, metadata, and Render entrypoint
# ============================================================

"""
FastAPI application entrypoint for Horseshoe Tavern AI.

Responsibilities:

- Create the FastAPI application
- Register the chat and widget routers
- Mount static widget assets
- Configure CORS
- Configure trusted hosts
- Add request-correlation middleware
- Add security and no-cache response headers
- Add safe exception handlers
- Initialize database tables on startup when configured
- Expose root, application health, readiness, and metadata endpoints
- Provide a stable `app` object for Uvicorn and Render
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Final

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.chat_routes import (
    CHAT_API_PHASE,
    CHAT_API_VERSION,
    router as chat_router,
)
from app.api.widget_routes import router as widget_router
from app.config import get_settings
from app.database.base import Base
from app.database.session import (
    create_all_tables,
    dispose_database_engine,
    get_engine,
)
from app.logging_config import get_logger


# ============================================================
# SECTION 01 - CONSTANTS
# ============================================================

APPLICATION_VERSION: Final[str] = "1.0.0"
APPLICATION_PHASE: Final[str] = "Phase 1 Part 1.22"
APPLICATION_NAME: Final[str] = "Horseshoe Tavern AI"
APPLICATION_DESCRIPTION: Final[str] = (
    "Embeddable AI concierge for Horseshoe Tavern."
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIRECTORY = Path(__file__).resolve().parent
STATIC_DIRECTORY = APP_DIRECTORY / "static"
WIDGET_DIRECTORY = STATIC_DIRECTORY / "widget"

settings = get_settings()
logger = get_logger(__name__)


# ============================================================
# SECTION 02 - SETTINGS HELPERS
# ============================================================

def _setting(
    name: str,
    default: Any,
) -> Any:
    return getattr(
        settings,
        name,
        default,
    )


def _environment() -> str:
    return str(
        _setting(
            "environment",
            os.getenv("ENVIRONMENT", "development"),
        )
    ).strip().lower()


def _debug_enabled() -> bool:
    configured = _setting(
        "debug",
        None,
    )

    if configured is not None:
        return bool(configured)

    return _environment() not in {
        "production",
        "prod",
    }


def _allowed_origins() -> list[str]:
    configured = _setting(
        "cors_allowed_origins",
        None,
    )

    if isinstance(configured, str):
        values = [
            value.strip()
            for value in configured.split(",")
            if value.strip()
        ]
    elif isinstance(
        configured,
        (list, tuple, set),
    ):
        values = [
            str(value).strip()
            for value in configured
            if str(value).strip()
        ]
    else:
        values = []

    if values:
        return values

    if _environment() in {
        "production",
        "prod",
    }:
        return [
            "https://www.thehorseshoetavern.com",
            "https://thehorseshoetavern.com",
        ]

    return [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "https://www.thehorseshoetavern.com",
        "https://thehorseshoetavern.com",
    ]


def _allowed_hosts() -> list[str]:
    configured = _setting(
        "allowed_hosts",
        None,
    )

    if isinstance(configured, str):
        values = [
            value.strip()
            for value in configured.split(",")
            if value.strip()
        ]
    elif isinstance(
        configured,
        (list, tuple, set),
    ):
        values = [
            str(value).strip()
            for value in configured
            if str(value).strip()
        ]
    else:
        values = []

    if values:
        return values

    return [
        "*",
    ]


def _initialize_database_enabled() -> bool:
    return bool(
        _setting(
            "initialize_database_on_startup",
            True,
        )
    )


# ============================================================
# SECTION 03 - DATABASE HEALTH
# ============================================================

def _check_database() -> tuple[bool, str | None]:
    try:
        with get_engine().connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        return True, None

    except Exception as exc:
        logger.exception(
            "Database health check failed."
        )

        return False, type(exc).__name__


def _initialize_database() -> None:
    if not _initialize_database_enabled():
        logger.info(
            "Database initialization on startup is disabled."
        )
        return

    logger.info(
        "Initializing database metadata."
    )

    create_all_tables()

    logger.info(
        "Database metadata initialization completed."
    )


# ============================================================
# SECTION 04 - LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    startup_started = time.perf_counter()

    logger.info(
        "%s startup beginning version=%s phase=%s environment=%s",
        APPLICATION_NAME,
        APPLICATION_VERSION,
        APPLICATION_PHASE,
        _environment(),
    )

    application.state.started_at = (
        datetime.now().astimezone()
    )

    application.state.startup_complete = False
    application.state.database_initialized = False
    application.state.database_available = False
    application.state.startup_error = None

    try:
        _initialize_database()

        application.state.database_initialized = True

        database_available, database_error = (
            _check_database()
        )

        application.state.database_available = (
            database_available
        )

        application.state.database_error = (
            database_error
        )

        application.state.startup_complete = True

        elapsed_ms = round(
            (
                time.perf_counter()
                - startup_started
            )
            * 1000.0,
            3,
        )

        logger.info(
            (
                "%s startup completed "
                "database_available=%s startup_ms=%s"
            ),
            APPLICATION_NAME,
            database_available,
            elapsed_ms,
        )

    except Exception as exc:
        application.state.startup_error = (
            type(exc).__name__
        )

        logger.exception(
            "%s startup failed.",
            APPLICATION_NAME,
        )

        if _environment() in {
            "production",
            "prod",
        }:
            raise

    yield

    logger.info(
        "%s shutdown beginning.",
        APPLICATION_NAME,
    )

    try:
        dispose_database_engine()
    except Exception:
        logger.exception(
            "Database engine disposal failed."
        )

    logger.info(
        "%s shutdown completed.",
        APPLICATION_NAME,
    )


# ============================================================
# SECTION 05 - APPLICATION FACTORY
# ============================================================

def create_application() -> FastAPI:
    application = FastAPI(
        title=APPLICATION_NAME,
        description=APPLICATION_DESCRIPTION,
        version=APPLICATION_VERSION,
        debug=_debug_enabled(),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=_allowed_hosts(),
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=[
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
        ],
        allow_headers=[
            "Accept",
            "Authorization",
            "Content-Type",
            "Origin",
            "User-Agent",
            "X-Requested-With",
            "X-Request-ID",
            "X-Horseshoe-Widget-Version",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Horseshoe-App-Version",
            "X-Horseshoe-Chat-API-Version",
            "X-Chat-Processing-Time-MS",
            "X-Conversation-ID",
            "X-Conversation-Restored",
            "X-Persistence-Mode",
        ],
        max_age=600,
    )

    _register_middleware(
        application
    )

    _register_exception_handlers(
        application
    )

    application.include_router(
        chat_router
    )

    application.include_router(
        widget_router
    )

    if STATIC_DIRECTORY.exists():
        application.mount(
            "/static",
            StaticFiles(
                directory=str(
                    STATIC_DIRECTORY
                )
            ),
            name="static",
        )
    else:
        logger.warning(
            "Static directory does not exist: %s",
            STATIC_DIRECTORY,
        )

    _register_application_routes(
        application
    )

    return application


# ============================================================
# SECTION 06 - REQUEST MIDDLEWARE
# ============================================================

def _register_middleware(
    application: FastAPI,
) -> None:
    @application.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Any,
    ) -> Response:
        started_at = time.perf_counter()

        supplied_request_id = (
            request.headers.get(
                "X-Request-ID"
            )
        )

        request_id = (
            supplied_request_id.strip()
            if supplied_request_id
            and supplied_request_id.strip()
            else f"request_{uuid.uuid4().hex}"
        )

        request.state.request_id = (
            request_id
        )

        try:
            response = await call_next(
                request
            )

        except Exception:
            logger.exception(
                (
                    "Unhandled request failure "
                    "request_id=%s method=%s path=%s"
                ),
                request_id,
                request.method,
                request.url.path,
            )
            raise

        elapsed_ms = round(
            (
                time.perf_counter()
                - started_at
            )
            * 1000.0,
            3,
        )

        response.headers[
            "X-Request-ID"
        ] = request_id

        response.headers[
            "X-Horseshoe-App-Version"
        ] = APPLICATION_VERSION

        response.headers[
            "X-Content-Type-Options"
        ] = "nosniff"

        response.headers[
            "X-Frame-Options"
        ] = "SAMEORIGIN"

        response.headers[
            "Referrer-Policy"
        ] = "strict-origin-when-cross-origin"

        response.headers[
            "Permissions-Policy"
        ] = (
            "camera=(), microphone=(), geolocation=()"
        )

        if request.url.path.startswith(
            "/api/"
        ):
            response.headers[
                "Cache-Control"
            ] = "no-store"

        logger.info(
            (
                "HTTP request completed "
                "request_id=%s method=%s path=%s "
                "status=%s duration_ms=%s"
            ),
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        return response


# ============================================================
# SECTION 07 - EXCEPTION HANDLERS
# ============================================================

def _register_exception_handlers(
    application: FastAPI,
) -> None:
    @application.exception_handler(
        RequestValidationError
    )
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = getattr(
            request.state,
            "request_id",
            f"request_{uuid.uuid4().hex}",
        )

        errors = []

        for error in exc.errors():
            errors.append(
                {
                    "location": [
                        str(value)
                        for value
                        in error.get(
                            "loc",
                            (),
                        )
                    ],
                    "message": error.get(
                        "msg",
                        "Invalid value.",
                    ),
                    "type": error.get(
                        "type",
                        "validation_error",
                    ),
                }
            )

        return JSONResponse(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            content={
                "request_id": request_id,
                "error": {
                    "code": "validation_error",
                    "message": (
                        "The request contains invalid values."
                    ),
                    "retryable": False,
                    "details": errors,
                },
                "timestamp": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
            },
            headers={
                "X-Request-ID": request_id,
                "Cache-Control": "no-store",
            },
        )

    @application.exception_handler(
        HTTPException
    )
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        request_id = getattr(
            request.state,
            "request_id",
            f"request_{uuid.uuid4().hex}",
        )

        detail = exc.detail

        if isinstance(
            detail,
            dict,
        ):
            content = detail
        else:
            content = {
                "request_id": request_id,
                "error": {
                    "code": "http_error",
                    "message": str(detail),
                    "retryable": (
                        exc.status_code >= 500
                    ),
                },
                "status_code": (
                    exc.status_code
                ),
                "timestamp": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
            }

        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers={
                "X-Request-ID": request_id,
                "Cache-Control": "no-store",
                **dict(
                    exc.headers or {}
                ),
            },
        )

    @application.exception_handler(
        SQLAlchemyError
    )
    async def database_exception_handler(
        request: Request,
        exc: SQLAlchemyError,
    ) -> JSONResponse:
        request_id = getattr(
            request.state,
            "request_id",
            f"request_{uuid.uuid4().hex}",
        )

        logger.exception(
            (
                "Database request failure "
                "request_id=%s path=%s"
            ),
            request_id,
            request.url.path,
        )

        return JSONResponse(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            content={
                "request_id": request_id,
                "error": {
                    "code": "database_unavailable",
                    "message": (
                        "The service is temporarily unable "
                        "to access its database."
                    ),
                    "retryable": True,
                },
                "timestamp": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
            },
            headers={
                "X-Request-ID": request_id,
                "Retry-After": "5",
                "Cache-Control": "no-store",
            },
        )

    @application.exception_handler(
        Exception
    )
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        request_id = getattr(
            request.state,
            "request_id",
            f"request_{uuid.uuid4().hex}",
        )

        logger.exception(
            (
                "Unhandled application exception "
                "request_id=%s method=%s path=%s"
            ),
            request_id,
            request.method,
            request.url.path,
        )

        return JSONResponse(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            content={
                "request_id": request_id,
                "error": {
                    "code": "internal_error",
                    "message": (
                        "The request could not be completed."
                    ),
                    "retryable": True,
                },
                "timestamp": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
            },
            headers={
                "X-Request-ID": request_id,
                "Cache-Control": "no-store",
            },
        )


# ============================================================
# SECTION 08 - APPLICATION ROUTES
# ============================================================

def _register_application_routes(
    application: FastAPI,
) -> None:
    @application.get(
        "/",
        include_in_schema=False,
    )
    async def root() -> RedirectResponse:
        return RedirectResponse(
            url="/static/widget/widget-preview.html",
            status_code=(
                status.HTTP_307_TEMPORARY_REDIRECT
            ),
        )

    @application.get(
        "/health",
        tags=["system"],
        summary="Application health",
    )
    async def application_health(
        request: Request,
    ) -> JSONResponse:
        database_available, database_error = (
            _check_database()
        )

        startup_complete = bool(
            getattr(
                request.app.state,
                "startup_complete",
                False,
            )
        )

        overall_ok = (
            startup_complete
            and database_available
        )

        payload = {
            "status": (
                "ok"
                if overall_ok
                else "degraded"
            ),
            "application": APPLICATION_NAME,
            "version": APPLICATION_VERSION,
            "phase": APPLICATION_PHASE,
            "environment": _environment(),
            "startup_complete": (
                startup_complete
            ),
            "database_available": (
                database_available
            ),
            "database_error": (
                database_error
            ),
            "chat_api_version": (
                CHAT_API_VERSION
            ),
            "chat_api_phase": (
                CHAT_API_PHASE
            ),
            "timestamp": (
                datetime.now()
                .astimezone()
                .isoformat()
            ),
        }

        return JSONResponse(
            status_code=(
                status.HTTP_200_OK
                if overall_ok
                else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            content=payload,
            headers={
                "Cache-Control": "no-store",
            },
        )

    @application.get(
        "/ready",
        tags=["system"],
        summary="Application readiness",
    )
    async def readiness(
        request: Request,
    ) -> JSONResponse:
        database_available, database_error = (
            _check_database()
        )

        static_available = (
            WIDGET_DIRECTORY.exists()
            and (
                WIDGET_DIRECTORY
                / "horseshoe-widget.js"
            ).exists()
            and (
                WIDGET_DIRECTORY
                / "horseshoe-widget.css"
            ).exists()
        )

        startup_complete = bool(
            getattr(
                request.app.state,
                "startup_complete",
                False,
            )
        )

        ready = (
            startup_complete
            and database_available
            and static_available
        )

        return JSONResponse(
            status_code=(
                status.HTTP_200_OK
                if ready
                else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            content={
                "status": (
                    "ready"
                    if ready
                    else "not_ready"
                ),
                "startup_complete": (
                    startup_complete
                ),
                "database_available": (
                    database_available
                ),
                "database_error": (
                    database_error
                ),
                "widget_assets_available": (
                    static_available
                ),
                "timestamp": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
            },
            headers={
                "Cache-Control": "no-store",
            },
        )

    @application.get(
        "/api/meta",
        tags=["system"],
        summary="Application metadata",
    )
    async def application_metadata() -> dict[str, Any]:
        return {
            "application": APPLICATION_NAME,
            "description": APPLICATION_DESCRIPTION,
            "version": APPLICATION_VERSION,
            "phase": APPLICATION_PHASE,
            "environment": _environment(),
            "chat_api": {
                "version": CHAT_API_VERSION,
                "phase": CHAT_API_PHASE,
                "base_path": "/api/chat",
            },
            "widget": {
                "script_url": (
                    "/static/widget/horseshoe-widget.js"
                ),
                "stylesheet_url": (
                    "/static/widget/horseshoe-widget.css"
                ),
                "preview_url": (
                    "/static/widget/widget-preview.html"
                ),
            },
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json",
            },
        }

    @application.get(
        "/favicon.ico",
        include_in_schema=False,
    )
    async def favicon() -> Response:
        return Response(
            status_code=(
                status.HTTP_204_NO_CONTENT
            )
        )

    @application.get(
        "/robots.txt",
        include_in_schema=False,
    )
    async def robots() -> Response:
        return Response(
            content=(
                "User-agent: *\n"
                "Disallow: /api/\n"
                "Disallow: /docs\n"
                "Disallow: /redoc\n"
            ),
            media_type="text/plain",
        )


# ============================================================
# SECTION 09 - APPLICATION INSTANCE
# ============================================================

app = create_application()


# ============================================================
# SECTION 10 - VALIDATION
# ============================================================

def validate_main_module() -> dict[str, Any]:
    paths = {
        route.path
        for route in app.routes
    }

    required_paths = {
        "/",
        "/health",
        "/ready",
        "/api/meta",
        "/api/chat/message",
        "/api/chat/restore",
        "/api/chat/widget-state",
        "/api/chat/feedback",
        "/api/chat/config",
        "/api/chat/health",
    }

    checks = {
        "application_created": (
            isinstance(
                app,
                FastAPI,
            )
        ),
        "application_title_valid": (
            app.title
            == APPLICATION_NAME
        ),
        "application_version_valid": (
            app.version
            == APPLICATION_VERSION
        ),
        "required_routes_present": (
            required_paths.issubset(
                paths
            )
        ),
        "static_directory_present": (
            STATIC_DIRECTORY.exists()
        ),
        "widget_directory_present": (
            WIDGET_DIRECTORY.exists()
        ),
        "widget_javascript_present": (
            (
                WIDGET_DIRECTORY
                / "horseshoe-widget.js"
            ).exists()
        ),
        "widget_stylesheet_present": (
            (
                WIDGET_DIRECTORY
                / "horseshoe-widget.css"
            ).exists()
        ),
        "openapi_available": bool(
            app.openapi()
        ),
        "application_version_present": bool(
            APPLICATION_VERSION
        ),
        "application_phase_present": bool(
            APPLICATION_PHASE
        ),
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
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
        "routes": sorted(paths),
        "application_version": (
            APPLICATION_VERSION
        ),
        "application_phase": (
            APPLICATION_PHASE
        ),
    }


# ============================================================
# SECTION 11 - LOCAL EXECUTION
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=str(
            _setting(
                "host",
                "0.0.0.0",
            )
        ),
        port=int(
            os.getenv(
                "PORT",
                str(
                    _setting(
                        "port",
                        8000,
                    )
                ),
            )
        ),
        reload=bool(
            _setting(
                "reload",
                _debug_enabled(),
            )
        ),
        log_level=str(
            _setting(
                "log_level",
                "info",
            )
        ).lower(),
    )

