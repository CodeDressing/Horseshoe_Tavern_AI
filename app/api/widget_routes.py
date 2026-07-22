# ============================================================
# Exact file location: app/api/widget_routes.py
# Horseshoe Tavern AI
# Phase 1 Part 1.7
# Widget session initialization, restoration, configuration,
# state synchronization, and static asset delivery routes
# ============================================================

"""
API routes supporting the Horseshoe Tavern embeddable chatbot widget.

Responsibilities:

- Initialize or resume anonymous widget sessions
- Create stable conversation identifiers
- Preserve conversation continuity across website pages
- Accept safe page context
- Return saved conversation history when available
- Synchronize widget open state and size
- Return public widget configuration
- Serve the JavaScript and CSS assets
- Validate session and conversation identifiers
- Apply strict origin checks
- Avoid exposing secrets or internal configuration
- Provide temporary in-memory persistence until the database layer is ready

The in-memory store in this phase is intentionally temporary. Durable
conversation, message, lead, and widget-state persistence will be moved to
SQLAlchemy repositories after app/database/models.py and session.py exist.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import STATIC_DIRECTORY, get_settings
from app.logging_config import (
    bind_logging_context,
    generate_request_id,
    get_logger,
    log_security_event,
)


# ============================================================
# SECTION 01 - ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/widget",
    tags=["Widget"],
)

asset_router = APIRouter(
    tags=["Widget Assets"],
)

logger = get_logger(__name__)
settings = get_settings()


# ============================================================
# SECTION 02 - CONSTANTS
# ============================================================

SESSION_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9_-]{8,160}$"
)

CONVERSATION_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9_-]{8,160}$"
)

ALLOWED_WIDGET_STATES: Final[set[str]] = {
    "collapsed",
    "minimized",
    "open",
}

ALLOWED_WIDGET_SIZES: Final[set[str]] = {
    "compact",
    "expanded",
    "fullscreen",
}

MAX_PAGE_TITLE_LENGTH: Final[int] = 300
MAX_PAGE_URL_LENGTH: Final[int] = 2048
MAX_REFERRER_LENGTH: Final[int] = 2048
MAX_LANGUAGE_LENGTH: Final[int] = 32
MAX_PAGE_CATEGORY_LENGTH: Final[int] = 100
MAX_STORED_MESSAGES: Final[int] = 100

WIDGET_DIRECTORY: Final[Path] = (
    STATIC_DIRECTORY / "widget"
)

WIDGET_JS_FILE: Final[Path] = (
    WIDGET_DIRECTORY / "horseshoe-widget.js"
)

WIDGET_CSS_FILE: Final[Path] = (
    WIDGET_DIRECTORY / "horseshoe-widget.css"
)


# ============================================================
# SECTION 03 - SCHEMAS
# ============================================================

class PageViewport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    width: int = Field(
        default=0,
        ge=0,
        le=20000,
    )

    height: int = Field(
        default=0,
        ge=0,
        le=20000,
    )


class PageContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str = Field(
        default="",
        max_length=MAX_PAGE_URL_LENGTH,
    )

    path: str = Field(
        default="/",
        max_length=1000,
    )

    title: str = Field(
        default="",
        max_length=MAX_PAGE_TITLE_LENGTH,
    )

    category: str = Field(
        default="home",
        max_length=MAX_PAGE_CATEGORY_LENGTH,
    )

    referrer: str = Field(
        default="",
        max_length=MAX_REFERRER_LENGTH,
    )

    language: str = Field(
        default="en-US",
        max_length=MAX_LANGUAGE_LENGTH,
    )

    viewport: PageViewport = Field(
        default_factory=PageViewport,
    )

    timestamp: datetime | None = None

    @field_validator(
        "url",
        "path",
        "title",
        "category",
        "referrer",
        "language",
        mode="before",
    )
    @classmethod
    def clean_text(cls, value: Any) -> str:
        if value is None:
            return ""

        return (
            str(value)
            .replace("\x00", "")
            .replace("\r", " ")
            .replace("\n", " ")
            .strip()
        )


class WidgetContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: str = Field(
        default="",
        max_length=50,
    )

    state: Literal[
        "collapsed",
        "minimized",
        "open",
    ] = "collapsed"

    size: Literal[
        "compact",
        "expanded",
        "fullscreen",
    ] = "compact"

    previous_page_url: str | None = Field(
        default=None,
        max_length=MAX_PAGE_URL_LENGTH,
    )

    last_active_at: datetime | None = None


class WidgetMessageResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    role: Literal[
        "assistant",
        "user",
        "system",
    ]
    text: str
    created_at: datetime
    page_url: str | None = None
    actions: list[dict[str, Any]] = Field(
        default_factory=list,
    )


class WidgetSessionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(
        min_length=8,
        max_length=160,
    )

    conversation_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=160,
    )

    business_slug: str = Field(
        default="horseshoe-tavern",
        min_length=2,
        max_length=100,
    )

    page_context: PageContext = Field(
        default_factory=PageContext,
    )

    widget_context: WidgetContext | None = None

    widget_version: str | None = Field(
        default=None,
        max_length=50,
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        candidate = value.strip()

        if not SESSION_ID_PATTERN.fullmatch(candidate):
            raise ValueError(
                "Session identifier contains unsupported characters."
            )

        return candidate

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        candidate = value.strip()

        if not CONVERSATION_ID_PATTERN.fullmatch(candidate):
            raise ValueError(
                "Conversation identifier contains unsupported characters."
            )

        return candidate

    @field_validator("business_slug")
    @classmethod
    def validate_business_slug(
        cls,
        value: str,
    ) -> str:
        candidate = value.strip().lower()

        if candidate != settings.business_slug:
            raise ValueError(
                "Unknown business slug."
            )

        return candidate


class WidgetStateUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(
        min_length=8,
        max_length=160,
    )

    conversation_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=160,
    )

    widget_state: Literal[
        "collapsed",
        "minimized",
        "open",
    ]

    widget_size: Literal[
        "compact",
        "expanded",
        "fullscreen",
    ]

    unread_count: int = Field(
        default=0,
        ge=0,
        le=999,
    )

    page_context: PageContext = Field(
        default_factory=PageContext,
    )

    last_active_at: datetime | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        candidate = value.strip()

        if not SESSION_ID_PATTERN.fullmatch(candidate):
            raise ValueError(
                "Invalid session identifier."
            )

        return candidate

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        candidate = value.strip()

        if not CONVERSATION_ID_PATTERN.fullmatch(candidate):
            raise ValueError(
                "Invalid conversation identifier."
            )

        return candidate


class WidgetSessionResponse(BaseModel):
    session_id: str
    conversation_id: str
    restored: bool
    messages: list[WidgetMessageResponse]
    widget_state: str
    widget_size: str
    unread_count: int
    server_time: datetime
    persistence: str
    page_context_received: bool


class WidgetPublicConfiguration(BaseModel):
    widget_version: str
    business_slug: str
    business_name: str
    api_base_url: str
    chat_endpoint: str
    session_endpoint: str
    state_endpoint: str
    restore_endpoint: str
    javascript_url: str
    stylesheet_url: str
    allowed_states: list[str]
    allowed_sizes: list[str]
    default_state: str
    default_size: str
    maximum_message_characters: int
    conversation_persistence: bool
    session_memory_enabled: bool
    server_time: datetime


# ============================================================
# SECTION 04 - TEMPORARY MEMORY STORE
# ============================================================

class TemporaryWidgetStore:
    """
    Thread-safe temporary widget session store.

    This allows the widget routes to function before the SQLAlchemy
    repositories are implemented. It is not durable across process restarts.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._conversations: dict[str, dict[str, Any]] = {}

    def get_or_create_session(
        self,
        *,
        session_id: str,
        conversation_id: str | None,
        page_context: PageContext,
        widget_context: WidgetContext | None,
    ) -> tuple[dict[str, Any], bool]:
        with self._lock:
            restored = session_id in self._sessions

            if restored:
                session = self._sessions[session_id]
                resolved_conversation_id = (
                    conversation_id
                    or session["conversation_id"]
                )
            else:
                resolved_conversation_id = (
                    conversation_id
                    or generate_conversation_id()
                )

                session = {
                    "session_id": session_id,
                    "conversation_id": resolved_conversation_id,
                    "created_at": utc_now(),
                    "last_active_at": utc_now(),
                    "widget_state": "collapsed",
                    "widget_size": "compact",
                    "unread_count": 0,
                    "last_page_context": {},
                }

                self._sessions[session_id] = session

            if resolved_conversation_id not in self._conversations:
                self._conversations[resolved_conversation_id] = {
                    "conversation_id": resolved_conversation_id,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "messages": [],
                }

            session["conversation_id"] = resolved_conversation_id
            session["last_active_at"] = utc_now()
            session["last_page_context"] = (
                page_context.model_dump(
                    mode="json"
                )
            )

            if widget_context is not None:
                session["widget_state"] = widget_context.state
                session["widget_size"] = widget_context.size

            return dict(session), restored

    def update_state(
        self,
        *,
        request: WidgetStateUpdateRequest,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(
                request.session_id
            )

            if session is None:
                conversation_id = (
                    request.conversation_id
                    or generate_conversation_id()
                )

                session = {
                    "session_id": request.session_id,
                    "conversation_id": conversation_id,
                    "created_at": utc_now(),
                    "last_active_at": utc_now(),
                    "widget_state": request.widget_state,
                    "widget_size": request.widget_size,
                    "unread_count": request.unread_count,
                    "last_page_context": (
                        request.page_context.model_dump(
                            mode="json"
                        )
                    ),
                }

                self._sessions[
                    request.session_id
                ] = session
            else:
                if request.conversation_id:
                    session["conversation_id"] = (
                        request.conversation_id
                    )

                session["widget_state"] = (
                    request.widget_state
                )
                session["widget_size"] = (
                    request.widget_size
                )
                session["unread_count"] = (
                    request.unread_count
                )
                session["last_active_at"] = (
                    request.last_active_at
                    or utc_now()
                )
                session["last_page_context"] = (
                    request.page_context.model_dump(
                        mode="json"
                    )
                )

            return dict(session)

    def get_session(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)

            return (
                dict(session)
                if session is not None
                else None
            )

    def get_messages(
        self,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conversation = self._conversations.get(
                conversation_id
            )

            if conversation is None:
                return []

            return [
                dict(message)
                for message in conversation["messages"][
                    -MAX_STORED_MESSAGES:
                ]
            ]

    def append_message(
        self,
        *,
        conversation_id: str,
        role: str,
        text: str,
        page_url: str | None = None,
        actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            conversation = self._conversations.setdefault(
                conversation_id,
                {
                    "conversation_id": conversation_id,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "messages": [],
                },
            )

            message = {
                "id": generate_message_id(),
                "role": role,
                "text": text,
                "created_at": utc_now(),
                "page_url": page_url,
                "actions": actions or [],
            }

            conversation["messages"].append(
                message
            )

            conversation["messages"] = (
                conversation["messages"][
                    -MAX_STORED_MESSAGES:
                ]
            )

            conversation["updated_at"] = utc_now()

            return dict(message)

    def statistics(self) -> dict[str, int]:
        with self._lock:
            message_count = sum(
                len(conversation["messages"])
                for conversation in (
                    self._conversations.values()
                )
            )

            return {
                "session_count": len(
                    self._sessions
                ),
                "conversation_count": len(
                    self._conversations
                ),
                "message_count": message_count,
            }

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._conversations.clear()


widget_store = TemporaryWidgetStore()


# ============================================================
# SECTION 05 - IDENTIFIER HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now().astimezone()


def generate_conversation_id() -> str:
    return f"conversation_{uuid4()}"


def generate_message_id() -> str:
    return f"message_{uuid4()}"


def normalize_origin(value: str | None) -> str | None:
    if value is None:
        return None

    return value.strip().rstrip("/")


def ensure_allowed_origin(
    request: Request,
    origin_header: str | None,
) -> None:
    """
    Validate browser-origin access.

    Non-browser requests without Origin are permitted for local testing and
    server-side health checks. Browser requests must match configured origins.
    """

    origin = normalize_origin(origin_header)

    if origin is None:
        return

    allowed = {
        normalize_origin(item)
        for item in settings.allowed_origins
    }

    if origin not in allowed:
        log_security_event(
            "widget_origin_rejected",
            severity="warning",
            source_ip=(
                request.client.host
                if request.client
                else None
            ),
            reason="Origin not present in configured allowlist.",
            metadata={
                "origin": origin,
                "path": request.url.path,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Widget origin is not authorized.",
        )


def safe_widget_asset(
    file_path: Path,
) -> Path:
    resolved = file_path.resolve()
    widget_root = WIDGET_DIRECTORY.resolve()

    if widget_root not in resolved.parents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Widget asset not found.",
        )

    if not resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Widget asset not found.",
        )

    return resolved


# ============================================================
# SECTION 06 - SESSION INITIALIZATION
# ============================================================

@router.post(
    "/session",
    response_model=WidgetSessionResponse,
    status_code=status.HTTP_200_OK,
)
def initialize_widget_session(
    payload: WidgetSessionRequest,
    request: Request,
    origin: str | None = Header(
        default=None,
        alias="Origin",
    ),
    x_request_id: str | None = Header(
        default=None,
        alias="X-Request-ID",
    ),
) -> WidgetSessionResponse:
    """
    Initialize or resume one browser widget session.
    """

    ensure_allowed_origin(
        request,
        origin,
    )

    request_id = (
        x_request_id.strip()
        if x_request_id
        else generate_request_id()
    )

    session, restored = (
        widget_store.get_or_create_session(
            session_id=payload.session_id,
            conversation_id=payload.conversation_id,
            page_context=payload.page_context,
            widget_context=payload.widget_context,
        )
    )

    conversation_id = session[
        "conversation_id"
    ]

    bind_logging_context(
        request_id=request_id,
        session_id=payload.session_id,
        conversation_id=conversation_id,
        route=request.url.path,
    )

    messages = widget_store.get_messages(
        conversation_id
    )

    logger.info(
        "widget_session_initialized",
        restored=restored,
        business_slug=payload.business_slug,
        widget_version=(
            payload.widget_version
            or (
                payload.widget_context.version
                if payload.widget_context
                else None
            )
        ),
        page_category=payload.page_context.category,
        page_path=payload.page_context.path,
        message_count=len(messages),
        persistence="temporary_memory",
    )

    return WidgetSessionResponse(
        session_id=payload.session_id,
        conversation_id=conversation_id,
        restored=restored,
        messages=[
            WidgetMessageResponse(
                **message
            )
            for message in messages
        ],
        widget_state=session[
            "widget_state"
        ],
        widget_size=session[
            "widget_size"
        ],
        unread_count=session[
            "unread_count"
        ],
        server_time=utc_now(),
        persistence="temporary_memory",
        page_context_received=True,
    )


# ============================================================
# SECTION 07 - STATE SYNCHRONIZATION
# ============================================================

@router.put(
    "/state",
    status_code=status.HTTP_200_OK,
)
def update_widget_state(
    payload: WidgetStateUpdateRequest,
    request: Request,
    origin: str | None = Header(
        default=None,
        alias="Origin",
    ),
) -> dict[str, Any]:
    """
    Synchronize open state, display size, unread count, and page context.
    """

    ensure_allowed_origin(
        request,
        origin,
    )

    session = widget_store.update_state(
        request=payload
    )

    bind_logging_context(
        request_id=generate_request_id(),
        session_id=payload.session_id,
        conversation_id=session[
            "conversation_id"
        ],
        route=request.url.path,
    )

    logger.info(
        "widget_state_updated",
        widget_state=payload.widget_state,
        widget_size=payload.widget_size,
        unread_count=payload.unread_count,
        page_category=payload.page_context.category,
        page_path=payload.page_context.path,
    )

    return {
        "status": "ok",
        "session_id": payload.session_id,
        "conversation_id": session[
            "conversation_id"
        ],
        "widget_state": session[
            "widget_state"
        ],
        "widget_size": session[
            "widget_size"
        ],
        "unread_count": session[
            "unread_count"
        ],
        "server_time": utc_now(),
    }


# ============================================================
# SECTION 08 - CONVERSATION RESTORATION
# ============================================================

@router.get(
    "/conversation/{session_id}",
    response_model=WidgetSessionResponse,
)
def restore_widget_conversation(
    session_id: str,
    request: Request,
    origin: str | None = Header(
        default=None,
        alias="Origin",
    ),
) -> WidgetSessionResponse:
    """
    Restore a previously initialized widget session.
    """

    ensure_allowed_origin(
        request,
        origin,
    )

    if not SESSION_ID_PATTERN.fullmatch(
        session_id
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid session identifier.",
        )

    session = widget_store.get_session(
        session_id
    )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Widget session was not found.",
        )

    conversation_id = session[
        "conversation_id"
    ]

    messages = widget_store.get_messages(
        conversation_id
    )

    bind_logging_context(
        request_id=generate_request_id(),
        session_id=session_id,
        conversation_id=conversation_id,
        route=request.url.path,
    )

    logger.info(
        "widget_conversation_restored",
        message_count=len(messages),
        persistence="temporary_memory",
    )

    return WidgetSessionResponse(
        session_id=session_id,
        conversation_id=conversation_id,
        restored=True,
        messages=[
            WidgetMessageResponse(
                **message
            )
            for message in messages
        ],
        widget_state=session[
            "widget_state"
        ],
        widget_size=session[
            "widget_size"
        ],
        unread_count=session[
            "unread_count"
        ],
        server_time=utc_now(),
        persistence="temporary_memory",
        page_context_received=bool(
            session.get(
                "last_page_context"
            )
        ),
    )


# ============================================================
# SECTION 09 - PUBLIC CONFIGURATION
# ============================================================

@router.get(
    "/configuration",
    response_model=WidgetPublicConfiguration,
)
def get_widget_configuration(
    request: Request,
    origin: str | None = Header(
        default=None,
        alias="Origin",
    ),
) -> WidgetPublicConfiguration:
    """
    Return non-secret configuration needed by the browser widget.
    """

    ensure_allowed_origin(
        request,
        origin,
    )

    base_url = (
        settings.public_base_url_string
    )

    return WidgetPublicConfiguration(
        widget_version="1.0.0",
        business_slug=settings.business_slug,
        business_name=settings.business_name,
        api_base_url=base_url,
        chat_endpoint="/api/chat",
        session_endpoint="/api/widget/session",
        state_endpoint="/api/widget/state",
        restore_endpoint="/api/widget/conversation",
        javascript_url=(
            f"{base_url}"
            "/static/widget/horseshoe-widget.js"
        ),
        stylesheet_url=(
            f"{base_url}"
            "/static/widget/horseshoe-widget.css"
        ),
        allowed_states=sorted(
            ALLOWED_WIDGET_STATES
        ),
        allowed_sizes=[
            "compact",
            "expanded",
            "fullscreen",
        ],
        default_state="collapsed",
        default_size="compact",
        maximum_message_characters=(
            settings.maximum_message_characters
        ),
        conversation_persistence=(
            settings.store_conversations
        ),
        session_memory_enabled=(
            settings.session_memory_enabled
        ),
        server_time=utc_now(),
    )


# ============================================================
# SECTION 10 - TEMPORARY STORE DIAGNOSTICS
# ============================================================

@router.get(
    "/diagnostics",
)
def get_widget_diagnostics() -> dict[str, Any]:
    """
    Return safe development diagnostics.

    This route must be protected or disabled before production.
    """

    if settings.is_production:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found.",
        )

    return {
        "status": "ok",
        "persistence": "temporary_memory",
        "statistics": (
            widget_store.statistics()
        ),
        "assets": {
            "javascript_exists": (
                WIDGET_JS_FILE.is_file()
            ),
            "stylesheet_exists": (
                WIDGET_CSS_FILE.is_file()
            ),
        },
        "server_time": utc_now(),
    }


# ============================================================
# SECTION 11 - STATIC ASSET ROUTES
# ============================================================

@asset_router.get(
    "/static/widget/horseshoe-widget.js",
    include_in_schema=False,
)
def serve_widget_javascript() -> FileResponse:
    """
    Serve the embeddable widget JavaScript with a stable content type.
    """

    file_path = safe_widget_asset(
        WIDGET_JS_FILE
    )

    response = FileResponse(
        path=file_path,
        media_type="application/javascript",
        filename=None,
    )

    response.headers[
        "Cache-Control"
    ] = "public, max-age=300"

    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    return response


@asset_router.get(
    "/static/widget/horseshoe-widget.css",
    include_in_schema=False,
)
def serve_widget_stylesheet() -> FileResponse:
    """
    Serve the isolated widget stylesheet.
    """

    file_path = safe_widget_asset(
        WIDGET_CSS_FILE
    )

    response = FileResponse(
        path=file_path,
        media_type="text/css",
        filename=None,
    )

    response.headers[
        "Cache-Control"
    ] = "public, max-age=300"

    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    return response


# ============================================================
# SECTION 12 - TEST HELPERS
# ============================================================

def validate_widget_routes_module() -> dict[str, Any]:
    """
    Run deterministic module-level verification.
    """

    widget_store.clear()

    request = WidgetSessionRequest(
        session_id=(
            "session_"
            "12345678-1234-1234-1234-123456789012"
        ),
        business_slug=settings.business_slug,
        page_context=PageContext(
            url=(
                "https://www.thehorseshoetavern.com/menu"
            ),
            path="/menu",
            title="Menu",
            category="menu",
        ),
        widget_context=WidgetContext(
            version="1.0.0",
            state="collapsed",
            size="compact",
        ),
    )

    session, restored = (
        widget_store.get_or_create_session(
            session_id=request.session_id,
            conversation_id=None,
            page_context=request.page_context,
            widget_context=request.widget_context,
        )
    )

    assert restored is False

    conversation_id = session[
        "conversation_id"
    ]

    message = widget_store.append_message(
        conversation_id=conversation_id,
        role="assistant",
        text="Welcome to Horseshoe Tavern.",
        page_url=request.page_context.url,
    )

    messages = widget_store.get_messages(
        conversation_id
    )

    restored_session, restored_again = (
        widget_store.get_or_create_session(
            session_id=request.session_id,
            conversation_id=conversation_id,
            page_context=PageContext(
                url=(
                    "https://www.thehorseshoetavern.com/events"
                ),
                path="/events",
                title="Events",
                category="events",
            ),
            widget_context=WidgetContext(
                version="1.0.0",
                state="open",
                size="expanded",
            ),
        )
    )

    checks = {
        "javascript_exists": (
            WIDGET_JS_FILE.is_file()
        ),
        "stylesheet_exists": (
            WIDGET_CSS_FILE.is_file()
        ),
        "session_created": (
            session[
                "session_id"
            ] == request.session_id
        ),
        "conversation_created": bool(
            conversation_id
        ),
        "message_stored": (
            len(messages) == 1
        ),
        "message_role_preserved": (
            message["role"] == "assistant"
        ),
        "session_restored": (
            restored_again is True
        ),
        "state_restored": (
            restored_session[
                "widget_state"
            ] == "open"
        ),
        "size_restored": (
            restored_session[
                "widget_size"
            ] == "expanded"
        ),
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
        "statistics": (
            widget_store.statistics()
        ),
        "sample_session_id": (
            request.session_id
        ),
        "sample_conversation_id": (
            conversation_id
        ),
    }


# ============================================================
# SECTION 13 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import json

    report = validate_widget_routes_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
