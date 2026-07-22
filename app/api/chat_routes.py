# ============================================================
# Exact file location: app/api/chat_routes.py
# Horseshoe Tavern AI
# Phase 1 Part 1.21
# FastAPI chat, restore, widget-state, feedback, health,
# configuration, validation, and safe error routes
# ============================================================

"""
FastAPI routes for Horseshoe Tavern AI chat operations.

Endpoints:

- POST /api/chat/message
- POST /api/chat/restore
- POST /api/chat/widget-state
- POST /api/chat/feedback
- GET  /api/chat/config
- GET  /api/chat/health

Responsibilities:

- Validate incoming Pydantic request models
- Resolve database sessions through dependency injection
- Execute the transactional ChatService
- Restore persisted conversation state
- Persist widget state
- Persist structured feedback
- Return safe API error payloads
- Attach request correlation identifiers
- Avoid exposing internal exceptions or database details
- Prevent public feedback from becoming verified knowledge
"""

from __future__ import annotations

import copy
import time
import uuid
from datetime import datetime
from typing import Any, Final, Mapping

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.repositories import (
    AnalyticsRepository,
    ConversationRepository,
    FeedbackRepository,
    WidgetStateRepository,
)
from app.database.session import get_database_session
from app.logging_config import get_logger
from app.schemas.chat import (
    APIErrorResponse,
    ChatFeedbackRequest,
    ChatFeedbackResponse,
    ChatPublicConfiguration,
    ChatRequest,
    ChatResponse,
    ConversationRestoreRequest,
    ConversationRestoreResponse,
    ErrorCode,
    ErrorDetail,
    WidgetSize,
    WidgetState,
)
from app.services.chat_service import (
    CHAT_SERVICE_PHASE,
    CHAT_SERVICE_VERSION,
    ChatService,
    ChatServiceDecision,
    PersistenceMode,
)


# ============================================================
# SECTION 01 - SETTINGS, LOGGER, AND ROUTER
# ============================================================

settings = get_settings()
logger = get_logger(__name__)

CHAT_API_VERSION: Final[str] = "1.0.0"
CHAT_API_PHASE: Final[str] = "Phase 1 Part 1.21"

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
)


# ============================================================
# SECTION 02 - REQUEST IDENTIFIERS
# ============================================================

def _request_id(
    request: Request,
    supplied_request_id: str | None = None,
) -> str:
    """
    Resolve a safe request correlation ID.
    """

    existing = getattr(
        request.state,
        "request_id",
        None,
    )

    if existing:
        return str(existing)

    candidate = (
        str(supplied_request_id).strip()
        if supplied_request_id
        else ""
    )

    if not candidate:
        candidate = (
            f"request_{uuid.uuid4().hex}"
        )

    request.state.request_id = candidate

    return candidate


def _set_response_headers(
    response: Response,
    *,
    request_id: str,
) -> None:
    response.headers[
        "X-Request-ID"
    ] = request_id

    response.headers[
        "X-Horseshoe-Chat-API-Version"
    ] = CHAT_API_VERSION

    response.headers[
        "Cache-Control"
    ] = "no-store"


# ============================================================
# SECTION 03 - SAFE ERROR BUILDING
# ============================================================

def _error_response(
    *,
    request_id: str,
    code: ErrorCode,
    message: str,
    status_code: int,
    field: str | None = None,
    retryable: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> APIErrorResponse:
    """
    Build a public-safe structured API error.
    """

    return APIErrorResponse(
        request_id=request_id,
        error=ErrorDetail(
            code=code,
            message=message,
            field=field,
            retryable=retryable,
            metadata=dict(
                metadata or {}
            ),
        ),
        status_code=status_code,
        timestamp=(
            datetime.now().astimezone()
        ),
    )


def _raise_http_error(
    *,
    request_id: str,
    code: ErrorCode,
    message: str,
    status_code: int,
    field: str | None = None,
    retryable: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    payload = _error_response(
        request_id=request_id,
        code=code,
        message=message,
        status_code=status_code,
        field=field,
        retryable=retryable,
        metadata=metadata,
    )

    raise HTTPException(
        status_code=status_code,
        detail=payload.model_dump(
            mode="json"
        ),
    )


# ============================================================
# SECTION 04 - CHAT MESSAGE ENDPOINT
# ============================================================

@router.post(
    "/message",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Process a chat message",
    description=(
        "Processes one customer message through persistence, "
        "NLU, verified knowledge retrieval, and grounded response generation."
    ),
)
def process_chat_message(
    payload: ChatRequest,
    request: Request,
    response: Response,
    database: Session = Depends(
        get_database_session
    ),
    x_request_id: str | None = Header(
        default=None,
        alias="X-Request-ID",
    ),
) -> ChatResponse:
    request_id = _request_id(
        request,
        x_request_id,
    )

    _set_response_headers(
        response,
        request_id=request_id,
    )

    started_at = time.perf_counter()

    try:
        service = ChatService(
            database,
            business_slug=(
                payload.business_slug
            ),
        )

        result = service.process(
            payload
        )

        response.headers[
            "X-Chat-Processing-Time-MS"
        ] = str(
            int(
                round(
                    result.processing_time_ms
                )
            )
        )

        response.headers[
            "X-Conversation-ID"
        ] = result.response.conversation_id

        logger.info(
            (
                "Chat request completed "
                "request_id=%s session_id=%s "
                "conversation_id=%s intent=%s "
                "decision=%s processing_ms=%s"
            ),
            request_id,
            result.response.session_id,
            result.response.conversation_id,
            result.response.detected_intent,
            result.decision.value,
            result.processing_time_ms,
        )

        return result.response

    except ValueError as exc:
        database.rollback()

        logger.warning(
            "Chat request validation failed request_id=%s error=%s",
            request_id,
            type(exc).__name__,
        )

        _raise_http_error(
            request_id=request_id,
            code=ErrorCode.VALIDATION_ERROR,
            message=(
                "The chat request could not be processed because "
                "one or more values were invalid."
            ),
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            retryable=False,
        )

    except HTTPException:
        database.rollback()
        raise

    except Exception as exc:
        database.rollback()

        logger.exception(
            "Chat request failed request_id=%s",
            request_id,
        )

        elapsed_ms = round(
            (
                time.perf_counter()
                - started_at
            )
            * 1000.0,
            3,
        )

        _raise_http_error(
            request_id=request_id,
            code=ErrorCode.INTERNAL_ERROR,
            message=(
                "The chat service could not complete the request."
            ),
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            retryable=True,
            metadata={
                "processing_time_ms": (
                    elapsed_ms
                ),
            },
        )

    raise RuntimeError(
        "Unreachable chat route state."
    )


# ============================================================
# SECTION 05 - RESTORE ENDPOINT
# ============================================================

@router.post(
    "/restore",
    response_model=ConversationRestoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Restore a chat conversation",
)
def restore_chat_conversation_route(
    payload: ConversationRestoreRequest,
    request: Request,
    response: Response,
    database: Session = Depends(
        get_database_session
    ),
    x_request_id: str | None = Header(
        default=None,
        alias="X-Request-ID",
    ),
) -> ConversationRestoreResponse:
    request_id = _request_id(
        request,
        x_request_id,
    )

    _set_response_headers(
        response,
        request_id=request_id,
    )

    try:
        service = ChatService(
            database
        )

        result = service.restore(
            session_id=payload.session_id,
            conversation_id=(
                payload.conversation_id
            ),
            page_context=(
                payload.page_context
            ),
            limit=payload.limit,
        )

        response.headers[
            "X-Conversation-Restored"
        ] = (
            "true"
            if result.response.restored
            else "false"
        )

        response.headers[
            "X-Persistence-Mode"
        ] = result.persistence_mode.value

        return result.response

    except HTTPException:
        raise

    except Exception:
        database.rollback()

        logger.exception(
            "Conversation restore failed request_id=%s",
            request_id,
        )

        _raise_http_error(
            request_id=request_id,
            code=ErrorCode.RESTORE_FAILED,
            message=(
                "The conversation could not be restored."
            ),
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            retryable=True,
        )

    raise RuntimeError(
        "Unreachable restore route state."
    )


# ============================================================
# SECTION 06 - WIDGET STATE REQUEST MODEL
# ============================================================

from pydantic import BaseModel, ConfigDict, Field


class WidgetStateUpdateRequest(BaseModel):
    """
    Persisted widget display state sent by the browser.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    session_id: str = Field(
        min_length=8,
        max_length=128,
    )

    conversation_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
    )

    state: WidgetState

    size: WidgetSize

    unread_count: int = Field(
        default=0,
        ge=0,
        le=10000,
    )

    draft_text: str | None = Field(
        default=None,
        max_length=5000,
    )

    current_page_url: str | None = Field(
        default=None,
        max_length=2048,
    )

    current_page_category: str | None = Field(
        default=None,
        max_length=100,
    )

    private_event_draft: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class WidgetStateUpdateResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    request_id: str
    session_id: str
    conversation_id: str | None = None
    state: WidgetState
    size: WidgetSize
    unread_count: int
    persisted: bool
    updated_at: datetime


# ============================================================
# SECTION 07 - WIDGET STATE ENDPOINT
# ============================================================

@router.post(
    "/widget-state",
    response_model=WidgetStateUpdateResponse,
    status_code=status.HTTP_200_OK,
    summary="Persist widget state",
)
def update_widget_state(
    payload: WidgetStateUpdateRequest,
    request: Request,
    response: Response,
    database: Session = Depends(
        get_database_session
    ),
    x_request_id: str | None = Header(
        default=None,
        alias="X-Request-ID",
    ),
) -> WidgetStateUpdateResponse:
    request_id = _request_id(
        request,
        x_request_id,
    )

    _set_response_headers(
        response,
        request_id=request_id,
    )

    reference = (
        datetime.now().astimezone()
    )

    try:
        repository = WidgetStateRepository(
            database
        )

        repository.upsert(
            browser_session_id=(
                payload.session_id
            ),
            conversation_id=(
                payload.conversation_id
            ),
            state=payload.state.value,
            size=payload.size.value,
            unread_count=(
                payload.unread_count
            ),
            draft_text=payload.draft_text,
            current_page_url=(
                payload.current_page_url
            ),
            current_page_category=(
                payload.current_page_category
            ),
            private_event_draft=(
                copy.deepcopy(
                    payload.private_event_draft
                )
            ),
            updated_at=reference,
        )

        AnalyticsRepository(
            database
        ).record(
            event_name="widget_state_updated",
            browser_session_id=(
                payload.session_id
            ),
            conversation_id=(
                payload.conversation_id
            ),
            page_url=(
                payload.current_page_url
            ),
            page_category=(
                payload.current_page_category
            ),
            event_value=1.0,
            occurred_at=reference,
            metadata_json={
                "state": payload.state.value,
                "size": payload.size.value,
                "unread_count": (
                    payload.unread_count
                ),
                "request_id": request_id,
                **copy.deepcopy(
                    payload.metadata
                ),
            },
        )

        database.commit()

        return WidgetStateUpdateResponse(
            request_id=request_id,
            session_id=payload.session_id,
            conversation_id=(
                payload.conversation_id
            ),
            state=payload.state,
            size=payload.size,
            unread_count=(
                payload.unread_count
            ),
            persisted=True,
            updated_at=reference,
        )

    except Exception:
        database.rollback()

        logger.exception(
            "Widget state update failed request_id=%s",
            request_id,
        )

        _raise_http_error(
            request_id=request_id,
            code=ErrorCode.PERSISTENCE_ERROR,
            message=(
                "The widget state could not be saved."
            ),
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            retryable=True,
        )

    raise RuntimeError(
        "Unreachable widget-state route state."
    )


# ============================================================
# SECTION 08 - FEEDBACK ENDPOINT
# ============================================================

@router.post(
    "/feedback",
    response_model=ChatFeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit chat feedback",
    description=(
        "Stores reviewable feedback without automatically "
        "promoting it into verified business knowledge."
    ),
)
def submit_chat_feedback(
    payload: ChatFeedbackRequest,
    request: Request,
    response: Response,
    database: Session = Depends(
        get_database_session
    ),
    x_request_id: str | None = Header(
        default=None,
        alias="X-Request-ID",
    ),
) -> ChatFeedbackResponse:
    request_id = _request_id(
        request,
        x_request_id,
    )

    _set_response_headers(
        response,
        request_id=request_id,
    )

    reference = (
        datetime.now().astimezone()
    )

    try:
        conversation_repository = (
            ConversationRepository(
                database
            )
        )

        conversation = (
            conversation_repository
            .get_by_id(
                payload.conversation_id
            )
        )

        if conversation is None:
            _raise_http_error(
                request_id=request_id,
                code=(
                    ErrorCode.CONVERSATION_NOT_FOUND
                ),
                message=(
                    "The referenced conversation was not found."
                ),
                status_code=(
                    status.HTTP_404_NOT_FOUND
                ),
                field="conversation_id",
                retryable=False,
            )

        repository = FeedbackRepository(
            database
        )

        feedback = repository.create(
            feedback_id=(
                f"feedback_{uuid.uuid4().hex}"
            ),
            conversation_id=(
                payload.conversation_id
            ),
            message_id=payload.message_id,
            session_id=payload.session_id,
            feedback_type=(
                payload.feedback_type.value
            ),
            rating=payload.rating,
            comment=payload.comment,
            expected_answer=(
                payload.expected_answer
            ),
            page_url=payload.page_url,
            page_category=(
                payload.page_category
            ),
            requires_review=True,
            review_status="pending",
            created_at=reference,
            metadata_json={
                "request_id": request_id,
                "public_feedback": True,
                "auto_promote_to_verified": False,
                **copy.deepcopy(
                    payload.metadata
                ),
            },
        )

        AnalyticsRepository(
            database
        ).record(
            event_name="chat_feedback_submitted",
            browser_session_id=(
                payload.session_id
            ),
            conversation_id=(
                payload.conversation_id
            ),
            message_id=payload.message_id,
            page_url=payload.page_url,
            page_category=(
                payload.page_category
            ),
            event_value=(
                float(payload.rating)
                if payload.rating
                is not None
                else 1.0
            ),
            occurred_at=reference,
            metadata_json={
                "feedback_id": feedback.id,
                "feedback_type": (
                    payload.feedback_type.value
                ),
                "requires_review": True,
                "request_id": request_id,
            },
        )

        database.commit()

        return ChatFeedbackResponse(
            request_id=request_id,
            feedback_id=feedback.id,
            accepted=True,
            requires_review=True,
            review_status="pending",
            message=(
                "Thank you. Your feedback was saved for review."
            ),
            created_at=reference,
        )

    except HTTPException:
        database.rollback()
        raise

    except Exception:
        database.rollback()

        logger.exception(
            "Feedback submission failed request_id=%s",
            request_id,
        )

        _raise_http_error(
            request_id=request_id,
            code=ErrorCode.PERSISTENCE_ERROR,
            message=(
                "The feedback could not be saved."
            ),
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            retryable=True,
        )

    raise RuntimeError(
        "Unreachable feedback route state."
    )


# ============================================================
# SECTION 09 - PUBLIC CONFIGURATION
# ============================================================

@router.get(
    "/config",
    response_model=ChatPublicConfiguration,
    status_code=status.HTTP_200_OK,
    summary="Get public chat configuration",
)
def get_chat_configuration(
    request: Request,
    response: Response,
    x_request_id: str | None = Header(
        default=None,
        alias="X-Request-ID",
    ),
) -> ChatPublicConfiguration:
    request_id = _request_id(
        request,
        x_request_id,
    )

    _set_response_headers(
        response,
        request_id=request_id,
    )

    return ChatPublicConfiguration(
        business_slug=(
            getattr(
                settings,
                "default_business_slug",
                "horseshoe-tavern",
            )
        ),
        business_name=(
            getattr(
                settings,
                "business_display_name",
                "Horseshoe Tavern",
            )
        ),
        api_base_path="/api/chat",
        message_endpoint="/api/chat/message",
        restore_endpoint="/api/chat/restore",
        feedback_endpoint="/api/chat/feedback",
        widget_state_endpoint=(
            "/api/chat/widget-state"
        ),
        maximum_message_characters=(
            getattr(
                settings,
                "maximum_message_characters",
                3000,
            )
        ),
        default_widget_state=(
            WidgetState.COLLAPSED
        ),
        default_widget_size=(
            WidgetSize.COMPACT
        ),
        restore_enabled=True,
        feedback_enabled=True,
        private_events_enabled=True,
        human_handoff_enabled=True,
        persistence_enabled=True,
        supported_languages=["en"],
        public_version=CHAT_API_VERSION,
        metadata={
            "chat_api_phase": (
                CHAT_API_PHASE
            ),
            "chat_service_version": (
                CHAT_SERVICE_VERSION
            ),
            "chat_service_phase": (
                CHAT_SERVICE_PHASE
            ),
        },
    )


# ============================================================
# SECTION 10 - HEALTH ENDPOINT
# ============================================================

class ChatHealthResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    status: str
    api_version: str
    api_phase: str
    chat_service_version: str
    database_available: bool
    timestamp: datetime


@router.get(
    "/health",
    response_model=ChatHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check chat API health",
)
def chat_health(
    request: Request,
    response: Response,
    database: Session = Depends(
        get_database_session
    ),
    x_request_id: str | None = Header(
        default=None,
        alias="X-Request-ID",
    ),
) -> ChatHealthResponse:
    request_id = _request_id(
        request,
        x_request_id,
    )

    _set_response_headers(
        response,
        request_id=request_id,
    )

    database_available = False

    try:
        database.execute(
            __import__(
                "sqlalchemy"
            ).text(
                "SELECT 1"
            )
        )

        database_available = True

    except Exception:
        logger.exception(
            "Chat health database check failed request_id=%s",
            request_id,
        )

    if not database_available:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return ChatHealthResponse(
        status=(
            "ok"
            if database_available
            else "degraded"
        ),
        api_version=CHAT_API_VERSION,
        api_phase=CHAT_API_PHASE,
        chat_service_version=(
            CHAT_SERVICE_VERSION
        ),
        database_available=(
            database_available
        ),
        timestamp=(
            datetime.now().astimezone()
        ),
    )


# ============================================================
# SECTION 11 - ROUTE VALIDATION
# ============================================================

def validate_chat_routes_module() -> dict[str, Any]:
    routes = {
        route.path: sorted(
            route.methods or []
        )
        for route in router.routes
    }

    required_routes = {
        "/api/chat/message": "POST",
        "/api/chat/restore": "POST",
        "/api/chat/widget-state": "POST",
        "/api/chat/feedback": "POST",
        "/api/chat/config": "GET",
        "/api/chat/health": "GET",
    }

    checks = {
        "router_prefix_valid": (
            router.prefix == "/api/chat"
        ),
        "message_route_present": (
            "POST"
            in routes.get(
                "/api/chat/message",
                []
            )
        ),
        "restore_route_present": (
            "POST"
            in routes.get(
                "/api/chat/restore",
                []
            )
        ),
        "widget_state_route_present": (
            "POST"
            in routes.get(
                "/api/chat/widget-state",
                []
            )
        ),
        "feedback_route_present": (
            "POST"
            in routes.get(
                "/api/chat/feedback",
                []
            )
        ),
        "config_route_present": (
            "GET"
            in routes.get(
                "/api/chat/config",
                []
            )
        ),
        "health_route_present": (
            "GET"
            in routes.get(
                "/api/chat/health",
                []
            )
        ),
        "api_version_present": bool(
            CHAT_API_VERSION
        ),
        "api_phase_present": bool(
            CHAT_API_PHASE
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
        "routes": routes,
    }


# ============================================================
# SECTION 12 - PUBLIC EXPORTS
# ============================================================

__all__ = [
    "CHAT_API_PHASE",
    "CHAT_API_VERSION",
    "ChatHealthResponse",
    "WidgetStateUpdateRequest",
    "WidgetStateUpdateResponse",
    "chat_health",
    "get_chat_configuration",
    "process_chat_message",
    "restore_chat_conversation_route",
    "router",
    "submit_chat_feedback",
    "update_widget_state",
    "validate_chat_routes_module",
]
