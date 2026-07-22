# ============================================================
# Exact file location: app/services/chat_service.py
# Horseshoe Tavern AI
# Phase 1 Part 1.37
# End-to-end chat orchestration, persistence, restoration,
# widget state, analytics, grounded responses, and transactions
# ============================================================

"""
End-to-end application service for Horseshoe Tavern AI.

This service coordinates:

- Browser session creation and restoration
- Conversation creation and restoration
- User-message persistence
- NLU processing
- Verified knowledge retrieval
- Grounded response generation
- Assistant-message persistence
- Conversation-context persistence
- Widget-state persistence
- Analytics-event recording
- Private-event draft persistence
- Safe transaction boundaries
- Failure capture and rollback
- Structured ChatResponse creation

The service intentionally separates:

- public user input
- verified business knowledge
- generated response content
- controlled review and learning data

Public messages never become verified business facts automatically.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time as clock_time
from decimal import Decimal
from enum import Enum
from typing import Any, Final, Mapping, Sequence

from sqlalchemy.orm import Session

from app.database.models import (
    ConversationStatus,
    MessageRole,
    MessageStatus,
    WidgetSizeValue,
    WidgetStateValue,
)
from app.database.repositories import (
    AnalyticsRepository,
    BrowserSessionRepository,
    ConversationRepository,
    MessageRepository,
    PrivateEventRepository,
    WidgetStateRepository,
)
from app.logging_config import get_logger
from app.nlu.context import ConversationContext
from app.nlu.orchestrator import (
    NLUResult,
    process_message,
)
from app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConfidenceBreakdown,
    ConversationRestoreResponse,
    PageContext,
    PrivateEventDraft,
    ResponseAction,
    ResponseSource,
    ResponseValidation,
    WidgetContext,
    WidgetSize,
    WidgetState,
)
from app.services.knowledge_service import (
    KnowledgeResult,
    KnowledgeService,
)
from app.services.response_service import (
    GroundedResponse,
    ResponseService,
)


# ============================================================
# SECTION 01 - LOGGER AND CONSTANTS
# ============================================================

logger = get_logger(__name__)

CHAT_SERVICE_VERSION: Final[str] = "1.1.0"
CHAT_SERVICE_PHASE: Final[str] = "Phase 1 Part 1.37"

DEFAULT_BUSINESS_SLUG: Final[str] = "horseshoe-tavern"
DEFAULT_RESTORE_LIMIT: Final[int] = 100
MAXIMUM_RESTORE_LIMIT: Final[int] = 200

SESSION_ID_PREFIX: Final[str] = "session"
CONVERSATION_ID_PREFIX: Final[str] = "conversation"
MESSAGE_ID_PREFIX: Final[str] = "message"
REQUEST_ID_PREFIX: Final[str] = "request"

PII_REDACTION_TOKEN: Final[str] = "[REDACTED]"


# ============================================================
# SECTION 02 - ENUMERATIONS
# ============================================================

class ChatServiceDecision(str, Enum):
    COMPLETED = "completed"
    RESTORED = "restored"
    PARTIAL = "partial"
    FAILED = "failed"


class PersistenceMode(str, Enum):
    DATABASE = "database"
    TEMPORARY_MEMORY = "temporary_memory"
    BROWSER_ONLY = "browser_only"


# ============================================================
# SECTION 03 - DATA CLASSES
# ============================================================

@dataclass(frozen=True, slots=True)
class ChatProcessingResult:
    response: ChatResponse
    nlu_result: NLUResult
    knowledge_result: KnowledgeResult
    grounded_response: GroundedResponse
    decision: ChatServiceDecision
    persisted: bool
    processing_time_ms: float
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "response": self.response.model_dump(
                mode="json"
            ),
            "nlu_result": self.nlu_result.as_dict(
                include_diagnostics=False
            ),
            "knowledge_result": (
                self.knowledge_result.as_dict()
            ),
            "grounded_response": (
                self.grounded_response.as_dict()
            ),
            "decision": self.decision.value,
            "persisted": self.persisted,
            "processing_time_ms": (
                self.processing_time_ms
            ),
            "warnings": list(self.warnings),
            "metadata": copy.deepcopy(
                self.metadata
            ),
        }


@dataclass(frozen=True, slots=True)
class RestoreResult:
    response: ConversationRestoreResponse
    decision: ChatServiceDecision
    persistence_mode: PersistenceMode
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "response": self.response.model_dump(
                mode="json"
            ),
            "decision": self.decision.value,
            "persistence_mode": (
                self.persistence_mode.value
            ),
            "warnings": list(self.warnings),
        }


# ============================================================
# SECTION 04 - CHAT SERVICE
# ============================================================

class ChatService:
    """
    End-to-end transactional chat application service.
    """

    def __init__(
        self,
        session: Session,
        *,
        business_slug: str = DEFAULT_BUSINESS_SLUG,
        response_service: ResponseService | None = None,
    ) -> None:
        self.session = session
        self.business_slug = (
            business_slug.strip()
            or DEFAULT_BUSINESS_SLUG
        )

        self.browser_sessions = (
            BrowserSessionRepository(session)
        )

        self.conversations = (
            ConversationRepository(session)
        )

        self.messages = (
            MessageRepository(session)
        )

        self.widget_states = (
            WidgetStateRepository(session)
        )

        self.analytics = (
            AnalyticsRepository(session)
        )

        self.private_events = (
            PrivateEventRepository(session)
        )

        self.knowledge_service = (
            KnowledgeService(session)
        )

        self.response_service = (
            response_service
            or ResponseService()
        )

    # ========================================================
    # SECTION 05 - PROCESS CHAT MESSAGE
    # ========================================================

    def process(
        self,
        request: ChatRequest,
        *,
        now: datetime | None = None,
    ) -> ChatProcessingResult:
        started_at = time.perf_counter()
        reference = (
            now
            or datetime.now().astimezone()
        )

        request_id = self._new_id(
            REQUEST_ID_PREFIX
        )

        warnings: list[str] = []

        browser_session = None
        conversation = None
        user_message = None
        assistant_message = None

        try:
            browser_session = (
                self._resolve_browser_session(
                    request,
                    reference,
                )
            )

            conversation = (
                self._resolve_conversation(
                    request,
                    browser_session,
                    reference,
                )
            )

            previous_context = (
                self._load_conversation_context(
                    conversation
                )
            )

            user_message = (
                self._persist_user_message(
                    request=request,
                    conversation=conversation,
                    reference=reference,
                )
            )

            nlu_result = process_message(
                request.message,
                previous_context=previous_context,
                conversation_id=(
                    conversation.id
                ),
                session_id=(
                    browser_session.id
                ),
                page_category=(
                    request.page_context.category
                ),
                page_url=(
                    request.page_context.url
                ),
                page_context={
                    **request.page_context.model_dump(
                        mode="json"
                    ),
                    "business_name": (
                        "Horseshoe Tavern"
                    ),
                },
                reference_datetime=reference,
                metadata={
                    "request_id": request_id,
                    "client_message_id": (
                        request.client_message_id
                    ),
                    "business_slug": (
                        request.business_slug
                    ),
                },
            )

            knowledge_result = (
                self.knowledge_service.retrieve(
                    nlu_result,
                    business_slug=(
                        request.business_slug
                    ),
                    now=reference,
                )
            )

            grounded_response = (
                self.response_service.compose(
                    nlu_result,
                    knowledge_result,
                    now=reference,
                    response_metadata={
                        "request_id": request_id,
                        "session_id": (
                            browser_session.id
                        ),
                        "conversation_id": (
                            conversation.id
                        ),
                    },
                )
            )

            assistant_message = (
                self._persist_assistant_message(
                    conversation=conversation,
                    grounded_response=(
                        grounded_response
                    ),
                    nlu_result=nlu_result,
                    knowledge_result=(
                        knowledge_result
                    ),
                    reference=reference,
                )
            )

            self._persist_context(
                conversation=conversation,
                nlu_result=nlu_result,
                reference=reference,
            )

            self._persist_widget_state(
                request=request,
                browser_session=browser_session,
                conversation=conversation,
                grounded_response=(
                    grounded_response
                ),
                reference=reference,
            )

            self._persist_private_event_draft(
                request=request,
                browser_session=browser_session,
                conversation=conversation,
                grounded_response=(
                    grounded_response
                ),
                reference=reference,
            )

            self._record_chat_analytics(
                request=request,
                browser_session=browser_session,
                conversation=conversation,
                user_message=user_message,
                assistant_message=assistant_message,
                nlu_result=nlu_result,
                knowledge_result=(
                    knowledge_result
                ),
                grounded_response=(
                    grounded_response
                ),
                request_id=request_id,
                reference=reference,
            )

            self._update_conversation_summary(
                conversation=conversation,
                nlu_result=nlu_result,
                grounded_response=(
                    grounded_response
                ),
                reference=reference,
            )

            self.session.commit()

            processing_time_ms = round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000.0,
                3,
            )

            response = self._build_chat_response(
                request_id=request_id,
                request=request,
                browser_session=browser_session,
                conversation=conversation,
                assistant_message=assistant_message,
                nlu_result=nlu_result,
                grounded_response=(
                    grounded_response
                ),
                processing_time_ms=(
                    processing_time_ms
                ),
                reference=reference,
            )

            return ChatProcessingResult(
                response=response,
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
                grounded_response=(
                    grounded_response
                ),
                decision=(
                    ChatServiceDecision.COMPLETED
                ),
                persisted=True,
                processing_time_ms=(
                    processing_time_ms
                ),
                warnings=tuple(warnings),
                metadata={
                    "request_id": request_id,
                    "browser_session_id": (
                        browser_session.id
                    ),
                    "conversation_id": (
                        conversation.id
                    ),
                    "user_message_id": (
                        user_message.id
                    ),
                    "assistant_message_id": (
                        assistant_message.id
                    ),
                    "destination_routing": (
                        self._destination_routing_metadata(
                            grounded_response
                        )
                    ),
                    "restaurant_schema": (
                        self._restaurant_schema_metadata(
                            grounded_response
                        )
                    ),
                },
            )

        except Exception as exc:
            self.session.rollback()

            logger.exception(
                "Chat processing failed for request %s",
                request_id,
            )

            processing_time_ms = round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000.0,
                3,
            )

            raise RuntimeError(
                "Chat processing failed."
            ) from exc

    # ========================================================
    # SECTION 06 - RESTORE CONVERSATION
    # ========================================================

    def restore(
        self,
        *,
        session_id: str,
        conversation_id: str | None = None,
        page_context: PageContext | None = None,
        limit: int = DEFAULT_RESTORE_LIMIT,
        now: datetime | None = None,
    ) -> RestoreResult:
        reference = (
            now
            or datetime.now().astimezone()
        )

        limit = min(
            max(int(limit), 1),
            MAXIMUM_RESTORE_LIMIT,
        )

        warnings: list[str] = []

        browser_session = (
            self.browser_sessions.get_by_id(
                session_id
            )
        )

        if browser_session is None:
            response = ConversationRestoreResponse(
                session_id=session_id,
                conversation_id=(
                    conversation_id
                    or self._new_id(
                        CONVERSATION_ID_PREFIX
                    )
                ),
                restored=False,
                messages=[],
                widget_state=(
                    WidgetState.COLLAPSED
                ),
                widget_size=(
                    WidgetSize.COMPACT
                ),
                unread_count=0,
                private_event_draft=None,
                current_intent=None,
                conversation_status="new",
                server_time=reference,
                persistence=(
                    PersistenceMode.DATABASE.value
                ),
            )

            warnings.append(
                "Browser session was not found."
            )

            return RestoreResult(
                response=response,
                decision=(
                    ChatServiceDecision.PARTIAL
                ),
                persistence_mode=(
                    PersistenceMode.DATABASE
                ),
                warnings=tuple(warnings),
            )

        conversation = None

        if conversation_id:
            conversation = (
                self.conversations.get_by_id(
                    conversation_id
                )
            )

        if conversation is None:
            conversation = (
                self.conversations
                .get_latest_for_session(
                    browser_session.id
                )
            )

        if conversation is None:
            response = ConversationRestoreResponse(
                session_id=browser_session.id,
                conversation_id=(
                    conversation_id
                    or self._new_id(
                        CONVERSATION_ID_PREFIX
                    )
                ),
                restored=False,
                messages=[],
                widget_state=(
                    WidgetState.COLLAPSED
                ),
                widget_size=(
                    WidgetSize.COMPACT
                ),
                unread_count=0,
                private_event_draft=None,
                current_intent=None,
                conversation_status="new",
                server_time=reference,
                persistence=(
                    PersistenceMode.DATABASE.value
                ),
            )

            return RestoreResult(
                response=response,
                decision=(
                    ChatServiceDecision.PARTIAL
                ),
                persistence_mode=(
                    PersistenceMode.DATABASE
                ),
                warnings=(),
            )

        messages = (
            self.messages.list_for_conversation(
                conversation.id,
                limit=limit,
            )
        )

        restored_messages = [
            self._message_to_schema(
                message
            )
            for message in messages
        ]

        widget_state_record = (
            self.widget_states.get_for_session(
                browser_session.id
            )
        )

        private_event_draft = (
            self._restore_private_event_draft(
                conversation.id
            )
        )

        widget_state = (
            self._coerce_widget_state(
                getattr(
                    widget_state_record,
                    "state",
                    None,
                )
            )
        )

        widget_size = (
            self._coerce_widget_size(
                getattr(
                    widget_state_record,
                    "size",
                    None,
                )
            )
        )

        unread_count = int(
            getattr(
                widget_state_record,
                "unread_count",
                0,
            )
            or 0
        )

        response = ConversationRestoreResponse(
            session_id=browser_session.id,
            conversation_id=conversation.id,
            restored=True,
            messages=restored_messages,
            widget_state=widget_state,
            widget_size=widget_size,
            unread_count=unread_count,
            private_event_draft=(
                private_event_draft
            ),
            current_intent=getattr(
                conversation,
                "current_intent",
                None,
            ),
            conversation_status=str(
                getattr(
                    conversation,
                    "status",
                    "active",
                )
            ),
            server_time=reference,
            persistence=(
                PersistenceMode.DATABASE.value
            ),
        )

        return RestoreResult(
            response=response,
            decision=(
                ChatServiceDecision.RESTORED
            ),
            persistence_mode=(
                PersistenceMode.DATABASE
            ),
            warnings=tuple(warnings),
        )

    # ========================================================
    # SECTION 07 - SESSION RESOLUTION
    # ========================================================

    def _resolve_browser_session(
        self,
        request: ChatRequest,
        reference: datetime,
    ) -> Any:
        browser_session = (
            self.browser_sessions.get_by_id(
                request.session_id
            )
        )

        if browser_session is None:
            browser_session = (
                self.browser_sessions.create(
                    session_id=request.session_id,
                    business_slug=(
                        request.business_slug
                    ),
                    first_page_url=(
                        request.page_context.url
                    ),
                    first_page_category=(
                        request.page_context.category
                    ),
                    user_agent=(
                        request.metadata.get(
                            "user_agent"
                        )
                    ),
                    language=(
                        request.page_context.language
                    ),
                    metadata_json={
                        "widget_version": (
                            request.widget_context.version
                        ),
                        "initial_page": (
                            request.page_context.model_dump(
                                mode="json"
                            )
                        ),
                    },
                )
            )

        if hasattr(
            browser_session,
            "touch",
        ):
            browser_session.touch(
                page_url=(
                    request.page_context.url
                ),
                page_category=(
                    request.page_context.category
                ),
                observed_at=reference,
            )
        else:
            if hasattr(
                browser_session,
                "last_seen_at",
            ):
                browser_session.last_seen_at = (
                    reference
                )

            if hasattr(
                browser_session,
                "last_page_url",
            ):
                browser_session.last_page_url = (
                    request.page_context.url
                )

            if hasattr(
                browser_session,
                "last_page_category",
            ):
                browser_session.last_page_category = (
                    request.page_context.category
                )

        return browser_session

    # ========================================================
    # SECTION 08 - CONVERSATION RESOLUTION
    # ========================================================

    def _resolve_conversation(
        self,
        request: ChatRequest,
        browser_session: Any,
        reference: datetime,
    ) -> Any:
        conversation = None

        if request.conversation_id:
            conversation = (
                self.conversations.get_by_id(
                    request.conversation_id
                )
            )

        if conversation is None:
            conversation = (
                self.conversations
                .get_latest_for_session(
                    browser_session.id
                )
            )

        if conversation is None:
            conversation_id = (
                request.conversation_id
                or self._new_id(
                    CONVERSATION_ID_PREFIX
                )
            )

            conversation = (
                self.conversations.create(
                    conversation_id=(
                        conversation_id
                    ),
                    browser_session_id=(
                        browser_session.id
                    ),
                    business_slug=(
                        request.business_slug
                    ),
                    status=(
                        ConversationStatus.ACTIVE.value
                    ),
                    started_at=reference,
                    metadata_json={
                        "page_context": (
                            request.page_context.model_dump(
                                mode="json"
                            )
                        )
                    },
                )
            )

        if hasattr(
            conversation,
            "last_activity_at",
        ):
            conversation.last_activity_at = (
                reference
            )

        if hasattr(
            conversation,
            "status",
        ):
            conversation.status = (
                ConversationStatus.ACTIVE.value
            )

        return conversation

    # ========================================================
    # SECTION 09 - USER MESSAGE PERSISTENCE
    # ========================================================

    def _persist_user_message(
        self,
        *,
        request: ChatRequest,
        conversation: Any,
        reference: datetime,
    ) -> Any:
        return self.messages.create(
            message_id=(
                request.client_message_id
                or self._new_id(
                    MESSAGE_ID_PREFIX
                )
            ),
            conversation_id=(
                conversation.id
            ),
            role=MessageRole.USER.value,
            status=(
                MessageStatus.COMPLETED.value
            ),
            text=request.message,
            original_text=request.message,
            normalized_text=None,
            corrected_text=None,
            page_url=(
                request.page_context.url
            ),
            page_category=(
                request.page_context.category
            ),
            created_at=reference,
            metadata_json={
                "client_timestamp": (
                    request.client_timestamp.isoformat()
                    if request.client_timestamp
                    else None
                ),
                "page_context": (
                    request.page_context.model_dump(
                        mode="json"
                    )
                ),
                "widget_context": (
                    request.widget_context.model_dump(
                        mode="json"
                    )
                ),
                "request_metadata": (
                    copy.deepcopy(
                        request.metadata
                    )
                ),
            },
        )

    # ========================================================
    # SECTION 10 - ASSISTANT MESSAGE PERSISTENCE
    # ========================================================

    def _persist_assistant_message(
        self,
        *,
        conversation: Any,
        grounded_response: GroundedResponse,
        nlu_result: NLUResult,
        knowledge_result: KnowledgeResult,
        reference: datetime,
    ) -> Any:
        return self.messages.create(
            message_id=self._new_id(
                MESSAGE_ID_PREFIX
            ),
            conversation_id=(
                conversation.id
            ),
            role=MessageRole.ASSISTANT.value,
            status=(
                MessageStatus.COMPLETED.value
            ),
            text=grounded_response.message,
            original_text=None,
            normalized_text=(
                nlu_result.normalized_text
            ),
            corrected_text=(
                nlu_result.corrected_text
            ),
            detected_intent=(
                nlu_result.primary_intent.value
            ),
            intent_confidence=(
                nlu_result.confidence.intent
            ),
            answer_confidence=(
                grounded_response
                .confidence
                .overall
            ),
            sources_json=[
                source.model_dump(
                    mode="json"
                )
                for source
                in grounded_response.sources
            ],
            actions_json=[
                action.model_dump(
                    mode="json"
                )
                for action
                in grounded_response.actions
            ],
            validation_json=(
                grounded_response
                .validation
                .model_dump(
                    mode="json"
                )
            ),
            created_at=reference,
            metadata_json={
                "nlu": nlu_result.as_dict(
                    include_diagnostics=False
                ),
                "knowledge": (
                    knowledge_result.as_dict()
                ),
                "response": (
                    grounded_response.as_dict()
                ),
            },
        )

    # ========================================================
    # SECTION 11 - CONTEXT PERSISTENCE
    # ========================================================

    def _persist_context(
        self,
        *,
        conversation: Any,
        nlu_result: NLUResult,
        reference: datetime,
    ) -> None:
        context_payload = (
            nlu_result.context.as_dict()
        )

        if hasattr(
            conversation,
            "context_json",
        ):
            conversation.context_json = (
                context_payload
            )

        if hasattr(
            conversation,
            "current_intent",
        ):
            conversation.current_intent = (
                nlu_result
                .context
                .current_intent
            )

        if hasattr(
            conversation,
            "active_flow",
        ):
            conversation.active_flow = (
                nlu_result
                .context
                .active_flow
            )

        if hasattr(
            conversation,
            "last_activity_at",
        ):
            conversation.last_activity_at = (
                reference
            )

    # ========================================================
    # SECTION 12 - WIDGET STATE PERSISTENCE
    # ========================================================

    def _persist_widget_state(
        self,
        *,
        request: ChatRequest,
        browser_session: Any,
        conversation: Any,
        grounded_response: GroundedResponse,
        reference: datetime,
    ) -> None:
        self.widget_states.upsert(
            browser_session_id=(
                browser_session.id
            ),
            conversation_id=(
                conversation.id
            ),
            state=self._widget_state_value(
                grounded_response.widget_state
            ),
            size=self._widget_size_value(
                grounded_response.widget_size
            ),
            unread_count=0,
            draft_text=None,
            current_page_url=(
                request.page_context.url
            ),
            current_page_category=(
                request.page_context.category
            ),
            private_event_draft=(
                grounded_response
                .private_event_draft
                .model_dump(
                    mode="json"
                )
                if grounded_response
                .private_event_draft
                else {}
            ),
            updated_at=reference,
        )

    # ========================================================
    # SECTION 13 - PRIVATE EVENT DRAFT PERSISTENCE
    # ========================================================

    def _persist_private_event_draft(
        self,
        *,
        request: ChatRequest,
        browser_session: Any,
        conversation: Any,
        grounded_response: GroundedResponse,
        reference: datetime,
    ) -> None:
        draft = (
            grounded_response
            .private_event_draft
        )

        if draft is None:
            return

        self.private_events.upsert_draft(
            conversation_id=(
                conversation.id
            ),
            browser_session_id=(
                browser_session.id
            ),
            business_slug=(
                request.business_slug
            ),
            event_type=draft.event_type,
            preferred_date=draft.preferred_date,
            alternate_date=draft.alternate_date,
            start_time=draft.start_time,
            end_time=draft.end_time,
            guest_count=draft.guest_count,
            budget_min=draft.budget_min,
            budget_max=draft.budget_max,
            customer_name=draft.customer_name,
            email=draft.email,
            phone=draft.phone,
            company_name=draft.company_name,
            space_preference=(
                draft.space_preference
            ),
            food_package=draft.food_package,
            bar_package=draft.bar_package,
            dietary_requirements=(
                draft.dietary_requirements
            ),
            notes=draft.notes,
            completed_fields=(
                draft.completed_fields
            ),
            missing_fields=(
                draft.missing_fields
            ),
            updated_at=reference,
        )

    # ========================================================
    # SECTION 14 - ANALYTICS
    # ========================================================

    def _record_chat_analytics(
        self,
        *,
        request: ChatRequest,
        browser_session: Any,
        conversation: Any,
        user_message: Any,
        assistant_message: Any,
        nlu_result: NLUResult,
        knowledge_result: KnowledgeResult,
        grounded_response: GroundedResponse,
        request_id: str,
        reference: datetime,
    ) -> None:
        self.analytics.record(
            event_name="chat_message_completed",
            business_slug=request.business_slug,
            browser_session_id=(
                browser_session.id
            ),
            conversation_id=(
                conversation.id
            ),
            message_id=(
                assistant_message.id
            ),
            page_url=(
                request.page_context.url
            ),
            page_category=(
                request.page_context.category
            ),
            intent=(
                nlu_result.primary_intent.value
            ),
            event_value=1.0,
            occurred_at=reference,
            metadata_json={
                "request_id": request_id,
                "user_message_id": (
                    user_message.id
                ),
                "assistant_message_id": (
                    assistant_message.id
                ),
                "nlu_decision": (
                    nlu_result
                    .nlu_decision
                    .value
                ),
                "nlu_confidence": (
                    nlu_result
                    .confidence
                    .overall
                ),
                "knowledge_decision": (
                    knowledge_result
                    .decision
                    .value
                ),
                "verified_fact_count": (
                    knowledge_result
                    .verified_fact_count
                ),
                "response_decision": (
                    grounded_response
                    .decision
                    .value
                ),
                "response_source_count": (
                    len(
                        grounded_response
                        .sources
                    )
                ),
                "response_action_count": (
                    len(
                        grounded_response
                        .actions
                    )
                ),
                "destination_match_count": (
                    self._destination_match_count(
                        grounded_response
                    )
                ),
                "destination_keys": (
                    self._destination_keys(
                        grounded_response
                    )
                ),
                "destination_urls": (
                    self._destination_urls(
                        grounded_response
                    )
                ),
                "destination_routing_version": (
                    self._destination_routing_version(
                        grounded_response
                    )
                ),
            },
        )

    # ========================================================
    # SECTION 15 - CONVERSATION SUMMARY
    # ========================================================

    def _update_conversation_summary(
        self,
        *,
        conversation: Any,
        nlu_result: NLUResult,
        grounded_response: GroundedResponse,
        reference: datetime,
    ) -> None:
        if hasattr(
            conversation,
            "message_count",
        ):
            conversation.message_count = int(
                getattr(
                    conversation,
                    "message_count",
                    0,
                )
                or 0
            ) + 2

        if hasattr(
            conversation,
            "summary",
        ):
            conversation.summary = (
                nlu_result.context.summary
            )

        if hasattr(
            conversation,
            "last_response_preview",
        ):
            conversation.last_response_preview = (
                grounded_response.message[
                    :500
                ]
            )

        if hasattr(
            conversation,
            "last_activity_at",
        ):
            conversation.last_activity_at = (
                reference
            )

    # ========================================================
    # SECTION 16 - CHAT RESPONSE BUILDING
    # ========================================================

    def _build_chat_response(
        self,
        *,
        request_id: str,
        request: ChatRequest,
        browser_session: Any,
        conversation: Any,
        assistant_message: Any,
        nlu_result: NLUResult,
        grounded_response: GroundedResponse,
        processing_time_ms: float,
        reference: datetime,
    ) -> ChatResponse:
        return ChatResponse(
            request_id=request_id,
            session_id=browser_session.id,
            conversation_id=conversation.id,
            message_id=assistant_message.id,
            message=grounded_response.message,
            detected_intent=(
                nlu_result.primary_intent.value
            ),
            normalized_message=(
                nlu_result.normalized_text
            ),
            corrected_message=(
                nlu_result.corrected_text
            ),
            language=(
                request.requested_language
                or request.page_context.language
            ),
            confidence=(
                grounded_response.confidence
            ),
            spelling_corrections=list(
                grounded_response
                .spelling_corrections
            ),
            entities=list(
                grounded_response.entities
            ),
            sources=list(
                grounded_response.sources
            ),
            actions=list(
                grounded_response.actions
            ),
            validation=(
                grounded_response.validation
            ),
            widget_state=(
                grounded_response.widget_state
            ),
            widget_size=(
                grounded_response.widget_size
            ),
            private_event_draft=(
                grounded_response
                .private_event_draft
            ),
            human_handoff_available=(
                grounded_response
                .human_handoff_available
            ),
            human_handoff_required=(
                grounded_response
                .human_handoff_required
            ),
            response_template_id=(
                grounded_response
                .response_template_id
            ),
            response_variant_id=(
                grounded_response
                .response_variant_id
            ),
            model_versions={
                "chat_service": (
                    CHAT_SERVICE_VERSION
                ),
                "nlu_engine": (
                    nlu_result.engine_version
                ),
                "knowledge_service": (
                    grounded_response
                    .processing_metadata
                    .get(
                        "knowledge_service_version",
                        "",
                    )
                ),
                "response_service": (
                    grounded_response
                    .service_version
                ),
                "destination_routing": (
                    self._destination_routing_version(
                        grounded_response
                    )
                ),
            },
            processing_time_ms=int(
                round(
                    processing_time_ms
                )
            ),
            created_at=reference,
            metadata={
                "chat_service_phase": (
                    CHAT_SERVICE_PHASE
                ),
                "grounded_response_decision": (
                    grounded_response
                    .decision
                    .value
                ),
                "nlu_decision": (
                    nlu_result
                    .nlu_decision
                    .value
                ),
                "active_flow": (
                    nlu_result.active_flow
                ),
                "pending_fields": list(
                    nlu_result.pending_fields
                ),
                "completed_fields": list(
                    nlu_result.completed_fields
                ),
                "destination_routing": (
                    self._destination_routing_metadata(
                        grounded_response
                    )
                ),
                "restaurant_schema": (
                    self._restaurant_schema_metadata(
                        grounded_response
                    )
                ),
                "official_destination_urls": (
                    self._destination_urls(
                        grounded_response
                    )
                ),
                "destination_match_count": (
                    self._destination_match_count(
                        grounded_response
                    )
                ),
            },
        )

    # ========================================================
    # SECTION 16A - DESTINATION ROUTING METADATA
    # ========================================================

    @staticmethod
    def _destination_routing_metadata(
        grounded_response: GroundedResponse,
    ) -> dict[str, Any]:
        """
        Return a defensive copy of destination-routing metadata.

        The response service owns destination matching. The chat service
        propagates the resulting verified navigation payload without
        recomputing or mutating it.
        """

        metadata = (
            grounded_response.processing_metadata
            if isinstance(
                grounded_response.processing_metadata,
                Mapping,
            )
            else {}
        )

        routing = metadata.get(
            "destination_routing",
            {},
        )

        if not isinstance(
            routing,
            Mapping,
        ):
            return {}

        return copy.deepcopy(
            dict(routing)
        )

    @staticmethod
    def _restaurant_schema_metadata(
        grounded_response: GroundedResponse,
    ) -> dict[str, Any]:
        """
        Return Schema.org Restaurant metadata when available.
        """

        metadata = (
            grounded_response.processing_metadata
            if isinstance(
                grounded_response.processing_metadata,
                Mapping,
            )
            else {}
        )

        schema = metadata.get(
            "restaurant_schema",
            {},
        )

        if not isinstance(
            schema,
            Mapping,
        ):
            return {}

        return copy.deepcopy(
            dict(schema)
        )

    @classmethod
    def _destination_matches(
        cls,
        grounded_response: GroundedResponse,
    ) -> list[dict[str, Any]]:
        routing = cls._destination_routing_metadata(
            grounded_response
        )

        matches = routing.get(
            "matches",
            [],
        )

        if not isinstance(
            matches,
            Sequence,
        ) or isinstance(
            matches,
            (str, bytes, bytearray),
        ):
            return []

        return [
            copy.deepcopy(
                dict(match)
            )
            for match in matches
            if isinstance(
                match,
                Mapping,
            )
        ]

    @classmethod
    def _destination_match_count(
        cls,
        grounded_response: GroundedResponse,
    ) -> int:
        routing = cls._destination_routing_metadata(
            grounded_response
        )

        raw_count = routing.get(
            "match_count",
        )

        if isinstance(
            raw_count,
            int,
        ):
            return max(
                raw_count,
                0,
            )

        return len(
            cls._destination_matches(
                grounded_response
            )
        )

    @classmethod
    def _destination_keys(
        cls,
        grounded_response: GroundedResponse,
    ) -> list[str]:
        keys: list[str] = []

        for match in cls._destination_matches(
            grounded_response
        ):
            destination = match.get(
                "destination",
                {},
            )

            if not isinstance(
                destination,
                Mapping,
            ):
                continue

            key = destination.get(
                "key"
            )

            if isinstance(
                key,
                str,
            ) and key.strip():
                keys.append(
                    key.strip()
                )

        return list(
            dict.fromkeys(
                keys
            )
        )

    @classmethod
    def _destination_urls(
        cls,
        grounded_response: GroundedResponse,
    ) -> list[str]:
        routing = cls._destination_routing_metadata(
            grounded_response
        )

        raw_urls = routing.get(
            "official_destinations",
            [],
        )

        urls: list[str] = []

        if isinstance(
            raw_urls,
            Sequence,
        ) and not isinstance(
            raw_urls,
            (str, bytes, bytearray),
        ):
            for value in raw_urls:
                if (
                    isinstance(
                        value,
                        str,
                    )
                    and value.startswith(
                        "https://"
                    )
                ):
                    urls.append(
                        value
                    )

        if not urls:
            for match in cls._destination_matches(
                grounded_response
            ):
                destination = match.get(
                    "destination",
                    {},
                )

                if not isinstance(
                    destination,
                    Mapping,
                ):
                    continue

                value = destination.get(
                    "url"
                )

                if (
                    isinstance(
                        value,
                        str,
                    )
                    and value.startswith(
                        "https://"
                    )
                ):
                    urls.append(
                        value
                    )

        return list(
            dict.fromkeys(
                urls
            )
        )

    @classmethod
    def _destination_routing_version(
        cls,
        grounded_response: GroundedResponse,
    ) -> str:
        routing = cls._destination_routing_metadata(
            grounded_response
        )

        value = routing.get(
            "service_version",
            "",
        )

        return (
            value.strip()
            if isinstance(
                value,
                str,
            )
            else ""
        )

    # ========================================================
    # SECTION 17 - RESTORE MESSAGE CONVERSION
    # ========================================================

    def _message_to_schema(
        self,
        message: Any,
    ) -> ChatMessage:
        role = str(
            getattr(
                message,
                "role",
                MessageRole.ASSISTANT.value,
            )
        )

        status = str(
            getattr(
                message,
                "status",
                MessageStatus.COMPLETED.value,
            )
        )

        return ChatMessage(
            id=message.id,
            conversation_id=(
                getattr(
                    message,
                    "conversation_id",
                    None,
                )
            ),
            sequence_number=(
                getattr(
                    message,
                    "sequence_number",
                    None,
                )
            ),
            role=role,
            status=status,
            text=str(
                getattr(
                    message,
                    "text",
                    "",
                )
            ),
            original_text=(
                getattr(
                    message,
                    "original_text",
                    None,
                )
            ),
            normalized_text=(
                getattr(
                    message,
                    "normalized_text",
                    None,
                )
            ),
            corrected_text=(
                getattr(
                    message,
                    "corrected_text",
                    None,
                )
            ),
            created_at=(
                getattr(
                    message,
                    "created_at",
                    datetime.now().astimezone(),
                )
            ),
            page_url=(
                getattr(
                    message,
                    "page_url",
                    None,
                )
            ),
            page_category=(
                getattr(
                    message,
                    "page_category",
                    None,
                )
            ),
            detected_intent=(
                getattr(
                    message,
                    "detected_intent",
                    None,
                )
            ),
            sources=(
                getattr(
                    message,
                    "sources_json",
                    [],
                )
                or []
            ),
            actions=(
                getattr(
                    message,
                    "actions_json",
                    [],
                )
                or []
            ),
            metadata=(
                getattr(
                    message,
                    "metadata_json",
                    {},
                )
                or {}
            ),
        )

    # ========================================================
    # SECTION 18 - CONTEXT LOADING
    # ========================================================

    @staticmethod
    def _load_conversation_context(
        conversation: Any,
    ) -> ConversationContext | None:
        payload = getattr(
            conversation,
            "context_json",
            None,
        )

        if not payload:
            return None

        if isinstance(
            payload,
            ConversationContext,
        ):
            return payload

        if isinstance(
            payload,
            Mapping,
        ):
            return ConversationContext.from_dict(
                payload
            )

        return None

    # ========================================================
    # SECTION 19 - PRIVATE EVENT RESTORATION
    # ========================================================

    def _restore_private_event_draft(
        self,
        conversation_id: str,
    ) -> PrivateEventDraft | None:
        record = (
            self.private_events
            .get_draft_for_conversation(
                conversation_id
            )
        )

        if record is None:
            return None

        return PrivateEventDraft(
            event_type=getattr(
                record,
                "event_type",
                None,
            ),
            preferred_date=getattr(
                record,
                "preferred_date",
                None,
            ),
            alternate_date=getattr(
                record,
                "alternate_date",
                None,
            ),
            start_time=getattr(
                record,
                "start_time",
                None,
            ),
            end_time=getattr(
                record,
                "end_time",
                None,
            ),
            guest_count=getattr(
                record,
                "guest_count",
                None,
            ),
            budget_min=getattr(
                record,
                "budget_min",
                None,
            ),
            budget_max=getattr(
                record,
                "budget_max",
                None,
            ),
            customer_name=getattr(
                record,
                "customer_name",
                None,
            ),
            email=getattr(
                record,
                "email",
                None,
            ),
            phone=getattr(
                record,
                "phone",
                None,
            ),
            company_name=getattr(
                record,
                "company_name",
                None,
            ),
            space_preference=getattr(
                record,
                "space_preference",
                None,
            ),
            food_package=getattr(
                record,
                "food_package",
                None,
            ),
            bar_package=getattr(
                record,
                "bar_package",
                None,
            ),
            dietary_requirements=getattr(
                record,
                "dietary_requirements",
                None,
            ),
            notes=getattr(
                record,
                "notes",
                None,
            ),
            completed_fields=list(
                getattr(
                    record,
                    "completed_fields",
                    [],
                )
                or []
            ),
            missing_fields=list(
                getattr(
                    record,
                    "missing_fields",
                    [],
                )
                or []
            ),
        )

    # ========================================================
    # SECTION 20 - WIDGET ENUM HELPERS
    # ========================================================

    @staticmethod
    def _widget_state_value(
        value: WidgetState,
    ) -> str:
        return str(value)

    @staticmethod
    def _widget_size_value(
        value: WidgetSize,
    ) -> str:
        return str(value)

    @staticmethod
    def _coerce_widget_state(
        value: Any,
    ) -> WidgetState:
        candidate = str(
            value
            or WidgetState.COLLAPSED.value
        )

        try:
            return WidgetState(candidate)
        except ValueError:
            return WidgetState.COLLAPSED

    @staticmethod
    def _coerce_widget_size(
        value: Any,
    ) -> WidgetSize:
        candidate = str(
            value
            or WidgetSize.COMPACT.value
        )

        try:
            return WidgetSize(candidate)
        except ValueError:
            return WidgetSize.COMPACT

    # ========================================================
    # SECTION 21 - IDENTIFIER HELPERS
    # ========================================================

    @staticmethod
    def _new_id(
        prefix: str,
    ) -> str:
        return (
            f"{prefix}_"
            f"{uuid.uuid4().hex}"
        )


# ============================================================
# SECTION 22 - MODULE-LEVEL HELPERS
# ============================================================

def process_chat_request(
    session: Session,
    request: ChatRequest,
    *,
    business_slug: str = DEFAULT_BUSINESS_SLUG,
    now: datetime | None = None,
) -> ChatProcessingResult:
    """
    Process one complete chat request.
    """

    return ChatService(
        session,
        business_slug=business_slug,
    ).process(
        request,
        now=now,
    )


def restore_chat_conversation(
    session: Session,
    *,
    session_id: str,
    conversation_id: str | None = None,
    page_context: PageContext | None = None,
    limit: int = DEFAULT_RESTORE_LIMIT,
    now: datetime | None = None,
) -> RestoreResult:
    """
    Restore one persisted chat conversation.
    """

    return ChatService(
        session
    ).restore(
        session_id=session_id,
        conversation_id=(
            conversation_id
        ),
        page_context=page_context,
        limit=limit,
        now=now,
    )


# ============================================================
# SECTION 23 - SELF-TEST
# ============================================================

def validate_chat_service_module() -> dict[str, Any]:
    from datetime import time as business_time

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.database.base import Base
    from app.database.models import (
        Business,
        BusinessHour,
    )

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
        future=True,
    )

    Base.metadata.create_all(engine)

    reference = datetime(
        2026,
        7,
        22,
        12,
        0,
    ).astimezone()

    checks: dict[str, bool] = {}

    with Session(engine) as session:
        business = Business(
            slug="horseshoe-tavern",
            display_name="Horseshoe Tavern",
            website_url=(
                "https://www.thehorseshoetavern.com/"
            ),
            source_type="official_website",
            source_name="Official website",
            source_url=(
                "https://www.thehorseshoetavern.com/"
            ),
        )

        business.mark_verified(
            verified_by="chat-service-test",
            notes="Verified test business",
        )

        hours = BusinessHour(
            business=business,
            service_type="kitchen",
            day_of_week=reference.weekday(),
            open_time=business_time(11, 0),
            close_time=business_time(22, 0),
            is_closed=False,
            source_type="official_hours",
            source_name="Official kitchen hours",
            source_url=(
                "https://www.thehorseshoetavern.com/hours"
            ),
        )

        hours.mark_verified(
            verified_by="chat-service-test",
            notes="Verified test hours",
        )

        session.add_all(
            [
                business,
                hours,
            ]
        )

        session.commit()

        service = ChatService(
            session
        )

        request = ChatRequest(
            session_id="session_chatservice123",
            conversation_id=None,
            message=(
                "wat time does the kichen close today"
            ),
            business_slug="horseshoe-tavern",
            page_context=PageContext(
                url=(
                    "https://www.thehorseshoetavern.com/"
                ),
                path="/",
                title="Horseshoe Tavern",
                category="home",
            ),
            widget_context=WidgetContext(
                state=WidgetState.OPEN,
                size=WidgetSize.COMPACT,
            ),
        )

        result = service.process(
            request,
            now=reference,
        )

        restore = service.restore(
            session_id=result.response.session_id,
            conversation_id=(
                result.response.conversation_id
            ),
            now=reference,
        )

        checks = {
            "chat_completed": (
                result.decision
                == ChatServiceDecision.COMPLETED
            ),
            "response_persisted": (
                result.persisted is True
            ),
            "intent_detected": (
                result.response.detected_intent
                == "HOURS_KITCHEN"
            ),
            "response_message_present": (
                bool(result.response.message)
            ),
            "source_present": (
                bool(result.response.sources)
            ),
            "session_id_present": (
                result.response.session_id
                == "session_chatservice123"
            ),
            "conversation_id_present": (
                bool(
                    result.response.conversation_id
                )
            ),
            "assistant_message_id_present": (
                bool(
                    result.response.message_id
                )
            ),
            "validation_passed": (
                result.response.validation.passed
            ),
            "restore_succeeded": (
                restore.response.restored
                is True
            ),
            "restore_has_messages": (
                len(
                    restore.response.messages
                )
                >= 2
            ),
            "restore_same_conversation": (
                restore.response.conversation_id
                == result.response.conversation_id
            ),
            "json_safe": bool(
                result.as_dict()
            ),
        }

    engine.dispose()

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
    }


# ============================================================
# SECTION 24 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    report = validate_chat_service_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
