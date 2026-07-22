# ============================================================
# Exact file location: app/database/repositories.py
# Horseshoe Tavern AI
# Phase 1 Part 1.10
# Repository layer for business knowledge, browser sessions,
# conversations, messages, widget state, learning, feedback,
# analytics, and private-event leads
# ============================================================

"""
Repository layer for Horseshoe Tavern AI.

This module centralizes durable database access for:

- Business records
- Verified business facts
- Browser sessions
- Cross-page widget state
- Conversations
- Messages
- Spelling observations
- Training candidates
- User feedback
- Analytics events
- Private-event inquiries
- Knowledge gaps
- Model versions

Routes and services should use repositories instead of writing direct
SQLAlchemy queries throughout the application.

Repository methods do not blindly commit by default. The caller controls
transaction boundaries through managed_database_session(), safe_commit(),
or an explicit Session commit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Generic, TypeVar

from sqlalchemy import Select, and_, desc, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from app.database.base import Base, utc_now
from app.database.models import (
    AnalyticsEvent,
    BrowserSession,
    Business,
    BusinessEvent,
    BusinessHour,
    Conversation,
    ConversationMessage,
    ConversationStatus,
    FailedQuery,
    FAQEntry,
    FeedbackType,
    KnowledgeDocument,
    KnowledgeGap,
    KnowledgeStatus,
    LeadStatus,
    MenuCategory,
    MenuItem,
    MessageRole,
    MessageStatus,
    ModelLifecycleStatus,
    ModelVersion,
    PrivateEventInquiry,
    PrivateEventPackage,
    SpellingVariant,
    TrainingExample,
    TrainingReviewStatus,
    UserFeedback,
    WidgetSizeValue,
    WidgetState,
    WidgetStateValue,
)
from app.logging_config import get_logger


# ============================================================
# SECTION 01 - TYPES AND LOGGER
# ============================================================

ModelT = TypeVar("ModelT", bound=Base)

logger = get_logger(__name__)


# ============================================================
# SECTION 02 - GENERAL HELPERS
# ============================================================

def _clean_text(
    value: str | None,
    *,
    maximum_length: int | None = None,
) -> str | None:
    if value is None:
        return None

    cleaned = (
        str(value)
        .replace("\x00", "")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )

    if not cleaned:
        return None

    if maximum_length is not None:
        cleaned = cleaned[:maximum_length]

    return cleaned


def _normalize_phrase(value: str) -> str:
    return " ".join(
        _clean_text(value).lower().split()
    )


def _stable_hash(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _json_safe_dict(
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    return dict(value or {})


def _json_safe_list(
    value: Sequence[Any] | None,
) -> list[Any]:
    return list(value or [])


# ============================================================
# SECTION 03 - GENERIC BASE REPOSITORY
# ============================================================

class BaseRepository(Generic[ModelT]):
    """
    Shared CRUD behavior for one SQLAlchemy model.
    """

    model: type[ModelT]

    def __init__(
        self,
        session: Session,
    ) -> None:
        self.session = session

    def add(
        self,
        instance: ModelT,
        *,
        flush: bool = True,
    ) -> ModelT:
        self.session.add(instance)

        if flush:
            self.session.flush()

        return instance

    def add_all(
        self,
        instances: Iterable[ModelT],
        *,
        flush: bool = True,
    ) -> list[ModelT]:
        values = list(instances)

        self.session.add_all(values)

        if flush:
            self.session.flush()

        return values

    def get(
        self,
        record_id: str,
    ) -> ModelT | None:
        return self.session.get(
            self.model,
            record_id,
        )

    def require(
        self,
        record_id: str,
    ) -> ModelT:
        instance = self.get(record_id)

        if instance is None:
            raise LookupError(
                f"{self.model.__name__} was not found: {record_id}"
            )

        return instance

    def delete(
        self,
        instance: ModelT,
        *,
        flush: bool = True,
    ) -> None:
        self.session.delete(instance)

        if flush:
            self.session.flush()

    def count(self) -> int:
        return int(
            self.session.scalar(
                select(func.count())
                .select_from(self.model)
            )
            or 0
        )

    def list_recent(
        self,
        *,
        limit: int = 50,
    ) -> list[ModelT]:
        limit = min(max(limit, 1), 500)

        statement = (
            select(self.model)
            .order_by(desc(self.model.created_at))
            .limit(limit)
        )

        return list(
            self.session.scalars(statement).all()
        )


# ============================================================
# SECTION 04 - BUSINESS REPOSITORY
# ============================================================

class BusinessRepository(
    BaseRepository[Business]
):
    model = Business

    def get_by_slug(
        self,
        slug: str,
    ) -> Business | None:
        normalized = _normalize_phrase(
            slug
        ).replace(" ", "-")

        return self.session.scalar(
            select(Business)
            .where(Business.slug == normalized)
        )

    def get_or_create(
        self,
        *,
        slug: str,
        display_name: str,
        website_url: str,
        timezone: str = "America/New_York",
        flush: bool = True,
    ) -> tuple[Business, bool]:
        existing = self.get_by_slug(slug)

        if existing is not None:
            return existing, False

        business = Business(
            slug=slug.strip().lower(),
            display_name=display_name.strip(),
            website_url=website_url.strip(),
            timezone=timezone.strip(),
            source_type="official_website",
            source_name=(
                f"{display_name.strip()} official website"
            ),
            source_url=website_url.strip(),
        )

        self.add(
            business,
            flush=flush,
        )

        return business, True

    def get_with_verified_knowledge(
        self,
        slug: str,
    ) -> Business | None:
        statement = (
            select(Business)
            .options(
                selectinload(
                    Business.hours
                ),
                selectinload(
                    Business.faq_entries
                ),
                selectinload(
                    Business.events
                ),
                selectinload(
                    Business.menu_categories
                ).selectinload(
                    MenuCategory.items
                ),
                selectinload(
                    Business.private_event_packages
                ),
                selectinload(
                    Business.knowledge_documents
                ),
            )
            .where(
                Business.slug
                == slug.strip().lower()
            )
        )

        return self.session.scalar(statement)


# ============================================================
# SECTION 05 - KNOWLEDGE REPOSITORY
# ============================================================

class KnowledgeRepository:
    def __init__(
        self,
        session: Session,
    ) -> None:
        self.session = session

    def list_verified_hours(
        self,
        business_id: str,
        *,
        service_type: str | None = None,
    ) -> list[BusinessHour]:
        conditions = [
            BusinessHour.business_id
            == business_id,
            BusinessHour.is_active.is_(True),
            BusinessHour.is_deleted.is_(False),
            BusinessHour.is_verified.is_(True),
        ]

        if service_type:
            conditions.append(
                BusinessHour.service_type
                == service_type.strip().lower()
            )

        statement = (
            select(BusinessHour)
            .where(and_(*conditions))
            .order_by(
                BusinessHour.service_type,
                BusinessHour.day_of_week,
            )
        )

        return list(
            self.session.scalars(statement).all()
        )

    def list_upcoming_events(
        self,
        business_id: str,
        *,
        after: datetime | None = None,
        limit: int = 25,
    ) -> list[BusinessEvent]:
        after = after or utc_now()
        limit = min(max(limit, 1), 100)

        statement = (
            select(BusinessEvent)
            .where(
                BusinessEvent.business_id
                == business_id,
                BusinessEvent.is_active.is_(True),
                BusinessEvent.is_deleted.is_(False),
                BusinessEvent.is_verified.is_(True),
                BusinessEvent.start_at >= after,
            )
            .order_by(
                BusinessEvent.start_at.asc()
            )
            .limit(limit)
        )

        return list(
            self.session.scalars(statement).all()
        )

    def list_verified_faqs(
        self,
        business_id: str,
        *,
        category: str | None = None,
    ) -> list[FAQEntry]:
        conditions = [
            FAQEntry.business_id
            == business_id,
            FAQEntry.is_active.is_(True),
            FAQEntry.is_deleted.is_(False),
            FAQEntry.is_verified.is_(True),
        ]

        if category:
            conditions.append(
                FAQEntry.category
                == category.strip().lower()
            )

        statement = (
            select(FAQEntry)
            .where(and_(*conditions))
            .order_by(
                FAQEntry.category,
                FAQEntry.display_order,
                FAQEntry.created_at,
            )
        )

        return list(
            self.session.scalars(statement).all()
        )

    def list_verified_menu_items(
        self,
        business_id: str,
        *,
        category_slug: str | None = None,
        featured_only: bool = False,
    ) -> list[MenuItem]:
        conditions = [
            MenuCategory.business_id
            == business_id,
            MenuCategory.is_active.is_(True),
            MenuCategory.is_deleted.is_(False),
            MenuItem.is_active.is_(True),
            MenuItem.is_deleted.is_(False),
            MenuItem.is_verified.is_(True),
        ]

        if category_slug:
            conditions.append(
                MenuCategory.slug
                == category_slug.strip().lower()
            )

        if featured_only:
            conditions.append(
                MenuItem.is_featured.is_(True)
            )

        statement = (
            select(MenuItem)
            .join(MenuItem.category)
            .options(
                selectinload(
                    MenuItem.category
                )
            )
            .where(and_(*conditions))
            .order_by(
                MenuCategory.display_order,
                MenuItem.display_order,
                MenuItem.name,
            )
        )

        return list(
            self.session.scalars(statement).all()
        )

    def search_documents(
        self,
        business_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[KnowledgeDocument]:
        normalized = _clean_text(query)

        if not normalized:
            return []

        limit = min(max(limit, 1), 100)
        pattern = f"%{normalized.lower()}%"

        statement = (
            select(KnowledgeDocument)
            .where(
                KnowledgeDocument.business_id
                == business_id,
                KnowledgeDocument.is_active.is_(True),
                KnowledgeDocument.is_deleted.is_(False),
                KnowledgeDocument.is_verified.is_(True),
                KnowledgeDocument.knowledge_status
                == KnowledgeStatus.VERIFIED.value,
                or_(
                    func.lower(
                        KnowledgeDocument.title
                    ).like(pattern),
                    func.lower(
                        KnowledgeDocument.content
                    ).like(pattern),
                    func.lower(
                        KnowledgeDocument.search_text
                    ).like(pattern),
                ),
            )
            .order_by(
                KnowledgeDocument.priority.desc(),
                KnowledgeDocument.updated_at.desc(),
            )
            .limit(limit)
        )

        return list(
            self.session.scalars(statement).all()
        )

    def list_private_event_packages(
        self,
        business_id: str,
    ) -> list[PrivateEventPackage]:
        statement = (
            select(PrivateEventPackage)
            .where(
                PrivateEventPackage.business_id
                == business_id,
                PrivateEventPackage.is_active.is_(True),
                PrivateEventPackage.is_deleted.is_(False),
                PrivateEventPackage.is_verified.is_(True),
            )
            .order_by(
                PrivateEventPackage.package_type,
                PrivateEventPackage.name,
            )
        )

        return list(
            self.session.scalars(statement).all()
        )


# ============================================================
# SECTION 06 - BROWSER SESSION REPOSITORY
# ============================================================

class BrowserSessionRepository(
    BaseRepository[BrowserSession]
):
    model = BrowserSession

    def get_by_external_id(
        self,
        external_session_id: str,
    ) -> BrowserSession | None:
        return self.session.scalar(
            select(BrowserSession)
            .options(
                selectinload(
                    BrowserSession.widget_state
                )
            )
            .where(
                BrowserSession.external_session_id
                == external_session_id
            )
        )

    def get_or_create(
        self,
        *,
        business_id: str,
        external_session_id: str,
        page_url: str | None = None,
        referrer: str | None = None,
        user_agent: str | None = None,
        ip_hash: str | None = None,
        attribution: dict[str, str | None] | None = None,
    ) -> tuple[BrowserSession, bool]:
        existing = self.get_by_external_id(
            external_session_id
        )

        now = utc_now()

        if existing is not None:
            existing.last_seen_at = now
            existing.last_page_url = (
                _clean_text(page_url, maximum_length=2048)
                or existing.last_page_url
            )

            if user_agent:
                existing.user_agent = _clean_text(
                    user_agent
                )

            if referrer and not existing.referrer:
                existing.referrer = _clean_text(
                    referrer,
                    maximum_length=2048,
                )

            self.session.flush()

            return existing, False

        attribution = attribution or {}

        browser_session = BrowserSession(
            business_id=business_id,
            external_session_id=external_session_id,
            first_seen_at=now,
            last_seen_at=now,
            first_page_url=_clean_text(
                page_url,
                maximum_length=2048,
            ),
            last_page_url=_clean_text(
                page_url,
                maximum_length=2048,
            ),
            referrer=_clean_text(
                referrer,
                maximum_length=2048,
            ),
            user_agent=_clean_text(
                user_agent
            ),
            ip_hash=_clean_text(
                ip_hash,
                maximum_length=128,
            ),
            utm_source=_clean_text(
                attribution.get("utm_source"),
                maximum_length=255,
            ),
            utm_medium=_clean_text(
                attribution.get("utm_medium"),
                maximum_length=255,
            ),
            utm_campaign=_clean_text(
                attribution.get("utm_campaign"),
                maximum_length=255,
            ),
        )

        self.add(browser_session)

        return browser_session, True


# ============================================================
# SECTION 07 - CONVERSATION REPOSITORY
# ============================================================

class ConversationRepository(
    BaseRepository[Conversation]
):
    model = Conversation

    def get_by_external_id(
        self,
        external_conversation_id: str,
        *,
        include_messages: bool = False,
    ) -> Conversation | None:
        statement = select(
            Conversation
        ).where(
            Conversation.external_conversation_id
            == external_conversation_id
        )

        if include_messages:
            statement = statement.options(
                selectinload(
                    Conversation.messages
                )
            )

        return self.session.scalar(statement)

    def create(
        self,
        *,
        business_id: str,
        external_conversation_id: str,
        browser_session_id: str | None = None,
        channel: str = "website_widget",
        initial_context: dict[str, Any] | None = None,
    ) -> Conversation:
        conversation = Conversation(
            business_id=business_id,
            browser_session_id=browser_session_id,
            external_conversation_id=(
                external_conversation_id
            ),
            status=ConversationStatus.ACTIVE.value,
            channel=channel,
            started_at=utc_now(),
            last_activity_at=utc_now(),
            current_context_json=_json_safe_dict(
                initial_context
            ),
        )

        return self.add(conversation)

    def get_or_create(
        self,
        *,
        business_id: str,
        external_conversation_id: str,
        browser_session_id: str | None = None,
        channel: str = "website_widget",
        initial_context: dict[str, Any] | None = None,
    ) -> tuple[Conversation, bool]:
        existing = self.get_by_external_id(
            external_conversation_id
        )

        if existing is not None:
            existing.last_activity_at = utc_now()

            if browser_session_id:
                existing.browser_session_id = (
                    browser_session_id
                )

            self.session.flush()

            return existing, False

        conversation = self.create(
            business_id=business_id,
            external_conversation_id=(
                external_conversation_id
            ),
            browser_session_id=browser_session_id,
            channel=channel,
            initial_context=initial_context,
        )

        return conversation, True

    def update_context(
        self,
        conversation: Conversation,
        *,
        current_intent: str | None = None,
        context_updates: dict[str, Any] | None = None,
    ) -> Conversation:
        context = dict(
            conversation.current_context_json
            or {}
        )

        if context_updates:
            context.update(context_updates)

        conversation.current_context_json = context
        conversation.last_activity_at = utc_now()

        if current_intent:
            conversation.current_intent = (
                current_intent
            )

        self.session.flush()

        return conversation

    def complete(
        self,
        conversation: Conversation,
        *,
        summary: str | None = None,
    ) -> Conversation:
        conversation.status = (
            ConversationStatus.COMPLETED.value
        )
        conversation.completed_at = utc_now()
        conversation.last_activity_at = utc_now()
        conversation.summary = _clean_text(summary)

        self.session.flush()

        return conversation

    def record_conversion(
        self,
        conversation: Conversation,
        *,
        conversion_type: str,
        value: Decimal | None = None,
    ) -> Conversation:
        conversation.converted = True
        conversation.conversion_type = (
            conversion_type.strip().lower()
        )
        conversation.conversion_value = value
        conversation.last_activity_at = utc_now()

        self.session.flush()

        return conversation


# ============================================================
# SECTION 08 - MESSAGE REPOSITORY
# ============================================================

class MessageRepository(
    BaseRepository[ConversationMessage]
):
    model = ConversationMessage

    def next_sequence_number(
        self,
        conversation_id: str,
    ) -> int:
        current = self.session.scalar(
            select(
                func.max(
                    ConversationMessage.sequence_number
                )
            )
            .where(
                ConversationMessage.conversation_id
                == conversation_id
            )
        )

        return int(current or 0) + 1

    def create_message(
        self,
        *,
        conversation: Conversation,
        role: str,
        original_text: str,
        normalized_text: str | None = None,
        corrected_text: str | None = None,
        status: str = MessageStatus.COMPLETED.value,
        detected_language: str | None = None,
        detected_intent: str | None = None,
        intent_confidence: Decimal | None = None,
        answer_confidence: Decimal | None = None,
        entities: Sequence[dict[str, Any]] | None = None,
        spelling_corrections: Sequence[dict[str, Any]] | None = None,
        retrieval_sources: Sequence[dict[str, Any]] | None = None,
        response_actions: Sequence[dict[str, Any]] | None = None,
        validation_result: dict[str, Any] | None = None,
        model_versions: dict[str, Any] | None = None,
        response_template_id: str | None = None,
        response_variant_id: str | None = None,
        latency_ms: int | None = None,
        token_count: int | None = None,
        page_url: str | None = None,
        page_category: str | None = None,
    ) -> ConversationMessage:
        sequence_number = (
            self.next_sequence_number(
                conversation.id
            )
        )

        message = ConversationMessage(
            conversation_id=conversation.id,
            sequence_number=sequence_number,
            role=role,
            status=status,
            original_text=original_text,
            normalized_text=normalized_text,
            corrected_text=corrected_text,
            detected_language=detected_language,
            detected_intent=detected_intent,
            intent_confidence=intent_confidence,
            answer_confidence=answer_confidence,
            detected_entities_json=_json_safe_list(
                entities
            ),
            spelling_corrections_json=_json_safe_list(
                spelling_corrections
            ),
            retrieval_sources_json=_json_safe_list(
                retrieval_sources
            ),
            response_actions_json=_json_safe_list(
                response_actions
            ),
            validation_result_json=_json_safe_dict(
                validation_result
            ),
            model_versions_json=_json_safe_dict(
                model_versions
            ),
            response_template_id=(
                response_template_id
            ),
            response_variant_id=(
                response_variant_id
            ),
            latency_ms=latency_ms,
            token_count=token_count,
            page_url=_clean_text(
                page_url,
                maximum_length=2048,
            ),
            page_category=_clean_text(
                page_category,
                maximum_length=100,
            ),
        )

        self.add(message)

        conversation.message_count = (
            int(conversation.message_count or 0)
            + 1
        )
        conversation.last_activity_at = utc_now()

        if detected_intent:
            conversation.current_intent = (
                detected_intent
            )

        self.session.flush()

        return message

    def create_user_message(
        self,
        *,
        conversation: Conversation,
        text: str,
        normalized_text: str | None = None,
        corrected_text: str | None = None,
        page_url: str | None = None,
        page_category: str | None = None,
        **metadata: Any,
    ) -> ConversationMessage:
        return self.create_message(
            conversation=conversation,
            role=MessageRole.USER.value,
            original_text=text,
            normalized_text=normalized_text,
            corrected_text=corrected_text,
            page_url=page_url,
            page_category=page_category,
            **metadata,
        )

    def create_assistant_message(
        self,
        *,
        conversation: Conversation,
        text: str,
        page_url: str | None = None,
        page_category: str | None = None,
        **metadata: Any,
    ) -> ConversationMessage:
        return self.create_message(
            conversation=conversation,
            role=MessageRole.ASSISTANT.value,
            original_text=text,
            page_url=page_url,
            page_category=page_category,
            **metadata,
        )

    def list_for_conversation(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
    ) -> list[ConversationMessage]:
        limit = min(max(limit, 1), 500)

        statement = (
            select(ConversationMessage)
            .where(
                ConversationMessage.conversation_id
                == conversation_id
            )
            .order_by(
                ConversationMessage.sequence_number
            )
            .limit(limit)
        )

        return list(
            self.session.scalars(statement).all()
        )


# ============================================================
# SECTION 09 - WIDGET STATE REPOSITORY
# ============================================================

class WidgetStateRepository(
    BaseRepository[WidgetState]
):
    model = WidgetState

    def get_by_browser_session(
        self,
        browser_session_id: str,
    ) -> WidgetState | None:
        return self.session.scalar(
            select(WidgetState)
            .where(
                WidgetState.browser_session_id
                == browser_session_id
            )
        )

    def get_or_create(
        self,
        *,
        browser_session_id: str,
        conversation_id: str | None = None,
        page_url: str | None = None,
        page_category: str | None = None,
    ) -> tuple[WidgetState, bool]:
        existing = self.get_by_browser_session(
            browser_session_id
        )

        if existing is not None:
            if conversation_id:
                existing.conversation_id = (
                    conversation_id
                )

            if page_url:
                existing.current_page_url = (
                    page_url
                )

            if page_category:
                existing.current_page_category = (
                    page_category
                )

            self.session.flush()

            return existing, False

        widget_state = WidgetState(
            browser_session_id=(
                browser_session_id
            ),
            conversation_id=conversation_id,
            widget_state=(
                WidgetStateValue.COLLAPSED.value
            ),
            widget_size=(
                WidgetSizeValue.COMPACT.value
            ),
            unread_count=0,
            current_page_url=page_url,
            current_page_category=page_category,
        )

        self.add(widget_state)

        return widget_state, True

    def update_state(
        self,
        widget_state: WidgetState,
        *,
        state_value: str,
        size_value: str,
        unread_count: int = 0,
        conversation_id: str | None = None,
        page_url: str | None = None,
        page_category: str | None = None,
        page_context: dict[str, Any] | None = None,
        private_event_draft: dict[str, Any] | None = None,
    ) -> WidgetState:
        if state_value not in {
            item.value
            for item in WidgetStateValue
        }:
            raise ValueError(
                f"Invalid widget state: {state_value}"
            )

        if size_value not in {
            item.value
            for item in WidgetSizeValue
        }:
            raise ValueError(
                f"Invalid widget size: {size_value}"
            )

        widget_state.widget_state = state_value
        widget_state.widget_size = size_value
        widget_state.unread_count = max(
            int(unread_count),
            0,
        )

        if conversation_id is not None:
            widget_state.conversation_id = (
                conversation_id
            )

        if page_url is not None:
            widget_state.current_page_url = (
                _clean_text(
                    page_url,
                    maximum_length=2048,
                )
            )

        if page_category is not None:
            widget_state.current_page_category = (
                _clean_text(
                    page_category,
                    maximum_length=100,
                )
            )

        if page_context is not None:
            widget_state.page_context_json = (
                dict(page_context)
            )

        if private_event_draft is not None:
            widget_state.private_event_draft_json = (
                dict(private_event_draft)
            )

        self.session.flush()

        return widget_state


# ============================================================
# SECTION 10 - SPELLING REPOSITORY
# ============================================================

class SpellingRepository(
    BaseRepository[SpellingVariant]
):
    model = SpellingVariant

    def observe(
        self,
        *,
        incorrect_text: str,
        corrected_text: str,
        language: str = "en",
        confidence: Decimal = Decimal("0.5"),
        source_kind: str = "observed",
    ) -> tuple[SpellingVariant, bool]:
        incorrect = _normalize_phrase(
            incorrect_text
        )
        corrected = _normalize_phrase(
            corrected_text
        )
        language = language.strip().lower()

        existing = self.session.scalar(
            select(SpellingVariant)
            .where(
                SpellingVariant.incorrect_text
                == incorrect,
                SpellingVariant.corrected_text
                == corrected,
                SpellingVariant.language
                == language,
            )
        )

        now = utc_now()

        if existing is not None:
            existing.occurrence_count = (
                int(existing.occurrence_count or 0)
                + 1
            )
            existing.last_seen_at = now

            if confidence > existing.confidence:
                existing.confidence = confidence

            self.session.flush()

            return existing, False

        variant = SpellingVariant(
            incorrect_text=incorrect,
            corrected_text=corrected,
            language=language,
            source_kind=source_kind,
            occurrence_count=1,
            confidence=confidence,
            first_seen_at=now,
            last_seen_at=now,
        )

        self.add(variant)

        return variant, True

    def lookup(
        self,
        incorrect_text: str,
        *,
        language: str = "en",
        verified_only: bool = True,
    ) -> list[SpellingVariant]:
        conditions = [
            SpellingVariant.incorrect_text
            == _normalize_phrase(
                incorrect_text
            ),
            SpellingVariant.language
            == language.strip().lower(),
            SpellingVariant.is_active.is_(True),
        ]

        if verified_only:
            conditions.append(
                SpellingVariant.is_verified.is_(True)
            )

        statement = (
            select(SpellingVariant)
            .where(and_(*conditions))
            .order_by(
                SpellingVariant.confidence.desc(),
                SpellingVariant.occurrence_count.desc(),
            )
        )

        return list(
            self.session.scalars(statement).all()
        )


# ============================================================
# SECTION 11 - TRAINING REPOSITORY
# ============================================================

class TrainingRepository(
    BaseRepository[TrainingExample]
):
    model = TrainingExample

    def create_candidate(
        self,
        *,
        example_type: str,
        input_text: str,
        normalized_text: str | None = None,
        target_intent: str | None = None,
        target_entities: Sequence[dict[str, Any]] | None = None,
        target_response: str | None = None,
        source_message_id: str | None = None,
        quality_score: Decimal | None = None,
        contamination_risk_score: Decimal | None = None,
    ) -> tuple[TrainingExample, bool]:
        normalized_for_hash = (
            normalized_text
            or input_text
        )

        duplicate_hash = _stable_hash(
            (
                f"{example_type}|"
                f"{target_intent or ''}|"
                f"{_normalize_phrase(normalized_for_hash)}"
            )
        )

        existing = self.session.scalar(
            select(TrainingExample)
            .where(
                TrainingExample.duplicate_hash
                == duplicate_hash
            )
        )

        if existing is not None:
            return existing, False

        candidate = TrainingExample(
            source_message_id=source_message_id,
            example_type=example_type,
            input_text=input_text,
            normalized_text=normalized_text,
            target_intent=target_intent,
            target_entities_json=(
                _json_safe_list(
                    target_entities
                )
            ),
            target_response=target_response,
            review_status=(
                TrainingReviewStatus.PENDING.value
            ),
            quality_score=quality_score,
            duplicate_hash=duplicate_hash,
            contamination_risk_score=(
                contamination_risk_score
            ),
        )

        self.add(candidate)

        return candidate, True

    def list_pending_review(
        self,
        *,
        limit: int = 100,
    ) -> list[TrainingExample]:
        limit = min(max(limit, 1), 500)

        statement = (
            select(TrainingExample)
            .where(
                TrainingExample.review_status
                == TrainingReviewStatus.PENDING.value
            )
            .order_by(
                TrainingExample.created_at.asc()
            )
            .limit(limit)
        )

        return list(
            self.session.scalars(statement).all()
        )


# ============================================================
# SECTION 12 - FEEDBACK REPOSITORY
# ============================================================

class FeedbackRepository(
    BaseRepository[UserFeedback]
):
    model = UserFeedback

    def create_feedback(
        self,
        *,
        feedback_type: str,
        conversation_id: str | None = None,
        message_id: str | None = None,
        rating: int | None = None,
        feedback_text: str | None = None,
        suggested_intent: str | None = None,
        suggested_response: str | None = None,
        correction: dict[str, Any] | None = None,
    ) -> UserFeedback:
        if feedback_type not in {
            item.value
            for item in FeedbackType
        }:
            raise ValueError(
                f"Invalid feedback type: {feedback_type}"
            )

        if rating is not None and not 1 <= rating <= 5:
            raise ValueError(
                "Feedback rating must be between 1 and 5."
            )

        feedback = UserFeedback(
            conversation_id=conversation_id,
            message_id=message_id,
            feedback_type=feedback_type,
            rating=rating,
            feedback_text=_clean_text(
                feedback_text
            ),
            suggested_intent=_clean_text(
                suggested_intent,
                maximum_length=255,
            ),
            suggested_response=_clean_text(
                suggested_response
            ),
            correction_json=_json_safe_dict(
                correction
            ),
        )

        return self.add(feedback)


# ============================================================
# SECTION 13 - ANALYTICS REPOSITORY
# ============================================================

class AnalyticsRepository(
    BaseRepository[AnalyticsEvent]
):
    model = AnalyticsEvent

    def record(
        self,
        *,
        event_name: str,
        business_id: str | None = None,
        browser_session_id: str | None = None,
        conversation_id: str | None = None,
        event_category: str = "chatbot",
        page_url: str | None = None,
        page_category: str | None = None,
        event_value: Decimal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AnalyticsEvent:
        event = AnalyticsEvent(
            business_id=business_id,
            browser_session_id=(
                browser_session_id
            ),
            conversation_id=conversation_id,
            event_name=event_name.strip(),
            event_category=(
                event_category.strip().lower()
            ),
            page_url=_clean_text(
                page_url,
                maximum_length=2048,
            ),
            page_category=_clean_text(
                page_category,
                maximum_length=100,
            ),
            event_value=event_value,
            metadata_json=_json_safe_dict(
                metadata
            ),
            occurred_at=utc_now(),
        )

        return self.add(event)


# ============================================================
# SECTION 14 - PRIVATE EVENT REPOSITORY
# ============================================================

class PrivateEventRepository(
    BaseRepository[PrivateEventInquiry]
):
    model = PrivateEventInquiry

    def create_inquiry(
        self,
        *,
        business_id: str,
        conversation_id: str | None = None,
        customer_name: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        event_type: str | None = None,
        preferred_date: date | None = None,
        guest_count: int | None = None,
        budget_min: Decimal | None = None,
        budget_max: Decimal | None = None,
        source_page_url: str | None = None,
        customer_message: str | None = None,
        source_campaign: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> PrivateEventInquiry:
        if guest_count is not None and guest_count <= 0:
            raise ValueError(
                "Guest count must be greater than zero."
            )

        details = details or {}

        inquiry = PrivateEventInquiry(
            business_id=business_id,
            conversation_id=conversation_id,
            lead_status=LeadStatus.NEW.value,
            customer_name=_clean_text(
                customer_name,
                maximum_length=255,
            ),
            email=_clean_text(
                email,
                maximum_length=320,
            ),
            phone=_clean_text(
                phone,
                maximum_length=64,
            ),
            company_name=_clean_text(
                details.get("company_name"),
                maximum_length=255,
            ),
            event_type=_clean_text(
                event_type,
                maximum_length=100,
            ),
            preferred_date=preferred_date,
            alternate_date=details.get(
                "alternate_date"
            ),
            start_time=details.get(
                "start_time"
            ),
            end_time=details.get(
                "end_time"
            ),
            guest_count=guest_count,
            budget_min=budget_min,
            budget_max=budget_max,
            space_preference=_clean_text(
                details.get(
                    "space_preference"
                ),
                maximum_length=255,
            ),
            food_package=_clean_text(
                details.get(
                    "food_package"
                ),
                maximum_length=255,
            ),
            bar_package=_clean_text(
                details.get(
                    "bar_package"
                ),
                maximum_length=255,
            ),
            entertainment_required=bool(
                details.get(
                    "entertainment_required",
                    False,
                )
            ),
            av_required=bool(
                details.get(
                    "av_required",
                    False,
                )
            ),
            dietary_requirements=_clean_text(
                details.get(
                    "dietary_requirements"
                )
            ),
            customer_message=_clean_text(
                customer_message
            ),
            source_campaign=_clean_text(
                source_campaign,
                maximum_length=255,
            ),
            source_page_url=_clean_text(
                source_page_url,
                maximum_length=2048,
            ),
        )

        return self.add(inquiry)

    def update_status(
        self,
        inquiry: PrivateEventInquiry,
        *,
        status_value: str,
        assigned_to: str | None = None,
        management_notes: str | None = None,
    ) -> PrivateEventInquiry:
        if status_value not in {
            item.value
            for item in LeadStatus
        }:
            raise ValueError(
                f"Invalid lead status: {status_value}"
            )

        inquiry.lead_status = status_value

        if assigned_to is not None:
            inquiry.assigned_to = _clean_text(
                assigned_to,
                maximum_length=255,
            )

        if management_notes is not None:
            inquiry.management_notes = (
                _clean_text(
                    management_notes
                )
            )

        if status_value == LeadStatus.CONTACTED.value:
            inquiry.contacted_at = utc_now()

        if status_value == LeadStatus.BOOKED.value:
            inquiry.booked_at = utc_now()

        self.session.flush()

        return inquiry


# ============================================================
# SECTION 15 - KNOWLEDGE GAP REPOSITORY
# ============================================================

class KnowledgeGapRepository(
    BaseRepository[KnowledgeGap]
):
    model = KnowledgeGap

    def observe(
        self,
        *,
        gap_type: str,
        canonical_question: str,
        example_query: str,
        business_id: str | None = None,
        priority_score: Decimal | None = None,
    ) -> tuple[KnowledgeGap, bool]:
        normalized_question = _normalize_phrase(
            canonical_question
        )

        existing = self.session.scalar(
            select(KnowledgeGap)
            .where(
                KnowledgeGap.business_id
                == business_id,
                KnowledgeGap.gap_type
                == gap_type.strip().lower(),
                func.lower(
                    KnowledgeGap.canonical_question
                )
                == normalized_question,
                KnowledgeGap.resolved.is_(False),
            )
        )

        if existing is not None:
            examples = list(
                existing.example_queries_json
                or []
            )

            if example_query not in examples:
                examples.append(example_query)

            existing.example_queries_json = (
                examples[-100:]
            )
            existing.occurrence_count = (
                int(existing.occurrence_count or 0)
                + 1
            )

            if priority_score is not None:
                existing.priority_score = (
                    priority_score
                )

            self.session.flush()

            return existing, False

        gap = KnowledgeGap(
            business_id=business_id,
            gap_type=gap_type.strip().lower(),
            canonical_question=(
                normalized_question
            ),
            example_queries_json=[
                example_query
            ],
            occurrence_count=1,
            priority_score=priority_score,
            resolved=False,
        )

        self.add(gap)

        return gap, True


# ============================================================
# SECTION 16 - MODEL VERSION REPOSITORY
# ============================================================

class ModelVersionRepository(
    BaseRepository[ModelVersion]
):
    model = ModelVersion

    def get_by_type_and_version(
        self,
        *,
        model_type: str,
        version: str,
    ) -> ModelVersion | None:
        return self.session.scalar(
            select(ModelVersion)
            .where(
                ModelVersion.model_type
                == model_type.strip().lower(),
                ModelVersion.version
                == version.strip(),
            )
        )

    def get_production_model(
        self,
        model_type: str,
    ) -> ModelVersion | None:
        return self.session.scalar(
            select(ModelVersion)
            .where(
                ModelVersion.model_type
                == model_type.strip().lower(),
                ModelVersion.lifecycle_status
                == ModelLifecycleStatus.PRODUCTION.value,
            )
            .order_by(
                ModelVersion.promoted_at.desc()
            )
        )


# ============================================================
# SECTION 17 - REPOSITORY BUNDLE
# ============================================================

class RepositoryBundle:
    """
    Convenience container exposing repositories for one Session.
    """

    def __init__(
        self,
        session: Session,
    ) -> None:
        self.session = session

        self.businesses = (
            BusinessRepository(session)
        )
        self.knowledge = (
            KnowledgeRepository(session)
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
        self.spelling = (
            SpellingRepository(session)
        )
        self.training = (
            TrainingRepository(session)
        )
        self.feedback = (
            FeedbackRepository(session)
        )
        self.analytics = (
            AnalyticsRepository(session)
        )
        self.private_events = (
            PrivateEventRepository(session)
        )
        self.knowledge_gaps = (
            KnowledgeGapRepository(session)
        )
        self.model_versions = (
            ModelVersionRepository(session)
        )


def get_repositories(
    session: Session,
) -> RepositoryBundle:
    return RepositoryBundle(session)


# ============================================================
# SECTION 18 - COMPLETE WIDGET SESSION WORKFLOW
# ============================================================

def get_or_create_widget_conversation(
    session: Session,
    *,
    business_slug: str,
    external_session_id: str,
    external_conversation_id: str,
    page_url: str | None = None,
    page_category: str | None = None,
    referrer: str | None = None,
    user_agent: str | None = None,
    ip_hash: str | None = None,
    attribution: dict[str, str | None] | None = None,
    page_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Initialize or restore a durable global widget conversation.
    """

    repositories = get_repositories(session)

    business = repositories.businesses.get_by_slug(
        business_slug
    )

    if business is None:
        raise LookupError(
            f"Business not found: {business_slug}"
        )

    browser_session, browser_created = (
        repositories.browser_sessions.get_or_create(
            business_id=business.id,
            external_session_id=(
                external_session_id
            ),
            page_url=page_url,
            referrer=referrer,
            user_agent=user_agent,
            ip_hash=ip_hash,
            attribution=attribution,
        )
    )

    conversation, conversation_created = (
        repositories.conversations.get_or_create(
            business_id=business.id,
            external_conversation_id=(
                external_conversation_id
            ),
            browser_session_id=(
                browser_session.id
            ),
            initial_context={
                "page_url": page_url,
                "page_category": page_category,
            },
        )
    )

    widget_state, widget_created = (
        repositories.widget_states.get_or_create(
            browser_session_id=(
                browser_session.id
            ),
            conversation_id=conversation.id,
            page_url=page_url,
            page_category=page_category,
        )
    )

    if page_context is not None:
        widget_state.page_context_json = (
            dict(page_context)
        )

    analytics_event = (
        repositories.analytics.record(
            event_name=(
                "widget_session_initialized"
            ),
            event_category="widget",
            business_id=business.id,
            browser_session_id=(
                browser_session.id
            ),
            conversation_id=conversation.id,
            page_url=page_url,
            page_category=page_category,
            metadata={
                "browser_created": (
                    browser_created
                ),
                "conversation_created": (
                    conversation_created
                ),
                "widget_created": (
                    widget_created
                ),
            },
        )
    )

    session.flush()

    return {
        "business": business,
        "browser_session": browser_session,
        "conversation": conversation,
        "widget_state": widget_state,
        "analytics_event": analytics_event,
        "browser_created": browser_created,
        "conversation_created": (
            conversation_created
        ),
        "widget_created": widget_created,
    }


# ============================================================
# SECTION 19 - SELF-TEST
# ============================================================

def validate_repositories_module() -> dict[str, Any]:
    """
    Run an in-memory repository workflow verification.
    """

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.database.base import Base

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
        future=True,
    )

    Base.metadata.create_all(engine)

    checks: dict[str, bool] = {}

    with Session(engine) as session:
        repositories = get_repositories(
            session
        )

        business, business_created = (
            repositories.businesses.get_or_create(
                slug="horseshoe-tavern",
                display_name="Horseshoe Tavern",
                website_url=(
                    "https://www.thehorseshoetavern.com/"
                ),
            )
        )

        checks[
            "business_created"
        ] = business_created

        workflow = (
            get_or_create_widget_conversation(
                session,
                business_slug=(
                    "horseshoe-tavern"
                ),
                external_session_id=(
                    "session_repository_test_12345678"
                ),
                external_conversation_id=(
                    "conversation_repository_test_12345678"
                ),
                page_url=(
                    "https://www.thehorseshoetavern.com/menu"
                ),
                page_category="menu",
                page_context={
                    "title": "Menu",
                },
            )
        )

        conversation = workflow[
            "conversation"
        ]

        user_message = (
            repositories.messages.create_user_message(
                conversation=conversation,
                text="wat time do u close",
                normalized_text=(
                    "what time do you close"
                ),
                corrected_text=(
                    "what time do you close"
                ),
                detected_intent=(
                    "HOURS_GENERAL"
                ),
                page_category="menu",
            )
        )

        assistant_message = (
            repositories.messages.create_assistant_message(
                conversation=conversation,
                text=(
                    "I can help with today's verified hours."
                ),
                detected_intent=(
                    "HOURS_GENERAL"
                ),
                retrieval_sources=[
                    {
                        "source_type": (
                            "verified_business_hours"
                        )
                    }
                ],
                validation_result={
                    "verified": True,
                },
            )
        )

        spelling, spelling_created = (
            repositories.spelling.observe(
                incorrect_text="wat",
                corrected_text="what",
            )
        )

        training, training_created = (
            repositories.training.create_candidate(
                example_type=(
                    "intent_classification"
                ),
                input_text=(
                    "wat time do u close"
                ),
                normalized_text=(
                    "what time do you close"
                ),
                target_intent=(
                    "HOURS_GENERAL"
                ),
                source_message_id=(
                    user_message.id
                ),
            )
        )

        feedback = (
            repositories.feedback.create_feedback(
                feedback_type=(
                    FeedbackType.POSITIVE.value
                ),
                conversation_id=(
                    conversation.id
                ),
                message_id=(
                    assistant_message.id
                ),
                rating=5,
            )
        )

        inquiry = (
            repositories.private_events.create_inquiry(
                business_id=business.id,
                conversation_id=(
                    conversation.id
                ),
                customer_name=(
                    "Repository Test"
                ),
                event_type="birthday",
                guest_count=40,
            )
        )

        gap, gap_created = (
            repositories.knowledge_gaps.observe(
                business_id=business.id,
                gap_type="missing_fact",
                canonical_question=(
                    "Do you offer valet parking?"
                ),
                example_query=(
                    "is there valet"
                ),
            )
        )

        session.commit()

        checks.update(
            {
                "browser_session_created": (
                    workflow[
                        "browser_created"
                    ]
                ),
                "conversation_created": (
                    workflow[
                        "conversation_created"
                    ]
                ),
                "widget_state_created": (
                    workflow[
                        "widget_created"
                    ]
                ),
                "user_message_created": (
                    user_message.sequence_number
                    == 1
                ),
                "assistant_message_created": (
                    assistant_message.sequence_number
                    == 2
                ),
                "message_count_updated": (
                    conversation.message_count
                    == 2
                ),
                "spelling_created": (
                    spelling_created
                    and spelling.occurrence_count
                    == 1
                ),
                "training_created": (
                    training_created
                    and training.review_status
                    == (
                        TrainingReviewStatus.PENDING.value
                    )
                ),
                "feedback_created": (
                    feedback.rating == 5
                ),
                "private_event_created": (
                    inquiry.guest_count == 40
                ),
                "knowledge_gap_created": (
                    gap_created
                    and gap.occurrence_count
                    == 1
                ),
                "analytics_created": (
                    repositories.analytics.count()
                    == 1
                ),
            }
        )

    engine.dispose()

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
    }


# ============================================================
# SECTION 20 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    report = validate_repositories_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
