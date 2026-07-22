# ============================================================
# Exact file location: app/database/models.py
# Horseshoe Tavern AI
# Phase 1 Part 1.9
# Durable business, conversation, learning, lead, model,
# analytics, widget-state, and verified-knowledge schema
# ============================================================

"""
Durable SQLAlchemy models for Horseshoe Tavern AI.

The schema separates five major categories of information:

1. Verified business truth
   Hours, menu items, events, parking, ordering links, FAQs,
   private-event packages, contact information, and source provenance.

2. Conversation operations
   Browser sessions, conversations, messages, detected intent,
   spelling corrections, retrieval evidence, response variants,
   widget state, page context, and conversion actions.

3. Controlled learning
   Candidate training examples, reviews, phrase clusters,
   spelling variants, feedback, model versions, evaluations,
   deployments, learning jobs, drift reports, and poisoning reports.

4. Revenue operations
   Private-event inquiries, lead status, lead scoring,
   communication consent, attribution, and conversion events.

5. Governance and observability
   Verification state, source history, audit events,
   knowledge gaps, failed queries, and analytics events.

Raw public messages may become reviewed language examples. They may never
directly overwrite verified business facts.
"""

from __future__ import annotations

import enum
import json
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Final

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Time,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.database.base import (
    ActiveStatusMixin,
    Base,
    CreatedAtMixin,
    FreshnessMixin,
    GovernedRecordMixin,
    PublicIdentifierMixin,
    SoftDeleteMixin,
    SourceProvenanceMixin,
    StandardRecordMixin,
    TimestampMixin,
    UpdatedAtMixin,
    VerificationMixin,
    VersionedMixin,
    new_uuid_string,
    utc_now,
)


# ============================================================
# SECTION 01 - CONSTANTS
# ============================================================

SHORT_TEXT_LENGTH: Final[int] = 100
MEDIUM_TEXT_LENGTH: Final[int] = 255
LONG_TEXT_LENGTH: Final[int] = 500
URL_LENGTH: Final[int] = 2048
EMAIL_LENGTH: Final[int] = 320
PHONE_LENGTH: Final[int] = 64
IDENTIFIER_LENGTH: Final[int] = 160
MODEL_VERSION_LENGTH: Final[int] = 128
HASH_LENGTH: Final[int] = 128
MONEY_PRECISION: Final[int] = 12
MONEY_SCALE: Final[int] = 2
CONFIDENCE_PRECISION: Final[int] = 8
CONFIDENCE_SCALE: Final[int] = 6


# ============================================================
# SECTION 02 - ENUMERATIONS
# ============================================================

class ConversationStatus(str, enum.Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ESCALATED = "escalated"
    ARCHIVED = "archived"


class MessageRole(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageStatus(str, enum.Enum):
    RECEIVED = "received"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class WidgetStateValue(str, enum.Enum):
    COLLAPSED = "collapsed"
    MINIMIZED = "minimized"
    OPEN = "open"


class WidgetSizeValue(str, enum.Enum):
    COMPACT = "compact"
    EXPANDED = "expanded"
    FULLSCREEN = "fullscreen"


class KnowledgeStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    VERIFIED = "verified"
    STALE = "stale"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class TrainingReviewStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class FeedbackType(str, enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    REPORT = "report"
    CONVERSION = "conversion"
    NEUTRAL = "neutral"


class ModelLifecycleStatus(str, enum.Enum):
    CANDIDATE = "candidate"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    PRODUCTION = "production"
    REJECTED = "rejected"
    ARCHIVED = "archived"
    ROLLED_BACK = "rolled_back"


class LeadStatus(str, enum.Enum):
    NEW = "new"
    QUALIFYING = "qualifying"
    QUALIFIED = "qualified"
    CONTACTED = "contacted"
    PROPOSAL_SENT = "proposal_sent"
    DEPOSIT_PENDING = "deposit_pending"
    BOOKED = "booked"
    LOST = "lost"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class ConsentStatus(str, enum.Enum):
    NOT_REQUESTED = "not_requested"
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"


class LearningJobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SeverityLevel(str, enum.Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# SECTION 03 - ENUM STORAGE HELPER
# ============================================================

def enum_values(enum_class: type[enum.Enum]) -> list[str]:
    return [str(member.value) for member in enum_class]


# ============================================================
# SECTION 04 - BUSINESS
# ============================================================

class Business(
    GovernedRecordMixin,
    Base,
):
    __tablename__ = "businesses"

    slug: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
        unique=True,
        index=True,
    )

    legal_name: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    display_name: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    website_url: Mapped[str] = mapped_column(
        String(URL_LENGTH),
        nullable=False,
    )

    phone: Mapped[str | None] = mapped_column(
        String(PHONE_LENGTH),
        nullable=True,
    )

    general_email: Mapped[str | None] = mapped_column(
        String(EMAIL_LENGTH),
        nullable=True,
    )

    events_email: Mapped[str | None] = mapped_column(
        String(EMAIL_LENGTH),
        nullable=True,
    )

    events_phone: Mapped[str | None] = mapped_column(
        String(PHONE_LENGTH),
        nullable=True,
    )

    address_line_1: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    address_line_2: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    city: Mapped[str | None] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=True,
    )

    state_code: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
    )

    postal_code: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )

    country_code: Mapped[str] = mapped_column(
        String(2),
        nullable=False,
        default="US",
    )

    latitude: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 7),
        nullable=True,
    )

    longitude: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 7),
        nullable=True,
    )

    timezone: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default="America/New_York",
    )

    default_language: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="en",
    )

    configuration_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    hours: Mapped[list["BusinessHour"]] = relationship(
        back_populates="business",
        cascade="all, delete-orphan",
    )

    contacts: Mapped[list["BusinessContact"]] = relationship(
        back_populates="business",
        cascade="all, delete-orphan",
    )

    knowledge_documents: Mapped[list["KnowledgeDocument"]] = relationship(
        back_populates="business",
        cascade="all, delete-orphan",
    )

    menu_categories: Mapped[list["MenuCategory"]] = relationship(
        back_populates="business",
        cascade="all, delete-orphan",
    )

    events: Mapped[list["BusinessEvent"]] = relationship(
        back_populates="business",
        cascade="all, delete-orphan",
    )

    faq_entries: Mapped[list["FAQEntry"]] = relationship(
        back_populates="business",
        cascade="all, delete-orphan",
    )

    private_event_packages: Mapped[list["PrivateEventPackage"]] = relationship(
        back_populates="business",
        cascade="all, delete-orphan",
    )

    browser_sessions: Mapped[list["BrowserSession"]] = relationship(
        back_populates="business",
        cascade="all, delete-orphan",
    )

    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="business",
        cascade="all, delete-orphan",
    )

    private_event_inquiries: Mapped[list["PrivateEventInquiry"]] = relationship(
        back_populates="business",
        cascade="all, delete-orphan",
    )


# ============================================================
# SECTION 05 - BUSINESS CONTACTS
# ============================================================

class BusinessContact(
    GovernedRecordMixin,
    Base,
):
    __tablename__ = "business_contacts"

    business_id: Mapped[str] = mapped_column(
        ForeignKey(
            "businesses.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    contact_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    label: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
    )

    value: Mapped[str] = mapped_column(
        String(LONG_TEXT_LENGTH),
        nullable=False,
    )

    display_value: Mapped[str | None] = mapped_column(
        String(LONG_TEXT_LENGTH),
        nullable=True,
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    business: Mapped["Business"] = relationship(
        back_populates="contacts",
    )

    __table_args__ = (
        UniqueConstraint(
            "business_id",
            "contact_type",
            "value",
            name="uq_business_contacts_type_value",
        ),
    )


# ============================================================
# SECTION 06 - BUSINESS HOURS
# ============================================================

class BusinessHour(
    GovernedRecordMixin,
    Base,
):
    __tablename__ = "business_hours"

    business_id: Mapped[str] = mapped_column(
        ForeignKey(
            "businesses.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    service_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default="tavern",
        index=True,
    )

    day_of_week: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    open_time: Mapped[time | None] = mapped_column(
        Time,
        nullable=True,
    )

    close_time: Mapped[time | None] = mapped_column(
        Time,
        nullable=True,
    )

    closes_next_day: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_closed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    business: Mapped["Business"] = relationship(
        back_populates="hours",
    )

    __table_args__ = (
        CheckConstraint(
            "day_of_week >= 0 AND day_of_week <= 6",
            name="valid_day_of_week",
        ),
        UniqueConstraint(
            "business_id",
            "service_type",
            "day_of_week",
            "effective_from",
            name="uq_business_hours_effective_day",
        ),
    )


# ============================================================
# SECTION 07 - KNOWLEDGE DOCUMENTS
# ============================================================

class KnowledgeDocument(
    GovernedRecordMixin,
    Base,
):
    __tablename__ = "knowledge_documents"

    business_id: Mapped[str] = mapped_column(
        ForeignKey(
            "businesses.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    document_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(LONG_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    slug: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
        index=True,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    structured_payload: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    search_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    knowledge_status: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=KnowledgeStatus.DRAFT.value,
        index=True,
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    business: Mapped["Business"] = relationship(
        back_populates="knowledge_documents",
    )

    chunks: Mapped[list["KnowledgeChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index(
            "ix_knowledge_documents_business_type_status",
            "business_id",
            "document_type",
            "knowledge_status",
        ),
    )


class KnowledgeChunk(
    StandardRecordMixin,
    SourceProvenanceMixin,
    VerificationMixin,
    FreshnessMixin,
    ActiveStatusMixin,
    Base,
):
    __tablename__ = "knowledge_chunks"

    document_id: Mapped[str] = mapped_column(
        ForeignKey(
            "knowledge_documents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    chunk_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(MODEL_VERSION_LENGTH),
        nullable=True,
    )

    embedding_reference: Mapped[str | None] = mapped_column(
        String(LONG_TEXT_LENGTH),
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    document: Mapped["KnowledgeDocument"] = relationship(
        back_populates="chunks",
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_knowledge_chunks_document_index",
        ),
    )


# ============================================================
# SECTION 08 - MENU
# ============================================================

class MenuCategory(
    GovernedRecordMixin,
    Base,
):
    __tablename__ = "menu_categories"

    business_id: Mapped[str] = mapped_column(
        ForeignKey(
            "businesses.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
    )

    slug: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    display_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    business: Mapped["Business"] = relationship(
        back_populates="menu_categories",
    )

    items: Mapped[list["MenuItem"]] = relationship(
        back_populates="category",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "business_id",
            "slug",
            name="uq_menu_categories_business_slug",
        ),
    )


class MenuItem(
    GovernedRecordMixin,
    Base,
):
    __tablename__ = "menu_items"

    category_id: Mapped[str] = mapped_column(
        ForeignKey(
            "menu_categories.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    slug: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    price: Mapped[Decimal | None] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE),
        nullable=True,
    )

    price_note: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    dietary_tags: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    allergen_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    online_order_url: Mapped[str | None] = mapped_column(
        String(URL_LENGTH),
        nullable=True,
    )

    image_url: Mapped[str | None] = mapped_column(
        String(URL_LENGTH),
        nullable=True,
    )

    is_featured: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    display_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    category: Mapped["MenuCategory"] = relationship(
        back_populates="items",
    )

    __table_args__ = (
        UniqueConstraint(
            "category_id",
            "slug",
            name="uq_menu_items_category_slug",
        ),
    )


# ============================================================
# SECTION 09 - EVENTS, FAQ, AND PRIVATE EVENT PACKAGES
# ============================================================

class BusinessEvent(
    GovernedRecordMixin,
    Base,
):
    __tablename__ = "business_events"

    business_id: Mapped[str] = mapped_column(
        ForeignKey(
            "businesses.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(LONG_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    slug: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    start_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    end_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    recurrence_rule: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    location_name: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    age_restriction: Mapped[str | None] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=True,
    )

    cover_charge: Mapped[Decimal | None] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE),
        nullable=True,
    )

    reservation_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    reservation_url: Mapped[str | None] = mapped_column(
        String(URL_LENGTH),
        nullable=True,
    )

    image_url: Mapped[str | None] = mapped_column(
        String(URL_LENGTH),
        nullable=True,
    )

    business: Mapped["Business"] = relationship(
        back_populates="events",
    )

    __table_args__ = (
        UniqueConstraint(
            "business_id",
            "slug",
            "start_at",
            name="uq_business_events_slug_start",
        ),
        Index(
            "ix_business_events_business_start_active",
            "business_id",
            "start_at",
            "is_active",
        ),
    )


class FAQEntry(
    GovernedRecordMixin,
    Base,
):
    __tablename__ = "faq_entries"

    business_id: Mapped[str] = mapped_column(
        ForeignKey(
            "businesses.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    category: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default="general",
        index=True,
    )

    question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    alternative_questions: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    display_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    business: Mapped["Business"] = relationship(
        back_populates="faq_entries",
    )


class PrivateEventPackage(
    GovernedRecordMixin,
    Base,
):
    __tablename__ = "private_event_packages"

    business_id: Mapped[str] = mapped_column(
        ForeignKey(
            "businesses.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    package_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    price_per_person: Mapped[Decimal | None] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE),
        nullable=True,
    )

    flat_price: Mapped[Decimal | None] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE),
        nullable=True,
    )

    minimum_guests: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    maximum_guests: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    duration_minutes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    included_items: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    upgrade_options: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    business: Mapped["Business"] = relationship(
        back_populates="private_event_packages",
    )


# ============================================================
# SECTION 10 - BROWSER SESSION AND WIDGET STATE
# ============================================================

class BrowserSession(
    StandardRecordMixin,
    ActiveStatusMixin,
    Base,
):
    __tablename__ = "browser_sessions"

    business_id: Mapped[str] = mapped_column(
        ForeignKey(
            "businesses.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    external_session_id: Mapped[str] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=False,
        unique=True,
        index=True,
    )

    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    first_page_url: Mapped[str | None] = mapped_column(
        String(URL_LENGTH),
        nullable=True,
    )

    last_page_url: Mapped[str | None] = mapped_column(
        String(URL_LENGTH),
        nullable=True,
    )

    referrer: Mapped[str | None] = mapped_column(
        String(URL_LENGTH),
        nullable=True,
    )

    user_agent: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    ip_hash: Mapped[str | None] = mapped_column(
        String(HASH_LENGTH),
        nullable=True,
        index=True,
    )

    utm_source: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
        index=True,
    )

    utm_medium: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
        index=True,
    )

    utm_campaign: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
        index=True,
    )

    business: Mapped["Business"] = relationship(
        back_populates="browser_sessions",
    )

    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="browser_session",
    )

    widget_state: Mapped["WidgetState | None"] = relationship(
        back_populates="browser_session",
        cascade="all, delete-orphan",
        uselist=False,
    )


class WidgetState(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "widget_states"

    browser_session_id: Mapped[str] = mapped_column(
        ForeignKey(
            "browser_sessions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
        index=True,
    )

    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "conversations.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    widget_state: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=WidgetStateValue.COLLAPSED.value,
    )

    widget_size: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=WidgetSizeValue.COMPACT.value,
    )

    unread_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    current_page_url: Mapped[str | None] = mapped_column(
        String(URL_LENGTH),
        nullable=True,
    )

    current_page_category: Mapped[str | None] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=True,
    )

    page_context_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    private_event_draft_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    browser_session: Mapped["BrowserSession"] = relationship(
        back_populates="widget_state",
    )

    conversation: Mapped["Conversation | None"] = relationship(
        back_populates="widget_states",
    )

    __table_args__ = (
        CheckConstraint(
            "unread_count >= 0",
            name="widget_states_nonnegative_unread",
        ),
    )


# ============================================================
# SECTION 11 - CONVERSATIONS AND MESSAGES
# ============================================================

class Conversation(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "conversations"

    business_id: Mapped[str] = mapped_column(
        ForeignKey(
            "businesses.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    browser_session_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "browser_sessions.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    external_conversation_id: Mapped[str] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=False,
        unique=True,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=ConversationStatus.ACTIVE.value,
        index=True,
    )

    channel: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default="website_widget",
        index=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    current_intent: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
        index=True,
    )

    current_context_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    converted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    conversion_type: Mapped[str | None] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=True,
        index=True,
    )

    conversion_value: Mapped[Decimal | None] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE),
        nullable=True,
    )

    human_handoff_requested: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    business: Mapped["Business"] = relationship(
        back_populates="conversations",
    )

    browser_session: Mapped["BrowserSession | None"] = relationship(
        back_populates="conversations",
    )

    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.sequence_number",
    )

    widget_states: Mapped[list["WidgetState"]] = relationship(
        back_populates="conversation",
    )

    feedback_entries: Mapped[list["UserFeedback"]] = relationship(
        back_populates="conversation",
    )

    private_event_inquiries: Mapped[list["PrivateEventInquiry"]] = relationship(
        back_populates="conversation",
    )

    analytics_events: Mapped[list["AnalyticsEvent"]] = relationship(
        back_populates="conversation",
    )

    __table_args__ = (
        CheckConstraint(
            "message_count >= 0",
            name="conversations_nonnegative_message_count",
        ),
        Index(
            "ix_conversations_business_status_activity",
            "business_id",
            "status",
            "last_activity_at",
        ),
    )


class ConversationMessage(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "conversation_messages"

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey(
            "conversations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=MessageStatus.COMPLETED.value,
        index=True,
    )

    original_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    normalized_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    corrected_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    detected_language: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )

    detected_intent: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
        index=True,
    )

    intent_confidence: Mapped[Decimal | None] = mapped_column(
        Numeric(CONFIDENCE_PRECISION, CONFIDENCE_SCALE),
        nullable=True,
    )

    answer_confidence: Mapped[Decimal | None] = mapped_column(
        Numeric(CONFIDENCE_PRECISION, CONFIDENCE_SCALE),
        nullable=True,
    )

    detected_entities_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    spelling_corrections_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    retrieval_sources_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    response_actions_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    validation_result_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    model_versions_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    response_template_id: Mapped[str | None] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=True,
        index=True,
    )

    response_variant_id: Mapped[str | None] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=True,
        index=True,
    )

    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    page_url: Mapped[str | None] = mapped_column(
        String(URL_LENGTH),
        nullable=True,
    )

    page_category: Mapped[str | None] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=True,
    )

    conversation: Mapped["Conversation"] = relationship(
        back_populates="messages",
    )

    feedback_entries: Mapped[list["UserFeedback"]] = relationship(
        back_populates="message",
    )

    training_examples: Mapped[list["TrainingExample"]] = relationship(
        back_populates="source_message",
    )

    failed_query: Mapped["FailedQuery | None"] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        uselist=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "sequence_number",
            name="uq_conversation_messages_sequence",
        ),
        CheckConstraint(
            "sequence_number >= 1",
            name="conversation_messages_positive_sequence",
        ),
        CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0",
            name="conversation_messages_nonnegative_latency",
        ),
    )


# ============================================================
# SECTION 12 - RESPONSE VARIANTS
# ============================================================

class ResponseTemplate(
    GovernedRecordMixin,
    Base,
):
    __tablename__ = "response_templates"

    intent_name: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    template_key: Mapped[str] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=False,
        unique=True,
        index=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    required_fact_types: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    allowed_action_types: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    variants: Mapped[list["ResponseVariant"]] = relationship(
        back_populates="template",
        cascade="all, delete-orphan",
    )


class ResponseVariant(
    StandardRecordMixin,
    ActiveStatusMixin,
    VersionedMixin,
    Base,
):
    __tablename__ = "response_variants"

    template_id: Mapped[str] = mapped_column(
        ForeignKey(
            "response_templates.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    variant_key: Mapped[str] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=False,
    )

    response_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    tone: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default="helpful",
    )

    weight: Mapped[Decimal] = mapped_column(
        Numeric(CONFIDENCE_PRECISION, CONFIDENCE_SCALE),
        nullable=False,
        default=Decimal("1.0"),
    )

    selection_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    positive_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    negative_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    conversion_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    template: Mapped["ResponseTemplate"] = relationship(
        back_populates="variants",
    )

    __table_args__ = (
        UniqueConstraint(
            "template_id",
            "variant_key",
            name="uq_response_variants_template_key",
        ),
        CheckConstraint(
            "selection_count >= 0",
            name="response_variants_nonnegative_selection",
        ),
    )


# ============================================================
# SECTION 13 - SPELLING, PHRASES, AND INTENT EXAMPLES
# ============================================================

class SpellingVariant(
    StandardRecordMixin,
    VerificationMixin,
    ActiveStatusMixin,
    VersionedMixin,
    Base,
):
    __tablename__ = "spelling_variants"

    incorrect_text: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    corrected_text: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    language: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="en",
        index=True,
    )

    source_kind: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default="observed",
    )

    occurrence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    confidence: Mapped[Decimal] = mapped_column(
        Numeric(CONFIDENCE_PRECISION, CONFIDENCE_SCALE),
        nullable=False,
        default=Decimal("0.5"),
    )

    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    __table_args__ = (
        UniqueConstraint(
            "incorrect_text",
            "corrected_text",
            "language",
            name="uq_spelling_variants_mapping_language",
        ),
        CheckConstraint(
            "occurrence_count >= 1",
            name="spelling_variants_positive_occurrence",
        ),
    )


class PhraseCluster(
    StandardRecordMixin,
    VersionedMixin,
    Base,
):
    __tablename__ = "phrase_clusters"

    cluster_key: Mapped[str] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=False,
        unique=True,
        index=True,
    )

    canonical_phrase: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    inferred_intent: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
        index=True,
    )

    phrase_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    phrases_json: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    cluster_metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )


# ============================================================
# SECTION 14 - USER FEEDBACK
# ============================================================

class UserFeedback(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "user_feedback"

    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "conversations.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    message_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "conversation_messages.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    feedback_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=FeedbackType.NEUTRAL.value,
        index=True,
    )

    rating: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    feedback_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    suggested_intent: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    suggested_response: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    correction_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    reviewed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    reviewed_by: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    conversation: Mapped["Conversation | None"] = relationship(
        back_populates="feedback_entries",
    )

    message: Mapped["ConversationMessage | None"] = relationship(
        back_populates="feedback_entries",
    )

    __table_args__ = (
        CheckConstraint(
            "rating IS NULL OR (rating >= 1 AND rating <= 5)",
            name="user_feedback_rating_range",
        ),
    )


# ============================================================
# SECTION 15 - TRAINING EXAMPLES AND REVIEWS
# ============================================================

class TrainingExample(
    StandardRecordMixin,
    VersionedMixin,
    Base,
):
    __tablename__ = "training_examples"

    source_message_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "conversation_messages.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    example_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    input_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    normalized_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    target_intent: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
        index=True,
    )

    target_entities_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    target_response: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    dataset_split: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        index=True,
    )

    review_status: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=TrainingReviewStatus.PENDING.value,
        index=True,
    )

    quality_score: Mapped[Decimal | None] = mapped_column(
        Numeric(CONFIDENCE_PRECISION, CONFIDENCE_SCALE),
        nullable=True,
    )

    duplicate_hash: Mapped[str | None] = mapped_column(
        String(HASH_LENGTH),
        nullable=True,
        index=True,
    )

    contamination_risk_score: Mapped[Decimal | None] = mapped_column(
        Numeric(CONFIDENCE_PRECISION, CONFIDENCE_SCALE),
        nullable=True,
    )

    source_message: Mapped["ConversationMessage | None"] = relationship(
        back_populates="training_examples",
    )

    reviews: Mapped[list["TrainingExampleReview"]] = relationship(
        back_populates="training_example",
        cascade="all, delete-orphan",
    )


class TrainingExampleReview(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "training_example_reviews"

    training_example_id: Mapped[str] = mapped_column(
        ForeignKey(
            "training_examples.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    reviewer: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
    )

    decision: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    reviewed_input_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reviewed_intent: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    reviewed_entities_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    reviewed_response: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    training_example: Mapped["TrainingExample"] = relationship(
        back_populates="reviews",
    )


# ============================================================
# SECTION 16 - MODEL REGISTRY AND EVALUATION
# ============================================================

class ModelVersion(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "model_versions"

    model_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    version: Mapped[str] = mapped_column(
        String(MODEL_VERSION_LENGTH),
        nullable=False,
        index=True,
    )

    lifecycle_status: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=ModelLifecycleStatus.CANDIDATE.value,
        index=True,
    )

    artifact_path: Mapped[str | None] = mapped_column(
        String(URL_LENGTH),
        nullable=True,
    )

    artifact_hash: Mapped[str | None] = mapped_column(
        String(HASH_LENGTH),
        nullable=True,
    )

    framework: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    framework_version: Mapped[str | None] = mapped_column(
        String(MODEL_VERSION_LENGTH),
        nullable=True,
    )

    dataset_version: Mapped[str | None] = mapped_column(
        String(MODEL_VERSION_LENGTH),
        nullable=True,
        index=True,
    )

    parameters_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    trained_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    promoted_by: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    evaluations: Mapped[list["ModelEvaluation"]] = relationship(
        back_populates="model_version",
        cascade="all, delete-orphan",
    )

    deployments: Mapped[list["ModelDeployment"]] = relationship(
        back_populates="model_version",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "model_type",
            "version",
            name="uq_model_versions_type_version",
        ),
    )


class ModelEvaluation(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "model_evaluations"

    model_version_id: Mapped[str] = mapped_column(
        ForeignKey(
            "model_versions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    evaluation_name: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
    )

    dataset_version: Mapped[str | None] = mapped_column(
        String(MODEL_VERSION_LENGTH),
        nullable=True,
    )

    baseline_model_version: Mapped[str | None] = mapped_column(
        String(MODEL_VERSION_LENGTH),
        nullable=True,
    )

    passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    regression_failures_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    model_version: Mapped["ModelVersion"] = relationship(
        back_populates="evaluations",
    )


class ModelDeployment(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "model_deployments"

    model_version_id: Mapped[str] = mapped_column(
        ForeignKey(
            "model_versions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    environment: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    deployment_status: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    deployed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    deployed_by: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
    )

    previous_model_version: Mapped[str | None] = mapped_column(
        String(MODEL_VERSION_LENGTH),
        nullable=True,
    )

    rollback_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    model_version: Mapped["ModelVersion"] = relationship(
        back_populates="deployments",
    )


# ============================================================
# SECTION 17 - LEARNING JOBS, DRIFT, AND POISONING
# ============================================================

class LearningJob(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "learning_jobs"

    job_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=LearningJobStatus.QUEUED.value,
        index=True,
    )

    requested_by: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    input_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    output_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


class DriftReport(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "drift_reports"

    model_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    model_version: Mapped[str | None] = mapped_column(
        String(MODEL_VERSION_LENGTH),
        nullable=True,
        index=True,
    )

    severity: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=SeverityLevel.INFO.value,
        index=True,
    )

    drift_score: Mapped[Decimal | None] = mapped_column(
        Numeric(CONFIDENCE_PRECISION, CONFIDENCE_SCALE),
        nullable=True,
    )

    sample_window_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    sample_window_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    recommended_action: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


class PoisoningReport(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "poisoning_reports"

    source_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    source_reference: Mapped[str | None] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=True,
        index=True,
    )

    severity: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=SeverityLevel.LOW.value,
        index=True,
    )

    risk_score: Mapped[Decimal | None] = mapped_column(
        Numeric(CONFIDENCE_PRECISION, CONFIDENCE_SCALE),
        nullable=True,
    )

    indicators_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    quarantined: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    reviewed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    reviewed_by: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


# ============================================================
# SECTION 18 - FAILED QUERIES AND KNOWLEDGE GAPS
# ============================================================

class FailedQuery(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "failed_queries"

    message_id: Mapped[str] = mapped_column(
        ForeignKey(
            "conversation_messages.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
        index=True,
    )

    failure_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    normalized_query: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    predicted_intent: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
        index=True,
    )

    confidence: Mapped[Decimal | None] = mapped_column(
        Numeric(CONFIDENCE_PRECISION, CONFIDENCE_SCALE),
        nullable=True,
    )

    reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    review_status: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=TrainingReviewStatus.PENDING.value,
        index=True,
    )

    message: Mapped["ConversationMessage"] = relationship(
        back_populates="failed_query",
    )


class KnowledgeGap(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "knowledge_gaps"

    business_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "businesses.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    gap_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    canonical_question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    example_queries_json: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    occurrence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    priority_score: Mapped[Decimal | None] = mapped_column(
        Numeric(CONFIDENCE_PRECISION, CONFIDENCE_SCALE),
        nullable=True,
    )

    resolved: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    resolution_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


# ============================================================
# SECTION 19 - PRIVATE EVENT INQUIRIES
# ============================================================

class PrivateEventInquiry(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "private_event_inquiries"

    business_id: Mapped[str] = mapped_column(
        ForeignKey(
            "businesses.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "conversations.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    lead_status: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=LeadStatus.NEW.value,
        index=True,
    )

    customer_name: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    email: Mapped[str | None] = mapped_column(
        String(EMAIL_LENGTH),
        nullable=True,
    )

    phone: Mapped[str | None] = mapped_column(
        String(PHONE_LENGTH),
        nullable=True,
    )

    company_name: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    event_type: Mapped[str | None] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=True,
        index=True,
    )

    preferred_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    alternate_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    start_time: Mapped[time | None] = mapped_column(
        Time,
        nullable=True,
    )

    end_time: Mapped[time | None] = mapped_column(
        Time,
        nullable=True,
    )

    guest_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    budget_min: Mapped[Decimal | None] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE),
        nullable=True,
    )

    budget_max: Mapped[Decimal | None] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE),
        nullable=True,
    )

    estimated_value: Mapped[Decimal | None] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE),
        nullable=True,
    )

    space_preference: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    food_package: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    bar_package: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    entertainment_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    av_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    dietary_requirements: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    customer_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    lead_score: Mapped[Decimal | None] = mapped_column(
        Numeric(CONFIDENCE_PRECISION, CONFIDENCE_SCALE),
        nullable=True,
        index=True,
    )

    source_campaign: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
        index=True,
    )

    source_page_url: Mapped[str | None] = mapped_column(
        String(URL_LENGTH),
        nullable=True,
    )

    email_consent_status: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=ConsentStatus.NOT_REQUESTED.value,
    )

    sms_consent_status: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default=ConsentStatus.NOT_REQUESTED.value,
    )

    assigned_to: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    contacted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    booked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    management_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    business: Mapped["Business"] = relationship(
        back_populates="private_event_inquiries",
    )

    conversation: Mapped["Conversation | None"] = relationship(
        back_populates="private_event_inquiries",
    )

    __repr_exclude__ = {
        "email",
        "phone",
        "customer_message",
        "management_notes",
    }

    __serialize_exclude__ = {
        "email",
        "phone",
        "customer_message",
        "management_notes",
    }

    __table_args__ = (
        CheckConstraint(
            "guest_count IS NULL OR guest_count > 0",
            name="private_event_inquiries_positive_guests",
        ),
        CheckConstraint(
            "budget_min IS NULL OR budget_min >= 0",
            name="private_event_inquiries_nonnegative_budget_min",
        ),
        CheckConstraint(
            "budget_max IS NULL OR budget_max >= 0",
            name="private_event_inquiries_nonnegative_budget_max",
        ),
    )


# ============================================================
# SECTION 20 - ANALYTICS EVENTS
# ============================================================

class AnalyticsEvent(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "analytics_events"

    business_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "businesses.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    browser_session_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "browser_sessions.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "conversations.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    event_name: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    event_category: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default="chatbot",
        index=True,
    )

    page_url: Mapped[str | None] = mapped_column(
        String(URL_LENGTH),
        nullable=True,
    )

    page_category: Mapped[str | None] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=True,
    )

    event_value: Mapped[Decimal | None] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE),
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    conversation: Mapped["Conversation | None"] = relationship(
        back_populates="analytics_events",
    )

    __table_args__ = (
        Index(
            "ix_analytics_events_name_occurred",
            "event_name",
            "occurred_at",
        ),
    )


# ============================================================
# SECTION 21 - AUDIT EVENTS
# ============================================================

class AuditEvent(
    StandardRecordMixin,
    Base,
):
    __tablename__ = "audit_events"

    event_name: Mapped[str] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=False,
        index=True,
    )

    actor_type: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
    )

    actor_id: Mapped[str | None] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=True,
        index=True,
    )

    target_type: Mapped[str | None] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=True,
    )

    target_id: Mapped[str | None] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=True,
        index=True,
    )

    action: Mapped[str | None] = mapped_column(
        String(MEDIUM_TEXT_LENGTH),
        nullable=True,
    )

    outcome: Mapped[str] = mapped_column(
        String(SHORT_TEXT_LENGTH),
        nullable=False,
        default="success",
        index=True,
    )

    request_id: Mapped[str | None] = mapped_column(
        String(IDENTIFIER_LENGTH),
        nullable=True,
        index=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )


# ============================================================
# SECTION 22 - MODEL INVENTORY
# ============================================================

ALL_MODELS: Final[tuple[type[Base], ...]] = (
    Business,
    BusinessContact,
    BusinessHour,
    KnowledgeDocument,
    KnowledgeChunk,
    MenuCategory,
    MenuItem,
    BusinessEvent,
    FAQEntry,
    PrivateEventPackage,
    BrowserSession,
    WidgetState,
    Conversation,
    ConversationMessage,
    ResponseTemplate,
    ResponseVariant,
    SpellingVariant,
    PhraseCluster,
    UserFeedback,
    TrainingExample,
    TrainingExampleReview,
    ModelVersion,
    ModelEvaluation,
    ModelDeployment,
    LearningJob,
    DriftReport,
    PoisoningReport,
    FailedQuery,
    KnowledgeGap,
    PrivateEventInquiry,
    AnalyticsEvent,
    AuditEvent,
)


# ============================================================
# SECTION 23 - SCHEMA VALIDATION
# ============================================================

def collect_model_schema_inventory() -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []

    for model in ALL_MODELS:
        inventory.append(
            {
                "model": model.__name__,
                "table": model.__tablename__,
                "columns": list(model.column_names()),
                "primary_keys": list(model.primary_key_names()),
                "relationships": list(model.relationship_names()),
            }
        )

    return inventory


def validate_database_models() -> dict[str, Any]:
    required_tables = {
        "businesses",
        "business_hours",
        "knowledge_documents",
        "knowledge_chunks",
        "menu_categories",
        "menu_items",
        "business_events",
        "faq_entries",
        "private_event_packages",
        "browser_sessions",
        "widget_states",
        "conversations",
        "conversation_messages",
        "response_templates",
        "response_variants",
        "spelling_variants",
        "phrase_clusters",
        "user_feedback",
        "training_examples",
        "training_example_reviews",
        "model_versions",
        "model_evaluations",
        "model_deployments",
        "learning_jobs",
        "drift_reports",
        "poisoning_reports",
        "failed_queries",
        "knowledge_gaps",
        "private_event_inquiries",
        "analytics_events",
        "audit_events",
    }

    actual_tables = {
        model.__tablename__
        for model in ALL_MODELS
    }

    duplicate_tables = {
        table
        for table in actual_tables
        if sum(
            1
            for model in ALL_MODELS
            if model.__tablename__ == table
        ) > 1
    }

    missing_tables = sorted(
        required_tables - actual_tables
    )

    checks = {
        "all_required_tables_present": not missing_tables,
        "no_duplicate_model_tables": not duplicate_tables,
        "business_relationships_present": {
            "hours",
            "knowledge_documents",
            "menu_categories",
            "events",
            "private_event_packages",
            "conversations",
        }.issubset(
            set(Business.relationship_names())
        ),
        "conversation_messages_present": (
            "messages"
            in Conversation.relationship_names()
        ),
        "message_learning_relationship_present": (
            "training_examples"
            in ConversationMessage.relationship_names()
        ),
        "private_event_contact_fields_excluded": {
            "email",
            "phone",
            "customer_message",
            "management_notes",
        }.issubset(
            PrivateEventInquiry.__serialize_exclude__
        ),
        "model_registry_relationships_present": {
            "evaluations",
            "deployments",
        }.issubset(
            set(ModelVersion.relationship_names())
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    return {
        "status": "ok" if not failed_checks else "failed",
        "model_count": len(ALL_MODELS),
        "table_count": len(actual_tables),
        "checks": checks,
        "failed_checks": failed_checks,
        "missing_tables": missing_tables,
        "duplicate_tables": sorted(duplicate_tables),
        "inventory": collect_model_schema_inventory(),
    }


# ============================================================
# SECTION 24 - SAFE SAMPLE CREATION
# ============================================================

def build_sample_business() -> Business:
    return Business(
        slug="horseshoe-tavern",
        display_name="Horseshoe Tavern",
        website_url="https://www.thehorseshoetavern.com/",
        timezone="America/New_York",
        source_type="official_website",
        source_name="Horseshoe Tavern official website",
        source_url="https://www.thehorseshoetavern.com/",
    )


def build_sample_conversation(
    business: Business,
    browser_session: BrowserSession,
) -> Conversation:
    return Conversation(
        business=business,
        browser_session=browser_session,
        external_conversation_id=(
            f"conversation_{new_uuid_string()}"
        ),
        status=ConversationStatus.ACTIVE.value,
        channel="website_widget",
    )


# ============================================================
# SECTION 25 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    report = validate_database_models()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
