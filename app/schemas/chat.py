# ============================================================
# Exact file location: app/schemas/chat.py
# Horseshoe Tavern AI
# Phase 1 Part 1.11
# Validated chat, widget, conversation, feedback, source,
# action, error, and restoration API contracts
# ============================================================

"""
Pydantic API contracts for the Horseshoe Tavern AI chatbot.

This module defines validated request and response schemas for:

- Website page context
- Widget state and size
- Chat requests
- Chat responses
- Conversation restoration
- Message history
- Response actions
- Source attribution
- Intent and answer confidence
- Spelling corrections
- Detected entities
- Human handoff
- Private-event draft state
- User feedback
- Safe API errors
- Health and diagnostics metadata

The schemas are transport contracts only. They do not contain business
logic, database queries, model inference, or verified business facts.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    computed_field,
    field_validator,
    model_validator,
)


# ============================================================
# SECTION 01 - CONSTANTS
# ============================================================

MAX_SESSION_ID_LENGTH = 160
MAX_CONVERSATION_ID_LENGTH = 160
MAX_MESSAGE_ID_LENGTH = 160
MAX_MESSAGE_CHARACTERS = 3000
MAX_RESPONSE_CHARACTERS = 12000
MAX_PAGE_URL_LENGTH = 2048
MAX_PAGE_TITLE_LENGTH = 300
MAX_PAGE_PATH_LENGTH = 1000
MAX_CATEGORY_LENGTH = 100
MAX_LANGUAGE_LENGTH = 32
MAX_ACTION_LABEL_LENGTH = 120
MAX_ACTION_URL_LENGTH = 2048
MAX_SOURCE_NAME_LENGTH = 255
MAX_SOURCE_REFERENCE_LENGTH = 500
MAX_ERROR_MESSAGE_LENGTH = 1000
MAX_FEEDBACK_CHARACTERS = 4000
MAX_ENTITY_VALUE_LENGTH = 1000
MAX_MODEL_VERSION_LENGTH = 128
MAX_PRIVATE_EVENT_NOTES_LENGTH = 5000
MAX_HISTORY_MESSAGES = 200

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}$"
)

LANGUAGE_PATTERN = re.compile(
    r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})?$"
)

SAFE_CATEGORY_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9_-]{0,99}$"
)


# ============================================================
# SECTION 02 - ENUMERATIONS
# ============================================================

class WidgetState(str, Enum):
    COLLAPSED = "collapsed"
    MINIMIZED = "minimized"
    OPEN = "open"


class WidgetSize(str, Enum):
    COMPACT = "compact"
    EXPANDED = "expanded"
    FULLSCREEN = "fullscreen"


class ChatMessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessageStatus(str, Enum):
    RECEIVED = "received"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class ResponseActionType(str, Enum):
    MESSAGE = "message"
    LINK = "link"
    PHONE = "phone"
    EMAIL = "email"
    FORM = "form"
    NAVIGATION = "navigation"
    HUMAN_HANDOFF = "human_handoff"
    PRIVATE_EVENT = "private_event"
    ORDERING = "ordering"


class SourceTrustLevel(str, Enum):
    VERIFIED = "verified"
    OFFICIAL = "official"
    REVIEWED = "reviewed"
    UNVERIFIED = "unverified"
    STALE = "stale"
    UNKNOWN = "unknown"


class FeedbackType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    REPORT = "report"
    CONVERSION = "conversion"
    NEUTRAL = "neutral"


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "validation_error"
    RATE_LIMITED = "rate_limited"
    UNAUTHORIZED_ORIGIN = "unauthorized_origin"
    SESSION_NOT_FOUND = "session_not_found"
    CONVERSATION_NOT_FOUND = "conversation_not_found"
    MESSAGE_TOO_LONG = "message_too_long"
    KNOWLEDGE_UNAVAILABLE = "knowledge_unavailable"
    HUMAN_HANDOFF_REQUIRED = "human_handoff_required"
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"


# ============================================================
# SECTION 03 - GENERAL VALIDATION HELPERS
# ============================================================

def clean_single_line_text(
    value: Any,
    *,
    maximum_length: int | None = None,
) -> str:
    if value is None:
        return ""

    cleaned = (
        str(value)
        .replace("\x00", "")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )

    cleaned = " ".join(cleaned.split())

    if maximum_length is not None:
        cleaned = cleaned[:maximum_length]

    return cleaned


def clean_multiline_text(
    value: Any,
    *,
    maximum_length: int | None = None,
) -> str:
    if value is None:
        return ""

    cleaned = (
        str(value)
        .replace("\x00", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )

    if maximum_length is not None:
        cleaned = cleaned[:maximum_length]

    return cleaned


def validate_identifier(
    value: str,
    *,
    field_name: str,
) -> str:
    candidate = clean_single_line_text(value)

    if not IDENTIFIER_PATTERN.fullmatch(candidate):
        raise ValueError(
            f"{field_name} contains unsupported characters or length."
        )

    return candidate


def normalize_category(value: Any) -> str:
    candidate = clean_single_line_text(
        value,
        maximum_length=MAX_CATEGORY_LENGTH,
    ).lower()

    candidate = candidate.replace(" ", "_").replace("-", "_")
    candidate = re.sub(r"_+", "_", candidate).strip("_")

    if not candidate:
        return "general"

    if not SAFE_CATEGORY_PATTERN.fullmatch(candidate):
        raise ValueError("Category contains unsupported characters.")

    return candidate


def normalize_language(value: Any) -> str:
    candidate = clean_single_line_text(
        value,
        maximum_length=MAX_LANGUAGE_LENGTH,
    )

    if not candidate:
        return "en-US"

    if not LANGUAGE_PATTERN.fullmatch(candidate):
        raise ValueError("Invalid language identifier.")

    parts = candidate.split("-", maxsplit=1)

    if len(parts) == 1:
        return parts[0].lower()

    return f"{parts[0].lower()}-{parts[1].upper()}"


def normalize_url_or_empty(value: Any) -> str:
    candidate = clean_single_line_text(
        value,
        maximum_length=MAX_PAGE_URL_LENGTH,
    )

    if not candidate:
        return ""

    parsed = urlparse(candidate)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are allowed.")

    if not parsed.netloc:
        raise ValueError("URL hostname is missing.")

    return candidate


# ============================================================
# SECTION 04 - BASE SCHEMA
# ============================================================

class APIModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )


# ============================================================
# SECTION 05 - PAGE CONTEXT
# ============================================================

class ViewportContext(APIModel):
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

    device_pixel_ratio: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
    )


class UTMContext(APIModel):
    source: str | None = Field(
        default=None,
        max_length=255,
    )

    medium: str | None = Field(
        default=None,
        max_length=255,
    )

    campaign: str | None = Field(
        default=None,
        max_length=255,
    )

    term: str | None = Field(
        default=None,
        max_length=255,
    )

    content: str | None = Field(
        default=None,
        max_length=255,
    )


class PageContext(APIModel):
    url: str = Field(
        default="",
        max_length=MAX_PAGE_URL_LENGTH,
    )

    path: str = Field(
        default="/",
        max_length=MAX_PAGE_PATH_LENGTH,
    )

    title: str = Field(
        default="",
        max_length=MAX_PAGE_TITLE_LENGTH,
    )

    category: str = Field(
        default="home",
        max_length=MAX_CATEGORY_LENGTH,
    )

    referrer: str = Field(
        default="",
        max_length=MAX_PAGE_URL_LENGTH,
    )

    language: str = Field(
        default="en-US",
        max_length=MAX_LANGUAGE_LENGTH,
    )

    viewport: ViewportContext = Field(
        default_factory=ViewportContext,
    )

    utm: UTMContext = Field(
        default_factory=UTMContext,
    )

    timestamp: datetime | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("url", "referrer", mode="before")
    @classmethod
    def validate_urls(cls, value: Any) -> str:
        return normalize_url_or_empty(value)

    @field_validator("path", mode="before")
    @classmethod
    def clean_path(cls, value: Any) -> str:
        candidate = clean_single_line_text(
            value,
            maximum_length=MAX_PAGE_PATH_LENGTH,
        )

        if not candidate:
            return "/"

        return candidate if candidate.startswith("/") else f"/{candidate}"

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, value: Any) -> str:
        return clean_single_line_text(
            value,
            maximum_length=MAX_PAGE_TITLE_LENGTH,
        )

    @field_validator("category", mode="before")
    @classmethod
    def clean_category(cls, value: Any) -> str:
        return normalize_category(value)

    @field_validator("language", mode="before")
    @classmethod
    def clean_language(cls, value: Any) -> str:
        return normalize_language(value)


# ============================================================
# SECTION 06 - WIDGET CONTEXT
# ============================================================

class WidgetContext(APIModel):
    version: str = Field(
        default="1.0.0",
        max_length=50,
    )

    state: WidgetState = WidgetState.COLLAPSED

    size: WidgetSize = WidgetSize.COMPACT

    previous_page_url: str | None = Field(
        default=None,
        max_length=MAX_PAGE_URL_LENGTH,
    )

    last_active_at: datetime | None = None

    unread_count: int = Field(
        default=0,
        ge=0,
        le=999,
    )

    private_event_draft: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("previous_page_url", mode="before")
    @classmethod
    def validate_previous_page_url(
        cls,
        value: Any,
    ) -> str | None:
        if value in {None, ""}:
            return None

        return normalize_url_or_empty(value)


# ============================================================
# SECTION 07 - SPELLING AND ENTITY SCHEMAS
# ============================================================

class SpellingCorrection(APIModel):
    original: str = Field(
        min_length=1,
        max_length=255,
    )

    corrected: str = Field(
        min_length=1,
        max_length=255,
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    method: str = Field(
        default="unknown",
        max_length=100,
    )

    approved_mapping: bool = False

    @model_validator(mode="after")
    def validate_change(self) -> "SpellingCorrection":
        if self.original.strip().lower() == self.corrected.strip().lower():
            raise ValueError(
                "Spelling correction must change the original value."
            )

        return self


class DetectedEntity(APIModel):
    entity_type: str = Field(
        min_length=1,
        max_length=100,
    )

    value: str = Field(
        min_length=1,
        max_length=MAX_ENTITY_VALUE_LENGTH,
    )

    normalized_value: str | None = Field(
        default=None,
        max_length=MAX_ENTITY_VALUE_LENGTH,
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    start_index: int | None = Field(
        default=None,
        ge=0,
    )

    end_index: int | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_offsets(self) -> "DetectedEntity":
        if (
            self.start_index is not None
            and self.end_index is not None
            and self.end_index < self.start_index
        ):
            raise ValueError(
                "Entity end_index cannot precede start_index."
            )

        return self


# ============================================================
# SECTION 08 - SOURCE ATTRIBUTION
# ============================================================

class ResponseSource(APIModel):
    source_type: str = Field(
        min_length=1,
        max_length=100,
    )

    source_name: str = Field(
        min_length=1,
        max_length=MAX_SOURCE_NAME_LENGTH,
    )

    source_reference: str | None = Field(
        default=None,
        max_length=MAX_SOURCE_REFERENCE_LENGTH,
    )

    source_url: str | None = Field(
        default=None,
        max_length=MAX_ACTION_URL_LENGTH,
    )

    trust_level: SourceTrustLevel = SourceTrustLevel.UNKNOWN

    verified: bool = False

    retrieved_at: datetime | None = None

    source_updated_at: datetime | None = None

    relevance_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("source_url", mode="before")
    @classmethod
    def validate_source_url(
        cls,
        value: Any,
    ) -> str | None:
        if value in {None, ""}:
            return None

        return normalize_url_or_empty(value)


# ============================================================
# SECTION 09 - RESPONSE ACTIONS
# ============================================================

class ResponseAction(APIModel):
    action_type: ResponseActionType

    label: str = Field(
        min_length=1,
        max_length=MAX_ACTION_LABEL_LENGTH,
    )

    message: str | None = Field(
        default=None,
        max_length=MAX_MESSAGE_CHARACTERS,
    )

    url: str | None = Field(
        default=None,
        max_length=MAX_ACTION_URL_LENGTH,
    )

    target: Literal[
        "_self",
        "_blank",
        "_parent",
        "_top",
    ] = "_self"

    phone_number: str | None = Field(
        default=None,
        max_length=64,
    )

    email_address: str | None = Field(
        default=None,
        max_length=320,
    )

    form_key: str | None = Field(
        default=None,
        max_length=100,
    )

    analytics_event: str | None = Field(
        default=None,
        max_length=255,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("label", mode="before")
    @classmethod
    def clean_label(cls, value: Any) -> str:
        candidate = clean_single_line_text(
            value,
            maximum_length=MAX_ACTION_LABEL_LENGTH,
        )

        if not candidate:
            raise ValueError("Action label cannot be empty.")

        return candidate

    @field_validator("message", mode="before")
    @classmethod
    def clean_message(
        cls,
        value: Any,
    ) -> str | None:
        if value in {None, ""}:
            return None

        return clean_multiline_text(
            value,
            maximum_length=MAX_MESSAGE_CHARACTERS,
        )

    @field_validator("url", mode="before")
    @classmethod
    def validate_action_url(
        cls,
        value: Any,
    ) -> str | None:
        if value in {None, ""}:
            return None

        return normalize_url_or_empty(value)

    @model_validator(mode="after")
    def validate_action_payload(self) -> "ResponseAction":
        if self.action_type in {
            ResponseActionType.LINK,
            ResponseActionType.NAVIGATION,
            ResponseActionType.ORDERING,
        } and not self.url:
            raise ValueError(
                f"{self.action_type} action requires a URL."
            )

        if (
            self.action_type == ResponseActionType.MESSAGE
            and not self.message
        ):
            raise ValueError(
                "Message action requires message text."
            )

        if (
            self.action_type == ResponseActionType.PHONE
            and not self.phone_number
        ):
            raise ValueError(
                "Phone action requires a phone number."
            )

        if (
            self.action_type == ResponseActionType.EMAIL
            and not self.email_address
        ):
            raise ValueError(
                "Email action requires an email address."
            )

        if (
            self.action_type == ResponseActionType.FORM
            and not self.form_key
        ):
            raise ValueError(
                "Form action requires a form key."
            )

        return self


# ============================================================
# SECTION 10 - CONFIDENCE AND VALIDATION
# ============================================================

class ConfidenceBreakdown(APIModel):
    intent: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    entity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    retrieval: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    answer: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    factuality: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    overall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class ResponseValidation(APIModel):
    verified_business_facts_only: bool = True

    source_count: int = Field(
        default=0,
        ge=0,
    )

    stale_source_count: int = Field(
        default=0,
        ge=0,
    )

    unsupported_claim_count: int = Field(
        default=0,
        ge=0,
    )

    hallucination_guard_passed: bool = True

    privacy_guard_passed: bool = True

    safety_guard_passed: bool = True

    requires_human_review: bool = False

    warnings: list[str] = Field(
        default_factory=list,
    )

    @computed_field
    @property
    def passed(self) -> bool:
        return (
            self.unsupported_claim_count == 0
            and self.hallucination_guard_passed
            and self.privacy_guard_passed
            and self.safety_guard_passed
        )


# ============================================================
# SECTION 11 - PRIVATE EVENT DRAFT
# ============================================================

class PrivateEventDraft(APIModel):
    event_type: str | None = Field(
        default=None,
        max_length=100,
    )

    preferred_date: date | None = None

    alternate_date: date | None = None

    start_time: time | None = None

    end_time: time | None = None

    guest_count: int | None = Field(
        default=None,
        ge=1,
        le=5000,
    )

    budget_min: Decimal | None = Field(
        default=None,
        ge=0,
    )

    budget_max: Decimal | None = Field(
        default=None,
        ge=0,
    )

    customer_name: str | None = Field(
        default=None,
        max_length=255,
    )

    email: str | None = Field(
        default=None,
        max_length=320,
    )

    phone: str | None = Field(
        default=None,
        max_length=64,
    )

    company_name: str | None = Field(
        default=None,
        max_length=255,
    )

    space_preference: str | None = Field(
        default=None,
        max_length=255,
    )

    food_package: str | None = Field(
        default=None,
        max_length=255,
    )

    bar_package: str | None = Field(
        default=None,
        max_length=255,
    )

    dietary_requirements: str | None = Field(
        default=None,
        max_length=MAX_PRIVATE_EVENT_NOTES_LENGTH,
    )

    notes: str | None = Field(
        default=None,
        max_length=MAX_PRIVATE_EVENT_NOTES_LENGTH,
    )

    completed_fields: list[str] = Field(
        default_factory=list,
    )

    missing_fields: list[str] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_budget(self) -> "PrivateEventDraft":
        if (
            self.budget_min is not None
            and self.budget_max is not None
            and self.budget_max < self.budget_min
        ):
            raise ValueError(
                "budget_max cannot be less than budget_min."
            )

        return self


# ============================================================
# SECTION 12 - CHAT REQUEST
# ============================================================

class ChatRequest(APIModel):
    session_id: str = Field(
        min_length=8,
        max_length=MAX_SESSION_ID_LENGTH,
    )

    conversation_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=MAX_CONVERSATION_ID_LENGTH,
    )

    message: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_CHARACTERS,
    )

    business_slug: str = Field(
        default="horseshoe-tavern",
        min_length=2,
        max_length=100,
    )

    page_context: PageContext = Field(
        default_factory=PageContext,
    )

    widget_context: WidgetContext = Field(
        default_factory=WidgetContext,
    )

    client_message_id: str | None = Field(
        default=None,
        max_length=MAX_MESSAGE_ID_LENGTH,
    )

    client_timestamp: datetime | None = None

    requested_language: str | None = Field(
        default=None,
        max_length=MAX_LANGUAGE_LENGTH,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return validate_identifier(
            value,
            field_name="session_id",
        )

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return validate_identifier(
            value,
            field_name="conversation_id",
        )

    @field_validator("client_message_id")
    @classmethod
    def validate_client_message_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return validate_identifier(
            value,
            field_name="client_message_id",
        )

    @field_validator("message", mode="before")
    @classmethod
    def clean_user_message(cls, value: Any) -> str:
        candidate = clean_multiline_text(
            value,
            maximum_length=MAX_MESSAGE_CHARACTERS,
        )

        if not candidate:
            raise ValueError("Message cannot be empty.")

        return candidate

    @field_validator("business_slug", mode="before")
    @classmethod
    def normalize_business_slug(
        cls,
        value: Any,
    ) -> str:
        candidate = clean_single_line_text(
            value,
            maximum_length=100,
        ).lower()

        candidate = candidate.replace("_", "-").replace(" ", "-")
        candidate = re.sub(r"-+", "-", candidate).strip("-")

        if not candidate:
            raise ValueError("Business slug cannot be empty.")

        return candidate

    @field_validator("requested_language", mode="before")
    @classmethod
    def normalize_requested_language(
        cls,
        value: Any,
    ) -> str | None:
        if value in {None, ""}:
            return None

        return normalize_language(value)


# ============================================================
# SECTION 13 - MESSAGE HISTORY
# ============================================================

class ChatMessage(APIModel):
    id: str = Field(
        min_length=8,
        max_length=MAX_MESSAGE_ID_LENGTH,
    )

    conversation_id: str | None = Field(
        default=None,
        max_length=MAX_CONVERSATION_ID_LENGTH,
    )

    sequence_number: int | None = Field(
        default=None,
        ge=1,
    )

    role: ChatMessageRole

    status: ChatMessageStatus = ChatMessageStatus.COMPLETED

    text: str = Field(
        min_length=1,
        max_length=MAX_RESPONSE_CHARACTERS,
    )

    original_text: str | None = Field(
        default=None,
        max_length=MAX_RESPONSE_CHARACTERS,
    )

    normalized_text: str | None = Field(
        default=None,
        max_length=MAX_RESPONSE_CHARACTERS,
    )

    corrected_text: str | None = Field(
        default=None,
        max_length=MAX_RESPONSE_CHARACTERS,
    )

    created_at: datetime

    page_url: str | None = Field(
        default=None,
        max_length=MAX_PAGE_URL_LENGTH,
    )

    page_category: str | None = Field(
        default=None,
        max_length=MAX_CATEGORY_LENGTH,
    )

    detected_intent: str | None = Field(
        default=None,
        max_length=255,
    )

    spelling_corrections: list[SpellingCorrection] = Field(
        default_factory=list,
    )

    entities: list[DetectedEntity] = Field(
        default_factory=list,
    )

    sources: list[ResponseSource] = Field(
        default_factory=list,
    )

    actions: list[ResponseAction] = Field(
        default_factory=list,
    )

    confidence: ConfidenceBreakdown = Field(
        default_factory=ConfidenceBreakdown,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# SECTION 14 - CHAT RESPONSE
# ============================================================

class ChatResponse(APIModel):
    request_id: str = Field(
        min_length=8,
        max_length=160,
    )

    session_id: str = Field(
        min_length=8,
        max_length=MAX_SESSION_ID_LENGTH,
    )

    conversation_id: str = Field(
        min_length=8,
        max_length=MAX_CONVERSATION_ID_LENGTH,
    )

    message_id: str = Field(
        min_length=8,
        max_length=MAX_MESSAGE_ID_LENGTH,
    )

    message: str = Field(
        min_length=1,
        max_length=MAX_RESPONSE_CHARACTERS,
    )

    detected_intent: str | None = Field(
        default=None,
        max_length=255,
    )

    normalized_message: str | None = Field(
        default=None,
        max_length=MAX_MESSAGE_CHARACTERS,
    )

    corrected_message: str | None = Field(
        default=None,
        max_length=MAX_MESSAGE_CHARACTERS,
    )

    language: str = Field(
        default="en-US",
        max_length=MAX_LANGUAGE_LENGTH,
    )

    confidence: ConfidenceBreakdown = Field(
        default_factory=ConfidenceBreakdown,
    )

    spelling_corrections: list[SpellingCorrection] = Field(
        default_factory=list,
    )

    entities: list[DetectedEntity] = Field(
        default_factory=list,
    )

    sources: list[ResponseSource] = Field(
        default_factory=list,
    )

    actions: list[ResponseAction] = Field(
        default_factory=list,
    )

    validation: ResponseValidation = Field(
        default_factory=ResponseValidation,
    )

    widget_state: WidgetState = WidgetState.OPEN

    widget_size: WidgetSize = WidgetSize.COMPACT

    private_event_draft: PrivateEventDraft | None = None

    human_handoff_available: bool = False

    human_handoff_required: bool = False

    response_template_id: str | None = Field(
        default=None,
        max_length=MAX_MESSAGE_ID_LENGTH,
    )

    response_variant_id: str | None = Field(
        default=None,
        max_length=MAX_MESSAGE_ID_LENGTH,
    )

    model_versions: dict[str, str] = Field(
        default_factory=dict,
    )

    processing_time_ms: int = Field(
        default=0,
        ge=0,
    )

    created_at: datetime

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator(
        "request_id",
        "session_id",
        "conversation_id",
        "message_id",
    )
    @classmethod
    def validate_required_identifiers(
        cls,
        value: str,
        info: Any,
    ) -> str:
        return validate_identifier(
            value,
            field_name=info.field_name,
        )

    @field_validator("message", mode="before")
    @classmethod
    def clean_response_message(cls, value: Any) -> str:
        candidate = clean_multiline_text(
            value,
            maximum_length=MAX_RESPONSE_CHARACTERS,
        )

        if not candidate:
            raise ValueError(
                "Assistant response cannot be empty."
            )

        return candidate

    @field_validator("language", mode="before")
    @classmethod
    def validate_response_language(cls, value: Any) -> str:
        return normalize_language(value)

    @model_validator(mode="after")
    def validate_handoff_state(self) -> "ChatResponse":
        if (
            self.human_handoff_required
            and not self.human_handoff_available
        ):
            raise ValueError(
                "Required human handoff must also be available."
            )

        return self


# ============================================================
# SECTION 15 - CONVERSATION RESTORATION
# ============================================================

class ConversationRestoreRequest(APIModel):
    session_id: str = Field(
        min_length=8,
        max_length=MAX_SESSION_ID_LENGTH,
    )

    conversation_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=MAX_CONVERSATION_ID_LENGTH,
    )

    page_context: PageContext = Field(
        default_factory=PageContext,
    )

    limit: int = Field(
        default=100,
        ge=1,
        le=MAX_HISTORY_MESSAGES,
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return validate_identifier(
            value,
            field_name="session_id",
        )

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return validate_identifier(
            value,
            field_name="conversation_id",
        )


class ConversationRestoreResponse(APIModel):
    session_id: str

    conversation_id: str

    restored: bool

    messages: list[ChatMessage] = Field(
        default_factory=list,
    )

    widget_state: WidgetState = WidgetState.COLLAPSED

    widget_size: WidgetSize = WidgetSize.COMPACT

    unread_count: int = Field(
        default=0,
        ge=0,
        le=999,
    )

    private_event_draft: PrivateEventDraft | None = None

    current_intent: str | None = Field(
        default=None,
        max_length=255,
    )

    conversation_status: str = Field(
        default="active",
        max_length=100,
    )

    server_time: datetime

    persistence: Literal[
        "database",
        "temporary_memory",
        "browser_only",
    ] = "database"


# ============================================================
# SECTION 16 - FEEDBACK
# ============================================================

class ChatFeedbackRequest(APIModel):
    session_id: str = Field(
        min_length=8,
        max_length=MAX_SESSION_ID_LENGTH,
    )

    conversation_id: str | None = Field(
        default=None,
        max_length=MAX_CONVERSATION_ID_LENGTH,
    )

    message_id: str | None = Field(
        default=None,
        max_length=MAX_MESSAGE_ID_LENGTH,
    )

    feedback_type: FeedbackType

    rating: int | None = Field(
        default=None,
        ge=1,
        le=5,
    )

    feedback_text: str | None = Field(
        default=None,
        max_length=MAX_FEEDBACK_CHARACTERS,
    )

    suggested_intent: str | None = Field(
        default=None,
        max_length=255,
    )

    suggested_response: str | None = Field(
        default=None,
        max_length=MAX_RESPONSE_CHARACTERS,
    )

    correction: dict[str, Any] = Field(
        default_factory=dict,
    )

    page_context: PageContext = Field(
        default_factory=PageContext,
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return validate_identifier(
            value,
            field_name="session_id",
        )

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return validate_identifier(
            value,
            field_name="conversation_id",
        )

    @field_validator("message_id")
    @classmethod
    def validate_message_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return validate_identifier(
            value,
            field_name="message_id",
        )

    @field_validator(
        "feedback_text",
        "suggested_response",
        mode="before",
    )
    @classmethod
    def clean_feedback_text(
        cls,
        value: Any,
    ) -> str | None:
        if value in {None, ""}:
            return None

        return clean_multiline_text(
            value,
            maximum_length=MAX_RESPONSE_CHARACTERS,
        )

    @model_validator(mode="after")
    def validate_feedback_payload(self) -> "ChatFeedbackRequest":
        if (
            self.feedback_type == FeedbackType.CORRECTION
            and not any(
                (
                    self.feedback_text,
                    self.suggested_intent,
                    self.suggested_response,
                    self.correction,
                )
            )
        ):
            raise ValueError(
                "Correction feedback requires correction details."
            )

        return self


class ChatFeedbackResponse(APIModel):
    status: Literal["accepted", "rejected"] = "accepted"

    feedback_id: str | None = None

    review_required: bool = True

    training_candidate_created: bool = False

    message: str = (
        "Thank you. Your feedback has been recorded for review."
    )

    server_time: datetime


# ============================================================
# SECTION 17 - SAFE ERROR RESPONSES
# ============================================================

class ErrorDetail(APIModel):
    field: str | None = Field(
        default=None,
        max_length=255,
    )

    issue: str = Field(
        min_length=1,
        max_length=MAX_ERROR_MESSAGE_LENGTH,
    )

    rejected_value: Any | None = None


class APIErrorResponse(APIModel):
    status: Literal["error"] = "error"

    error_code: ErrorCode

    message: str = Field(
        min_length=1,
        max_length=MAX_ERROR_MESSAGE_LENGTH,
    )

    request_id: str | None = Field(
        default=None,
        max_length=160,
    )

    retryable: bool = False

    retry_after_seconds: int | None = Field(
        default=None,
        ge=0,
        le=86400,
    )

    details: list[ErrorDetail] = Field(
        default_factory=list,
    )

    support_reference: str | None = Field(
        default=None,
        max_length=160,
    )

    timestamp: datetime

    @model_validator(mode="after")
    def validate_retry_metadata(self) -> "APIErrorResponse":
        if (
            self.retry_after_seconds is not None
            and not self.retryable
        ):
            raise ValueError(
                "retry_after_seconds requires retryable=True."
            )

        return self


# ============================================================
# SECTION 18 - PUBLIC CONFIGURATION
# ============================================================

class ChatPublicConfiguration(APIModel):
    business_slug: str

    business_name: str

    maximum_message_characters: int = Field(
        ge=100,
        le=20000,
    )

    supported_widget_states: list[WidgetState]

    supported_widget_sizes: list[WidgetSize]

    feedback_enabled: bool = True

    conversation_persistence_enabled: bool = True

    response_sources_enabled: bool = True

    spelling_correction_enabled: bool = True

    human_handoff_enabled: bool = True

    private_events_enabled: bool = True

    default_language: str = "en-US"

    server_time: datetime


# ============================================================
# SECTION 19 - MODULE SELF-TEST
# ============================================================

def validate_chat_schemas() -> dict[str, Any]:
    now = datetime.now().astimezone()

    request = ChatRequest(
        session_id="session_12345678",
        conversation_id="conversation_12345678",
        message="wat time do u close",
        page_context=PageContext(
            url="https://www.thehorseshoetavern.com/menu",
            path="/menu",
            title="Menu",
            category="menu",
            language="en-us",
            viewport=ViewportContext(
                width=1440,
                height=900,
            ),
        ),
        widget_context=WidgetContext(
            state=WidgetState.OPEN,
            size=WidgetSize.EXPANDED,
        ),
    )

    correction = SpellingCorrection(
        original="wat",
        corrected="what",
        confidence=0.98,
        method="reviewed_dictionary",
        approved_mapping=True,
    )

    source = ResponseSource(
        source_type="business_hours",
        source_name="Verified Horseshoe Tavern hours",
        source_url="https://www.thehorseshoetavern.com/",
        trust_level=SourceTrustLevel.VERIFIED,
        verified=True,
        relevance_score=0.99,
    )

    action = ResponseAction(
        action_type=ResponseActionType.MESSAGE,
        label="Show today's hours",
        message="Show me today's hours.",
    )

    response = ChatResponse(
        request_id="request_12345678",
        session_id=request.session_id,
        conversation_id=request.conversation_id,
        message_id="message_12345678",
        message="I can help with today's verified tavern hours.",
        detected_intent="HOURS_GENERAL",
        normalized_message="what time do you close",
        corrected_message="what time do you close",
        language="en-US",
        confidence=ConfidenceBreakdown(
            intent=0.96,
            retrieval=0.99,
            answer=0.98,
            factuality=1.0,
            overall=0.98,
        ),
        spelling_corrections=[correction],
        sources=[source],
        actions=[action],
        validation=ResponseValidation(
            verified_business_facts_only=True,
            source_count=1,
        ),
        widget_state=WidgetState.OPEN,
        widget_size=WidgetSize.EXPANDED,
        processing_time_ms=42,
        created_at=now,
    )

    feedback = ChatFeedbackRequest(
        session_id=request.session_id,
        conversation_id=request.conversation_id,
        message_id=response.message_id,
        feedback_type=FeedbackType.POSITIVE,
        rating=5,
        page_context=request.page_context,
    )

    draft = PrivateEventDraft(
        event_type="birthday",
        guest_count=45,
        budget_min=Decimal("1500.00"),
        budget_max=Decimal("3000.00"),
    )

    checks = {
        "request_session_valid": (
            request.session_id == "session_12345678"
        ),
        "message_preserved": (
            request.message == "wat time do u close"
        ),
        "language_normalized": (
            request.page_context.language == "en-US"
        ),
        "widget_state_valid": (
            request.widget_context.state == "open"
        ),
        "widget_size_valid": (
            request.widget_context.size == "expanded"
        ),
        "correction_valid": (
            correction.corrected == "what"
        ),
        "verified_source_valid": (
            source.verified is True
        ),
        "response_valid": (
            response.validation.passed is True
        ),
        "feedback_valid": (
            feedback.rating == 5
        ),
        "private_event_budget_valid": (
            draft.budget_max >= draft.budget_min
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    return {
        "status": "ok" if not failed_checks else "failed",
        "checks": checks,
        "failed_checks": failed_checks,
        "sample_request": request.model_dump(
            mode="json"
        ),
        "sample_response": response.model_dump(
            mode="json"
        ),
        "sample_feedback": feedback.model_dump(
            mode="json"
        ),
        "sample_private_event_draft": draft.model_dump(
            mode="json"
        ),
    }


# ============================================================
# SECTION 20 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import json

    report = validate_chat_schemas()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
