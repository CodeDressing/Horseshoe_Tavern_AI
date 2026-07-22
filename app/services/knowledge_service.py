# ============================================================
# Exact file location: app/services/knowledge_service.py
# Horseshoe Tavern AI
# Phase 1 Part 1.18
# Verified business-fact retrieval, ranking, source attribution,
# freshness controls, ambiguity handling, and answer evidence
# ============================================================

"""
Verified knowledge retrieval service for Horseshoe Tavern AI.

This service connects NLU output to durable business knowledge.

Responsibilities:

- Retrieve only active and verified business records
- Resolve the business by stable slug
- Retrieve business hours by service type and requested day
- Retrieve menu categories and items
- Retrieve dietary and allergen information
- Retrieve upcoming public events
- Retrieve private-event packages
- Retrieve FAQs and verified knowledge documents
- Retrieve location, contact, parking, ordering, and accessibility facts
- Rank evidence against the corrected user message
- Produce structured source attribution
- Track freshness and stale-source warnings
- Return safe "knowledge unavailable" results when facts do not exist
- Keep public conversation text separate from verified business truth

This module does not directly generate the final conversational prose.
It returns verified evidence and answer guidance for the response service.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.database.models import (
    Business,
    BusinessContact,
    BusinessEvent,
    BusinessHour,
    FAQEntry,
    KnowledgeDocument,
    KnowledgeStatus,
    MenuCategory,
    MenuItem,
    PrivateEventPackage,
)
from app.database.repositories import (
    BusinessRepository,
    KnowledgeRepository,
)
from app.logging_config import get_logger
from app.nlu.entities import EntityType, ExtractedEntity
from app.nlu.intent import IntentName
from app.nlu.orchestrator import NLUResult


# ============================================================
# SECTION 01 - LOGGER AND CONSTANTS
# ============================================================

logger = get_logger(__name__)

KNOWLEDGE_SERVICE_VERSION: Final[str] = "1.0.0"
KNOWLEDGE_SERVICE_PHASE: Final[str] = "Phase 1 Part 1.18"

DEFAULT_BUSINESS_SLUG: Final[str] = "horseshoe-tavern"
DEFAULT_RESULT_LIMIT: Final[int] = 12
MAXIMUM_RESULT_LIMIT: Final[int] = 100
DEFAULT_STALE_AFTER_DAYS: Final[int] = 45
DEFAULT_MINIMUM_RELEVANCE: Final[float] = 0.20

WORD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[a-z0-9]+(?:[-'][a-z0-9]+)*",
    re.IGNORECASE,
)

STOP_WORDS: Final[set[str]] = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "at",
    "be",
    "can",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "show",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    "your",
}

INTENT_DOCUMENT_TYPES: Final[dict[IntentName, tuple[str, ...]]] = {
    IntentName.HOURS_GENERAL: (
        "hours",
        "business_hours",
        "operations",
    ),
    IntentName.HOURS_TODAY: (
        "hours",
        "business_hours",
        "operations",
    ),
    IntentName.HOURS_KITCHEN: (
        "hours",
        "kitchen_hours",
        "food_service",
    ),
    IntentName.HOURS_HAPPY_HOUR: (
        "hours",
        "happy_hour",
        "specials",
    ),
    IntentName.MENU_GENERAL: (
        "menu",
        "food",
        "beverage",
    ),
    IntentName.MENU_ITEM_LOOKUP: (
        "menu",
        "food",
        "beverage",
    ),
    IntentName.MENU_DIETARY: (
        "menu",
        "dietary",
        "allergen",
    ),
    IntentName.MENU_ALLERGEN: (
        "menu",
        "allergen",
        "dietary",
    ),
    IntentName.MENU_PRICE: (
        "menu",
        "pricing",
    ),
    IntentName.EVENTS_GENERAL: (
        "events",
        "entertainment",
    ),
    IntentName.EVENTS_TONIGHT: (
        "events",
        "entertainment",
    ),
    IntentName.LIVE_MUSIC: (
        "events",
        "live_music",
        "entertainment",
    ),
    IntentName.SPORTS_VIEWING: (
        "events",
        "sports",
        "television",
    ),
    IntentName.PRIVATE_EVENT: (
        "private_events",
        "events",
        "packages",
    ),
    IntentName.PRIVATE_EVENT_PRICING: (
        "private_events",
        "pricing",
        "packages",
    ),
    IntentName.PRIVATE_EVENT_AVAILABILITY: (
        "private_events",
        "availability",
    ),
    IntentName.PRIVATE_EVENT_CONTACT: (
        "private_events",
        "contact",
    ),
    IntentName.PARKING: (
        "parking",
        "location",
        "transportation",
    ),
    IntentName.LOCATION: (
        "location",
        "address",
    ),
    IntentName.DIRECTIONS: (
        "location",
        "directions",
        "transportation",
    ),
    IntentName.CONTACT: (
        "contact",
        "business",
    ),
    IntentName.ORDERING: (
        "ordering",
        "takeout",
        "delivery",
    ),
    IntentName.TAKEOUT: (
        "ordering",
        "takeout",
    ),
    IntentName.DELIVERY: (
        "ordering",
        "delivery",
    ),
    IntentName.ACCESSIBILITY: (
        "accessibility",
        "location",
    ),
    IntentName.JOBS: (
        "employment",
        "jobs",
    ),
    IntentName.LOST_AND_FOUND: (
        "lost_and_found",
        "contact",
    ),
}


# ============================================================
# SECTION 02 - ENUMERATIONS
# ============================================================

class KnowledgeRecordType(str, Enum):
    BUSINESS = "business"
    CONTACT = "contact"
    HOURS = "hours"
    MENU_CATEGORY = "menu_category"
    MENU_ITEM = "menu_item"
    EVENT = "event"
    FAQ = "faq"
    DOCUMENT = "document"
    PRIVATE_EVENT_PACKAGE = "private_event_package"


class KnowledgeTrustLevel(str, Enum):
    VERIFIED = "verified"
    VERIFIED_STALE = "verified_stale"
    REVIEWED = "reviewed"
    UNKNOWN = "unknown"


class KnowledgeDecision(str, Enum):
    FOUND = "found"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    BUSINESS_NOT_FOUND = "business_not_found"
    UNSUPPORTED_INTENT = "unsupported_intent"
    ERROR = "error"


# ============================================================
# SECTION 03 - DATA CLASSES
# ============================================================

@dataclass(frozen=True, slots=True)
class KnowledgeSource:
    source_type: str
    source_name: str
    source_reference: str | None
    source_url: str | None
    trust_level: KnowledgeTrustLevel
    verified: bool
    retrieved_at: datetime
    source_updated_at: datetime | None
    stale: bool
    relevance_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_name": self.source_name,
            "source_reference": self.source_reference,
            "source_url": self.source_url,
            "trust_level": self.trust_level.value,
            "verified": self.verified,
            "retrieved_at": self.retrieved_at.isoformat(),
            "source_updated_at": (
                self.source_updated_at.isoformat()
                if self.source_updated_at
                else None
            ),
            "stale": self.stale,
            "relevance_score": self.relevance_score,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class KnowledgeRecord:
    record_type: KnowledgeRecordType
    record_id: str
    title: str
    content: str
    structured_data: dict[str, Any]
    source: KnowledgeSource
    priority: int = 0
    relevance_score: float = 0.0
    confidence: float = 0.0
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type.value,
            "record_id": self.record_id,
            "title": self.title,
            "content": self.content,
            "structured_data": copy.deepcopy(
                self.structured_data
            ),
            "source": self.source.as_dict(),
            "priority": self.priority,
            "relevance_score": self.relevance_score,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class KnowledgeQuery:
    business_slug: str
    intent: IntentName
    text: str
    corrected_text: str
    page_category: str | None
    entities: tuple[ExtractedEntity, ...]
    requested_limit: int
    requested_date: date | None
    service_type: str | None
    category_slug: str | None
    menu_terms: tuple[str, ...]
    event_terms: tuple[str, ...]
    dietary_terms: tuple[str, ...]
    allergen_terms: tuple[str, ...]
    sports_terms: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "business_slug": self.business_slug,
            "intent": self.intent.value,
            "text": self.text,
            "corrected_text": self.corrected_text,
            "page_category": self.page_category,
            "entities": [
                entity.as_dict()
                for entity in self.entities
            ],
            "requested_limit": self.requested_limit,
            "requested_date": (
                self.requested_date.isoformat()
                if self.requested_date
                else None
            ),
            "service_type": self.service_type,
            "category_slug": self.category_slug,
            "menu_terms": list(self.menu_terms),
            "event_terms": list(self.event_terms),
            "dietary_terms": list(self.dietary_terms),
            "allergen_terms": list(self.allergen_terms),
            "sports_terms": list(self.sports_terms),
        }


@dataclass(frozen=True, slots=True)
class KnowledgeResult:
    decision: KnowledgeDecision
    business_id: str | None
    business_slug: str
    intent: IntentName
    records: tuple[KnowledgeRecord, ...]
    sources: tuple[KnowledgeSource, ...]
    query: KnowledgeQuery
    answer_guidance: str | None
    verified_fact_count: int
    stale_source_count: int
    unsupported_claim_count: int
    requires_human_review: bool
    warnings: tuple[str, ...]
    retrieved_at: datetime
    service_version: str
    service_phase: str

    @property
    def has_verified_knowledge(self) -> bool:
        return self.verified_fact_count > 0

    @property
    def is_complete(self) -> bool:
        return (
            self.decision == KnowledgeDecision.FOUND
            and self.has_verified_knowledge
            and self.unsupported_claim_count == 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "business_id": self.business_id,
            "business_slug": self.business_slug,
            "intent": self.intent.value,
            "records": [
                record.as_dict()
                for record in self.records
            ],
            "sources": [
                source.as_dict()
                for source in self.sources
            ],
            "query": self.query.as_dict(),
            "answer_guidance": self.answer_guidance,
            "verified_fact_count": self.verified_fact_count,
            "stale_source_count": self.stale_source_count,
            "unsupported_claim_count": self.unsupported_claim_count,
            "requires_human_review": self.requires_human_review,
            "warnings": list(self.warnings),
            "retrieved_at": self.retrieved_at.isoformat(),
            "service_version": self.service_version,
            "service_phase": self.service_phase,
            "has_verified_knowledge": self.has_verified_knowledge,
            "is_complete": self.is_complete,
        }


# ============================================================
# SECTION 04 - KNOWLEDGE SERVICE
# ============================================================

class KnowledgeService:
    """
    Verified business-knowledge retrieval service.
    """

    def __init__(
        self,
        session: Session,
        *,
        stale_after_days: int = DEFAULT_STALE_AFTER_DAYS,
        minimum_relevance: float = DEFAULT_MINIMUM_RELEVANCE,
    ) -> None:
        self.session = session
        self.business_repository = BusinessRepository(
            session
        )
        self.knowledge_repository = KnowledgeRepository(
            session
        )
        self.stale_after_days = max(
            1,
            int(stale_after_days),
        )
        self.minimum_relevance = min(
            max(
                float(minimum_relevance),
                0.0,
            ),
            1.0,
        )

    # ========================================================
    # SECTION 05 - PUBLIC RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        nlu_result: NLUResult,
        *,
        business_slug: str = DEFAULT_BUSINESS_SLUG,
        limit: int = DEFAULT_RESULT_LIMIT,
        now: datetime | None = None,
    ) -> KnowledgeResult:
        retrieved_at = (
            now
            or datetime.now().astimezone()
        )

        limit = min(
            max(int(limit), 1),
            MAXIMUM_RESULT_LIMIT,
        )

        query = self._build_query(
            nlu_result,
            business_slug=business_slug,
            limit=limit,
        )

        business = (
            self.business_repository
            .get_by_slug(
                business_slug
            )
        )

        if business is None:
            return self._empty_result(
                query=query,
                decision=(
                    KnowledgeDecision
                    .BUSINESS_NOT_FOUND
                ),
                retrieved_at=retrieved_at,
                warning=(
                    "The configured business record was not found."
                ),
            )

        records: list[KnowledgeRecord] = []
        warnings: list[str] = []

        try:
            records.extend(
                self._retrieve_for_intent(
                    business=business,
                    query=query,
                    retrieved_at=retrieved_at,
                    limit=limit,
                )
            )
        except Exception as exc:
            logger.exception(
                "Knowledge retrieval failed for intent %s",
                query.intent.value,
            )

            return self._empty_result(
                query=query,
                decision=KnowledgeDecision.ERROR,
                retrieved_at=retrieved_at,
                business=business,
                warning=(
                    "Verified knowledge retrieval failed."
                ),
                requires_human_review=True,
                metadata={
                    "exception_type": (
                        type(exc).__name__
                    ),
                },
            )

        records = self._rank_records(
            records,
            query=query,
            limit=limit,
        )

        if not records:
            fallback_documents = (
                self._retrieve_document_fallback(
                    business=business,
                    query=query,
                    retrieved_at=retrieved_at,
                    limit=limit,
                )
            )

            records = self._rank_records(
                fallback_documents,
                query=query,
                limit=limit,
            )

        if not records:
            return self._empty_result(
                query=query,
                decision=KnowledgeDecision.NOT_FOUND,
                retrieved_at=retrieved_at,
                business=business,
                warning=(
                    "No verified business fact matched the request."
                ),
            )

        stale_source_count = sum(
            record.source.stale
            for record in records
        )

        if stale_source_count:
            warnings.append(
                (
                    f"{stale_source_count} verified source"
                    f"{'s are' if stale_source_count != 1 else ' is'} "
                    "older than the configured freshness threshold."
                )
            )

        sources = self._deduplicate_sources(
            record.source
            for record in records
        )

        answer_guidance = (
            self._build_answer_guidance(
                query=query,
                records=records,
            )
        )

        decision = (
            KnowledgeDecision.FOUND
            if all(
                record.source.verified
                for record in records
            )
            else KnowledgeDecision.PARTIAL
        )

        return KnowledgeResult(
            decision=decision,
            business_id=business.id,
            business_slug=business.slug,
            intent=query.intent,
            records=tuple(records),
            sources=tuple(sources),
            query=query,
            answer_guidance=answer_guidance,
            verified_fact_count=sum(
                record.source.verified
                for record in records
            ),
            stale_source_count=stale_source_count,
            unsupported_claim_count=0,
            requires_human_review=(
                stale_source_count
                == len(records)
                and bool(records)
            ),
            warnings=tuple(warnings),
            retrieved_at=retrieved_at,
            service_version=(
                KNOWLEDGE_SERVICE_VERSION
            ),
            service_phase=(
                KNOWLEDGE_SERVICE_PHASE
            ),
        )

    # ========================================================
    # SECTION 06 - INTENT ROUTING
    # ========================================================

    def _retrieve_for_intent(
        self,
        *,
        business: Business,
        query: KnowledgeQuery,
        retrieved_at: datetime,
        limit: int,
    ) -> list[KnowledgeRecord]:
        intent = query.intent

        if intent in {
            IntentName.HOURS_GENERAL,
            IntentName.HOURS_TODAY,
            IntentName.HOURS_KITCHEN,
            IntentName.HOURS_HAPPY_HOUR,
            IntentName.HAPPY_HOUR,
        }:
            return self._retrieve_hours(
                business=business,
                query=query,
                retrieved_at=retrieved_at,
            )

        if intent in {
            IntentName.MENU_GENERAL,
            IntentName.MENU_ITEM_LOOKUP,
            IntentName.MENU_DIETARY,
            IntentName.MENU_ALLERGEN,
            IntentName.MENU_PRICE,
        }:
            return self._retrieve_menu(
                business=business,
                query=query,
                retrieved_at=retrieved_at,
                limit=limit,
            )

        if intent in {
            IntentName.EVENTS_GENERAL,
            IntentName.EVENTS_TONIGHT,
            IntentName.LIVE_MUSIC,
            IntentName.SPORTS_VIEWING,
        }:
            return self._retrieve_events(
                business=business,
                query=query,
                retrieved_at=retrieved_at,
                limit=limit,
            )

        if intent in {
            IntentName.PRIVATE_EVENT,
            IntentName.PRIVATE_EVENT_PRICING,
            IntentName.PRIVATE_EVENT_AVAILABILITY,
            IntentName.PRIVATE_EVENT_CONTACT,
        }:
            return self._retrieve_private_events(
                business=business,
                query=query,
                retrieved_at=retrieved_at,
                limit=limit,
            )

        if intent in {
            IntentName.LOCATION,
            IntentName.DIRECTIONS,
            IntentName.PARKING,
            IntentName.CONTACT,
            IntentName.ACCESSIBILITY,
            IntentName.JOBS,
            IntentName.LOST_AND_FOUND,
        }:
            return self._retrieve_business_facts(
                business=business,
                query=query,
                retrieved_at=retrieved_at,
                limit=limit,
            )

        if intent in {
            IntentName.ORDERING,
            IntentName.TAKEOUT,
            IntentName.DELIVERY,
            IntentName.RESERVATION,
            IntentName.RESERVATION_CHANGE,
            IntentName.RESERVATION_CANCEL,
            IntentName.SPECIALS,
        }:
            return self._retrieve_faqs_and_documents(
                business=business,
                query=query,
                retrieved_at=retrieved_at,
                limit=limit,
            )

        return []

    # ========================================================
    # SECTION 07 - HOURS
    # ========================================================

    def _retrieve_hours(
        self,
        *,
        business: Business,
        query: KnowledgeQuery,
        retrieved_at: datetime,
    ) -> list[KnowledgeRecord]:
        service_type = query.service_type

        hours = (
            self.knowledge_repository
            .list_verified_hours(
                business.id,
                service_type=service_type,
            )
        )

        if query.requested_date:
            weekday = (
                query.requested_date.weekday()
            )

            hours = [
                item
                for item in hours
                if item.day_of_week == weekday
            ]

        records: list[KnowledgeRecord] = []

        for item in hours:
            day_name = (
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            )[item.day_of_week]

            if item.is_closed:
                content = (
                    f"{item.service_type.replace('_', ' ').title()} "
                    f"is closed on {day_name}."
                )
            else:
                opening = (
                    item.open_time.strftime("%-I:%M %p")
                    if item.open_time
                    else "not published"
                )

                closing = (
                    item.close_time.strftime("%-I:%M %p")
                    if item.close_time
                    else "not published"
                )

                next_day_note = (
                    " the following day"
                    if item.closes_next_day
                    else ""
                )

                content = (
                    f"{item.service_type.replace('_', ' ').title()} "
                    f"hours on {day_name}: {opening} to "
                    f"{closing}{next_day_note}."
                )

                if item.notes:
                    content += f" {item.notes.strip()}"

            records.append(
                self._record_from_model(
                    record_type=KnowledgeRecordType.HOURS,
                    model=item,
                    title=(
                        f"{day_name} "
                        f"{item.service_type.replace('_', ' ').title()} Hours"
                    ),
                    content=content,
                    structured_data={
                        "service_type": item.service_type,
                        "day_of_week": item.day_of_week,
                        "day_name": day_name,
                        "open_time": (
                            item.open_time.isoformat(
                                timespec="minutes"
                            )
                            if item.open_time
                            else None
                        ),
                        "close_time": (
                            item.close_time.isoformat(
                                timespec="minutes"
                            )
                            if item.close_time
                            else None
                        ),
                        "closes_next_day": item.closes_next_day,
                        "is_closed": item.is_closed,
                        "notes": item.notes,
                    },
                    retrieved_at=retrieved_at,
                    query=query,
                    priority=100,
                )
            )

        return records

    # ========================================================
    # SECTION 08 - MENU
    # ========================================================

    def _retrieve_menu(
        self,
        *,
        business: Business,
        query: KnowledgeQuery,
        retrieved_at: datetime,
        limit: int,
    ) -> list[KnowledgeRecord]:
        statement = (
            select(MenuItem)
            .join(MenuItem.category)
            .options(
                selectinload(
                    MenuItem.category
                )
            )
            .where(
                MenuCategory.business_id
                == business.id,
                MenuCategory.is_active.is_(True),
                MenuCategory.is_deleted.is_(False),
                MenuItem.is_active.is_(True),
                MenuItem.is_deleted.is_(False),
                MenuItem.is_verified.is_(True),
            )
        )

        if query.category_slug:
            statement = statement.where(
                MenuCategory.slug
                == query.category_slug
            )

        statement = statement.order_by(
            MenuItem.is_featured.desc(),
            MenuCategory.display_order.asc(),
            MenuItem.display_order.asc(),
            MenuItem.name.asc(),
        ).limit(
            max(
                limit * 4,
                20,
            )
        )

        items = list(
            self.session.scalars(
                statement
            ).all()
        )

        records: list[KnowledgeRecord] = []

        query_terms = set(
            query.menu_terms
            + query.dietary_terms
            + query.allergen_terms
        )

        for item in items:
            searchable = " ".join(
                [
                    item.name or "",
                    item.description or "",
                    " ".join(
                        item.dietary_tags
                        or []
                    ),
                    item.allergen_notes
                    or "",
                    item.category.name
                    if item.category
                    else "",
                ]
            ).casefold()

            if (
                query_terms
                and not any(
                    term.casefold()
                    in searchable
                    for term in query_terms
                )
            ):
                continue

            price_text = (
                f"${item.price:.2f}"
                if item.price is not None
                else (
                    item.price_note
                    or "Price not published"
                )
            )

            content_parts = [
                item.name,
            ]

            if item.description:
                content_parts.append(
                    item.description.strip()
                )

            content_parts.append(
                f"Price: {price_text}."
            )

            if item.dietary_tags:
                content_parts.append(
                    "Dietary tags: "
                    + ", ".join(
                        item.dietary_tags
                    )
                    + "."
                )

            if item.allergen_notes:
                content_parts.append(
                    "Allergen notes: "
                    + item.allergen_notes.strip()
                )

            records.append(
                self._record_from_model(
                    record_type=(
                        KnowledgeRecordType.MENU_ITEM
                    ),
                    model=item,
                    title=item.name,
                    content=" ".join(
                        content_parts
                    ),
                    structured_data={
                        "name": item.name,
                        "slug": item.slug,
                        "description": item.description,
                        "category": (
                            item.category.name
                            if item.category
                            else None
                        ),
                        "category_slug": (
                            item.category.slug
                            if item.category
                            else None
                        ),
                        "price": (
                            str(item.price)
                            if item.price is not None
                            else None
                        ),
                        "price_note": item.price_note,
                        "dietary_tags": list(
                            item.dietary_tags
                            or []
                        ),
                        "allergen_notes": (
                            item.allergen_notes
                        ),
                        "online_order_url": (
                            item.online_order_url
                        ),
                        "is_featured": (
                            item.is_featured
                        ),
                    },
                    retrieved_at=retrieved_at,
                    query=query,
                    priority=(
                        100
                        if item.is_featured
                        else 60
                    ),
                )
            )

        if records:
            return records

        categories = list(
            self.session.scalars(
                select(MenuCategory)
                .where(
                    MenuCategory.business_id
                    == business.id,
                    MenuCategory.is_active.is_(True),
                    MenuCategory.is_deleted.is_(False),
                    MenuCategory.is_verified.is_(True),
                )
                .order_by(
                    MenuCategory.display_order.asc(),
                    MenuCategory.name.asc(),
                )
                .limit(limit)
            ).all()
        )

        for category in categories:
            records.append(
                self._record_from_model(
                    record_type=(
                        KnowledgeRecordType
                        .MENU_CATEGORY
                    ),
                    model=category,
                    title=category.name,
                    content=(
                        category.description
                        or (
                            f"Verified menu category: "
                            f"{category.name}."
                        )
                    ),
                    structured_data={
                        "name": category.name,
                        "slug": category.slug,
                        "description": (
                            category.description
                        ),
                    },
                    retrieved_at=retrieved_at,
                    query=query,
                    priority=40,
                )
            )

        return records

    # ========================================================
    # SECTION 09 - EVENTS
    # ========================================================

    def _retrieve_events(
        self,
        *,
        business: Business,
        query: KnowledgeQuery,
        retrieved_at: datetime,
        limit: int,
    ) -> list[KnowledgeRecord]:
        after = retrieved_at

        if query.requested_date:
            start_of_day = datetime.combine(
                query.requested_date,
                datetime.min.time(),
            ).replace(
                tzinfo=retrieved_at.tzinfo
            )

            end_of_day = (
                start_of_day
                + timedelta(days=1)
            )

            statement = (
                select(BusinessEvent)
                .where(
                    BusinessEvent.business_id
                    == business.id,
                    BusinessEvent.is_active.is_(True),
                    BusinessEvent.is_deleted.is_(False),
                    BusinessEvent.is_verified.is_(True),
                    BusinessEvent.start_at
                    >= start_of_day,
                    BusinessEvent.start_at
                    < end_of_day,
                )
                .order_by(
                    BusinessEvent.start_at.asc()
                )
                .limit(limit)
            )

            events = list(
                self.session.scalars(
                    statement
                ).all()
            )
        else:
            events = (
                self.knowledge_repository
                .list_upcoming_events(
                    business.id,
                    after=after,
                    limit=limit,
                )
            )

        records: list[KnowledgeRecord] = []

        event_terms = set(
            term.casefold()
            for term in query.event_terms
            + query.sports_terms
        )

        for event in events:
            searchable = " ".join(
                [
                    event.title or "",
                    event.description or "",
                    event.event_type or "",
                    event.location_name or "",
                ]
            ).casefold()

            if (
                event_terms
                and not any(
                    term in searchable
                    for term in event_terms
                )
            ):
                continue

            start_text = event.start_at.strftime(
                "%A, %B %d, %Y at %I:%M %p"
            )

            content = (
                f"{event.title} starts {start_text}."
            )

            if event.description:
                content += (
                    " "
                    + event.description.strip()
                )

            if event.cover_charge is not None:
                content += (
                    f" Cover charge: "
                    f"${event.cover_charge:.2f}."
                )

            if event.reservation_required:
                content += (
                    " A reservation is required."
                )

            records.append(
                self._record_from_model(
                    record_type=(
                        KnowledgeRecordType.EVENT
                    ),
                    model=event,
                    title=event.title,
                    content=content,
                    structured_data={
                        "title": event.title,
                        "slug": event.slug,
                        "description": (
                            event.description
                        ),
                        "event_type": (
                            event.event_type
                        ),
                        "start_at": (
                            event.start_at.isoformat()
                        ),
                        "end_at": (
                            event.end_at.isoformat()
                            if event.end_at
                            else None
                        ),
                        "location_name": (
                            event.location_name
                        ),
                        "age_restriction": (
                            event.age_restriction
                        ),
                        "cover_charge": (
                            str(event.cover_charge)
                            if event.cover_charge
                            is not None
                            else None
                        ),
                        "reservation_required": (
                            event.reservation_required
                        ),
                        "reservation_url": (
                            event.reservation_url
                        ),
                    },
                    retrieved_at=retrieved_at,
                    query=query,
                    priority=90,
                )
            )

        return records

    # ========================================================
    # SECTION 10 - PRIVATE EVENTS
    # ========================================================

    def _retrieve_private_events(
        self,
        *,
        business: Business,
        query: KnowledgeQuery,
        retrieved_at: datetime,
        limit: int,
    ) -> list[KnowledgeRecord]:
        records: list[KnowledgeRecord] = []

        packages = (
            self.knowledge_repository
            .list_private_event_packages(
                business.id
            )
        )

        for package in packages[:limit]:
            pricing_parts: list[str] = []

            if (
                package.price_per_person
                is not None
            ):
                pricing_parts.append(
                    (
                        f"${package.price_per_person:.2f} "
                        "per person"
                    )
                )

            if package.flat_price is not None:
                pricing_parts.append(
                    (
                        f"${package.flat_price:.2f} "
                        "flat price"
                    )
                )

            pricing_text = (
                ", ".join(pricing_parts)
                if pricing_parts
                else "Pricing is not published."
            )

            content = (
                f"{package.name}. "
                f"{package.description or ''} "
                f"{pricing_text}"
            ).strip()

            records.append(
                self._record_from_model(
                    record_type=(
                        KnowledgeRecordType
                        .PRIVATE_EVENT_PACKAGE
                    ),
                    model=package,
                    title=package.name,
                    content=content,
                    structured_data={
                        "package_type": (
                            package.package_type
                        ),
                        "name": package.name,
                        "description": (
                            package.description
                        ),
                        "price_per_person": (
                            str(
                                package.price_per_person
                            )
                            if package.price_per_person
                            is not None
                            else None
                        ),
                        "flat_price": (
                            str(package.flat_price)
                            if package.flat_price
                            is not None
                            else None
                        ),
                        "minimum_guests": (
                            package.minimum_guests
                        ),
                        "maximum_guests": (
                            package.maximum_guests
                        ),
                        "duration_minutes": (
                            package.duration_minutes
                        ),
                        "included_items": list(
                            package.included_items
                            or []
                        ),
                        "upgrade_options": list(
                            package.upgrade_options
                            or []
                        ),
                    },
                    retrieved_at=retrieved_at,
                    query=query,
                    priority=100,
                )
            )

        if (
            query.intent
            == IntentName.PRIVATE_EVENT_CONTACT
        ):
            records.extend(
                self._retrieve_contacts(
                    business=business,
                    query=query,
                    retrieved_at=retrieved_at,
                    contact_types={
                        "events",
                        "private_events",
                        "email",
                        "phone",
                    },
                )
            )

        records.extend(
            self._retrieve_faqs_and_documents(
                business=business,
                query=query,
                retrieved_at=retrieved_at,
                limit=limit,
            )
        )

        return records

    # ========================================================
    # SECTION 11 - BUSINESS FACTS
    # ========================================================

    def _retrieve_business_facts(
        self,
        *,
        business: Business,
        query: KnowledgeQuery,
        retrieved_at: datetime,
        limit: int,
    ) -> list[KnowledgeRecord]:
        records: list[KnowledgeRecord] = []

        if query.intent in {
            IntentName.LOCATION,
            IntentName.DIRECTIONS,
        }:
            address_parts = [
                business.address_line_1,
                business.address_line_2,
                business.city,
                business.state_code,
                business.postal_code,
            ]

            address = ", ".join(
                part.strip()
                for part in address_parts
                if part and part.strip()
            )

            if address:
                records.append(
                    self._record_from_model(
                        record_type=(
                            KnowledgeRecordType.BUSINESS
                        ),
                        model=business,
                        title=(
                            f"{business.display_name} Address"
                        ),
                        content=address,
                        structured_data={
                            "display_name": (
                                business.display_name
                            ),
                            "address": address,
                            "latitude": (
                                str(business.latitude)
                                if business.latitude
                                is not None
                                else None
                            ),
                            "longitude": (
                                str(business.longitude)
                                if business.longitude
                                is not None
                                else None
                            ),
                            "website_url": (
                                business.website_url
                            ),
                        },
                        retrieved_at=retrieved_at,
                        query=query,
                        priority=100,
                    )
                )

        if query.intent == IntentName.CONTACT:
            records.extend(
                self._retrieve_contacts(
                    business=business,
                    query=query,
                    retrieved_at=retrieved_at,
                    contact_types=None,
                )
            )

            if business.phone:
                records.append(
                    self._record_from_model(
                        record_type=(
                            KnowledgeRecordType.BUSINESS
                        ),
                        model=business,
                        title=(
                            f"{business.display_name} Phone"
                        ),
                        content=business.phone,
                        structured_data={
                            "phone": business.phone,
                        },
                        retrieved_at=retrieved_at,
                        query=query,
                        priority=100,
                    )
                )

            if business.general_email:
                records.append(
                    self._record_from_model(
                        record_type=(
                            KnowledgeRecordType.BUSINESS
                        ),
                        model=business,
                        title=(
                            f"{business.display_name} Email"
                        ),
                        content=business.general_email,
                        structured_data={
                            "email": (
                                business.general_email
                            ),
                        },
                        retrieved_at=retrieved_at,
                        query=query,
                        priority=100,
                    )
                )

        records.extend(
            self._retrieve_faqs_and_documents(
                business=business,
                query=query,
                retrieved_at=retrieved_at,
                limit=limit,
            )
        )

        return records

    # ========================================================
    # SECTION 12 - CONTACTS
    # ========================================================

    def _retrieve_contacts(
        self,
        *,
        business: Business,
        query: KnowledgeQuery,
        retrieved_at: datetime,
        contact_types: set[str] | None,
    ) -> list[KnowledgeRecord]:
        statement = (
            select(BusinessContact)
            .where(
                BusinessContact.business_id
                == business.id,
                BusinessContact.is_active.is_(True),
                BusinessContact.is_deleted.is_(False),
                BusinessContact.is_verified.is_(True),
            )
            .order_by(
                BusinessContact.priority.desc(),
                BusinessContact.contact_type.asc(),
            )
        )

        contacts = list(
            self.session.scalars(
                statement
            ).all()
        )

        records: list[KnowledgeRecord] = []

        for contact in contacts:
            if (
                contact_types
                and contact.contact_type.casefold()
                not in contact_types
            ):
                continue

            records.append(
                self._record_from_model(
                    record_type=(
                        KnowledgeRecordType.CONTACT
                    ),
                    model=contact,
                    title=contact.label,
                    content=(
                        contact.display_value
                        or contact.value
                    ),
                    structured_data={
                        "contact_type": (
                            contact.contact_type
                        ),
                        "label": contact.label,
                        "value": contact.value,
                        "display_value": (
                            contact.display_value
                        ),
                        "priority": contact.priority,
                        "metadata": copy.deepcopy(
                            contact.metadata_json
                            or {}
                        ),
                    },
                    retrieved_at=retrieved_at,
                    query=query,
                    priority=contact.priority,
                )
            )

        return records

    # ========================================================
    # SECTION 13 - FAQ AND DOCUMENTS
    # ========================================================

    def _retrieve_faqs_and_documents(
        self,
        *,
        business: Business,
        query: KnowledgeQuery,
        retrieved_at: datetime,
        limit: int,
    ) -> list[KnowledgeRecord]:
        records: list[KnowledgeRecord] = []

        faqs = (
            self.knowledge_repository
            .list_verified_faqs(
                business.id
            )
        )

        for faq in faqs:
            searchable = (
                f"{faq.question} {faq.answer} "
                + " ".join(
                    faq.alternative_questions
                    or []
                )
            )

            relevance = self._text_relevance(
                query.corrected_text,
                searchable,
            )

            if (
                relevance
                < self.minimum_relevance
            ):
                continue

            records.append(
                self._record_from_model(
                    record_type=KnowledgeRecordType.FAQ,
                    model=faq,
                    title=faq.question,
                    content=faq.answer,
                    structured_data={
                        "category": faq.category,
                        "question": faq.question,
                        "answer": faq.answer,
                        "alternative_questions": list(
                            faq.alternative_questions
                            or []
                        ),
                    },
                    retrieved_at=retrieved_at,
                    query=query,
                    priority=70,
                    forced_relevance=relevance,
                )
            )

        document_types = (
            INTENT_DOCUMENT_TYPES.get(
                query.intent,
                (),
            )
        )

        statement = (
            select(KnowledgeDocument)
            .where(
                KnowledgeDocument.business_id
                == business.id,
                KnowledgeDocument.is_active.is_(True),
                KnowledgeDocument.is_deleted.is_(False),
                KnowledgeDocument.is_verified.is_(True),
                KnowledgeDocument.knowledge_status
                == KnowledgeStatus.VERIFIED.value,
            )
            .order_by(
                KnowledgeDocument.priority.desc(),
                KnowledgeDocument.updated_at.desc(),
            )
            .limit(
                max(
                    limit * 4,
                    20,
                )
            )
        )

        documents = list(
            self.session.scalars(
                statement
            ).all()
        )

        for document in documents:
            type_match = (
                not document_types
                or document.document_type
                in document_types
            )

            relevance = self._text_relevance(
                query.corrected_text,
                " ".join(
                    [
                        document.title or "",
                        document.content or "",
                        document.search_text or "",
                        document.document_type
                        or "",
                    ]
                ),
            )

            if (
                not type_match
                and relevance
                < self.minimum_relevance
            ):
                continue

            records.append(
                self._record_from_model(
                    record_type=(
                        KnowledgeRecordType.DOCUMENT
                    ),
                    model=document,
                    title=document.title,
                    content=document.content,
                    structured_data={
                        "document_type": (
                            document.document_type
                        ),
                        "slug": document.slug,
                        "structured_payload": (
                            copy.deepcopy(
                                document.structured_payload
                                or {}
                            )
                        ),
                        "priority": document.priority,
                    },
                    retrieved_at=retrieved_at,
                    query=query,
                    priority=document.priority,
                    forced_relevance=max(
                        relevance,
                        0.45
                        if type_match
                        else 0.0,
                    ),
                )
            )

        return records

    # ========================================================
    # SECTION 14 - DOCUMENT FALLBACK
    # ========================================================

    def _retrieve_document_fallback(
        self,
        *,
        business: Business,
        query: KnowledgeQuery,
        retrieved_at: datetime,
        limit: int,
    ) -> list[KnowledgeRecord]:
        documents = (
            self.knowledge_repository
            .search_documents(
                business.id,
                query.corrected_text,
                limit=limit,
            )
        )

        records: list[KnowledgeRecord] = []

        for document in documents:
            records.append(
                self._record_from_model(
                    record_type=(
                        KnowledgeRecordType.DOCUMENT
                    ),
                    model=document,
                    title=document.title,
                    content=document.content,
                    structured_data={
                        "document_type": (
                            document.document_type
                        ),
                        "slug": document.slug,
                        "structured_payload": (
                            copy.deepcopy(
                                document.structured_payload
                                or {}
                            )
                        ),
                    },
                    retrieved_at=retrieved_at,
                    query=query,
                    priority=document.priority,
                )
            )

        return records

    # ========================================================
    # SECTION 15 - QUERY CONSTRUCTION
    # ========================================================

    def _build_query(
        self,
        nlu_result: NLUResult,
        *,
        business_slug: str,
        limit: int,
    ) -> KnowledgeQuery:
        requested_date = self._requested_date(
            nlu_result.entities
        )

        service_type = self._service_type(
            nlu_result
        )

        category_slug = self._first_entity_value(
            nlu_result.entities,
            EntityType.MENU_CATEGORY,
        )

        menu_terms = self._entity_values(
            nlu_result.entities,
            {
                EntityType.MENU_ITEM,
                EntityType.MENU_CATEGORY,
                EntityType.BEVERAGE_TYPE,
            },
        )

        dietary_terms = self._entity_values(
            nlu_result.entities,
            {
                EntityType.DIETARY_REQUIREMENT,
            },
        )

        allergen_terms = self._entity_values(
            nlu_result.entities,
            {
                EntityType.ALLERGEN,
            },
        )

        sports_terms = self._entity_values(
            nlu_result.entities,
            {
                EntityType.SPORTS_TEAM,
                EntityType.SPORTS_LEAGUE,
                EntityType.SPORTS_TYPE,
            },
        )

        event_terms = self._entity_values(
            nlu_result.entities,
            {
                EntityType.EVENT_TYPE,
                EntityType.OCCASION,
            },
        )

        return KnowledgeQuery(
            business_slug=business_slug,
            intent=nlu_result.primary_intent,
            text=nlu_result.original_text,
            corrected_text=(
                nlu_result.corrected_text
            ),
            page_category=(
                nlu_result.metadata.get(
                    "page_category"
                )
                if nlu_result.metadata
                else None
            ),
            entities=nlu_result.entities,
            requested_limit=limit,
            requested_date=requested_date,
            service_type=service_type,
            category_slug=(
                str(category_slug)
                if category_slug
                else None
            ),
            menu_terms=tuple(
                str(value)
                for value in menu_terms
            ),
            event_terms=tuple(
                str(value)
                for value in event_terms
            ),
            dietary_terms=tuple(
                str(value)
                for value in dietary_terms
            ),
            allergen_terms=tuple(
                str(value)
                for value in allergen_terms
            ),
            sports_terms=tuple(
                str(value)
                for value in sports_terms
            ),
        )

    # ========================================================
    # SECTION 16 - RECORD CONSTRUCTION
    # ========================================================

    def _record_from_model(
        self,
        *,
        record_type: KnowledgeRecordType,
        model: Any,
        title: str,
        content: str,
        structured_data: dict[str, Any],
        retrieved_at: datetime,
        query: KnowledgeQuery,
        priority: int,
        forced_relevance: float | None = None,
    ) -> KnowledgeRecord:
        source_updated_at = (
            getattr(
                model,
                "source_updated_at",
                None,
            )
            or getattr(
                model,
                "verified_at",
                None,
            )
            or getattr(
                model,
                "updated_at",
                None,
            )
        )

        stale = self._is_stale(
            source_updated_at,
            retrieved_at,
        )

        verified = bool(
            getattr(
                model,
                "is_verified",
                False,
            )
        )

        trust_level = (
            KnowledgeTrustLevel.VERIFIED_STALE
            if verified and stale
            else (
                KnowledgeTrustLevel.VERIFIED
                if verified
                else KnowledgeTrustLevel.UNKNOWN
            )
        )

        source_name = (
            getattr(
                model,
                "source_name",
                None,
            )
            or title
        )

        source_url = (
            getattr(
                model,
                "source_url",
                None,
            )
            or structured_data.get(
                "reservation_url"
            )
            or structured_data.get(
                "online_order_url"
            )
        )

        relevance = (
            forced_relevance
            if forced_relevance
            is not None
            else self._text_relevance(
                query.corrected_text,
                f"{title} {content}",
            )
        )

        relevance = round(
            min(
                max(relevance, 0.0),
                1.0,
            ),
            6,
        )

        confidence = round(
            min(
                (
                    0.65
                    + relevance * 0.25
                    + (
                        0.10
                        if verified
                        else 0.0
                    )
                    - (
                        0.15
                        if stale
                        else 0.0
                    )
                ),
                1.0,
            ),
            6,
        )

        warnings: list[str] = []

        if stale:
            warnings.append(
                "Verified source may be stale."
            )

        if not verified:
            warnings.append(
                "Record is not verified."
            )

        source = KnowledgeSource(
            source_type=(
                getattr(
                    model,
                    "source_type",
                    None,
                )
                or record_type.value
            ),
            source_name=str(source_name),
            source_reference=str(
                getattr(model, "id", "")
            )
            or None,
            source_url=(
                str(source_url)
                if source_url
                else None
            ),
            trust_level=trust_level,
            verified=verified,
            retrieved_at=retrieved_at,
            source_updated_at=(
                source_updated_at
                if isinstance(
                    source_updated_at,
                    datetime,
                )
                else None
            ),
            stale=stale,
            relevance_score=relevance,
            metadata={
                "record_type": (
                    record_type.value
                ),
                "model_name": (
                    type(model).__name__
                ),
            },
        )

        return KnowledgeRecord(
            record_type=record_type,
            record_id=str(
                getattr(model, "id", "")
            ),
            title=title.strip(),
            content=content.strip(),
            structured_data=structured_data,
            source=source,
            priority=priority,
            relevance_score=relevance,
            confidence=confidence,
            warnings=tuple(warnings),
        )

    # ========================================================
    # SECTION 17 - RANKING
    # ========================================================

    def _rank_records(
        self,
        records: Sequence[KnowledgeRecord],
        *,
        query: KnowledgeQuery,
        limit: int,
    ) -> list[KnowledgeRecord]:
        unique: dict[
            tuple[str, str],
            KnowledgeRecord,
        ] = {}

        for record in records:
            key = (
                record.record_type.value,
                record.record_id,
            )

            existing = unique.get(key)

            if (
                existing is None
                or self._ranking_score(record)
                > self._ranking_score(existing)
            ):
                unique[key] = record

        ranked = sorted(
            unique.values(),
            key=lambda record: (
                self._ranking_score(record),
                record.priority,
                record.source.verified,
                not record.source.stale,
                record.title.casefold(),
            ),
            reverse=True,
        )

        return ranked[:limit]

    @staticmethod
    def _ranking_score(
        record: KnowledgeRecord,
    ) -> float:
        return (
            record.relevance_score * 0.50
            + record.confidence * 0.30
            + min(
                max(record.priority, 0),
                100,
            )
            / 100.0
            * 0.20
        )

    # ========================================================
    # SECTION 18 - RELEVANCE
    # ========================================================

    @staticmethod
    def _text_relevance(
        query_text: str,
        candidate_text: str,
    ) -> float:
        query_tokens = {
            token.casefold()
            for token in WORD_PATTERN.findall(
                query_text
            )
            if token.casefold()
            not in STOP_WORDS
        }

        candidate_tokens = {
            token.casefold()
            for token in WORD_PATTERN.findall(
                candidate_text
            )
            if token.casefold()
            not in STOP_WORDS
        }

        if not query_tokens:
            return 0.50

        if not candidate_tokens:
            return 0.0

        intersection = (
            query_tokens
            & candidate_tokens
        )

        coverage = (
            len(intersection)
            / len(query_tokens)
        )

        union = (
            query_tokens
            | candidate_tokens
        )

        jaccard = (
            len(intersection)
            / len(union)
            if union
            else 0.0
        )

        phrase_bonus = (
            0.20
            if query_text.casefold()
            in candidate_text.casefold()
            else 0.0
        )

        return min(
            coverage * 0.65
            + jaccard * 0.25
            + phrase_bonus,
            1.0,
        )

    # ========================================================
    # SECTION 19 - ANSWER GUIDANCE
    # ========================================================

    @staticmethod
    def _build_answer_guidance(
        *,
        query: KnowledgeQuery,
        records: Sequence[KnowledgeRecord],
    ) -> str:
        if not records:
            return (
                "Do not invent an answer. State that verified "
                "information is unavailable and offer an appropriate "
                "contact or human-handoff action."
            )

        stale_count = sum(
            record.source.stale
            for record in records
        )

        guidance = [
            (
                "Answer using only the returned verified records. "
                "Do not add unsupported prices, times, dates, menu "
                "items, event details, or availability."
            )
        ]

        if stale_count:
            guidance.append(
                (
                    "Clearly indicate that some information may "
                    "require confirmation because its verified source "
                    "is older than the freshness threshold."
                )
            )

        if query.intent in {
            IntentName.MENU_ALLERGEN,
            IntentName.MENU_DIETARY,
        }:
            guidance.append(
                (
                    "Do not guarantee allergen safety. Present the "
                    "recorded allergen or dietary information and "
                    "recommend confirming with restaurant staff."
                )
            )

        if query.intent in {
            IntentName.PRIVATE_EVENT_AVAILABILITY,
            IntentName.RESERVATION,
        }:
            guidance.append(
                (
                    "Do not promise availability unless a verified "
                    "availability source explicitly confirms it."
                )
            )

        return " ".join(guidance)

    # ========================================================
    # SECTION 20 - SOURCE DEDUPLICATION
    # ========================================================

    @staticmethod
    def _deduplicate_sources(
        sources: Iterable[KnowledgeSource],
    ) -> list[KnowledgeSource]:
        unique: dict[
            tuple[
                str,
                str | None,
                str | None,
            ],
            KnowledgeSource,
        ] = {}

        for source in sources:
            key = (
                source.source_type,
                source.source_reference,
                source.source_url,
            )

            existing = unique.get(key)

            if (
                existing is None
                or source.relevance_score
                > existing.relevance_score
            ):
                unique[key] = source

        return sorted(
            unique.values(),
            key=lambda source: (
                source.relevance_score,
                source.verified,
                not source.stale,
            ),
            reverse=True,
        )

    # ========================================================
    # SECTION 21 - ENTITY HELPERS
    # ========================================================

    @staticmethod
    def _requested_date(
        entities: Sequence[ExtractedEntity],
    ) -> date | None:
        for entity_type in (
            EntityType.DATE,
            EntityType.RELATIVE_DATE,
            EntityType.WEEKDAY,
        ):
            for entity in entities:
                if entity.entity_type != entity_type:
                    continue

                value = entity.normalized_value

                if isinstance(value, date):
                    return value

                if isinstance(value, str):
                    try:
                        return date.fromisoformat(
                            value
                        )
                    except ValueError:
                        continue

        return None

    @staticmethod
    def _service_type(
        nlu_result: NLUResult,
    ) -> str | None:
        intent_map = {
            IntentName.HOURS_KITCHEN: "kitchen",
            IntentName.HOURS_HAPPY_HOUR: "happy_hour",
            IntentName.HAPPY_HOUR: "happy_hour",
            IntentName.TAKEOUT: "takeout",
            IntentName.DELIVERY: "delivery",
        }

        if (
            nlu_result.primary_intent
            in intent_map
        ):
            return intent_map[
                nlu_result.primary_intent
            ]

        for entity in nlu_result.entities:
            if (
                entity.entity_type
                == EntityType.SERVICE_TYPE
            ):
                return str(
                    entity.normalized_value
                )

        return None

    @staticmethod
    def _entity_values(
        entities: Sequence[ExtractedEntity],
        types: set[EntityType],
    ) -> list[Any]:
        values: list[Any] = []

        for entity in entities:
            if entity.entity_type not in types:
                continue

            value = entity.normalized_value

            if isinstance(value, list):
                for item in value:
                    if item not in values:
                        values.append(item)
            elif value not in values:
                values.append(value)

        return values

    @staticmethod
    def _first_entity_value(
        entities: Sequence[ExtractedEntity],
        entity_type: EntityType,
    ) -> Any | None:
        for entity in entities:
            if entity.entity_type == entity_type:
                return entity.normalized_value

        return None

    # ========================================================
    # SECTION 22 - FRESHNESS
    # ========================================================

    def _is_stale(
        self,
        source_updated_at: datetime | None,
        retrieved_at: datetime,
    ) -> bool:
        if source_updated_at is None:
            return True

        comparison_time = retrieved_at

        if (
            source_updated_at.tzinfo is None
            and comparison_time.tzinfo
            is not None
        ):
            comparison_time = (
                comparison_time.replace(
                    tzinfo=None
                )
            )

        return (
            comparison_time
            - source_updated_at
            > timedelta(
                days=self.stale_after_days
            )
        )

    # ========================================================
    # SECTION 23 - EMPTY RESULT
    # ========================================================

    def _empty_result(
        self,
        *,
        query: KnowledgeQuery,
        decision: KnowledgeDecision,
        retrieved_at: datetime,
        warning: str,
        business: Business | None = None,
        requires_human_review: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> KnowledgeResult:
        warnings = [warning]

        if metadata:
            warnings.append(
                "Diagnostic metadata was recorded internally."
            )

        return KnowledgeResult(
            decision=decision,
            business_id=(
                business.id
                if business
                else None
            ),
            business_slug=query.business_slug,
            intent=query.intent,
            records=(),
            sources=(),
            query=query,
            answer_guidance=(
                "Do not invent a business fact. Explain that verified "
                "information is unavailable and provide a contact or "
                "human-handoff option when appropriate."
            ),
            verified_fact_count=0,
            stale_source_count=0,
            unsupported_claim_count=0,
            requires_human_review=(
                requires_human_review
            ),
            warnings=tuple(warnings),
            retrieved_at=retrieved_at,
            service_version=(
                KNOWLEDGE_SERVICE_VERSION
            ),
            service_phase=(
                KNOWLEDGE_SERVICE_PHASE
            ),
        )


# ============================================================
# SECTION 24 - MODULE-LEVEL HELPER
# ============================================================

def retrieve_verified_knowledge(
    session: Session,
    nlu_result: NLUResult,
    *,
    business_slug: str = DEFAULT_BUSINESS_SLUG,
    limit: int = DEFAULT_RESULT_LIMIT,
    now: datetime | None = None,
) -> KnowledgeResult:
    """
    Retrieve verified business knowledge for one NLU result.
    """

    return KnowledgeService(
        session
    ).retrieve(
        nlu_result,
        business_slug=business_slug,
        limit=limit,
        now=now,
    )


# ============================================================
# SECTION 25 - SELF-TEST
# ============================================================

def validate_knowledge_service_module() -> dict[str, Any]:
    from datetime import time as clock_time

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.database.base import Base
    from app.database.models import (
        Business,
        BusinessEvent,
        BusinessHour,
        FAQEntry,
        MenuCategory,
        MenuItem,
        PrivateEventPackage,
    )
    from app.nlu.orchestrator import process_message

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
            phone="973-555-0100",
            general_email=(
                "info@thehorseshoetavern.com"
            ),
            address_line_1="Test Street",
            city="Morristown",
            state_code="NJ",
            postal_code="07960",
            source_type="official_website",
            source_name=(
                "Horseshoe Tavern official website"
            ),
            source_url=(
                "https://www.thehorseshoetavern.com/"
            ),
        )

        business.mark_verified(
            verified_by="knowledge-test",
            notes="Verified test business",
        )

        hours = BusinessHour(
            business=business,
            service_type="kitchen",
            day_of_week=2,
            open_time=clock_time(11, 0),
            close_time=clock_time(22, 0),
            is_closed=False,
            source_type="official_website",
            source_name="Official kitchen hours",
            source_url=(
                "https://www.thehorseshoetavern.com/"
            ),
        )

        hours.mark_verified(
            verified_by="knowledge-test",
            notes="Verified test hours",
        )

        category = MenuCategory(
            business=business,
            name="Appetizers",
            slug="appetizers",
            description="Starter dishes.",
            source_type="official_menu",
            source_name="Official menu",
            source_url=(
                "https://www.thehorseshoetavern.com/menu"
            ),
        )

        category.mark_verified(
            verified_by="knowledge-test",
            notes="Verified test category",
        )

        menu_item = MenuItem(
            category=category,
            name="Tavern Wings",
            slug="tavern-wings",
            description="Crispy chicken wings.",
            price=Decimal("14.00"),
            dietary_tags=["gluten-aware"],
            allergen_notes=(
                "Prepared in a shared kitchen."
            ),
            source_type="official_menu",
            source_name="Official menu",
            source_url=(
                "https://www.thehorseshoetavern.com/menu"
            ),
        )

        menu_item.mark_verified(
            verified_by="knowledge-test",
            notes="Verified test item",
        )

        event = BusinessEvent(
            business=business,
            title="Friday Live Music",
            slug="friday-live-music",
            event_type="live_music",
            description="Live band performance.",
            start_at=datetime(
                2026,
                7,
                24,
                20,
                0,
                tzinfo=reference.tzinfo,
            ),
            source_type="official_events",
            source_name="Official event calendar",
            source_url=(
                "https://www.thehorseshoetavern.com/events"
            ),
        )

        event.mark_verified(
            verified_by="knowledge-test",
            notes="Verified test event",
        )

        faq = FAQEntry(
            business=business,
            category="parking",
            question="Where can guests park?",
            answer="Use nearby public parking.",
            alternative_questions=[
                "Is parking available?"
            ],
            source_type="official_website",
            source_name="Official parking FAQ",
            source_url=(
                "https://www.thehorseshoetavern.com/"
            ),
        )

        faq.mark_verified(
            verified_by="knowledge-test",
            notes="Verified test FAQ",
        )

        package = PrivateEventPackage(
            business=business,
            package_type="buffet",
            name="Private Event Buffet",
            description="Buffet package for groups.",
            price_per_person=Decimal("45.00"),
            minimum_guests=25,
            maximum_guests=100,
            source_type="official_private_events",
            source_name="Official private-event package",
            source_url=(
                "https://www.thehorseshoetavern.com/private-events"
            ),
        )

        package.mark_verified(
            verified_by="knowledge-test",
            notes="Verified test package",
        )

        session.add_all(
            [
                business,
                hours,
                category,
                menu_item,
                event,
                faq,
                package,
            ]
        )

        session.commit()

        service = KnowledgeService(
            session,
            stale_after_days=3650,
        )

        hours_nlu = process_message(
            "What time does the kitchen close today?",
            reference_datetime=reference,
        )

        hours_result = service.retrieve(
            hours_nlu,
            now=reference,
        )

        menu_nlu = process_message(
            "Do you have wings and how much are they?",
            reference_datetime=reference,
        )

        menu_result = service.retrieve(
            menu_nlu,
            now=reference,
        )

        event_nlu = process_message(
            "What live music is coming up?",
            reference_datetime=reference,
        )

        event_result = service.retrieve(
            event_nlu,
            now=reference,
        )

        parking_nlu = process_message(
            "Where can I park?",
            reference_datetime=reference,
        )

        parking_result = service.retrieve(
            parking_nlu,
            now=reference,
        )

        private_nlu = process_message(
            "How much is a private event package?",
            reference_datetime=reference,
        )

        private_result = service.retrieve(
            private_nlu,
            now=reference,
        )

        checks = {
            "hours_found": (
                hours_result.has_verified_knowledge
            ),
            "hours_record_type": any(
                record.record_type
                == KnowledgeRecordType.HOURS
                for record
                in hours_result.records
            ),
            "menu_found": (
                menu_result.has_verified_knowledge
            ),
            "menu_item_found": any(
                record.record_type
                == KnowledgeRecordType.MENU_ITEM
                for record
                in menu_result.records
            ),
            "event_found": (
                event_result.has_verified_knowledge
            ),
            "event_record_found": any(
                record.record_type
                == KnowledgeRecordType.EVENT
                for record
                in event_result.records
            ),
            "parking_found": (
                parking_result.has_verified_knowledge
            ),
            "faq_record_found": any(
                record.record_type
                == KnowledgeRecordType.FAQ
                for record
                in parking_result.records
            ),
            "private_event_found": (
                private_result.has_verified_knowledge
            ),
            "private_package_found": any(
                record.record_type
                == KnowledgeRecordType.PRIVATE_EVENT_PACKAGE
                for record
                in private_result.records
            ),
            "sources_present": all(
                bool(result.sources)
                for result in (
                    hours_result,
                    menu_result,
                    event_result,
                    parking_result,
                    private_result,
                )
            ),
            "unsupported_claims_zero": all(
                result.unsupported_claim_count
                == 0
                for result in (
                    hours_result,
                    menu_result,
                    event_result,
                    parking_result,
                    private_result,
                )
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
# SECTION 26 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import json

    report = validate_knowledge_service_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
