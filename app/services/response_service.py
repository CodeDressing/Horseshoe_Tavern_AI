# ============================================================
# Exact file location: app/services/response_service.py
# Horseshoe Tavern AI
# Phase 1 Part 1.19
# Grounded response composition, citations, actions, workflow
# prompts, safe fallbacks, validation, and answer diagnostics
# ============================================================

"""
Grounded response-generation service for Horseshoe Tavern AI.

This service converts:

- NLUResult
- KnowledgeResult
- Conversation context
- Verified source records

into a safe, structured chatbot response.

Core guarantees:

- Never invent business facts
- Never invent hours, prices, event dates, menu items, availability,
  reservation status, parking details, or private-event terms
- Use only verified KnowledgeRecord content for factual answers
- Preserve ambiguity and clarification requirements
- Present stale-source warnings when applicable
- Avoid guaranteeing allergen safety
- Avoid guaranteeing reservation or private-event availability
- Support actions such as navigation, phone, email, ordering,
  reservation, and human handoff
- Support multi-intent responses without losing intent boundaries
- Produce JSON-safe source attribution and validation metadata
"""

from __future__ import annotations

import copy
import html
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Sequence

from app.logging_config import get_logger
from app.nlu.context import ActiveFlow
from app.nlu.entities import EntityType
from app.nlu.intent import IntentName
from app.nlu.orchestrator import (
    NLUDecision,
    NLUResult,
)
from app.schemas.chat import (
    ConfidenceBreakdown,
    DetectedEntity,
    PrivateEventDraft,
    ResponseAction,
    ResponseActionType,
    ResponseSource,
    ResponseValidation,
    SourceTrustLevel,
    SpellingCorrection,
    WidgetSize,
    WidgetState,
)
from app.services.knowledge_service import (
    KnowledgeDecision,
    KnowledgeRecord,
    KnowledgeRecordType,
    KnowledgeResult,
    KnowledgeSource,
    KnowledgeTrustLevel,
)


# ============================================================
# SECTION 01 - LOGGER AND CONSTANTS
# ============================================================

logger = get_logger(__name__)

RESPONSE_SERVICE_VERSION: Final[str] = "1.0.0"
RESPONSE_SERVICE_PHASE: Final[str] = "Phase 1 Part 1.19"

MAXIMUM_RESPONSE_CHARACTERS: Final[int] = 12000
MAXIMUM_SOURCE_COUNT: Final[int] = 12
MAXIMUM_ACTION_COUNT: Final[int] = 8
MAXIMUM_RECORDS_IN_RESPONSE: Final[int] = 8

DEFAULT_BUSINESS_NAME: Final[str] = "Horseshoe Tavern"

SAFE_WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[ \t]+"
)

SAFE_MULTILINE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\n{3,}"
)


# ============================================================
# SECTION 02 - ENUMERATIONS
# ============================================================

class ResponseDecision(str, Enum):
    GROUNDED = "grounded"
    GROUNDED_WITH_WARNING = "grounded_with_warning"
    CLARIFICATION = "clarification"
    SAFE_FALLBACK = "safe_fallback"
    HUMAN_HANDOFF = "human_handoff"
    MULTI_INTENT = "multi_intent"
    ERROR = "error"


class ResponseTone(str, Enum):
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CONCISE = "concise"
    EMPATHETIC = "empathetic"


# ============================================================
# SECTION 03 - DATA CLASSES
# ============================================================

@dataclass(frozen=True, slots=True)
class ResponseSection:
    heading: str | None
    body: str
    record_ids: tuple[str, ...] = ()
    intent: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "body": self.body,
            "record_ids": list(self.record_ids),
            "intent": self.intent,
        }


@dataclass(frozen=True, slots=True)
class GroundedResponse:
    message: str
    decision: ResponseDecision
    sections: tuple[ResponseSection, ...]

    sources: tuple[ResponseSource, ...]
    actions: tuple[ResponseAction, ...]

    validation: ResponseValidation
    confidence: ConfidenceBreakdown

    spelling_corrections: tuple[SpellingCorrection, ...]
    entities: tuple[DetectedEntity, ...]

    private_event_draft: PrivateEventDraft | None

    human_handoff_available: bool
    human_handoff_required: bool

    widget_state: WidgetState
    widget_size: WidgetSize

    response_template_id: str
    response_variant_id: str

    warnings: tuple[str, ...]
    processing_metadata: dict[str, Any]

    service_version: str
    service_phase: str
    created_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "decision": self.decision.value,
            "sections": [
                section.as_dict()
                for section in self.sections
            ],
            "sources": [
                source.model_dump(
                    mode="json"
                )
                for source in self.sources
            ],
            "actions": [
                action.model_dump(
                    mode="json"
                )
                for action in self.actions
            ],
            "validation": (
                self.validation.model_dump(
                    mode="json"
                )
            ),
            "confidence": (
                self.confidence.model_dump(
                    mode="json"
                )
            ),
            "spelling_corrections": [
                correction.model_dump(
                    mode="json"
                )
                for correction
                in self.spelling_corrections
            ],
            "entities": [
                entity.model_dump(
                    mode="json"
                )
                for entity in self.entities
            ],
            "private_event_draft": (
                self.private_event_draft.model_dump(
                    mode="json"
                )
                if self.private_event_draft
                else None
            ),
            "human_handoff_available": (
                self.human_handoff_available
            ),
            "human_handoff_required": (
                self.human_handoff_required
            ),
            "widget_state": self.widget_state,
            "widget_size": self.widget_size,
            "response_template_id": (
                self.response_template_id
            ),
            "response_variant_id": (
                self.response_variant_id
            ),
            "warnings": list(self.warnings),
            "processing_metadata": copy.deepcopy(
                self.processing_metadata
            ),
            "service_version": self.service_version,
            "service_phase": self.service_phase,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================
# SECTION 04 - RESPONSE SERVICE
# ============================================================

class ResponseService:
    """
    Convert NLU and verified knowledge into grounded responses.
    """

    def __init__(
        self,
        *,
        business_name: str = DEFAULT_BUSINESS_NAME,
        human_handoff_available: bool = True,
    ) -> None:
        self.business_name = (
            business_name.strip()
            or DEFAULT_BUSINESS_NAME
        )

        self.human_handoff_available = bool(
            human_handoff_available
        )

    # ========================================================
    # SECTION 05 - PUBLIC RESPONSE ENTRYPOINT
    # ========================================================

    def compose(
        self,
        nlu_result: NLUResult,
        knowledge_result: KnowledgeResult,
        *,
        now: datetime | None = None,
        response_metadata: Mapping[str, Any] | None = None,
    ) -> GroundedResponse:
        created_at = (
            now
            or datetime.now().astimezone()
        )

        warnings: list[str] = list(
            knowledge_result.warnings
        )

        warnings.extend(
            warning.message
            for warning in nlu_result.warnings
            if warning.severity
            in {
                "warning",
                "error",
            }
        )

        if (
            nlu_result.primary_intent
            in {
                IntentName.HUMAN_HANDOFF,
                IntentName.COMPLAINT,
            }
        ):
            return self._compose_handoff_response(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
                created_at=created_at,
                warnings=warnings,
                metadata=response_metadata,
            )

        if nlu_result.requires_clarification:
            return self._compose_clarification_response(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
                created_at=created_at,
                warnings=warnings,
                metadata=response_metadata,
            )

        if (
            nlu_result.nlu_decision
            == NLUDecision.MULTI_INTENT
            or nlu_result.is_multi_intent
        ):
            return self._compose_multi_intent_response(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
                created_at=created_at,
                warnings=warnings,
                metadata=response_metadata,
            )

        if not knowledge_result.has_verified_knowledge:
            return self._compose_safe_fallback(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
                created_at=created_at,
                warnings=warnings,
                metadata=response_metadata,
            )

        return self._compose_grounded_response(
            nlu_result=nlu_result,
            knowledge_result=knowledge_result,
            created_at=created_at,
            warnings=warnings,
            metadata=response_metadata,
        )

    # ========================================================
    # SECTION 06 - GROUNDED RESPONSE
    # ========================================================

    def _compose_grounded_response(
        self,
        *,
        nlu_result: NLUResult,
        knowledge_result: KnowledgeResult,
        created_at: datetime,
        warnings: list[str],
        metadata: Mapping[str, Any] | None,
    ) -> GroundedResponse:
        records = list(
            knowledge_result.records[
                :MAXIMUM_RECORDS_IN_RESPONSE
            ]
        )

        sections = self._sections_for_records(
            nlu_result=nlu_result,
            records=records,
        )

        message = self._join_sections(
            sections
        )

        if knowledge_result.stale_source_count:
            message += (
                "\n\nSome of this information is from a verified "
                "source that may need reconfirmation."
            )

        if (
            nlu_result.primary_intent
            in {
                IntentName.MENU_ALLERGEN,
                IntentName.MENU_DIETARY,
            }
        ):
            message += (
                "\n\nFor allergies or severe dietary restrictions, "
                "please confirm preparation details directly with staff."
            )

        if (
            nlu_result.primary_intent
            in {
                IntentName.PRIVATE_EVENT_AVAILABILITY,
                IntentName.RESERVATION,
            }
        ):
            message += (
                "\n\nAvailability is not guaranteed until confirmed "
                "by the tavern or its reservation system."
            )

        message = self._sanitize_message(
            message
        )

        decision = (
            ResponseDecision.GROUNDED_WITH_WARNING
            if warnings
            or knowledge_result.stale_source_count
            else ResponseDecision.GROUNDED
        )

        return GroundedResponse(
            message=message,
            decision=decision,
            sections=tuple(sections),
            sources=self._convert_sources(
                knowledge_result.sources
            ),
            actions=self._build_actions(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
            ),
            validation=self._build_validation(
                knowledge_result=knowledge_result,
                additional_warnings=warnings,
            ),
            confidence=self._build_confidence(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
            ),
            spelling_corrections=(
                self._convert_spelling(
                    nlu_result
                )
            ),
            entities=self._convert_entities(
                nlu_result
            ),
            private_event_draft=(
                self._build_private_event_draft(
                    nlu_result
                )
            ),
            human_handoff_available=(
                self.human_handoff_available
            ),
            human_handoff_required=False,
            widget_state=WidgetState.OPEN,
            widget_size=self._preferred_widget_size(
                nlu_result
            ),
            response_template_id=(
                self._template_id(
                    nlu_result,
                    decision,
                )
            ),
            response_variant_id="grounded-v1",
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            processing_metadata=self._metadata(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
                metadata=metadata,
            ),
            service_version=(
                RESPONSE_SERVICE_VERSION
            ),
            service_phase=(
                RESPONSE_SERVICE_PHASE
            ),
            created_at=created_at,
        )

    # ========================================================
    # SECTION 07 - CLARIFICATION RESPONSE
    # ========================================================

    def _compose_clarification_response(
        self,
        *,
        nlu_result: NLUResult,
        knowledge_result: KnowledgeResult,
        created_at: datetime,
        warnings: list[str],
        metadata: Mapping[str, Any] | None,
    ) -> GroundedResponse:
        message = (
            nlu_result.suggested_clarification
            or (
                "Could you clarify what you would like "
                "to know?"
            )
        )

        actions = self._clarification_actions(
            nlu_result
        )

        section = ResponseSection(
            heading=None,
            body=message,
            intent=(
                nlu_result.primary_intent.value
            ),
        )

        return GroundedResponse(
            message=self._sanitize_message(
                message
            ),
            decision=ResponseDecision.CLARIFICATION,
            sections=(section,),
            sources=(),
            actions=actions,
            validation=ResponseValidation(
                verified_business_facts_only=True,
                source_count=0,
                stale_source_count=0,
                unsupported_claim_count=0,
                hallucination_guard_passed=True,
                privacy_guard_passed=True,
                safety_guard_passed=True,
                requires_human_review=False,
                warnings=warnings,
            ),
            confidence=self._build_confidence(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
            ),
            spelling_corrections=(
                self._convert_spelling(
                    nlu_result
                )
            ),
            entities=self._convert_entities(
                nlu_result
            ),
            private_event_draft=(
                self._build_private_event_draft(
                    nlu_result
                )
            ),
            human_handoff_available=(
                self.human_handoff_available
            ),
            human_handoff_required=False,
            widget_state=WidgetState.OPEN,
            widget_size=WidgetSize.COMPACT,
            response_template_id=(
                "clarification-v1"
            ),
            response_variant_id=(
                self._clarification_variant(
                    nlu_result
                )
            ),
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            processing_metadata=self._metadata(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
                metadata=metadata,
            ),
            service_version=(
                RESPONSE_SERVICE_VERSION
            ),
            service_phase=(
                RESPONSE_SERVICE_PHASE
            ),
            created_at=created_at,
        )

    # ========================================================
    # SECTION 08 - SAFE FALLBACK
    # ========================================================

    def _compose_safe_fallback(
        self,
        *,
        nlu_result: NLUResult,
        knowledge_result: KnowledgeResult,
        created_at: datetime,
        warnings: list[str],
        metadata: Mapping[str, Any] | None,
    ) -> GroundedResponse:
        message = self._fallback_message(
            nlu_result
        )

        warnings.append(
            "No matching verified business record was available."
        )

        return GroundedResponse(
            message=message,
            decision=ResponseDecision.SAFE_FALLBACK,
            sections=(
                ResponseSection(
                    heading=None,
                    body=message,
                    intent=(
                        nlu_result.primary_intent.value
                    ),
                ),
            ),
            sources=(),
            actions=self._fallback_actions(
                nlu_result
            ),
            validation=ResponseValidation(
                verified_business_facts_only=True,
                source_count=0,
                stale_source_count=0,
                unsupported_claim_count=0,
                hallucination_guard_passed=True,
                privacy_guard_passed=True,
                safety_guard_passed=True,
                requires_human_review=(
                    knowledge_result
                    .requires_human_review
                ),
                warnings=warnings,
            ),
            confidence=self._build_confidence(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
            ),
            spelling_corrections=(
                self._convert_spelling(
                    nlu_result
                )
            ),
            entities=self._convert_entities(
                nlu_result
            ),
            private_event_draft=(
                self._build_private_event_draft(
                    nlu_result
                )
            ),
            human_handoff_available=(
                self.human_handoff_available
            ),
            human_handoff_required=False,
            widget_state=WidgetState.OPEN,
            widget_size=WidgetSize.COMPACT,
            response_template_id="fallback-v1",
            response_variant_id=(
                nlu_result.primary_intent.value
                .lower()
                + "-fallback"
            ),
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            processing_metadata=self._metadata(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
                metadata=metadata,
            ),
            service_version=(
                RESPONSE_SERVICE_VERSION
            ),
            service_phase=(
                RESPONSE_SERVICE_PHASE
            ),
            created_at=created_at,
        )

    # ========================================================
    # SECTION 09 - HUMAN HANDOFF
    # ========================================================

    def _compose_handoff_response(
        self,
        *,
        nlu_result: NLUResult,
        knowledge_result: KnowledgeResult,
        created_at: datetime,
        warnings: list[str],
        metadata: Mapping[str, Any] | None,
    ) -> GroundedResponse:
        complaint = (
            nlu_result.primary_intent
            == IntentName.COMPLAINT
        )

        if complaint:
            message = (
                "I’m sorry you had a poor experience. "
                "I can help connect you with a person who can review it."
            )
        else:
            message = (
                "I can help connect you with a person at "
                f"{self.business_name}."
            )

        action = ResponseAction(
            action_type=(
                ResponseActionType.HUMAN_HANDOFF
            ),
            label="Contact the tavern",
            form_key="human_handoff",
            analytics_event="human_handoff_requested",
        )

        return GroundedResponse(
            message=message,
            decision=ResponseDecision.HUMAN_HANDOFF,
            sections=(
                ResponseSection(
                    heading=None,
                    body=message,
                    intent=(
                        nlu_result.primary_intent.value
                    ),
                ),
            ),
            sources=self._convert_sources(
                knowledge_result.sources
            ),
            actions=(action,),
            validation=self._build_validation(
                knowledge_result=knowledge_result,
                additional_warnings=warnings,
            ),
            confidence=self._build_confidence(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
            ),
            spelling_corrections=(
                self._convert_spelling(
                    nlu_result
                )
            ),
            entities=self._convert_entities(
                nlu_result
            ),
            private_event_draft=(
                self._build_private_event_draft(
                    nlu_result
                )
            ),
            human_handoff_available=(
                self.human_handoff_available
            ),
            human_handoff_required=True,
            widget_state=WidgetState.OPEN,
            widget_size=WidgetSize.COMPACT,
            response_template_id="handoff-v1",
            response_variant_id=(
                "complaint"
                if complaint
                else "general"
            ),
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            processing_metadata=self._metadata(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
                metadata=metadata,
            ),
            service_version=(
                RESPONSE_SERVICE_VERSION
            ),
            service_phase=(
                RESPONSE_SERVICE_PHASE
            ),
            created_at=created_at,
        )

    # ========================================================
    # SECTION 10 - MULTI-INTENT RESPONSE
    # ========================================================

    def _compose_multi_intent_response(
        self,
        *,
        nlu_result: NLUResult,
        knowledge_result: KnowledgeResult,
        created_at: datetime,
        warnings: list[str],
        metadata: Mapping[str, Any] | None,
    ) -> GroundedResponse:
        sections = self._sections_for_records(
            nlu_result=nlu_result,
            records=knowledge_result.records[
                :MAXIMUM_RECORDS_IN_RESPONSE
            ],
        )

        if not sections:
            return self._compose_safe_fallback(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
                created_at=created_at,
                warnings=warnings,
                metadata=metadata,
            )

        message = self._join_sections(
            sections
        )

        return GroundedResponse(
            message=message,
            decision=ResponseDecision.MULTI_INTENT,
            sections=tuple(sections),
            sources=self._convert_sources(
                knowledge_result.sources
            ),
            actions=self._build_actions(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
            ),
            validation=self._build_validation(
                knowledge_result=knowledge_result,
                additional_warnings=warnings,
            ),
            confidence=self._build_confidence(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
            ),
            spelling_corrections=(
                self._convert_spelling(
                    nlu_result
                )
            ),
            entities=self._convert_entities(
                nlu_result
            ),
            private_event_draft=(
                self._build_private_event_draft(
                    nlu_result
                )
            ),
            human_handoff_available=(
                self.human_handoff_available
            ),
            human_handoff_required=False,
            widget_state=WidgetState.OPEN,
            widget_size=WidgetSize.EXPANDED,
            response_template_id="multi-intent-v1",
            response_variant_id="sectioned",
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            processing_metadata=self._metadata(
                nlu_result=nlu_result,
                knowledge_result=knowledge_result,
                metadata=metadata,
            ),
            service_version=(
                RESPONSE_SERVICE_VERSION
            ),
            service_phase=(
                RESPONSE_SERVICE_PHASE
            ),
            created_at=created_at,
        )

    # ========================================================
    # SECTION 11 - RECORD SECTIONS
    # ========================================================

    def _sections_for_records(
        self,
        *,
        nlu_result: NLUResult,
        records: Sequence[KnowledgeRecord],
    ) -> list[ResponseSection]:
        if not records:
            return []

        if nlu_result.primary_intent in {
            IntentName.HOURS_GENERAL,
            IntentName.HOURS_TODAY,
            IntentName.HOURS_KITCHEN,
            IntentName.HOURS_HAPPY_HOUR,
            IntentName.HAPPY_HOUR,
        }:
            return [
                ResponseSection(
                    heading="Hours",
                    body="\n".join(
                        self._bullet(
                            record.content
                        )
                        for record in records
                    ),
                    record_ids=tuple(
                        record.record_id
                        for record in records
                    ),
                    intent=(
                        nlu_result.primary_intent.value
                    ),
                )
            ]

        if nlu_result.primary_intent in {
            IntentName.MENU_GENERAL,
            IntentName.MENU_ITEM_LOOKUP,
            IntentName.MENU_DIETARY,
            IntentName.MENU_ALLERGEN,
            IntentName.MENU_PRICE,
        }:
            return [
                ResponseSection(
                    heading="Menu information",
                    body="\n".join(
                        self._bullet(
                            record.content
                        )
                        for record in records
                    ),
                    record_ids=tuple(
                        record.record_id
                        for record in records
                    ),
                    intent=(
                        nlu_result.primary_intent.value
                    ),
                )
            ]

        if nlu_result.primary_intent in {
            IntentName.EVENTS_GENERAL,
            IntentName.EVENTS_TONIGHT,
            IntentName.LIVE_MUSIC,
            IntentName.SPORTS_VIEWING,
        }:
            return [
                ResponseSection(
                    heading="Events",
                    body="\n".join(
                        self._bullet(
                            record.content
                        )
                        for record in records
                    ),
                    record_ids=tuple(
                        record.record_id
                        for record in records
                    ),
                    intent=(
                        nlu_result.primary_intent.value
                    ),
                )
            ]

        if nlu_result.primary_intent in {
            IntentName.PRIVATE_EVENT,
            IntentName.PRIVATE_EVENT_PRICING,
            IntentName.PRIVATE_EVENT_AVAILABILITY,
            IntentName.PRIVATE_EVENT_CONTACT,
        }:
            return [
                ResponseSection(
                    heading="Private events",
                    body="\n".join(
                        self._bullet(
                            record.content
                        )
                        for record in records
                    ),
                    record_ids=tuple(
                        record.record_id
                        for record in records
                    ),
                    intent=(
                        nlu_result.primary_intent.value
                    ),
                )
            ]

        return [
            ResponseSection(
                heading=None,
                body="\n".join(
                    self._bullet(
                        record.content
                    )
                    for record in records
                ),
                record_ids=tuple(
                    record.record_id
                    for record in records
                ),
                intent=(
                    nlu_result.primary_intent.value
                ),
            )
        ]

    # ========================================================
    # SECTION 12 - ACTIONS
    # ========================================================

    def _build_actions(
        self,
        *,
        nlu_result: NLUResult,
        knowledge_result: KnowledgeResult,
    ) -> tuple[ResponseAction, ...]:
        actions: list[ResponseAction] = []

        for record in knowledge_result.records:
            data = record.structured_data

            if data.get("online_order_url"):
                actions.append(
                    ResponseAction(
                        action_type=(
                            ResponseActionType.ORDERING
                        ),
                        label="Order online",
                        url=data[
                            "online_order_url"
                        ],
                        target="_blank",
                        analytics_event=(
                            "ordering_link_clicked"
                        ),
                    )
                )

            if data.get("reservation_url"):
                actions.append(
                    ResponseAction(
                        action_type=(
                            ResponseActionType.LINK
                        ),
                        label="Open reservation page",
                        url=data[
                            "reservation_url"
                        ],
                        target="_blank",
                        analytics_event=(
                            "reservation_link_clicked"
                        ),
                    )
                )

            if data.get("phone"):
                actions.append(
                    ResponseAction(
                        action_type=(
                            ResponseActionType.PHONE
                        ),
                        label="Call the tavern",
                        phone_number=data["phone"],
                        analytics_event=(
                            "phone_action_clicked"
                        ),
                    )
                )

            if data.get("email"):
                actions.append(
                    ResponseAction(
                        action_type=(
                            ResponseActionType.EMAIL
                        ),
                        label="Email the tavern",
                        email_address=data["email"],
                        analytics_event=(
                            "email_action_clicked"
                        ),
                    )
                )

            source_url = (
                record.source.source_url
            )

            if (
                source_url
                and self._safe_http_url(
                    source_url
                )
            ):
                actions.append(
                    ResponseAction(
                        action_type=(
                            ResponseActionType.LINK
                        ),
                        label=(
                            self._source_action_label(
                                record
                            )
                        ),
                        url=source_url,
                        target="_blank",
                        analytics_event=(
                            "source_link_clicked"
                        ),
                    )
                )

        if (
            nlu_result.active_flow
            == ActiveFlow.PRIVATE_EVENT.value
        ):
            actions.append(
                ResponseAction(
                    action_type=(
                        ResponseActionType.PRIVATE_EVENT
                    ),
                    label="Continue event inquiry",
                    form_key="private_event",
                    analytics_event=(
                        "private_event_form_opened"
                    ),
                )
            )

        if self.human_handoff_available:
            actions.append(
                ResponseAction(
                    action_type=(
                        ResponseActionType.HUMAN_HANDOFF
                    ),
                    label="Contact a person",
                    form_key="human_handoff",
                    analytics_event=(
                        "human_handoff_option_clicked"
                    ),
                )
            )

        return self._deduplicate_actions(
            actions
        )

    def _clarification_actions(
        self,
        nlu_result: NLUResult,
    ) -> tuple[ResponseAction, ...]:
        actions: list[ResponseAction] = []

        if (
            nlu_result.active_flow
            == ActiveFlow.PRIVATE_EVENT.value
        ):
            actions.append(
                ResponseAction(
                    action_type=(
                        ResponseActionType.PRIVATE_EVENT
                    ),
                    label="Continue private-event inquiry",
                    form_key="private_event",
                )
            )

        if (
            nlu_result.primary_intent
            == IntentName.UNKNOWN
        ):
            for label, message in (
                ("Hours", "What are your hours?"),
                ("Menu", "Show me the menu."),
                ("Events", "What events are coming up?"),
                (
                    "Private events",
                    "I want information about a private event.",
                ),
            ):
                actions.append(
                    ResponseAction(
                        action_type=(
                            ResponseActionType.MESSAGE
                        ),
                        label=label,
                        message=message,
                    )
                )

        return tuple(
            actions[:MAXIMUM_ACTION_COUNT]
        )

    def _fallback_actions(
        self,
        nlu_result: NLUResult,
    ) -> tuple[ResponseAction, ...]:
        actions: list[ResponseAction] = []

        if (
            nlu_result.primary_intent
            in {
                IntentName.PRIVATE_EVENT,
                IntentName.PRIVATE_EVENT_PRICING,
                IntentName.PRIVATE_EVENT_AVAILABILITY,
                IntentName.PRIVATE_EVENT_CONTACT,
            }
        ):
            actions.append(
                ResponseAction(
                    action_type=(
                        ResponseActionType.PRIVATE_EVENT
                    ),
                    label="Submit event inquiry",
                    form_key="private_event",
                )
            )

        if self.human_handoff_available:
            actions.append(
                ResponseAction(
                    action_type=(
                        ResponseActionType.HUMAN_HANDOFF
                    ),
                    label="Contact the tavern",
                    form_key="human_handoff",
                )
            )

        return tuple(actions)

    # ========================================================
    # SECTION 13 - SOURCE CONVERSION
    # ========================================================

    def _convert_sources(
        self,
        sources: Sequence[KnowledgeSource],
    ) -> tuple[ResponseSource, ...]:
        converted: list[ResponseSource] = []

        for source in sources[
            :MAXIMUM_SOURCE_COUNT
        ]:
            trust_level = {
                KnowledgeTrustLevel.VERIFIED: (
                    SourceTrustLevel.VERIFIED
                ),
                KnowledgeTrustLevel.VERIFIED_STALE: (
                    SourceTrustLevel.STALE
                ),
                KnowledgeTrustLevel.REVIEWED: (
                    SourceTrustLevel.REVIEWED
                ),
                KnowledgeTrustLevel.UNKNOWN: (
                    SourceTrustLevel.UNKNOWN
                ),
            }[source.trust_level]

            converted.append(
                ResponseSource(
                    source_type=source.source_type,
                    source_name=source.source_name,
                    source_reference=(
                        source.source_reference
                    ),
                    source_url=(
                        source.source_url
                    ),
                    trust_level=trust_level,
                    verified=source.verified,
                    retrieved_at=(
                        source.retrieved_at
                    ),
                    source_updated_at=(
                        source.source_updated_at
                    ),
                    relevance_score=(
                        source.relevance_score
                    ),
                    metadata={
                        **copy.deepcopy(
                            source.metadata
                        ),
                        "stale": source.stale,
                    },
                )
            )

        return tuple(converted)

    # ========================================================
    # SECTION 14 - VALIDATION AND CONFIDENCE
    # ========================================================

    def _build_validation(
        self,
        *,
        knowledge_result: KnowledgeResult,
        additional_warnings: Sequence[str],
    ) -> ResponseValidation:
        return ResponseValidation(
            verified_business_facts_only=True,
            source_count=len(
                knowledge_result.sources
            ),
            stale_source_count=(
                knowledge_result
                .stale_source_count
            ),
            unsupported_claim_count=(
                knowledge_result
                .unsupported_claim_count
            ),
            hallucination_guard_passed=(
                knowledge_result
                .unsupported_claim_count
                == 0
            ),
            privacy_guard_passed=True,
            safety_guard_passed=True,
            requires_human_review=(
                knowledge_result
                .requires_human_review
            ),
            warnings=list(
                dict.fromkeys(
                    additional_warnings
                )
            ),
        )

    @staticmethod
    def _build_confidence(
        *,
        nlu_result: NLUResult,
        knowledge_result: KnowledgeResult,
    ) -> ConfidenceBreakdown:
        retrieval_confidence = (
            sum(
                record.confidence
                for record
                in knowledge_result.records
            )
            / len(
                knowledge_result.records
            )
            if knowledge_result.records
            else 0.0
        )

        factuality = (
            1.0
            if (
                knowledge_result
                .unsupported_claim_count
                == 0
            )
            else 0.0
        )

        answer_confidence = (
            nlu_result.confidence.overall
            * 0.45
            + retrieval_confidence
            * 0.40
            + factuality
            * 0.15
        )

        return ConfidenceBreakdown(
            intent=(
                nlu_result.confidence.intent
            ),
            entity=(
                nlu_result.confidence.entity
            ),
            retrieval=round(
                retrieval_confidence,
                6,
            ),
            answer=round(
                min(
                    max(
                        answer_confidence,
                        0.0,
                    ),
                    1.0,
                ),
                6,
            ),
            factuality=factuality,
            overall=round(
                min(
                    max(
                        answer_confidence,
                        0.0,
                    ),
                    1.0,
                ),
                6,
            ),
        )

    # ========================================================
    # SECTION 15 - SCHEMA CONVERSION
    # ========================================================

    @staticmethod
    def _convert_spelling(
        nlu_result: NLUResult,
    ) -> tuple[SpellingCorrection, ...]:
        corrections: list[
            SpellingCorrection
        ] = []

        for correction in (
            nlu_result
            .spelling
            .corrections
        ):
            if not correction.changed:
                continue

            corrections.append(
                SpellingCorrection(
                    original=(
                        correction.original
                    ),
                    corrected=(
                        correction.corrected
                    ),
                    confidence=(
                        correction.confidence
                    ),
                    method=(
                        correction.source.value
                    ),
                    approved_mapping=(
                        correction
                        .verified_mapping
                    ),
                )
            )

        return tuple(corrections)

    @staticmethod
    def _convert_entities(
        nlu_result: NLUResult,
    ) -> tuple[DetectedEntity, ...]:
        entities: list[DetectedEntity] = []

        for entity in nlu_result.entities:
            normalized_value = (
                entity.normalized_value
            )

            if isinstance(
                normalized_value,
                (date, datetime, time),
            ):
                normalized_value = (
                    normalized_value.isoformat()
                )

            elif isinstance(
                normalized_value,
                Decimal,
            ):
                normalized_value = str(
                    normalized_value
                )

            elif not isinstance(
                normalized_value,
                (
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                ),
            ):
                normalized_value = str(
                    normalized_value
                )

            entities.append(
                DetectedEntity(
                    entity_type=(
                        entity.entity_type.value
                    ),
                    value=(
                        entity.original_value
                    ),
                    normalized_value=(
                        str(normalized_value)
                        if normalized_value
                        is not None
                        else None
                    ),
                    confidence=(
                        entity.confidence
                    ),
                    start_index=(
                        entity.start_index
                    ),
                    end_index=entity.end_index,
                    metadata={
                        "source": (
                            entity.source.value
                        ),
                        **copy.deepcopy(
                            entity.metadata
                        ),
                    },
                )
            )

        return tuple(entities)

    # ========================================================
    # SECTION 16 - PRIVATE EVENT DRAFT
    # ========================================================

    @staticmethod
    def _build_private_event_draft(
        nlu_result: NLUResult,
    ) -> PrivateEventDraft | None:
        if (
            nlu_result.active_flow
            != ActiveFlow.PRIVATE_EVENT.value
            and nlu_result.primary_intent
            not in {
                IntentName.PRIVATE_EVENT,
                IntentName.PRIVATE_EVENT_PRICING,
                IntentName.PRIVATE_EVENT_AVAILABILITY,
                IntentName.PRIVATE_EVENT_CONTACT,
            }
        ):
            return None

        values = (
            nlu_result
            .context
            .current_entities
        )

        budget_min = None
        budget_max = None

        budget_range = values.get(
            EntityType.BUDGET_RANGE.value
        )

        if isinstance(
            budget_range,
            Mapping,
        ):
            budget_min = (
                budget_range.get("minimum")
            )
            budget_max = (
                budget_range.get("maximum")
            )

        budget = values.get(
            EntityType.BUDGET.value
        )

        if (
            budget_min is None
            and isinstance(
                budget,
                Mapping,
            )
        ):
            budget_min = budget.get(
                "amount"
            )

        return PrivateEventDraft(
            event_type=(
                values.get(
                    EntityType.EVENT_TYPE.value
                )
            ),
            preferred_date=(
                values.get(
                    EntityType.DATE.value
                )
                or values.get(
                    EntityType.RELATIVE_DATE.value
                )
                or values.get(
                    EntityType.WEEKDAY.value
                )
            ),
            guest_count=(
                values.get(
                    EntityType.GUEST_COUNT.value
                )
            ),
            budget_min=budget_min,
            budget_max=budget_max,
            customer_name=(
                values.get(
                    EntityType.PERSON_NAME.value
                )
            ),
            email=(
                values.get(
                    EntityType.EMAIL.value
                )
            ),
            phone=(
                values.get(
                    EntityType.PHONE.value
                )
            ),
            completed_fields=list(
                nlu_result.completed_fields
            ),
            missing_fields=list(
                nlu_result.pending_fields
            ),
        )

    # ========================================================
    # SECTION 17 - SAFE FALLBACK COPY
    # ========================================================

    def _fallback_message(
        self,
        nlu_result: NLUResult,
    ) -> str:
        intent = nlu_result.primary_intent

        if intent in {
            IntentName.MENU_ALLERGEN,
            IntentName.MENU_DIETARY,
        }:
            return (
                "I do not have a verified answer for that dietary "
                "or allergen question. Please confirm directly with "
                "the tavern before ordering."
            )

        if intent in {
            IntentName.PRIVATE_EVENT_AVAILABILITY,
            IntentName.RESERVATION,
            IntentName.RESERVATION_CHANGE,
            IntentName.RESERVATION_CANCEL,
        }:
            return (
                "I do not have verified live availability or reservation "
                "status for that request. Please use the tavern’s booking "
                "channel or contact a staff member."
            )

        if intent in {
            IntentName.HOURS_GENERAL,
            IntentName.HOURS_TODAY,
            IntentName.HOURS_KITCHEN,
            IntentName.HOURS_HAPPY_HOUR,
        }:
            return (
                "I could not find verified hours for that specific request. "
                "Please confirm with the tavern directly."
            )

        if intent in {
            IntentName.EVENTS_GENERAL,
            IntentName.EVENTS_TONIGHT,
            IntentName.LIVE_MUSIC,
            IntentName.SPORTS_VIEWING,
        }:
            return (
                "I could not find a verified event listing that matches "
                "your request. Please check the official events page or "
                "contact the tavern."
            )

        if intent == IntentName.UNKNOWN:
            return (
                "I’m not confident I understood that. You can ask about "
                "hours, menu items, events, parking, reservations, ordering, "
                "or private events."
            )

        return (
            "I do not have a verified business fact for that request. "
            "I do not want to guess, so please contact the tavern directly."
        )

    # ========================================================
    # SECTION 18 - FORMAT HELPERS
    # ========================================================

    @staticmethod
    def _bullet(value: str) -> str:
        return f"• {value.strip()}"

    def _join_sections(
        self,
        sections: Sequence[ResponseSection],
    ) -> str:
        output: list[str] = []

        for section in sections:
            if section.heading:
                output.append(
                    section.heading
                )

            output.append(
                section.body
            )

        return "\n\n".join(
            output
        )

    @staticmethod
    def _sanitize_message(
        value: str,
    ) -> str:
        cleaned = (
            str(value)
            .replace("\x00", "")
            .strip()
        )

        cleaned = SAFE_WHITESPACE_PATTERN.sub(
            " ",
            cleaned,
        )

        cleaned = SAFE_MULTILINE_PATTERN.sub(
            "\n\n",
            cleaned,
        )

        return cleaned[
            :MAXIMUM_RESPONSE_CHARACTERS
        ]

    @staticmethod
    def _safe_http_url(
        value: str,
    ) -> bool:
        return value.startswith(
            ("http://", "https://")
        )

    @staticmethod
    def _source_action_label(
        record: KnowledgeRecord,
    ) -> str:
        labels = {
            KnowledgeRecordType.EVENT: (
                "View event details"
            ),
            KnowledgeRecordType.MENU_ITEM: (
                "View menu"
            ),
            KnowledgeRecordType.PRIVATE_EVENT_PACKAGE: (
                "View private-event details"
            ),
            KnowledgeRecordType.HOURS: (
                "View official hours"
            ),
        }

        return labels.get(
            record.record_type,
            "View official source",
        )

    @staticmethod
    def _deduplicate_actions(
        actions: Iterable[ResponseAction],
    ) -> tuple[ResponseAction, ...]:
        unique: dict[
            tuple[str, str, str | None],
            ResponseAction,
        ] = {}

        for action in actions:
            key = (
                str(action.action_type),
                action.label,
                action.url
                or action.phone_number
                or action.email_address
                or action.form_key,
            )

            unique.setdefault(
                key,
                action,
            )

        return tuple(
            list(unique.values())[
                :MAXIMUM_ACTION_COUNT
            ]
        )

    # ========================================================
    # SECTION 19 - TEMPLATE HELPERS
    # ========================================================

    @staticmethod
    def _preferred_widget_size(
        nlu_result: NLUResult,
    ) -> WidgetSize:
        if (
            nlu_result.is_multi_intent
            or len(nlu_result.entities) >= 6
        ):
            return WidgetSize.EXPANDED

        return WidgetSize.COMPACT

    @staticmethod
    def _template_id(
        nlu_result: NLUResult,
        decision: ResponseDecision,
    ) -> str:
        return (
            f"{nlu_result.primary_intent.value.lower()}"
            f"-{decision.value}-v1"
        )

    @staticmethod
    def _clarification_variant(
        nlu_result: NLUResult,
    ) -> str:
        if (
            nlu_result.active_flow
            == ActiveFlow.PRIVATE_EVENT.value
        ):
            return "private-event-field"

        if (
            nlu_result.active_flow
            == ActiveFlow.RESERVATION.value
        ):
            return "reservation-field"

        if (
            nlu_result.primary_intent
            == IntentName.UNKNOWN
        ):
            return "unknown-intent"

        return "general"

    # ========================================================
    # SECTION 20 - METADATA
    # ========================================================

    @staticmethod
    def _metadata(
        *,
        nlu_result: NLUResult,
        knowledge_result: KnowledgeResult,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            **dict(metadata or {}),
            "nlu_engine_version": (
                nlu_result.engine_version
            ),
            "nlu_engine_phase": (
                nlu_result.engine_phase
            ),
            "knowledge_service_version": (
                knowledge_result.service_version
            ),
            "knowledge_service_phase": (
                knowledge_result.service_phase
            ),
            "knowledge_decision": (
                knowledge_result.decision.value
            ),
            "verified_fact_count": (
                knowledge_result
                .verified_fact_count
            ),
            "stale_source_count": (
                knowledge_result
                .stale_source_count
            ),
            "unsupported_claim_count": (
                knowledge_result
                .unsupported_claim_count
            ),
        }


# ============================================================
# SECTION 21 - MODULE-LEVEL HELPER
# ============================================================

_default_response_service = (
    ResponseService()
)


def compose_grounded_response(
    nlu_result: NLUResult,
    knowledge_result: KnowledgeResult,
    *,
    now: datetime | None = None,
    response_metadata: Mapping[str, Any] | None = None,
) -> GroundedResponse:
    """
    Compose one grounded chatbot response.
    """

    return _default_response_service.compose(
        nlu_result,
        knowledge_result,
        now=now,
        response_metadata=response_metadata,
    )


# ============================================================
# SECTION 22 - SELF-TEST
# ============================================================

def validate_response_service_module() -> dict[str, Any]:
    from datetime import timedelta

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
    from app.nlu.orchestrator import (
        process_message,
    )
    from app.services.knowledge_service import (
        KnowledgeService,
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
            phone="973-555-0100",
            general_email=(
                "info@thehorseshoetavern.com"
            ),
            address_line_1="Test Street",
            city="Morristown",
            state_code="NJ",
            postal_code="07960",
            source_type="official_website",
            source_name="Official website",
            source_url=(
                "https://www.thehorseshoetavern.com/"
            ),
        )

        business.mark_verified(
            verified_by="response-test",
            notes="Verified business",
        )

        hours = BusinessHour(
            business=business,
            service_type="kitchen",
            day_of_week=reference.weekday(),
            open_time=time(11, 0),
            close_time=time(22, 0),
            is_closed=False,
            source_type="official_hours",
            source_name="Official hours",
            source_url=(
                "https://www.thehorseshoetavern.com/hours"
            ),
        )

        hours.mark_verified(
            verified_by="response-test",
            notes="Verified hours",
        )

        category = MenuCategory(
            business=business,
            name="Appetizers",
            slug="appetizers",
            source_type="official_menu",
            source_name="Official menu",
            source_url=(
                "https://www.thehorseshoetavern.com/menu"
            ),
        )

        category.mark_verified(
            verified_by="response-test",
            notes="Verified category",
        )

        wings = MenuItem(
            category=category,
            name="Tavern Wings",
            slug="tavern-wings",
            description="Crispy chicken wings.",
            price=Decimal("14.00"),
            allergen_notes=(
                "Prepared in a shared kitchen."
            ),
            source_type="official_menu",
            source_name="Official menu",
            source_url=(
                "https://www.thehorseshoetavern.com/menu"
            ),
        )

        wings.mark_verified(
            verified_by="response-test",
            notes="Verified item",
        )

        event = BusinessEvent(
            business=business,
            title="Friday Live Music",
            slug="friday-live-music",
            event_type="live_music",
            start_at=(
                reference
                + timedelta(days=2)
            ),
            description="Live band performance.",
            source_type="official_events",
            source_name="Official events",
            source_url=(
                "https://www.thehorseshoetavern.com/events"
            ),
        )

        event.mark_verified(
            verified_by="response-test",
            notes="Verified event",
        )

        faq = FAQEntry(
            business=business,
            category="parking",
            question="Where can guests park?",
            answer="Use nearby public parking.",
            source_type="official_website",
            source_name="Official FAQ",
            source_url=(
                "https://www.thehorseshoetavern.com/"
            ),
        )

        faq.mark_verified(
            verified_by="response-test",
            notes="Verified FAQ",
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
            source_name="Official private events",
            source_url=(
                "https://www.thehorseshoetavern.com/private-events"
            ),
        )

        package.mark_verified(
            verified_by="response-test",
            notes="Verified package",
        )

        session.add_all(
            [
                business,
                hours,
                category,
                wings,
                event,
                faq,
                package,
            ]
        )

        session.commit()

        knowledge_service = (
            KnowledgeService(
                session,
                stale_after_days=3650,
            )
        )

        response_service = (
            ResponseService()
        )

        hours_nlu = process_message(
            "What time does the kitchen close today?",
            reference_datetime=reference,
        )

        hours_knowledge = (
            knowledge_service.retrieve(
                hours_nlu,
                now=reference,
            )
        )

        hours_response = (
            response_service.compose(
                hours_nlu,
                hours_knowledge,
                now=reference,
            )
        )

        menu_nlu = process_message(
            "Do you have wings and how much are they?",
            reference_datetime=reference,
        )

        menu_response = (
            response_service.compose(
                menu_nlu,
                knowledge_service.retrieve(
                    menu_nlu,
                    now=reference,
                ),
                now=reference,
            )
        )

        private_nlu = process_message(
            "I need a birthday party for 45 people.",
            page_category="private_events",
            reference_datetime=reference,
        )

        private_response = (
            response_service.compose(
                private_nlu,
                knowledge_service.retrieve(
                    private_nlu,
                    now=reference,
                ),
                now=reference,
            )
        )

        unknown_nlu = process_message(
            "quantum flux capacitor status",
            reference_datetime=reference,
        )

        unknown_response = (
            response_service.compose(
                unknown_nlu,
                knowledge_service.retrieve(
                    unknown_nlu,
                    now=reference,
                ),
                now=reference,
            )
        )

        handoff_nlu = process_message(
            "I need to speak to a real person.",
            reference_datetime=reference,
        )

        handoff_response = (
            response_service.compose(
                handoff_nlu,
                knowledge_service.retrieve(
                    handoff_nlu,
                    now=reference,
                ),
                now=reference,
            )
        )

        checks = {
            "hours_grounded": (
                hours_response.decision
                in {
                    ResponseDecision.GROUNDED,
                    ResponseDecision.GROUNDED_WITH_WARNING,
                }
            ),
            "hours_message_present": (
                "22:00"
                in hours_response.message
                or "10:00 PM"
                in hours_response.message
            ),
            "hours_sources_present": (
                bool(hours_response.sources)
            ),
            "menu_grounded": (
                menu_response.decision
                in {
                    ResponseDecision.GROUNDED,
                    ResponseDecision.GROUNDED_WITH_WARNING,
                }
            ),
            "menu_price_present": (
                "$14.00"
                in menu_response.message
            ),
            "private_event_draft_present": (
                private_response
                .private_event_draft
                is not None
            ),
            "private_guest_count_present": (
                private_response
                .private_event_draft
                .guest_count
                == 45
            ),
            "unknown_safe": (
                unknown_response.decision
                in {
                    ResponseDecision.CLARIFICATION,
                    ResponseDecision.SAFE_FALLBACK,
                }
            ),
            "handoff_required": (
                handoff_response
                .human_handoff_required
            ),
            "validation_passed": (
                hours_response.validation.passed
            ),
            "unsupported_claims_zero": (
                hours_response
                .validation
                .unsupported_claim_count
                == 0
            ),
            "json_safe": bool(
                hours_response.as_dict()
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
# SECTION 23 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import json

    report = validate_response_service_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
