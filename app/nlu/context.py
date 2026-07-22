# ============================================================
# Exact file location: app/nlu/context.py
# Horseshoe Tavern AI
# Phase 1 Part 1.16
# Lightweight multi-turn context, explicit overrides, entity
# memory, active workflow state, pronoun resolution, and expiry
# ============================================================

"""
Conversation context engine for Horseshoe Tavern AI.

Responsibilities:

- Maintain lightweight multi-turn conversation context
- Preserve useful entities between turns
- Track active operational workflows
- Track pending questions and required fields
- Resolve pronouns only when required and sufficiently confident
- Prevent previous-intent lock
- Let explicit new requests override old context immediately
- Let newly supplied names, dates, times, counts, and budgets override old ones
- Retain page context without making page context authoritative
- Expire stale context automatically
- Preserve conversation summaries and recent-turn metadata
- Support private-event, reservation, ordering, and human-handoff flows
- Return deterministic, JSON-safe context payloads
- Never convert unverified user input into verified business facts

Context is supporting evidence only. Current-message evidence remains primary.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Final, Iterable, Mapping, MutableMapping, Sequence

from app.logging_config import get_logger
from app.nlu.entities import (
    EntityExtractionResult,
    EntityType,
    ExtractedEntity,
)
from app.nlu.intent import (
    IntentDecision,
    IntentName,
    IntentResult,
)


# ============================================================
# SECTION 01 - LOGGER AND CONSTANTS
# ============================================================

logger = get_logger(__name__)

CONTEXT_VERSION: Final[str] = "1.0.0"

DEFAULT_CONTEXT_TTL_MINUTES: Final[int] = 45
DEFAULT_MAXIMUM_RECENT_TURNS: Final[int] = 12
DEFAULT_MAXIMUM_ENTITY_HISTORY: Final[int] = 50
DEFAULT_MAXIMUM_PENDING_FIELDS: Final[int] = 25
DEFAULT_MAXIMUM_PAGE_HISTORY: Final[int] = 10

EXPLICIT_RESET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bstart over\b", re.IGNORECASE),
    re.compile(r"\breset\b", re.IGNORECASE),
    re.compile(r"\bclear that\b", re.IGNORECASE),
    re.compile(r"\bforget that\b", re.IGNORECASE),
    re.compile(r"\bnever mind\b", re.IGNORECASE),
    re.compile(r"\bnew conversation\b", re.IGNORECASE),
)

EXPLICIT_OVERRIDE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bactually\b", re.IGNORECASE),
    re.compile(r"\binstead\b", re.IGNORECASE),
    re.compile(r"\bnew question\b", re.IGNORECASE),
    re.compile(r"\bseparately\b", re.IGNORECASE),
    re.compile(r"\banother question\b", re.IGNORECASE),
    re.compile(r"\bwhat about\b", re.IGNORECASE),
    re.compile(r"\bchanging the subject\b", re.IGNORECASE),
)

PRONOUN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(it|that|this|they|them|there|those|these|he|she|him|her)\b",
    re.IGNORECASE,
)

ELLIPSIS_FOLLOWUP_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^\s*(?:and|also|what about|how about)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:what time|how much|when|where|which one)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:yes|no|sure|okay|ok)\b", re.IGNORECASE),
)


# ============================================================
# SECTION 02 - ENUMERATIONS
# ============================================================

class ActiveFlow(str, Enum):
    NONE = "none"
    PRIVATE_EVENT = "private_event"
    RESERVATION = "reservation"
    ORDERING = "ordering"
    HUMAN_HANDOFF = "human_handoff"
    FEEDBACK = "feedback"


class ContextDecision(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    OVERRIDDEN = "overridden"
    RESET = "reset"
    EXPIRED = "expired"
    UNCHANGED = "unchanged"


class ContextSource(str, Enum):
    CURRENT_MESSAGE = "current_message"
    PRIOR_TURN = "prior_turn"
    PAGE_CONTEXT = "page_context"
    ACTIVE_FLOW = "active_flow"
    SYSTEM = "system"


# ============================================================
# SECTION 03 - DATA CLASSES
# ============================================================

@dataclass(frozen=True, slots=True)
class ContextEntity:
    entity_type: str
    value: Any
    original_value: str | None
    confidence: float
    source: ContextSource
    observed_at: datetime
    turn_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        value = self.value

        if isinstance(value, (date, datetime, time)):
            value = value.isoformat()

        if isinstance(value, Decimal):
            value = str(value)

        return {
            "entity_type": self.entity_type,
            "value": value,
            "original_value": self.original_value,
            "confidence": self.confidence,
            "source": self.source.value,
            "observed_at": self.observed_at.isoformat(),
            "turn_index": self.turn_index,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RecentTurn:
    turn_index: int
    user_text: str
    normalized_text: str
    corrected_text: str
    primary_intent: str
    confidence: float
    decision: str
    entity_types: tuple[str, ...]
    page_category: str | None
    occurred_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "user_text": self.user_text,
            "normalized_text": self.normalized_text,
            "corrected_text": self.corrected_text,
            "primary_intent": self.primary_intent,
            "confidence": self.confidence,
            "decision": self.decision,
            "entity_types": list(self.entity_types),
            "page_category": self.page_category,
            "occurred_at": self.occurred_at.isoformat(),
        }


@dataclass(slots=True)
class ConversationContext:
    version: str = CONTEXT_VERSION
    conversation_id: str | None = None
    session_id: str | None = None

    active_flow: str = ActiveFlow.NONE.value
    current_intent: str | None = None
    previous_intent: str | None = None

    current_entities: dict[str, Any] = field(default_factory=dict)
    entity_history: list[ContextEntity] = field(default_factory=list)

    pending_fields: list[str] = field(default_factory=list)
    completed_fields: list[str] = field(default_factory=list)

    page_category: str | None = None
    page_url: str | None = None
    page_history: list[dict[str, Any]] = field(default_factory=list)

    recent_turns: list[RecentTurn] = field(default_factory=list)

    summary: str | None = None
    human_handoff_requested: bool = False

    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    updated_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    expires_at: datetime = field(
        default_factory=lambda: (
            datetime.now().astimezone()
            + timedelta(minutes=DEFAULT_CONTEXT_TTL_MINUTES)
        )
    )

    turn_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(
        self,
        *,
        now: datetime | None = None,
    ) -> bool:
        reference = now or datetime.now().astimezone()
        return reference >= self.expires_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "active_flow": self.active_flow,
            "current_intent": self.current_intent,
            "previous_intent": self.previous_intent,
            "current_entities": _json_safe(self.current_entities),
            "entity_history": [
                item.as_dict()
                for item in self.entity_history
            ],
            "pending_fields": list(self.pending_fields),
            "completed_fields": list(self.completed_fields),
            "page_category": self.page_category,
            "page_url": self.page_url,
            "page_history": copy.deepcopy(self.page_history),
            "recent_turns": [
                turn.as_dict()
                for turn in self.recent_turns
            ],
            "summary": self.summary,
            "human_handoff_requested": self.human_handoff_requested,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "turn_count": self.turn_count,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "ConversationContext":
        if not payload:
            return cls()

        context = cls(
            version=str(payload.get("version") or CONTEXT_VERSION),
            conversation_id=_optional_text(payload.get("conversation_id")),
            session_id=_optional_text(payload.get("session_id")),
            active_flow=str(
                payload.get("active_flow")
                or ActiveFlow.NONE.value
            ),
            current_intent=_optional_text(payload.get("current_intent")),
            previous_intent=_optional_text(payload.get("previous_intent")),
            current_entities=copy.deepcopy(
                payload.get("current_entities") or {}
            ),
            pending_fields=list(
                payload.get("pending_fields") or []
            ),
            completed_fields=list(
                payload.get("completed_fields") or []
            ),
            page_category=_optional_text(payload.get("page_category")),
            page_url=_optional_text(payload.get("page_url")),
            page_history=list(
                payload.get("page_history") or []
            ),
            summary=_optional_text(payload.get("summary")),
            human_handoff_requested=bool(
                payload.get("human_handoff_requested", False)
            ),
            created_at=_parse_datetime(
                payload.get("created_at")
            ) or datetime.now().astimezone(),
            updated_at=_parse_datetime(
                payload.get("updated_at")
            ) or datetime.now().astimezone(),
            expires_at=_parse_datetime(
                payload.get("expires_at")
            )
            or (
                datetime.now().astimezone()
                + timedelta(minutes=DEFAULT_CONTEXT_TTL_MINUTES)
            ),
            turn_count=int(
                payload.get("turn_count") or 0
            ),
            metadata=copy.deepcopy(
                payload.get("metadata") or {}
            ),
        )

        entity_history_payload = payload.get(
            "entity_history"
        ) or []

        for item in entity_history_payload:
            if not isinstance(item, Mapping):
                continue

            observed_at = _parse_datetime(
                item.get("observed_at")
            ) or context.updated_at

            source_text = str(
                item.get("source")
                or ContextSource.PRIOR_TURN.value
            )

            try:
                source = ContextSource(source_text)
            except ValueError:
                source = ContextSource.PRIOR_TURN

            context.entity_history.append(
                ContextEntity(
                    entity_type=str(
                        item.get("entity_type") or ""
                    ),
                    value=copy.deepcopy(
                        item.get("value")
                    ),
                    original_value=_optional_text(
                        item.get("original_value")
                    ),
                    confidence=float(
                        item.get("confidence") or 0.0
                    ),
                    source=source,
                    observed_at=observed_at,
                    turn_index=int(
                        item.get("turn_index") or 0
                    ),
                    metadata=copy.deepcopy(
                        item.get("metadata") or {}
                    ),
                )
            )

        recent_turn_payload = payload.get(
            "recent_turns"
        ) or []

        for item in recent_turn_payload:
            if not isinstance(item, Mapping):
                continue

            context.recent_turns.append(
                RecentTurn(
                    turn_index=int(
                        item.get("turn_index") or 0
                    ),
                    user_text=str(
                        item.get("user_text") or ""
                    ),
                    normalized_text=str(
                        item.get("normalized_text") or ""
                    ),
                    corrected_text=str(
                        item.get("corrected_text") or ""
                    ),
                    primary_intent=str(
                        item.get("primary_intent")
                        or IntentName.UNKNOWN.value
                    ),
                    confidence=float(
                        item.get("confidence") or 0.0
                    ),
                    decision=str(
                        item.get("decision")
                        or IntentDecision.UNKNOWN.value
                    ),
                    entity_types=tuple(
                        str(value)
                        for value in (
                            item.get("entity_types") or []
                        )
                    ),
                    page_category=_optional_text(
                        item.get("page_category")
                    ),
                    occurred_at=_parse_datetime(
                        item.get("occurred_at")
                    ) or context.updated_at,
                )
            )

        return context


@dataclass(frozen=True, slots=True)
class ContextUpdateResult:
    context: ConversationContext
    decision: ContextDecision
    explicit_reset: bool
    explicit_override: bool
    expired_previous_context: bool
    previous_intent_used: bool
    resolved_references: dict[str, Any]
    changed_fields: tuple[str, ...]
    removed_fields: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.as_dict(),
            "decision": self.decision.value,
            "explicit_reset": self.explicit_reset,
            "explicit_override": self.explicit_override,
            "expired_previous_context": self.expired_previous_context,
            "previous_intent_used": self.previous_intent_used,
            "resolved_references": _json_safe(
                self.resolved_references
            ),
            "changed_fields": list(self.changed_fields),
            "removed_fields": list(self.removed_fields),
            "warnings": list(self.warnings),
        }


# ============================================================
# SECTION 04 - FIELD REQUIREMENTS
# ============================================================

FLOW_REQUIRED_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    ActiveFlow.PRIVATE_EVENT.value: (
        EntityType.EVENT_TYPE.value,
        EntityType.DATE.value,
        EntityType.GUEST_COUNT.value,
        EntityType.EMAIL.value,
    ),
    ActiveFlow.RESERVATION.value: (
        EntityType.DATE.value,
        EntityType.TIME.value,
        EntityType.GUEST_COUNT.value,
    ),
    ActiveFlow.ORDERING.value: (),
    ActiveFlow.HUMAN_HANDOFF.value: (),
    ActiveFlow.FEEDBACK.value: (),
}


FLOW_INTENTS: Final[dict[str, set[IntentName]]] = {
    ActiveFlow.PRIVATE_EVENT.value: {
        IntentName.PRIVATE_EVENT,
        IntentName.PRIVATE_EVENT_PRICING,
        IntentName.PRIVATE_EVENT_AVAILABILITY,
        IntentName.PRIVATE_EVENT_CONTACT,
    },
    ActiveFlow.RESERVATION.value: {
        IntentName.RESERVATION,
        IntentName.RESERVATION_CHANGE,
        IntentName.RESERVATION_CANCEL,
    },
    ActiveFlow.ORDERING.value: {
        IntentName.ORDERING,
        IntentName.TAKEOUT,
        IntentName.DELIVERY,
    },
    ActiveFlow.HUMAN_HANDOFF.value: {
        IntentName.HUMAN_HANDOFF,
        IntentName.COMPLAINT,
    },
    ActiveFlow.FEEDBACK.value: {
        IntentName.FEEDBACK,
    },
}


REPLACEABLE_ENTITY_TYPES: Final[set[str]] = {
    EntityType.DATE.value,
    EntityType.RELATIVE_DATE.value,
    EntityType.WEEKDAY.value,
    EntityType.TIME.value,
    EntityType.TIME_RANGE.value,
    EntityType.GUEST_COUNT.value,
    EntityType.BUDGET.value,
    EntityType.BUDGET_RANGE.value,
    EntityType.EMAIL.value,
    EntityType.PHONE.value,
    EntityType.PERSON_NAME.value,
    EntityType.EVENT_TYPE.value,
    EntityType.RESERVATION_REFERENCE.value,
    EntityType.SERVICE_TYPE.value,
}


MULTI_VALUE_ENTITY_TYPES: Final[set[str]] = {
    EntityType.MENU_ITEM.value,
    EntityType.MENU_CATEGORY.value,
    EntityType.DIETARY_REQUIREMENT.value,
    EntityType.ALLERGEN.value,
    EntityType.BEVERAGE_TYPE.value,
    EntityType.SPORTS_TEAM.value,
    EntityType.SPORTS_LEAGUE.value,
    EntityType.SPORTS_TYPE.value,
    EntityType.LOCATION.value,
}


# ============================================================
# SECTION 05 - CONTEXT MANAGER
# ============================================================

class ConversationContextManager:
    """
    Deterministic multi-turn context manager.
    """

    def __init__(
        self,
        *,
        ttl_minutes: int = DEFAULT_CONTEXT_TTL_MINUTES,
        maximum_recent_turns: int = DEFAULT_MAXIMUM_RECENT_TURNS,
        maximum_entity_history: int = DEFAULT_MAXIMUM_ENTITY_HISTORY,
        maximum_pending_fields: int = DEFAULT_MAXIMUM_PENDING_FIELDS,
        maximum_page_history: int = DEFAULT_MAXIMUM_PAGE_HISTORY,
    ) -> None:
        self.ttl_minutes = max(
            1,
            int(ttl_minutes),
        )

        self.maximum_recent_turns = max(
            1,
            int(maximum_recent_turns),
        )

        self.maximum_entity_history = max(
            1,
            int(maximum_entity_history),
        )

        self.maximum_pending_fields = max(
            1,
            int(maximum_pending_fields),
        )

        self.maximum_page_history = max(
            1,
            int(maximum_page_history),
        )

    # ========================================================
    # SECTION 06 - UPDATE ENTRYPOINT
    # ========================================================

    def update(
        self,
        *,
        user_text: str,
        intent_result: IntentResult,
        entity_result: EntityExtractionResult,
        previous_context: ConversationContext | Mapping[str, Any] | None = None,
        conversation_id: str | None = None,
        session_id: str | None = None,
        page_category: str | None = None,
        page_url: str | None = None,
        now: datetime | None = None,
    ) -> ContextUpdateResult:
        reference = (
            now
            or datetime.now().astimezone()
        )

        context = self._coerce_context(
            previous_context
        )

        expired_previous_context = False
        explicit_reset = self._matches_any(
            user_text,
            EXPLICIT_RESET_PATTERNS,
        )
        explicit_override = (
            intent_result.explicit_override
            or self._matches_any(
                user_text,
                EXPLICIT_OVERRIDE_PATTERNS,
            )
        )

        changed_fields: list[str] = []
        removed_fields: list[str] = []
        warnings: list[str] = []
        resolved_references: dict[str, Any] = {}

        if context.is_expired(now=reference):
            context = ConversationContext(
                conversation_id=(
                    conversation_id
                    or context.conversation_id
                ),
                session_id=(
                    session_id
                    or context.session_id
                ),
                created_at=reference,
                updated_at=reference,
                expires_at=(
                    reference
                    + timedelta(
                        minutes=self.ttl_minutes
                    )
                ),
            )

            expired_previous_context = True

        if explicit_reset:
            prior_conversation_id = (
                conversation_id
                or context.conversation_id
            )
            prior_session_id = (
                session_id
                or context.session_id
            )

            context = ConversationContext(
                conversation_id=prior_conversation_id,
                session_id=prior_session_id,
                created_at=reference,
                updated_at=reference,
                expires_at=(
                    reference
                    + timedelta(
                        minutes=self.ttl_minutes
                    )
                ),
            )

            self._record_page_context(
                context,
                page_category=page_category,
                page_url=page_url,
                observed_at=reference,
            )

            return ContextUpdateResult(
                context=context,
                decision=ContextDecision.RESET,
                explicit_reset=True,
                explicit_override=explicit_override,
                expired_previous_context=expired_previous_context,
                previous_intent_used=False,
                resolved_references={},
                changed_fields=(),
                removed_fields=(),
                warnings=(),
            )

        if conversation_id:
            context.conversation_id = conversation_id

        if session_id:
            context.session_id = session_id

        context.turn_count += 1
        turn_index = context.turn_count

        previous_current_intent = context.current_intent
        previous_active_flow = context.active_flow

        new_intent = intent_result.primary_intent

        if explicit_override:
            self._handle_explicit_override(
                context,
                new_intent,
                removed_fields,
            )

        previous_intent_used = self._should_use_previous_intent(
            user_text=user_text,
            intent_result=intent_result,
            context=context,
            explicit_override=explicit_override,
        )

        effective_intent = self._resolve_effective_intent(
            intent_result=intent_result,
            context=context,
            previous_intent_used=previous_intent_used,
        )

        if effective_intent != IntentName.UNKNOWN:
            if context.current_intent != effective_intent.value:
                context.previous_intent = (
                    context.current_intent
                )
                context.current_intent = (
                    effective_intent.value
                )
                changed_fields.append(
                    "current_intent"
                )

        active_flow = self._flow_for_intent(
            effective_intent
        )

        if active_flow != ActiveFlow.NONE.value:
            if context.active_flow != active_flow:
                context.active_flow = active_flow
                changed_fields.append(
                    "active_flow"
                )
        elif (
            explicit_override
            and effective_intent != IntentName.UNKNOWN
        ):
            context.active_flow = (
                ActiveFlow.NONE.value
            )

            if previous_active_flow != (
                ActiveFlow.NONE.value
            ):
                changed_fields.append(
                    "active_flow"
                )

        extracted_updates = (
            self._entities_to_context_updates(
                entity_result.entities
            )
        )

        entity_changed = self._merge_entities(
            context=context,
            updates=extracted_updates,
            observed_at=reference,
            turn_index=turn_index,
            changed_fields=changed_fields,
        )

        resolved_references = self._resolve_references(
            user_text=user_text,
            context=context,
            entity_result=entity_result,
            explicit_override=explicit_override,
        )

        self._update_flow_fields(
            context,
            changed_fields,
        )

        if effective_intent in {
            IntentName.HUMAN_HANDOFF,
            IntentName.COMPLAINT,
        }:
            if not context.human_handoff_requested:
                context.human_handoff_requested = True
                changed_fields.append(
                    "human_handoff_requested"
                )

        self._record_page_context(
            context,
            page_category=page_category,
            page_url=page_url,
            observed_at=reference,
        )

        self._record_turn(
            context,
            user_text=user_text,
            intent_result=intent_result,
            entity_result=entity_result,
            page_category=page_category,
            observed_at=reference,
        )

        context.updated_at = reference
        context.expires_at = (
            reference
            + timedelta(
                minutes=self.ttl_minutes
            )
        )

        context.summary = self._build_summary(
            context
        )

        context.entity_history = (
            context.entity_history[
                -self.maximum_entity_history:
            ]
        )

        context.recent_turns = (
            context.recent_turns[
                -self.maximum_recent_turns:
            ]
        )

        context.page_history = (
            context.page_history[
                -self.maximum_page_history:
            ]
        )

        if (
            intent_result.decision
            == IntentDecision.AMBIGUOUS
        ):
            warnings.append(
                "Current intent is ambiguous; context was not allowed to override explicit current-message evidence."
            )

        if (
            intent_result.primary_intent
            == IntentName.UNKNOWN
            and not previous_intent_used
        ):
            warnings.append(
                "No reliable current or prior intent was available."
            )

        decision = self._determine_decision(
            expired_previous_context=expired_previous_context,
            explicit_override=explicit_override,
            previous_current_intent=previous_current_intent,
            context=context,
            changed_fields=changed_fields,
            entity_changed=entity_changed,
        )

        return ContextUpdateResult(
            context=context,
            decision=decision,
            explicit_reset=False,
            explicit_override=explicit_override,
            expired_previous_context=expired_previous_context,
            previous_intent_used=previous_intent_used,
            resolved_references=resolved_references,
            changed_fields=tuple(
                dict.fromkeys(changed_fields)
            ),
            removed_fields=tuple(
                dict.fromkeys(removed_fields)
            ),
            warnings=tuple(warnings),
        )

    # ========================================================
    # SECTION 07 - CONTEXT COERCION
    # ========================================================

    @staticmethod
    def _coerce_context(
        value: ConversationContext | Mapping[str, Any] | None,
    ) -> ConversationContext:
        if value is None:
            return ConversationContext()

        if isinstance(
            value,
            ConversationContext,
        ):
            return copy.deepcopy(value)

        return ConversationContext.from_dict(
            value
        )

    # ========================================================
    # SECTION 08 - INTENT RESOLUTION
    # ========================================================

    def _should_use_previous_intent(
        self,
        *,
        user_text: str,
        intent_result: IntentResult,
        context: ConversationContext,
        explicit_override: bool,
    ) -> bool:
        if explicit_override:
            return False

        if not context.current_intent:
            return False

        if (
            intent_result.primary_intent
            != IntentName.UNKNOWN
            and intent_result.decision
            in {
                IntentDecision.ACCEPTED,
                IntentDecision.MULTI_INTENT,
            }
        ):
            return False

        if not self._looks_like_followup(
            user_text
        ):
            return False

        try:
            previous_intent = IntentName(
                context.current_intent
            )
        except ValueError:
            return False

        active_flow = context.active_flow
        compatible = FLOW_INTENTS.get(
            active_flow,
            set(),
        )

        if compatible and previous_intent not in compatible:
            return False

        return True

    @staticmethod
    def _resolve_effective_intent(
        *,
        intent_result: IntentResult,
        context: ConversationContext,
        previous_intent_used: bool,
    ) -> IntentName:
        if (
            intent_result.primary_intent
            != IntentName.UNKNOWN
        ):
            return intent_result.primary_intent

        if (
            previous_intent_used
            and context.current_intent
        ):
            try:
                return IntentName(
                    context.current_intent
                )
            except ValueError:
                return IntentName.UNKNOWN

        return IntentName.UNKNOWN

    @staticmethod
    def _flow_for_intent(
        intent: IntentName,
    ) -> str:
        for flow_name, intents in (
            FLOW_INTENTS.items()
        ):
            if intent in intents:
                return flow_name

        return ActiveFlow.NONE.value

    # ========================================================
    # SECTION 09 - EXPLICIT OVERRIDE HANDLING
    # ========================================================

    @staticmethod
    def _handle_explicit_override(
        context: ConversationContext,
        new_intent: IntentName,
        removed_fields: list[str],
    ) -> None:
        if new_intent == IntentName.UNKNOWN:
            return

        new_flow = (
            ConversationContextManager
            ._flow_for_intent(
                new_intent
            )
        )

        if (
            context.active_flow
            != ActiveFlow.NONE.value
            and new_flow
            != context.active_flow
        ):
            context.active_flow = (
                ActiveFlow.NONE.value
            )
            context.pending_fields.clear()
            context.completed_fields.clear()

            removed_fields.extend(
                [
                    "active_flow",
                    "pending_fields",
                    "completed_fields",
                ]
            )

    # ========================================================
    # SECTION 10 - ENTITY MERGING
    # ========================================================

    @staticmethod
    def _entities_to_context_updates(
        entities: Sequence[ExtractedEntity],
    ) -> dict[str, list[ExtractedEntity]]:
        updates: dict[
            str,
            list[ExtractedEntity],
        ] = {}

        for entity in entities:
            if entity.metadata.get(
                "synthetic"
            ):
                continue

            updates.setdefault(
                entity.entity_type.value,
                [],
            ).append(entity)

        return updates

    def _merge_entities(
        self,
        *,
        context: ConversationContext,
        updates: Mapping[
            str,
            Sequence[ExtractedEntity],
        ],
        observed_at: datetime,
        turn_index: int,
        changed_fields: list[str],
    ) -> bool:
        changed = False

        for entity_type, values in (
            updates.items()
        ):
            if not values:
                continue

            if (
                entity_type
                in MULTI_VALUE_ENTITY_TYPES
            ):
                existing = context.current_entities.get(
                    entity_type,
                    [],
                )

                if not isinstance(existing, list):
                    existing = [existing]

                merged = list(existing)

                for entity in values:
                    normalized = _json_safe(
                        entity.normalized_value
                    )

                    if normalized not in merged:
                        merged.append(normalized)

                    context.entity_history.append(
                        ContextEntity(
                            entity_type=entity_type,
                            value=normalized,
                            original_value=(
                                entity.original_value
                            ),
                            confidence=entity.confidence,
                            source=(
                                ContextSource.CURRENT_MESSAGE
                            ),
                            observed_at=observed_at,
                            turn_index=turn_index,
                            metadata=copy.deepcopy(
                                entity.metadata
                            ),
                        )
                    )

                if merged != existing:
                    context.current_entities[
                        entity_type
                    ] = merged
                    changed_fields.append(
                        f"entity:{entity_type}"
                    )
                    changed = True

                continue

            best = sorted(
                values,
                key=lambda item: (
                    item.confidence,
                    item.length,
                ),
                reverse=True,
            )[0]

            normalized_value = _json_safe(
                best.normalized_value
            )

            previous_value = (
                context.current_entities.get(
                    entity_type
                )
            )

            if (
                entity_type
                in REPLACEABLE_ENTITY_TYPES
                or previous_value is None
            ):
                if (
                    previous_value
                    != normalized_value
                ):
                    context.current_entities[
                        entity_type
                    ] = normalized_value
                    changed_fields.append(
                        f"entity:{entity_type}"
                    )
                    changed = True

            context.entity_history.append(
                ContextEntity(
                    entity_type=entity_type,
                    value=normalized_value,
                    original_value=(
                        best.original_value
                    ),
                    confidence=best.confidence,
                    source=(
                        ContextSource.CURRENT_MESSAGE
                    ),
                    observed_at=observed_at,
                    turn_index=turn_index,
                    metadata=copy.deepcopy(
                        best.metadata
                    ),
                )
            )

        return changed

    # ========================================================
    # SECTION 11 - REFERENCE RESOLUTION
    # ========================================================

    def _resolve_references(
        self,
        *,
        user_text: str,
        context: ConversationContext,
        entity_result: EntityExtractionResult,
        explicit_override: bool,
    ) -> dict[str, Any]:
        if explicit_override:
            return {}

        if not PRONOUN_PATTERN.search(
            user_text
        ):
            return {}

        if entity_result.entities:
            return {}

        resolved: dict[str, Any] = {}

        priority_types = self._reference_priority_for_flow(
            context.active_flow
        )

        for entity_type in priority_types:
            if entity_type in context.current_entities:
                resolved[entity_type] = copy.deepcopy(
                    context.current_entities[
                        entity_type
                    ]
                )

        return resolved

    @staticmethod
    def _reference_priority_for_flow(
        active_flow: str,
    ) -> tuple[str, ...]:
        if (
            active_flow
            == ActiveFlow.PRIVATE_EVENT.value
        ):
            return (
                EntityType.EVENT_TYPE.value,
                EntityType.DATE.value,
                EntityType.GUEST_COUNT.value,
                EntityType.BUDGET.value,
                EntityType.BUDGET_RANGE.value,
            )

        if (
            active_flow
            == ActiveFlow.RESERVATION.value
        ):
            return (
                EntityType.RESERVATION_REFERENCE.value,
                EntityType.DATE.value,
                EntityType.TIME.value,
                EntityType.GUEST_COUNT.value,
            )

        if (
            active_flow
            == ActiveFlow.ORDERING.value
        ):
            return (
                EntityType.MENU_ITEM.value,
                EntityType.SERVICE_TYPE.value,
            )

        return (
            EntityType.BUSINESS_NAME.value,
            EntityType.LOCATION.value,
            EntityType.MENU_ITEM.value,
            EntityType.SPORTS_TEAM.value,
        )

    # ========================================================
    # SECTION 12 - FLOW FIELD TRACKING
    # ========================================================

    def _update_flow_fields(
        self,
        context: ConversationContext,
        changed_fields: list[str],
    ) -> None:
        required = FLOW_REQUIRED_FIELDS.get(
            context.active_flow,
            (),
        )

        if not required:
            if context.pending_fields:
                context.pending_fields = []
                changed_fields.append(
                    "pending_fields"
                )

            return

        completed = [
            field_name
            for field_name in required
            if field_name
            in context.current_entities
        ]

        pending = [
            field_name
            for field_name in required
            if field_name
            not in context.current_entities
        ]

        completed = completed[
            : self.maximum_pending_fields
        ]
        pending = pending[
            : self.maximum_pending_fields
        ]

        if completed != context.completed_fields:
            context.completed_fields = completed
            changed_fields.append(
                "completed_fields"
            )

        if pending != context.pending_fields:
            context.pending_fields = pending
            changed_fields.append(
                "pending_fields"
            )

    # ========================================================
    # SECTION 13 - PAGE CONTEXT
    # ========================================================

    def _record_page_context(
        self,
        context: ConversationContext,
        *,
        page_category: str | None,
        page_url: str | None,
        observed_at: datetime,
    ) -> None:
        normalized_category = (
            str(page_category).strip()
            if page_category
            else None
        )

        normalized_url = (
            str(page_url).strip()
            if page_url
            else None
        )

        if normalized_category:
            context.page_category = (
                normalized_category
            )

        if normalized_url:
            context.page_url = normalized_url

        if (
            normalized_category
            or normalized_url
        ):
            page_record = {
                "page_category": normalized_category,
                "page_url": normalized_url,
                "observed_at": observed_at.isoformat(),
            }

            if (
                not context.page_history
                or context.page_history[-1]
                != page_record
            ):
                context.page_history.append(
                    page_record
                )

    # ========================================================
    # SECTION 14 - TURN HISTORY
    # ========================================================

    def _record_turn(
        self,
        context: ConversationContext,
        *,
        user_text: str,
        intent_result: IntentResult,
        entity_result: EntityExtractionResult,
        page_category: str | None,
        observed_at: datetime,
    ) -> None:
        context.recent_turns.append(
            RecentTurn(
                turn_index=context.turn_count,
                user_text=user_text,
                normalized_text=(
                    intent_result.normalized_text
                ),
                corrected_text=(
                    intent_result.corrected_text
                ),
                primary_intent=(
                    intent_result.primary_intent.value
                ),
                confidence=intent_result.confidence,
                decision=(
                    intent_result.decision.value
                ),
                entity_types=tuple(
                    dict.fromkeys(
                        entity.entity_type.value
                        for entity
                        in entity_result.entities
                        if not entity.metadata.get(
                            "synthetic"
                        )
                    )
                ),
                page_category=page_category,
                occurred_at=observed_at,
            )
        )

    # ========================================================
    # SECTION 15 - SUMMARY
    # ========================================================

    @staticmethod
    def _build_summary(
        context: ConversationContext,
    ) -> str:
        parts: list[str] = []

        if context.active_flow != (
            ActiveFlow.NONE.value
        ):
            parts.append(
                f"Active flow: {context.active_flow}."
            )

        if context.current_intent:
            parts.append(
                f"Current intent: {context.current_intent}."
            )

        if context.completed_fields:
            parts.append(
                "Collected: "
                + ", ".join(
                    context.completed_fields
                )
                + "."
            )

        if context.pending_fields:
            parts.append(
                "Still needed: "
                + ", ".join(
                    context.pending_fields
                )
                + "."
            )

        if context.human_handoff_requested:
            parts.append(
                "Human handoff requested."
            )

        return " ".join(parts) or (
            "No active workflow."
        )

    # ========================================================
    # SECTION 16 - GENERAL HELPERS
    # ========================================================

    @staticmethod
    def _matches_any(
        text: str,
        patterns: Sequence[
            re.Pattern[str]
        ],
    ) -> bool:
        return any(
            pattern.search(text)
            for pattern in patterns
        )

    @staticmethod
    def _looks_like_followup(
        text: str,
    ) -> bool:
        cleaned = text.strip()

        if not cleaned:
            return False

        if PRONOUN_PATTERN.search(
            cleaned
        ):
            return True

        if any(
            pattern.search(cleaned)
            for pattern
            in ELLIPSIS_FOLLOWUP_PATTERNS
        ):
            return True

        word_count = len(
            re.findall(
                r"\b\w+\b",
                cleaned,
            )
        )

        return word_count <= 5

    @staticmethod
    def _determine_decision(
        *,
        expired_previous_context: bool,
        explicit_override: bool,
        previous_current_intent: str | None,
        context: ConversationContext,
        changed_fields: Sequence[str],
        entity_changed: bool,
    ) -> ContextDecision:
        if expired_previous_context:
            return ContextDecision.EXPIRED

        if explicit_override:
            return ContextDecision.OVERRIDDEN

        if (
            previous_current_intent is None
            and context.current_intent
        ):
            return ContextDecision.CREATED

        if changed_fields or entity_changed:
            return ContextDecision.UPDATED

        return ContextDecision.UNCHANGED


# ============================================================
# SECTION 17 - MODULE-LEVEL MANAGER
# ============================================================

_default_context_manager = (
    ConversationContextManager()
)


def update_conversation_context(
    *,
    user_text: str,
    intent_result: IntentResult,
    entity_result: EntityExtractionResult,
    previous_context: ConversationContext | Mapping[str, Any] | None = None,
    conversation_id: str | None = None,
    session_id: str | None = None,
    page_category: str | None = None,
    page_url: str | None = None,
    now: datetime | None = None,
) -> ContextUpdateResult:
    """
    Update conversation context through the shared manager.
    """

    return _default_context_manager.update(
        user_text=user_text,
        intent_result=intent_result,
        entity_result=entity_result,
        previous_context=previous_context,
        conversation_id=conversation_id,
        session_id=session_id,
        page_category=page_category,
        page_url=page_url,
        now=now,
    )


def context_as_dict(
    context: ConversationContext,
) -> dict[str, Any]:
    """
    Return the context as a JSON-safe dictionary.
    """

    return context.as_dict()


# ============================================================
# SECTION 18 - SERIALIZATION HELPERS
# ============================================================

def _json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (date, datetime, time),
    ):
        return value.isoformat()

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    return copy.deepcopy(value)


def _optional_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()

    return cleaned or None


def _parse_datetime(
    value: Any,
) -> datetime | None:
    if isinstance(value, datetime):
        return value

    if not value:
        return None

    try:
        return datetime.fromisoformat(
            str(value)
            .replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError:
        return None


# ============================================================
# SECTION 19 - VALIDATION
# ============================================================

def validate_context_module() -> dict[str, Any]:
    from datetime import datetime

    from app.nlu.entities import (
        extract_entities,
    )
    from app.nlu.intent import (
        classify_intent,
    )

    manager = ConversationContextManager(
        ttl_minutes=45
    )

    reference = datetime(
        2026,
        7,
        22,
        12,
        0,
    ).astimezone()

    first_text = (
        "I need a birthday party for 45 people."
    )

    first_intent = classify_intent(
        first_text,
        page_category="private_events",
    )

    first_entities = extract_entities(
        first_text,
        intent=first_intent.primary_intent,
        page_category="private_events",
        reference_datetime=reference,
    )

    first_update = manager.update(
        user_text=first_text,
        intent_result=first_intent,
        entity_result=first_entities,
        conversation_id="conversation_context_test",
        session_id="session_context_test",
        page_category="private_events",
        page_url=(
            "https://www.thehorseshoetavern.com/private-events"
        ),
        now=reference,
    )

    second_text = (
        "Make it 60 people on July 25 at 7pm."
    )

    second_intent = classify_intent(
        second_text,
        previous_intent=(
            first_update.context.current_intent
        ),
        page_category="private_events",
        conversation_context={
            "active_flow": (
                first_update.context.active_flow
            ),
        },
    )

    second_entities = extract_entities(
        second_text,
        intent=(
            first_update.context.current_intent
        ),
        page_category="private_events",
        reference_datetime=reference,
    )

    second_update = manager.update(
        user_text=second_text,
        intent_result=second_intent,
        entity_result=second_entities,
        previous_context=first_update.context,
        page_category="private_events",
        now=(
            reference
            + timedelta(minutes=2)
        ),
    )

    override_text = (
        "Actually, what time does the kitchen close?"
    )

    override_intent = classify_intent(
        override_text,
        previous_intent=(
            second_update.context.current_intent
        ),
        conversation_context={
            "active_flow": (
                second_update.context.active_flow
            ),
        },
    )

    override_entities = extract_entities(
        override_text,
        intent=override_intent.primary_intent,
        reference_datetime=reference,
    )

    override_update = manager.update(
        user_text=override_text,
        intent_result=override_intent,
        entity_result=override_entities,
        previous_context=second_update.context,
        now=(
            reference
            + timedelta(minutes=4)
        ),
    )

    reset_text = "Never mind, start over."

    reset_intent = classify_intent(
        reset_text
    )

    reset_entities = extract_entities(
        reset_text,
        reference_datetime=reference,
    )

    reset_update = manager.update(
        user_text=reset_text,
        intent_result=reset_intent,
        entity_result=reset_entities,
        previous_context=override_update.context,
        now=(
            reference
            + timedelta(minutes=5)
        ),
    )

    checks = {
        "private_event_flow_created": (
            first_update.context.active_flow
            == ActiveFlow.PRIVATE_EVENT.value
        ),
        "initial_guest_count_saved": (
            first_update.context.current_entities.get(
                EntityType.GUEST_COUNT.value
            )
            == 45
        ),
        "new_guest_count_overrode_old": (
            second_update.context.current_entities.get(
                EntityType.GUEST_COUNT.value
            )
            == 60
        ),
        "new_date_saved": (
            second_update.context.current_entities.get(
                EntityType.DATE.value
            )
            == "2026-07-25"
        ),
        "new_time_saved": (
            second_update.context.current_entities.get(
                EntityType.TIME.value
            )
            == "19:00:00"
        ),
        "previous_intent_not_locked": (
            override_update.context.current_intent
            == IntentName.HOURS_KITCHEN.value
        ),
        "override_cleared_private_flow": (
            override_update.context.active_flow
            == ActiveFlow.NONE.value
        ),
        "explicit_override_detected": (
            override_update.explicit_override
        ),
        "reset_cleared_entities": (
            reset_update.context.current_entities
            == {}
        ),
        "reset_cleared_flow": (
            reset_update.context.active_flow
            == ActiveFlow.NONE.value
        ),
        "context_json_safe": bool(
            reset_update.context.as_dict()
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
        "first_update": (
            first_update.as_dict()
        ),
        "second_update": (
            second_update.as_dict()
        ),
        "override_update": (
            override_update.as_dict()
        ),
        "reset_update": (
            reset_update.as_dict()
        ),
    }


# ============================================================
# SECTION 20 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import json

    report = validate_context_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
