# ============================================================
# Exact file location: app/nlu/orchestrator.py
# Horseshoe Tavern AI
# Phase 1 Part 1.17
# Unified normalization, spelling, intent, entity, context,
# ambiguity, workflow, confidence, and diagnostics pipeline
# ============================================================

"""
Unified natural-language understanding orchestration.

This module is the single public entrypoint for processing chatbot input.

Pipeline:

1. Validate and preserve the original message
2. Normalize Unicode, shorthand, spacing, and business terminology
3. Apply controlled spelling correction
4. Classify intent using current-message evidence first
5. Extract restaurant, reservation, event, menu, contact, and time entities
6. Update lightweight multi-turn context
7. Re-evaluate follow-up intent only when prior context is legitimately needed
8. Produce confidence, ambiguity, workflow, diagnostics, and safe persistence data

Core guarantees:

- Previous intent never locks the next message
- Explicit current-message evidence outranks historical context
- New entity values replace stale entity values
- Public input never becomes verified business truth
- Unknown and ambiguous messages remain unknown or ambiguous
- Pronouns are resolved only when needed and supported
- Page context is weak supporting evidence, not an authoritative instruction
- Every output is deterministic and JSON-safe
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as datetime_time
from decimal import Decimal
from enum import Enum
from typing import Any, Final, Mapping, Sequence

from app.logging_config import get_logger
from app.nlu.context import (
    ActiveFlow,
    ContextUpdateResult,
    ConversationContext,
    ConversationContextManager,
)
from app.nlu.entities import (
    EntityExtractionResult,
    EntityExtractor,
    EntityType,
    ExtractedEntity,
)
from app.nlu.intent import (
    IntentCandidate,
    IntentClassifier,
    IntentDecision,
    IntentName,
    IntentResult,
)
from app.nlu.normalizer import (
    NormalizationResult,
    TextNormalizer,
)
from app.nlu.spelling import (
    SpellingEngine,
    SpellingResult,
)


# ============================================================
# SECTION 01 - LOGGER AND CONSTANTS
# ============================================================

logger = get_logger(__name__)

NLU_ENGINE_VERSION: Final[str] = "1.0.0"
NLU_ENGINE_PHASE: Final[str] = "Phase 1 Part 1.17"

DEFAULT_LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.44
DEFAULT_HIGH_CONFIDENCE_THRESHOLD: Final[float] = 0.80
DEFAULT_AMBIGUITY_MARGIN: Final[float] = 0.10
DEFAULT_MAXIMUM_MESSAGE_CHARACTERS: Final[int] = 3000
DEFAULT_MAXIMUM_DIAGNOSTIC_ITEMS: Final[int] = 100


# ============================================================
# SECTION 02 - ENUMERATIONS
# ============================================================

class NLUDecision(str, Enum):
    ACCEPTED = "accepted"
    ACCEPTED_WITH_CONTEXT = "accepted_with_context"
    MULTI_INTENT = "multi_intent"
    AMBIGUOUS = "ambiguous"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"
    EMPTY = "empty"
    REJECTED = "rejected"


class ConfidenceBand(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ProcessingStage(str, Enum):
    VALIDATION = "validation"
    NORMALIZATION = "normalization"
    SPELLING = "spelling"
    INTENT = "intent"
    ENTITIES = "entities"
    CONTEXT = "context"
    FINALIZATION = "finalization"


# ============================================================
# SECTION 03 - DATA CLASSES
# ============================================================

@dataclass(frozen=True, slots=True)
class StageTiming:
    stage: ProcessingStage
    duration_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True, slots=True)
class NLUWarning:
    code: str
    message: str
    severity: str = "warning"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class NLUConfidence:
    intent: float
    entity: float
    context: float
    overall: float
    band: ConfidenceBand

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "entity": self.entity,
            "context": self.context,
            "overall": self.overall,
            "band": self.band.value,
        }


@dataclass(frozen=True, slots=True)
class NLUResult:
    original_text: str
    normalized_text: str
    corrected_text: str

    primary_intent: IntentName
    detected_intents: tuple[IntentName, ...]
    intent_decision: IntentDecision
    nlu_decision: NLUDecision

    confidence: NLUConfidence

    entities: tuple[ExtractedEntity, ...]
    context: ConversationContext
    context_update: ContextUpdateResult

    normalization: NormalizationResult
    spelling: SpellingResult
    intent: IntentResult
    entity_extraction: EntityExtractionResult

    resolved_references: dict[str, Any]

    requires_clarification: bool
    clarification_reason: str | None
    suggested_clarification: str | None

    active_flow: str
    pending_fields: tuple[str, ...]
    completed_fields: tuple[str, ...]

    warnings: tuple[NLUWarning, ...]
    timings: tuple[StageTiming, ...]

    processing_time_ms: float
    engine_version: str
    engine_phase: str

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        return (
            self.nlu_decision
            in {
                NLUDecision.ACCEPTED,
                NLUDecision.ACCEPTED_WITH_CONTEXT,
                NLUDecision.MULTI_INTENT,
            }
            and self.primary_intent
            != IntentName.UNKNOWN
        )

    @property
    def is_multi_intent(self) -> bool:
        return len(self.detected_intents) > 1

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    def entities_of_type(
        self,
        entity_type: EntityType,
    ) -> tuple[ExtractedEntity, ...]:
        return tuple(
            entity
            for entity in self.entities
            if entity.entity_type == entity_type
        )

    def first_entity(
        self,
        entity_type: EntityType,
    ) -> ExtractedEntity | None:
        values = self.entities_of_type(
            entity_type
        )

        return values[0] if values else None

    def as_dict(
        self,
        *,
        include_diagnostics: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "corrected_text": self.corrected_text,
            "primary_intent": self.primary_intent.value,
            "detected_intents": [
                intent.value
                for intent in self.detected_intents
            ],
            "intent_decision": self.intent_decision.value,
            "nlu_decision": self.nlu_decision.value,
            "confidence": self.confidence.as_dict(),
            "entities": [
                entity.as_dict()
                for entity in self.entities
            ],
            "context": self.context.as_dict(),
            "resolved_references": _json_safe(
                self.resolved_references
            ),
            "requires_clarification": (
                self.requires_clarification
            ),
            "clarification_reason": (
                self.clarification_reason
            ),
            "suggested_clarification": (
                self.suggested_clarification
            ),
            "active_flow": self.active_flow,
            "pending_fields": list(
                self.pending_fields
            ),
            "completed_fields": list(
                self.completed_fields
            ),
            "warnings": [
                warning.as_dict()
                for warning in self.warnings
            ],
            "timings": [
                timing.as_dict()
                for timing in self.timings
            ],
            "processing_time_ms": (
                self.processing_time_ms
            ),
            "engine_version": self.engine_version,
            "engine_phase": self.engine_phase,
            "metadata": copy.deepcopy(
                self.metadata
            ),
            "is_confident": self.is_confident,
            "is_multi_intent": self.is_multi_intent,
            "entity_count": self.entity_count,
        }

        if include_diagnostics:
            payload["diagnostics"] = {
                "normalization": (
                    self.normalization.as_dict()
                ),
                "spelling": (
                    self.spelling.as_dict()
                ),
                "intent": (
                    self.intent.as_dict()
                ),
                "entity_extraction": (
                    self.entity_extraction.as_dict()
                ),
                "context_update": (
                    self.context_update.as_dict()
                ),
            }

        return payload


# ============================================================
# SECTION 04 - ORCHESTRATOR
# ============================================================

class NLUOrchestrator:
    """
    Single-entrypoint NLU orchestration service.
    """

    def __init__(
        self,
        *,
        normalizer: TextNormalizer | None = None,
        spelling_engine: SpellingEngine | None = None,
        intent_classifier: IntentClassifier | None = None,
        entity_extractor: EntityExtractor | None = None,
        context_manager: ConversationContextManager | None = None,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        high_confidence_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
        ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
        maximum_message_characters: int = DEFAULT_MAXIMUM_MESSAGE_CHARACTERS,
    ) -> None:
        self.normalizer = (
            normalizer
            or TextNormalizer(
                maximum_characters=(
                    maximum_message_characters
                )
            )
        )

        self.spelling_engine = (
            spelling_engine
            or SpellingEngine()
        )

        self.intent_classifier = (
            intent_classifier
            or IntentClassifier()
        )

        self.entity_extractor = (
            entity_extractor
            or EntityExtractor()
        )

        self.context_manager = (
            context_manager
            or ConversationContextManager()
        )

        self.low_confidence_threshold = float(
            low_confidence_threshold
        )

        self.high_confidence_threshold = float(
            high_confidence_threshold
        )

        self.ambiguity_margin = float(
            ambiguity_margin
        )

        self.maximum_message_characters = max(
            1,
            int(maximum_message_characters),
        )

    # ========================================================
    # SECTION 05 - PUBLIC PROCESSING ENTRYPOINT
    # ========================================================

    def process(
        self,
        text: str,
        *,
        previous_context: ConversationContext | Mapping[str, Any] | None = None,
        conversation_id: str | None = None,
        session_id: str | None = None,
        page_category: str | None = None,
        page_url: str | None = None,
        page_context: Mapping[str, Any] | None = None,
        reference_datetime: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> NLUResult:
        overall_started = time.perf_counter()
        timings: list[StageTiming] = []
        warnings: list[NLUWarning] = []

        original_text = (
            ""
            if text is None
            else str(text)
        )

        validation_started = (
            time.perf_counter()
        )

        validation_error = self._validate_input(
            original_text
        )

        timings.append(
            self._timing(
                ProcessingStage.VALIDATION,
                validation_started,
            )
        )

        context = self._coerce_context(
            previous_context
        )

        if validation_error is not None:
            return self._build_rejected_result(
                original_text=original_text,
                context=context,
                validation_error=validation_error,
                timings=timings,
                overall_started=overall_started,
                metadata=metadata,
            )

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        normalization_started = (
            time.perf_counter()
        )

        normalization = self.normalizer.normalize(
            original_text
        )

        timings.append(
            self._timing(
                ProcessingStage.NORMALIZATION,
                normalization_started,
            )
        )

        if normalization.was_truncated:
            warnings.append(
                NLUWarning(
                    code="message_truncated",
                    message=(
                        "The input exceeded the configured "
                        "maximum length and was truncated."
                    ),
                    metadata={
                        "maximum_characters": (
                            self.maximum_message_characters
                        )
                    },
                )
            )

        if (
            normalization
            .suspicious_repetition_score
            >= 0.70
        ):
            warnings.append(
                NLUWarning(
                    code="suspicious_repetition",
                    message=(
                        "The message contains unusually "
                        "repetitive text."
                    ),
                    metadata={
                        "score": (
                            normalization
                            .suspicious_repetition_score
                        )
                    },
                )
            )

        # ----------------------------------------------------
        # SPELLING
        # ----------------------------------------------------

        spelling_started = (
            time.perf_counter()
        )

        spelling = self.spelling_engine.correct(
            normalization.normalized_text
        )

        timings.append(
            self._timing(
                ProcessingStage.SPELLING,
                spelling_started,
            )
        )

        if spelling.ambiguous_count > 0:
            warnings.append(
                NLUWarning(
                    code="ambiguous_spelling",
                    message=(
                        "One or more spelling candidates "
                        "were ambiguous and were not forced."
                    ),
                    metadata={
                        "ambiguous_count": (
                            spelling.ambiguous_count
                        )
                    },
                )
            )

        corrected_text = spelling.corrected_text

        # ----------------------------------------------------
        # INTENT
        # ----------------------------------------------------

        intent_started = (
            time.perf_counter()
        )

        current_context_payload = (
            context.as_dict()
        )

        previous_intent = (
            context.current_intent
        )

        effective_page_category = (
            page_category
            or self._page_category_from_context(
                page_context
            )
            or context.page_category
        )

        initial_intent = (
            self.intent_classifier.classify(
                corrected_text,
                previous_intent=previous_intent,
                page_category=(
                    effective_page_category
                ),
                conversation_context={
                    "active_flow": (
                        context.active_flow
                    ),
                    "current_entities": (
                        context.current_entities
                    ),
                },
            )
        )

        timings.append(
            self._timing(
                ProcessingStage.INTENT,
                intent_started,
            )
        )

        # ----------------------------------------------------
        # ENTITIES
        # ----------------------------------------------------

        entities_started = (
            time.perf_counter()
        )

        entity_extraction = (
            self.entity_extractor.extract(
                corrected_text,
                intent=(
                    initial_intent.primary_intent
                ),
                page_category=(
                    effective_page_category
                ),
                reference_datetime=(
                    reference_datetime
                ),
                context={
                    "business_name": (
                        self._business_name_from_context(
                            page_context
                        )
                    ),
                },
            )
        )

        timings.append(
            self._timing(
                ProcessingStage.ENTITIES,
                entities_started,
            )
        )

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        context_started = (
            time.perf_counter()
        )

        context_update = (
            self.context_manager.update(
                user_text=original_text,
                intent_result=initial_intent,
                entity_result=entity_extraction,
                previous_context=context,
                conversation_id=conversation_id,
                session_id=session_id,
                page_category=(
                    effective_page_category
                ),
                page_url=(
                    page_url
                    or self._page_url_from_context(
                        page_context
                    )
                ),
                now=reference_datetime,
            )
        )

        final_intent = self._reconcile_intent(
            original_text=original_text,
            initial_intent=initial_intent,
            context_update=context_update,
        )

        if final_intent is not initial_intent:
            entity_extraction = (
                self.entity_extractor.extract(
                    corrected_text,
                    intent=(
                        final_intent.primary_intent
                    ),
                    page_category=(
                        effective_page_category
                    ),
                    reference_datetime=(
                        reference_datetime
                    ),
                    context={
                        "business_name": (
                            self._business_name_from_context(
                                page_context
                            )
                        ),
                    },
                )
            )

            context_update = (
                self.context_manager.update(
                    user_text=original_text,
                    intent_result=final_intent,
                    entity_result=(
                        entity_extraction
                    ),
                    previous_context=context,
                    conversation_id=(
                        conversation_id
                    ),
                    session_id=session_id,
                    page_category=(
                        effective_page_category
                    ),
                    page_url=(
                        page_url
                        or self._page_url_from_context(
                            page_context
                        )
                    ),
                    now=reference_datetime,
                )
            )

        timings.append(
            self._timing(
                ProcessingStage.CONTEXT,
                context_started,
            )
        )

        # ----------------------------------------------------
        # FINALIZATION
        # ----------------------------------------------------

        finalization_started = (
            time.perf_counter()
        )

        confidence = self._calculate_confidence(
            intent_result=final_intent,
            entity_result=entity_extraction,
            context_update=context_update,
        )

        nlu_decision = self._resolve_nlu_decision(
            intent_result=final_intent,
            context_update=context_update,
            confidence=confidence,
        )

        clarification = self._build_clarification(
            intent_result=final_intent,
            entity_result=entity_extraction,
            context_update=context_update,
            nlu_decision=nlu_decision,
        )

        warnings.extend(
            self._warnings_from_results(
                intent_result=final_intent,
                context_update=context_update,
                entity_result=entity_extraction,
            )
        )

        timings.append(
            self._timing(
                ProcessingStage.FINALIZATION,
                finalization_started,
            )
        )

        processing_time_ms = round(
            (
                time.perf_counter()
                - overall_started
            )
            * 1000.0,
            3,
        )

        return NLUResult(
            original_text=original_text,
            normalized_text=(
                normalization.normalized_text
            ),
            corrected_text=corrected_text,
            primary_intent=(
                final_intent.primary_intent
            ),
            detected_intents=(
                final_intent.detected_intents
            ),
            intent_decision=(
                final_intent.decision
            ),
            nlu_decision=nlu_decision,
            confidence=confidence,
            entities=entity_extraction.entities,
            context=context_update.context,
            context_update=context_update,
            normalization=normalization,
            spelling=spelling,
            intent=final_intent,
            entity_extraction=(
                entity_extraction
            ),
            resolved_references=(
                context_update
                .resolved_references
            ),
            requires_clarification=(
                clarification[
                    "requires_clarification"
                ]
            ),
            clarification_reason=(
                clarification[
                    "clarification_reason"
                ]
            ),
            suggested_clarification=(
                clarification[
                    "suggested_clarification"
                ]
            ),
            active_flow=(
                context_update
                .context
                .active_flow
            ),
            pending_fields=tuple(
                context_update
                .context
                .pending_fields
            ),
            completed_fields=tuple(
                context_update
                .context
                .completed_fields
            ),
            warnings=tuple(
                warnings[
                    :DEFAULT_MAXIMUM_DIAGNOSTIC_ITEMS
                ]
            ),
            timings=tuple(timings),
            processing_time_ms=(
                processing_time_ms
            ),
            engine_version=(
                NLU_ENGINE_VERSION
            ),
            engine_phase=(
                NLU_ENGINE_PHASE
            ),
            metadata={
                **dict(metadata or {}),
                "page_category": (
                    effective_page_category
                ),
                "page_context_present": bool(
                    page_context
                ),
                "previous_context_present": (
                    previous_context
                    is not None
                ),
                "previous_context_snapshot": (
                    current_context_payload
                    if previous_context
                    is not None
                    else None
                ),
            },
        )

    # ========================================================
    # SECTION 06 - INPUT VALIDATION
    # ========================================================

    def _validate_input(
        self,
        text: str,
    ) -> str | None:
        if not text.strip():
            return "Message cannot be empty."

        if len(text) > (
            self.maximum_message_characters
            * 10
        ):
            return (
                "Message exceeds the absolute "
                "processing limit."
            )

        return None

    # ========================================================
    # SECTION 07 - INTENT RECONCILIATION
    # ========================================================

    def _reconcile_intent(
        self,
        *,
        original_text: str,
        initial_intent: IntentResult,
        context_update: ContextUpdateResult,
    ) -> IntentResult:
        """
        Permit context-backed follow-up resolution only when the current
        message did not already provide a reliable explicit intent.
        """

        if (
            initial_intent.primary_intent
            != IntentName.UNKNOWN
        ):
            return initial_intent

        if not context_update.previous_intent_used:
            return initial_intent

        context_intent = (
            context_update
            .context
            .current_intent
        )

        if not context_intent:
            return initial_intent

        try:
            resolved_intent = IntentName(
                context_intent
            )
        except ValueError:
            return initial_intent

        synthetic_candidate = IntentCandidate(
            intent=resolved_intent,
            raw_score=0.75,
            confidence=0.60,
            positive_score=0.0,
            negative_score=0.0,
            precedence_bonus=0.0,
            context_bonus=0.60,
            page_context_bonus=0.0,
            evidence=(),
        )

        return IntentResult(
            original_text=(
                initial_intent.original_text
            ),
            normalized_text=(
                initial_intent.normalized_text
            ),
            corrected_text=(
                initial_intent.corrected_text
            ),
            primary_intent=resolved_intent,
            confidence=0.60,
            decision=(
                IntentDecision.ACCEPTED
            ),
            alternatives=(
                synthetic_candidate,
            ),
            detected_intents=(
                resolved_intent,
            ),
            ambiguity_margin=0.60,
            evidence=(),
            previous_intent_used=True,
            page_context_used=False,
            explicit_override=False,
        )

    # ========================================================
    # SECTION 08 - CONFIDENCE
    # ========================================================

    def _calculate_confidence(
        self,
        *,
        intent_result: IntentResult,
        entity_result: EntityExtractionResult,
        context_update: ContextUpdateResult,
    ) -> NLUConfidence:
        intent_confidence = round(
            min(
                max(
                    intent_result.confidence,
                    0.0,
                ),
                1.0,
            ),
            6,
        )

        non_synthetic_entities = [
            entity
            for entity
            in entity_result.entities
            if not entity.metadata.get(
                "synthetic"
            )
        ]

        if non_synthetic_entities:
            entity_confidence = (
                sum(
                    entity.confidence
                    for entity
                    in non_synthetic_entities
                )
                / len(
                    non_synthetic_entities
                )
            )
        else:
            entity_confidence = (
                0.50
                if intent_result.primary_intent
                not in {
                    IntentName.UNKNOWN,
                    IntentName.PRIVATE_EVENT,
                    IntentName.RESERVATION,
                }
                else 0.0
            )

        if (
            context_update.previous_intent_used
            or context_update.resolved_references
        ):
            context_confidence = 0.70
        elif (
            context_update.context.turn_count
            > 1
        ):
            context_confidence = 0.55
        else:
            context_confidence = 0.40

        if (
            intent_result.primary_intent
            == IntentName.UNKNOWN
        ):
            overall = (
                intent_confidence * 0.70
                + entity_confidence * 0.20
                + context_confidence * 0.10
            )
        else:
            overall = (
                intent_confidence * 0.65
                + entity_confidence * 0.20
                + context_confidence * 0.15
            )

        if (
            intent_result.decision
            == IntentDecision.AMBIGUOUS
        ):
            overall *= 0.75

        if (
            intent_result.decision
            == IntentDecision.MULTI_INTENT
        ):
            overall *= 0.92

        overall = round(
            min(
                max(overall, 0.0),
                1.0,
            ),
            6,
        )

        return NLUConfidence(
            intent=intent_confidence,
            entity=round(
                entity_confidence,
                6,
            ),
            context=round(
                context_confidence,
                6,
            ),
            overall=overall,
            band=self._confidence_band(
                overall
            ),
        )

    @staticmethod
    def _confidence_band(
        value: float,
    ) -> ConfidenceBand:
        if value < 0.20:
            return ConfidenceBand.VERY_LOW

        if value < 0.40:
            return ConfidenceBand.LOW

        if value < 0.65:
            return ConfidenceBand.MEDIUM

        if value < 0.85:
            return ConfidenceBand.HIGH

        return ConfidenceBand.VERY_HIGH

    # ========================================================
    # SECTION 09 - FINAL DECISION
    # ========================================================

    def _resolve_nlu_decision(
        self,
        *,
        intent_result: IntentResult,
        context_update: ContextUpdateResult,
        confidence: NLUConfidence,
    ) -> NLUDecision:
        if (
            intent_result.primary_intent
            == IntentName.UNKNOWN
        ):
            return NLUDecision.UNKNOWN

        if (
            intent_result.decision
            == IntentDecision.AMBIGUOUS
        ):
            return NLUDecision.AMBIGUOUS

        if (
            intent_result.decision
            == IntentDecision.MULTI_INTENT
            or len(
                intent_result.detected_intents
            )
            > 1
        ):
            return NLUDecision.MULTI_INTENT

        if (
            confidence.overall
            < self.low_confidence_threshold
        ):
            return NLUDecision.LOW_CONFIDENCE

        if (
            context_update.previous_intent_used
            or context_update.resolved_references
        ):
            return (
                NLUDecision
                .ACCEPTED_WITH_CONTEXT
            )

        return NLUDecision.ACCEPTED

    # ========================================================
    # SECTION 10 - CLARIFICATION
    # ========================================================

    def _build_clarification(
        self,
        *,
        intent_result: IntentResult,
        entity_result: EntityExtractionResult,
        context_update: ContextUpdateResult,
        nlu_decision: NLUDecision,
    ) -> dict[str, Any]:
        if nlu_decision == NLUDecision.UNKNOWN:
            return {
                "requires_clarification": True,
                "clarification_reason": (
                    "No reliable intent was detected."
                ),
                "suggested_clarification": (
                    "Could you rephrase that and mention whether "
                    "you are asking about hours, the menu, events, "
                    "reservations, ordering, parking, or a private event?"
                ),
            }

        if nlu_decision == NLUDecision.AMBIGUOUS:
            alternatives = [
                candidate.intent.value
                for candidate
                in intent_result.alternatives[:3]
            ]

            return {
                "requires_clarification": True,
                "clarification_reason": (
                    "Multiple intents received similar scores."
                ),
                "suggested_clarification": (
                    "Did you mean "
                    + ", ".join(alternatives)
                    + "?"
                ),
            }

        if nlu_decision == NLUDecision.MULTI_INTENT:
            return {
                "requires_clarification": False,
                "clarification_reason": None,
                "suggested_clarification": None,
            }

        if (
            context_update.context.active_flow
            == ActiveFlow.PRIVATE_EVENT.value
            and context_update.context.pending_fields
        ):
            next_field = (
                context_update
                .context
                .pending_fields[0]
            )

            prompts = {
                EntityType.EVENT_TYPE.value: (
                    "What type of private event are you planning?"
                ),
                EntityType.DATE.value: (
                    "What date would you prefer?"
                ),
                EntityType.GUEST_COUNT.value: (
                    "Approximately how many guests will attend?"
                ),
                EntityType.EMAIL.value: (
                    "What email address should the event team use to contact you?"
                ),
            }

            return {
                "requires_clarification": True,
                "clarification_reason": (
                    f"Private-event field missing: {next_field}"
                ),
                "suggested_clarification": (
                    prompts.get(
                        next_field,
                        f"Please provide {next_field.replace('_', ' ')}."
                    )
                ),
            }

        if (
            context_update.context.active_flow
            == ActiveFlow.RESERVATION.value
            and context_update.context.pending_fields
        ):
            next_field = (
                context_update
                .context
                .pending_fields[0]
            )

            prompts = {
                EntityType.DATE.value: (
                    "What date would you like the reservation?"
                ),
                EntityType.TIME.value: (
                    "What time would you prefer?"
                ),
                EntityType.GUEST_COUNT.value: (
                    "How many people are in your party?"
                ),
            }

            return {
                "requires_clarification": True,
                "clarification_reason": (
                    f"Reservation field missing: {next_field}"
                ),
                "suggested_clarification": (
                    prompts.get(
                        next_field,
                        f"Please provide {next_field.replace('_', ' ')}."
                    )
                ),
            }

        return {
            "requires_clarification": False,
            "clarification_reason": None,
            "suggested_clarification": None,
        }

    # ========================================================
    # SECTION 11 - WARNINGS
    # ========================================================

    def _warnings_from_results(
        self,
        *,
        intent_result: IntentResult,
        context_update: ContextUpdateResult,
        entity_result: EntityExtractionResult,
    ) -> list[NLUWarning]:
        warnings: list[NLUWarning] = []

        if (
            intent_result.primary_intent
            == IntentName.UNKNOWN
        ):
            warnings.append(
                NLUWarning(
                    code="unknown_intent",
                    message=(
                        "No supported operational intent "
                        "was identified."
                    ),
                )
            )

        if (
            intent_result.decision
            == IntentDecision.AMBIGUOUS
        ):
            warnings.append(
                NLUWarning(
                    code="ambiguous_intent",
                    message=(
                        "The top intent candidates were "
                        "too close for a definitive result."
                    ),
                    metadata={
                        "alternatives": [
                            candidate.intent.value
                            for candidate
                            in intent_result.alternatives[:3]
                        ]
                    },
                )
            )

        if context_update.expired_previous_context:
            warnings.append(
                NLUWarning(
                    code="context_expired",
                    message=(
                        "The previous conversation context "
                        "expired and was not reused."
                    ),
                    severity="information",
                )
            )

        if context_update.explicit_override:
            warnings.append(
                NLUWarning(
                    code="explicit_context_override",
                    message=(
                        "The current message explicitly "
                        "overrode previous context."
                    ),
                    severity="information",
                )
            )

        if context_update.resolved_references:
            warnings.append(
                NLUWarning(
                    code="context_reference_resolution",
                    message=(
                        "One or more references were resolved "
                        "using recent conversation context."
                    ),
                    severity="information",
                    metadata={
                        "resolved_types": list(
                            context_update
                            .resolved_references
                            .keys()
                        )
                    },
                )
            )

        synthetic_count = sum(
            bool(
                entity.metadata.get(
                    "synthetic"
                )
            )
            for entity
            in entity_result.entities
        )

        if synthetic_count:
            warnings.append(
                NLUWarning(
                    code="synthetic_context_entities",
                    message=(
                        "Some entities were inferred from "
                        "intent or page context."
                    ),
                    severity="information",
                    metadata={
                        "count": synthetic_count
                    },
                )
            )

        return warnings

    # ========================================================
    # SECTION 12 - REJECTED RESULT
    # ========================================================

    def _build_rejected_result(
        self,
        *,
        original_text: str,
        context: ConversationContext,
        validation_error: str,
        timings: list[StageTiming],
        overall_started: float,
        metadata: Mapping[str, Any] | None,
    ) -> NLUResult:
        normalization = self.normalizer.normalize(
            ""
        )

        spelling = self.spelling_engine.correct(
            ""
        )

        intent = self.intent_classifier.classify(
            ""
        )

        entity_extraction = (
            self.entity_extractor.extract(
                ""
            )
        )

        context_update = (
            self.context_manager.update(
                user_text="",
                intent_result=intent,
                entity_result=(
                    entity_extraction
                ),
                previous_context=context,
            )
        )

        processing_time_ms = round(
            (
                time.perf_counter()
                - overall_started
            )
            * 1000.0,
            3,
        )

        decision = (
            NLUDecision.EMPTY
            if not original_text.strip()
            else NLUDecision.REJECTED
        )

        return NLUResult(
            original_text=original_text,
            normalized_text="",
            corrected_text="",
            primary_intent=IntentName.UNKNOWN,
            detected_intents=(),
            intent_decision=(
                IntentDecision.UNKNOWN
            ),
            nlu_decision=decision,
            confidence=NLUConfidence(
                intent=0.0,
                entity=0.0,
                context=0.0,
                overall=0.0,
                band=ConfidenceBand.VERY_LOW,
            ),
            entities=(),
            context=context_update.context,
            context_update=context_update,
            normalization=normalization,
            spelling=spelling,
            intent=intent,
            entity_extraction=(
                entity_extraction
            ),
            resolved_references={},
            requires_clarification=True,
            clarification_reason=(
                validation_error
            ),
            suggested_clarification=(
                "Please enter a question or request."
            ),
            active_flow=(
                context_update
                .context
                .active_flow
            ),
            pending_fields=tuple(
                context_update
                .context
                .pending_fields
            ),
            completed_fields=tuple(
                context_update
                .context
                .completed_fields
            ),
            warnings=(
                NLUWarning(
                    code="input_rejected",
                    message=validation_error,
                    severity="error",
                ),
            ),
            timings=tuple(timings),
            processing_time_ms=(
                processing_time_ms
            ),
            engine_version=(
                NLU_ENGINE_VERSION
            ),
            engine_phase=(
                NLU_ENGINE_PHASE
            ),
            metadata=dict(
                metadata or {}
            ),
        )

    # ========================================================
    # SECTION 13 - GENERAL HELPERS
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

    @staticmethod
    def _timing(
        stage: ProcessingStage,
        started_at: float,
    ) -> StageTiming:
        return StageTiming(
            stage=stage,
            duration_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000.0,
                3,
            ),
        )

    @staticmethod
    def _page_category_from_context(
        page_context: Mapping[str, Any] | None,
    ) -> str | None:
        if not page_context:
            return None

        value = page_context.get(
            "category"
        )

        if not value:
            return None

        return str(value).strip() or None

    @staticmethod
    def _page_url_from_context(
        page_context: Mapping[str, Any] | None,
    ) -> str | None:
        if not page_context:
            return None

        value = page_context.get("url")

        if not value:
            return None

        return str(value).strip() or None

    @staticmethod
    def _business_name_from_context(
        page_context: Mapping[str, Any] | None,
    ) -> str | None:
        if not page_context:
            return None

        value = page_context.get(
            "business_name"
        )

        if not value:
            return None

        return str(value).strip() or None


# ============================================================
# SECTION 14 - MODULE-LEVEL ORCHESTRATOR
# ============================================================

_default_orchestrator = NLUOrchestrator()


def process_message(
    text: str,
    *,
    previous_context: ConversationContext | Mapping[str, Any] | None = None,
    conversation_id: str | None = None,
    session_id: str | None = None,
    page_category: str | None = None,
    page_url: str | None = None,
    page_context: Mapping[str, Any] | None = None,
    reference_datetime: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> NLUResult:
    """
    Process one message through the full NLU pipeline.
    """

    return _default_orchestrator.process(
        text,
        previous_context=previous_context,
        conversation_id=conversation_id,
        session_id=session_id,
        page_category=page_category,
        page_url=page_url,
        page_context=page_context,
        reference_datetime=reference_datetime,
        metadata=metadata,
    )


def analyze_message(
    text: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Process a message and return a JSON-safe dictionary.
    """

    return process_message(
        text,
        **kwargs,
    ).as_dict()


# ============================================================
# SECTION 15 - SERIALIZATION
# ============================================================

def _json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (
            date,
            datetime,
            datetime_time,
        ),
    ):
        return value.isoformat()

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Enum):
        return value.value

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


# ============================================================
# SECTION 16 - VALIDATION
# ============================================================

def validate_orchestrator_module() -> dict[str, Any]:
    from datetime import timedelta

    orchestrator = NLUOrchestrator()

    reference = datetime(
        2026,
        7,
        22,
        12,
        0,
    ).astimezone()

    hours = orchestrator.process(
        "wat time does the kichen close",
        conversation_id=(
            "conversation_orchestrator_test"
        ),
        session_id=(
            "session_orchestrator_test"
        ),
        reference_datetime=reference,
    )

    private_event = orchestrator.process(
        (
            "I need a birtday party for 45 ppl "
            "on July 25 at 7pm."
        ),
        conversation_id=(
            "conversation_private_event_test"
        ),
        session_id=(
            "session_private_event_test"
        ),
        page_category="private_events",
        reference_datetime=reference,
    )

    private_followup = orchestrator.process(
        "Make it 60 people instead.",
        previous_context=(
            private_event.context
        ),
        conversation_id=(
            "conversation_private_event_test"
        ),
        session_id=(
            "session_private_event_test"
        ),
        page_category="private_events",
        reference_datetime=(
            reference
            + timedelta(minutes=1)
        ),
    )

    override = orchestrator.process(
        (
            "Actually, what events are "
            "happening tonight?"
        ),
        previous_context=(
            private_followup.context
        ),
        conversation_id=(
            "conversation_private_event_test"
        ),
        session_id=(
            "session_private_event_test"
        ),
        page_category="private_events",
        reference_datetime=(
            reference
            + timedelta(minutes=2)
        ),
    )

    unknown = orchestrator.process(
        "quantum flux capacitor status",
        reference_datetime=reference,
    )

    multi = orchestrator.process(
        (
            "What time do you close and "
            "what events are happening tonight?"
        ),
        reference_datetime=reference,
    )

    empty = orchestrator.process(
        "   ",
        reference_datetime=reference,
    )

    guest_entity = (
        private_followup.first_entity(
            EntityType.GUEST_COUNT
        )
    )

    checks = {
        "normalization_applied": (
            "kitchen"
            in hours.corrected_text.casefold()
        ),
        "hours_intent_detected": (
            hours.primary_intent
            == IntentName.HOURS_KITCHEN
        ),
        "hours_confident": (
            hours.confidence.overall > 0
        ),
        "private_event_detected": (
            private_event.primary_intent
            == IntentName.PRIVATE_EVENT
        ),
        "private_event_flow_started": (
            private_event.active_flow
            == ActiveFlow.PRIVATE_EVENT.value
        ),
        "private_event_entities_saved": (
            private_event.context.current_entities.get(
                EntityType.GUEST_COUNT.value
            )
            == 45
        ),
        "followup_guest_count_replaced": (
            private_followup.context.current_entities.get(
                EntityType.GUEST_COUNT.value
            )
            == 60
        ),
        "previous_intent_not_locked": (
            override.primary_intent
            == IntentName.EVENTS_TONIGHT
        ),
        "explicit_override_cleared_flow": (
            override.active_flow
            == ActiveFlow.NONE.value
        ),
        "unknown_input_safe": (
            unknown.primary_intent
            == IntentName.UNKNOWN
        ),
        "multi_intent_detected": (
            multi.is_multi_intent
            or multi.nlu_decision
            == NLUDecision.MULTI_INTENT
        ),
        "empty_input_rejected": (
            empty.nlu_decision
            == NLUDecision.EMPTY
        ),
        "json_safe_output": bool(
            private_event.as_dict()
        ),
        "timings_available": (
            len(hours.timings) >= 6
        ),
        "engine_metadata_available": (
            hours.engine_version
            == NLU_ENGINE_VERSION
            and hours.engine_phase
            == NLU_ENGINE_PHASE
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
        "hours": hours.as_dict(
            include_diagnostics=False
        ),
        "private_event": (
            private_event.as_dict(
                include_diagnostics=False
            )
        ),
        "private_followup": (
            private_followup.as_dict(
                include_diagnostics=False
            )
        ),
        "override": override.as_dict(
            include_diagnostics=False
        ),
        "unknown": unknown.as_dict(
            include_diagnostics=False
        ),
        "multi": multi.as_dict(
            include_diagnostics=False
        ),
        "empty": empty.as_dict(
            include_diagnostics=False
        ),
    }


# ============================================================
# SECTION 17 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    report = validate_orchestrator_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
