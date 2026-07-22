# ============================================================
# Exact file location: app/nlu/__init__.py
# Horseshoe Tavern AI
# Public natural-language understanding API
# ============================================================

from app.nlu.context import (
    ActiveFlow,
    ContextDecision,
    ContextUpdateResult,
    ConversationContext,
    ConversationContextManager,
    update_conversation_context,
)
from app.nlu.entities import (
    EntityExtractionResult,
    EntityExtractor,
    EntitySource,
    EntityType,
    ExtractedEntity,
    extract_entities,
)
from app.nlu.intent import (
    IntentCandidate,
    IntentClassifier,
    IntentDecision,
    IntentName,
    IntentResult,
    classify_intent,
)
from app.nlu.normalizer import (
    NormalizationResult,
    TextNormalizer,
    normalize_text,
)
from app.nlu.orchestrator import (
    ConfidenceBand,
    NLUConfidence,
    NLUDecision,
    NLUOrchestrator,
    NLUResult,
    NLU_ENGINE_PHASE,
    NLU_ENGINE_VERSION,
    analyze_message,
    process_message,
)
from app.nlu.spelling import (
    SpellingEngine,
    SpellingResult,
    correct_spelling,
)

__all__ = [
    "ActiveFlow",
    "ConfidenceBand",
    "ContextDecision",
    "ContextUpdateResult",
    "ConversationContext",
    "ConversationContextManager",
    "EntityExtractionResult",
    "EntityExtractor",
    "EntitySource",
    "EntityType",
    "ExtractedEntity",
    "IntentCandidate",
    "IntentClassifier",
    "IntentDecision",
    "IntentName",
    "IntentResult",
    "NLUConfidence",
    "NLUDecision",
    "NLUOrchestrator",
    "NLUResult",
    "NLU_ENGINE_PHASE",
    "NLU_ENGINE_VERSION",
    "NormalizationResult",
    "SpellingEngine",
    "SpellingResult",
    "TextNormalizer",
    "analyze_message",
    "classify_intent",
    "correct_spelling",
    "extract_entities",
    "normalize_text",
    "process_message",
    "update_conversation_context",
]
