# ============================================================
# Exact file location: app/nlu/intent.py
# Horseshoe Tavern AI
# Phase 1 Part 1.14
# Weighted, explainable, context-aware intent classification
# with explicit precedence, negative evidence, and ambiguity
# ============================================================

"""
Intent classification engine for Horseshoe Tavern AI.

Responsibilities:

- Classify customer messages into operational restaurant intents
- Use normalized and spelling-corrected text
- Apply weighted positive evidence
- Apply weighted negative evidence
- Support explicit intent precedence
- Detect multi-intent customer messages
- Avoid fragile raw-substring classification
- Avoid previous-intent lock
- Allow a new explicit request to override conversation context
- Use page context as weak supporting evidence only
- Return confidence, evidence, alternatives, and ambiguity
- Distinguish factual lookup, navigation, ordering, reservations,
  private events, live entertainment, sports viewing, feedback,
  contact requests, and human handoff
- Reject low-confidence classifications safely
- Remain deterministic and dependency-free

This module does not retrieve business facts and does not generate answers.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Sequence

from app.logging_config import get_logger
from app.nlu.normalizer import normalize_text
from app.nlu.spelling import correct_spelling


# ============================================================
# SECTION 01 - LOGGER AND CONSTANTS
# ============================================================

logger = get_logger(__name__)

DEFAULT_MINIMUM_CONFIDENCE: Final[float] = 0.44
DEFAULT_AMBIGUITY_MARGIN: Final[float] = 0.10
DEFAULT_MAX_ALTERNATIVES: Final[int] = 5
DEFAULT_MULTI_INTENT_THRESHOLD: Final[float] = 0.58
DEFAULT_CONTEXT_WEIGHT: Final[float] = 0.12
DEFAULT_PAGE_CONTEXT_WEIGHT: Final[float] = 0.08

TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[a-z0-9]+(?:[-'][a-z0-9]+)*",
    re.IGNORECASE,
)

CLAUSE_SPLIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\s*(?:[.!?;]+|\b(?:and also|also|plus|as well as)\b)\s*",
    re.IGNORECASE,
)

WORD_BOUNDARY_TEMPLATE: Final[str] = r"(?<!\w){}(?!\w)"


# ============================================================
# SECTION 02 - INTENT ENUMERATION
# ============================================================

class IntentName(str, Enum):
    GREETING = "GREETING"
    GOODBYE = "GOODBYE"
    THANKS = "THANKS"

    HOURS_GENERAL = "HOURS_GENERAL"
    HOURS_TODAY = "HOURS_TODAY"
    HOURS_KITCHEN = "HOURS_KITCHEN"
    HOURS_HAPPY_HOUR = "HOURS_HAPPY_HOUR"

    MENU_GENERAL = "MENU_GENERAL"
    MENU_ITEM_LOOKUP = "MENU_ITEM_LOOKUP"
    MENU_DIETARY = "MENU_DIETARY"
    MENU_ALLERGEN = "MENU_ALLERGEN"
    MENU_PRICE = "MENU_PRICE"

    LOCATION = "LOCATION"
    DIRECTIONS = "DIRECTIONS"
    PARKING = "PARKING"
    CONTACT = "CONTACT"

    RESERVATION = "RESERVATION"
    RESERVATION_CHANGE = "RESERVATION_CHANGE"
    RESERVATION_CANCEL = "RESERVATION_CANCEL"

    ORDERING = "ORDERING"
    TAKEOUT = "TAKEOUT"
    DELIVERY = "DELIVERY"

    EVENTS_GENERAL = "EVENTS_GENERAL"
    EVENTS_TONIGHT = "EVENTS_TONIGHT"
    LIVE_MUSIC = "LIVE_MUSIC"
    SPORTS_VIEWING = "SPORTS_VIEWING"

    PRIVATE_EVENT = "PRIVATE_EVENT"
    PRIVATE_EVENT_PRICING = "PRIVATE_EVENT_PRICING"
    PRIVATE_EVENT_AVAILABILITY = "PRIVATE_EVENT_AVAILABILITY"
    PRIVATE_EVENT_CONTACT = "PRIVATE_EVENT_CONTACT"

    SPECIALS = "SPECIALS"
    HAPPY_HOUR = "HAPPY_HOUR"

    HUMAN_HANDOFF = "HUMAN_HANDOFF"
    FEEDBACK = "FEEDBACK"
    COMPLAINT = "COMPLAINT"
    LOST_AND_FOUND = "LOST_AND_FOUND"
    JOBS = "JOBS"
    ACCESSIBILITY = "ACCESSIBILITY"

    WEBSITE_NAVIGATION = "WEBSITE_NAVIGATION"
    UNKNOWN = "UNKNOWN"


class IntentDecision(str, Enum):
    ACCEPTED = "accepted"
    AMBIGUOUS = "ambiguous"
    LOW_CONFIDENCE = "low_confidence"
    MULTI_INTENT = "multi_intent"
    UNKNOWN = "unknown"


# ============================================================
# SECTION 03 - DATA STRUCTURES
# ============================================================

@dataclass(frozen=True, slots=True)
class EvidenceMatch:
    intent: IntentName
    evidence_type: str
    pattern: str
    matched_text: str
    weight: float
    start_index: int | None = None
    end_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "evidence_type": self.evidence_type,
            "pattern": self.pattern,
            "matched_text": self.matched_text,
            "weight": self.weight,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class IntentCandidate:
    intent: IntentName
    raw_score: float
    confidence: float
    positive_score: float
    negative_score: float
    precedence_bonus: float
    context_bonus: float
    page_context_bonus: float
    evidence: tuple[EvidenceMatch, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "raw_score": self.raw_score,
            "confidence": self.confidence,
            "positive_score": self.positive_score,
            "negative_score": self.negative_score,
            "precedence_bonus": self.precedence_bonus,
            "context_bonus": self.context_bonus,
            "page_context_bonus": self.page_context_bonus,
            "evidence": [
                item.as_dict()
                for item in self.evidence
            ],
        }


@dataclass(frozen=True, slots=True)
class IntentResult:
    original_text: str
    normalized_text: str
    corrected_text: str
    primary_intent: IntentName
    confidence: float
    decision: IntentDecision
    alternatives: tuple[IntentCandidate, ...]
    detected_intents: tuple[IntentName, ...]
    ambiguity_margin: float
    evidence: tuple[EvidenceMatch, ...]
    previous_intent_used: bool
    page_context_used: bool
    explicit_override: bool

    @property
    def is_confident(self) -> bool:
        return (
            self.primary_intent != IntentName.UNKNOWN
            and self.decision in {
                IntentDecision.ACCEPTED,
                IntentDecision.MULTI_INTENT,
            }
        )

    @property
    def is_multi_intent(self) -> bool:
        return len(self.detected_intents) > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "corrected_text": self.corrected_text,
            "primary_intent": self.primary_intent.value,
            "confidence": self.confidence,
            "decision": self.decision.value,
            "alternatives": [
                candidate.as_dict()
                for candidate in self.alternatives
            ],
            "detected_intents": [
                intent.value
                for intent in self.detected_intents
            ],
            "ambiguity_margin": self.ambiguity_margin,
            "evidence": [
                item.as_dict()
                for item in self.evidence
            ],
            "previous_intent_used": self.previous_intent_used,
            "page_context_used": self.page_context_used,
            "explicit_override": self.explicit_override,
            "is_confident": self.is_confident,
            "is_multi_intent": self.is_multi_intent,
        }


@dataclass(frozen=True, slots=True)
class IntentRule:
    intent: IntentName
    positive_phrases: tuple[tuple[str, float], ...] = ()
    positive_regex: tuple[tuple[str, float], ...] = ()
    negative_phrases: tuple[tuple[str, float], ...] = ()
    required_any: tuple[str, ...] = ()
    precedence: int = 0
    page_categories: tuple[str, ...] = ()
    context_compatible: tuple[IntentName, ...] = ()


# ============================================================
# SECTION 04 - INTENT RULES
# ============================================================

INTENT_RULES: Final[tuple[IntentRule, ...]] = (
    IntentRule(
        intent=IntentName.GREETING,
        positive_phrases=(
            ("hello", 1.0),
            ("hi", 0.9),
            ("hey", 0.9),
            ("good morning", 1.0),
            ("good afternoon", 1.0),
            ("good evening", 1.0),
        ),
        negative_phrases=(
            ("say hello to", 0.8),
        ),
        precedence=5,
    ),
    IntentRule(
        intent=IntentName.GOODBYE,
        positive_phrases=(
            ("goodbye", 1.0),
            ("bye", 0.9),
            ("see you", 0.9),
            ("talk later", 0.8),
        ),
        precedence=5,
    ),
    IntentRule(
        intent=IntentName.THANKS,
        positive_phrases=(
            ("thank you", 1.0),
            ("thanks", 0.9),
            ("appreciate it", 0.9),
        ),
        precedence=5,
    ),
    IntentRule(
        intent=IntentName.HOURS_KITCHEN,
        positive_phrases=(
            ("kitchen hours", 1.4),
            ("kitchen close", 1.3),
            ("kitchen closes", 1.3),
            ("food service hours", 1.2),
            ("serving food", 1.0),
            ("food until", 1.1),
        ),
        positive_regex=(
            (r"\bwhat time\b.*\bkitchen\b.*\bclose", 1.5),
            (r"\bhow late\b.*\bfood\b", 1.2),
        ),
        precedence=90,
        page_categories=("menu",),
        context_compatible=(IntentName.HOURS_GENERAL,),
    ),
    IntentRule(
        intent=IntentName.HOURS_HAPPY_HOUR,
        positive_phrases=(
            ("happy hour hours", 1.5),
            ("happy hour time", 1.4),
            ("when is happy hour", 1.4),
            ("what time is happy hour", 1.5),
        ),
        precedence=95,
        page_categories=("specials",),
    ),
    IntentRule(
        intent=IntentName.HOURS_TODAY,
        positive_phrases=(
            ("hours today", 1.4),
            ("today's hours", 1.4),
            ("today hours", 1.3),
            ("open today", 1.3),
            ("close today", 1.3),
            ("tonight hours", 1.2),
        ),
        positive_regex=(
            (r"\bwhat time\b.*\b(?:open|close)\b.*\btoday\b", 1.5),
            (r"\bare you open\b.*\btoday\b", 1.4),
        ),
        negative_phrases=(
            ("kitchen", 0.5),
            ("happy hour", 0.7),
        ),
        precedence=80,
        context_compatible=(IntentName.HOURS_GENERAL,),
    ),
    IntentRule(
        intent=IntentName.HOURS_GENERAL,
        positive_phrases=(
            ("hours", 1.0),
            ("opening time", 1.2),
            ("closing time", 1.2),
            ("what time do you open", 1.4),
            ("what time do you close", 1.4),
            ("when do you open", 1.3),
            ("when do you close", 1.3),
            ("are you open", 1.1),
            ("how late are you open", 1.3),
        ),
        positive_regex=(
            (r"\bwhat time\b.*\b(?:open|close)\b", 1.2),
            (r"\bhow late\b.*\bopen\b", 1.2),
        ),
        negative_phrases=(
            ("kitchen", 0.5),
            ("happy hour", 0.7),
            ("private event", 0.4),
        ),
        precedence=50,
    ),
    IntentRule(
        intent=IntentName.MENU_DIETARY,
        positive_phrases=(
            ("vegan", 1.2),
            ("vegetarian", 1.2),
            ("gluten-free", 1.3),
            ("gluten free", 1.3),
            ("dairy-free", 1.3),
            ("dairy free", 1.3),
            ("dietary options", 1.3),
            ("diet restrictions", 1.2),
        ),
        positive_regex=(
            (r"\bdo you have\b.*\b(?:vegan|vegetarian|gluten)", 1.4),
        ),
        precedence=85,
        page_categories=("menu",),
    ),
    IntentRule(
        intent=IntentName.MENU_ALLERGEN,
        positive_phrases=(
            ("allergy", 1.3),
            ("allergies", 1.3),
            ("allergen", 1.4),
            ("allergens", 1.4),
            ("contains nuts", 1.4),
            ("peanut", 1.1),
            ("shellfish", 1.1),
            ("cross contamination", 1.3),
        ),
        precedence=90,
        page_categories=("menu",),
    ),
    IntentRule(
        intent=IntentName.MENU_PRICE,
        positive_phrases=(
            ("how much", 1.1),
            ("price", 1.0),
            ("prices", 1.0),
            ("cost", 1.0),
            ("menu pricing", 1.4),
        ),
        required_any=("menu", "burger", "food", "drink", "beer", "cocktail"),
        precedence=75,
        page_categories=("menu",),
    ),
    IntentRule(
        intent=IntentName.MENU_ITEM_LOOKUP,
        positive_phrases=(
            ("do you have", 0.6),
            ("serve", 0.8),
            ("food item", 0.9),
            ("menu item", 1.2),
            ("burger", 0.8),
            ("wings", 0.8),
            ("pizza", 0.8),
            ("sandwich", 0.8),
            ("salad", 0.8),
            ("beer", 0.7),
            ("cocktail", 0.7),
            ("dessert", 0.8),
        ),
        negative_phrases=(
            ("private event", 0.7),
            ("catering package", 0.7),
        ),
        precedence=55,
        page_categories=("menu",),
    ),
    IntentRule(
        intent=IntentName.MENU_GENERAL,
        positive_phrases=(
            ("menu", 1.3),
            ("show me the menu", 1.5),
            ("food menu", 1.4),
            ("drink menu", 1.4),
            ("what food do you have", 1.2),
            ("what do you serve", 1.1),
        ),
        negative_phrases=(
            ("private event menu", 0.8),
            ("catering menu", 0.8),
        ),
        precedence=50,
        page_categories=("menu",),
    ),
    IntentRule(
        intent=IntentName.DIRECTIONS,
        positive_phrases=(
            ("directions", 1.4),
            ("how do i get there", 1.4),
            ("how do i get to", 1.3),
            ("navigate to", 1.2),
            ("driving directions", 1.4),
        ),
        precedence=75,
        page_categories=("contact",),
    ),
    IntentRule(
        intent=IntentName.PARKING,
        positive_phrases=(
            ("parking", 1.4),
            ("where can i park", 1.5),
            ("place to park", 1.3),
            ("parking lot", 1.3),
            ("street parking", 1.3),
            ("valet", 1.2),
        ),
        precedence=85,
        page_categories=("contact",),
    ),
    IntentRule(
        intent=IntentName.LOCATION,
        positive_phrases=(
            ("address", 1.4),
            ("location", 1.2),
            ("where are you located", 1.5),
            ("where is horseshoe", 1.4),
            ("what town", 1.0),
        ),
        negative_phrases=(
            ("parking", 0.8),
            ("directions", 0.8),
        ),
        precedence=65,
        page_categories=("contact",),
    ),
    IntentRule(
        intent=IntentName.CONTACT,
        positive_phrases=(
            ("phone number", 1.4),
            ("email address", 1.4),
            ("contact information", 1.3),
            ("contact you", 1.1),
            ("call you", 1.0),
            ("email you", 1.0),
        ),
        precedence=60,
        page_categories=("contact",),
    ),
    IntentRule(
        intent=IntentName.RESERVATION_CANCEL,
        positive_phrases=(
            ("cancel reservation", 1.6),
            ("cancel my reservation", 1.7),
            ("cancel booking", 1.5),
            ("remove reservation", 1.4),
        ),
        precedence=100,
        page_categories=("reservations",),
    ),
    IntentRule(
        intent=IntentName.RESERVATION_CHANGE,
        positive_phrases=(
            ("change reservation", 1.6),
            ("modify reservation", 1.5),
            ("move reservation", 1.5),
            ("update reservation", 1.5),
            ("change booking", 1.4),
        ),
        precedence=95,
        page_categories=("reservations",),
    ),
    IntentRule(
        intent=IntentName.RESERVATION,
        positive_phrases=(
            ("reservation", 1.2),
            ("book a table", 1.5),
            ("reserve a table", 1.5),
            ("table for", 1.2),
            ("make a booking", 1.3),
            ("dinner reservation", 1.4),
        ),
        negative_phrases=(
            ("private event", 0.9),
            ("party for", 0.6),
            ("cancel", 0.7),
            ("change", 0.5),
        ),
        precedence=70,
        page_categories=("reservations",),
    ),
    IntentRule(
        intent=IntentName.DELIVERY,
        positive_phrases=(
            ("delivery", 1.4),
            ("deliver", 1.2),
            ("doordash", 1.3),
            ("grubhub", 1.3),
            ("uber eats", 1.3),
            ("food delivered", 1.3),
        ),
        precedence=90,
        page_categories=("order", "ordering"),
    ),
    IntentRule(
        intent=IntentName.TAKEOUT,
        positive_phrases=(
            ("takeout", 1.4),
            ("take out", 1.4),
            ("pickup order", 1.4),
            ("pick up food", 1.3),
            ("order for pickup", 1.5),
            ("to go order", 1.3),
        ),
        precedence=85,
        page_categories=("order", "ordering", "menu"),
    ),
    IntentRule(
        intent=IntentName.ORDERING,
        positive_phrases=(
            ("order online", 1.5),
            ("place an order", 1.4),
            ("order food", 1.3),
            ("how do i order", 1.4),
            ("online ordering", 1.4),
        ),
        negative_phrases=(
            ("private event order", 0.7),
        ),
        precedence=70,
        page_categories=("order", "ordering", "menu"),
    ),
    IntentRule(
        intent=IntentName.EVENTS_TONIGHT,
        positive_phrases=(
            ("what is happening tonight", 1.6),
            ("events tonight", 1.6),
            ("tonight's events", 1.6),
            ("anything tonight", 1.3),
            ("what is on tonight", 1.4),
        ),
        precedence=95,
        page_categories=("events",),
    ),
    IntentRule(
        intent=IntentName.LIVE_MUSIC,
        positive_phrases=(
            ("live music", 1.5),
            ("band", 1.0),
            ("bands", 1.0),
            ("music tonight", 1.3),
            ("who is playing", 1.4),
            ("live entertainment", 1.3),
        ),
        precedence=90,
        page_categories=("events",),
    ),
    IntentRule(
        intent=IntentName.SPORTS_VIEWING,
        positive_phrases=(
            ("watch the game", 1.5),
            ("showing the game", 1.5),
            ("sports on tv", 1.4),
            ("football game", 1.1),
            ("baseball game", 1.1),
            ("basketball game", 1.1),
            ("hockey game", 1.1),
            ("game on", 1.0),
        ),
        negative_phrases=(
            ("private game room", 0.6),
        ),
        precedence=85,
        page_categories=("events",),
    ),
    IntentRule(
        intent=IntentName.EVENTS_GENERAL,
        positive_phrases=(
            ("events", 1.3),
            ("event calendar", 1.4),
            ("upcoming events", 1.5),
            ("what is happening", 1.2),
            ("what is going on", 1.1),
            ("entertainment schedule", 1.4),
        ),
        negative_phrases=(
            ("private event", 1.0),
            ("book an event", 0.8),
        ),
        precedence=60,
        page_categories=("events",),
    ),
    IntentRule(
        intent=IntentName.PRIVATE_EVENT_PRICING,
        positive_phrases=(
            ("private event price", 1.6),
            ("private event pricing", 1.7),
            ("party package price", 1.6),
            ("how much for a party", 1.5),
            ("event package cost", 1.5),
            ("cost per person", 1.4),
            ("minimum spend", 1.4),
        ),
        precedence=110,
        page_categories=("private_events",),
    ),
    IntentRule(
        intent=IntentName.PRIVATE_EVENT_AVAILABILITY,
        positive_phrases=(
            ("private event availability", 1.7),
            ("is the event space available", 1.7),
            ("available for a party", 1.5),
            ("book the event space", 1.5),
            ("party date available", 1.5),
        ),
        precedence=108,
        page_categories=("private_events",),
    ),
    IntentRule(
        intent=IntentName.PRIVATE_EVENT_CONTACT,
        positive_phrases=(
            ("event coordinator", 1.5),
            ("private events contact", 1.6),
            ("who handles private events", 1.5),
            ("speak to event manager", 1.5),
        ),
        precedence=106,
        page_categories=("private_events",),
    ),
    IntentRule(
        intent=IntentName.PRIVATE_EVENT,
        positive_phrases=(
            ("private event", 1.5),
            ("private party", 1.5),
            ("birthday party", 1.3),
            ("corporate event", 1.4),
            ("rehearsal dinner", 1.4),
            ("baby shower", 1.3),
            ("bridal shower", 1.3),
            ("event space", 1.4),
            ("party room", 1.3),
            ("book a party", 1.4),
            ("group event", 1.2),
        ),
        negative_phrases=(
            ("event tonight", 0.8),
            ("upcoming events", 0.8),
            ("live music", 0.7),
        ),
        precedence=100,
        page_categories=("private_events",),
    ),
    IntentRule(
        intent=IntentName.HAPPY_HOUR,
        positive_phrases=(
            ("happy hour", 1.5),
            ("drink specials", 1.2),
            ("happy hour specials", 1.6),
        ),
        negative_phrases=(
            ("what time", 0.4),
            ("hours", 0.4),
        ),
        precedence=80,
        page_categories=("specials",),
    ),
    IntentRule(
        intent=IntentName.SPECIALS,
        positive_phrases=(
            ("specials", 1.3),
            ("today's specials", 1.5),
            ("daily specials", 1.4),
            ("food specials", 1.3),
            ("drink specials", 1.2),
            ("deals", 1.0),
        ),
        negative_phrases=(
            ("private event package", 0.6),
        ),
        precedence=60,
        page_categories=("specials",),
    ),
    IntentRule(
        intent=IntentName.HUMAN_HANDOFF,
        positive_phrases=(
            ("speak to a person", 1.7),
            ("talk to a person", 1.7),
            ("human agent", 1.6),
            ("real person", 1.6),
            ("manager", 1.1),
            ("call me", 1.0),
            ("contact management", 1.4),
        ),
        precedence=120,
    ),
    IntentRule(
        intent=IntentName.COMPLAINT,
        positive_phrases=(
            ("complaint", 1.5),
            ("bad experience", 1.4),
            ("terrible service", 1.4),
            ("very disappointed", 1.3),
            ("food was cold", 1.2),
            ("wrong order", 1.2),
            ("speak to manager", 1.4),
        ),
        precedence=115,
    ),
    IntentRule(
        intent=IntentName.FEEDBACK,
        positive_phrases=(
            ("feedback", 1.4),
            ("leave a review", 1.3),
            ("suggestion", 1.2),
            ("tell you about my experience", 1.3),
        ),
        negative_phrases=(
            ("complaint", 0.8),
            ("terrible", 0.6),
        ),
        precedence=75,
    ),
    IntentRule(
        intent=IntentName.LOST_AND_FOUND,
        positive_phrases=(
            ("lost and found", 1.6),
            ("i lost", 1.3),
            ("left my", 1.2),
            ("forgot my", 1.2),
            ("found an item", 1.3),
        ),
        precedence=100,
    ),
    IntentRule(
        intent=IntentName.JOBS,
        positive_phrases=(
            ("jobs", 1.3),
            ("job opening", 1.5),
            ("hiring", 1.4),
            ("apply for a job", 1.5),
            ("employment", 1.2),
            ("work at horseshoe", 1.4),
        ),
        precedence=85,
    ),
    IntentRule(
        intent=IntentName.ACCESSIBILITY,
        positive_phrases=(
            ("wheelchair accessible", 1.6),
            ("accessible entrance", 1.5),
            ("accessibility", 1.4),
            ("handicap accessible", 1.5),
            ("accessible bathroom", 1.4),
        ),
        precedence=90,
    ),
    IntentRule(
        intent=IntentName.WEBSITE_NAVIGATION,
        positive_phrases=(
            ("take me to", 1.2),
            ("open the page", 1.2),
            ("go to the", 1.0),
            ("show the page", 1.0),
            ("website page", 1.0),
        ),
        precedence=40,
    ),
)


# ============================================================
# SECTION 05 - EXPLICIT OVERRIDE TERMS
# ============================================================

EXPLICIT_OVERRIDE_PATTERNS: Final[tuple[str, ...]] = (
    r"\bactually\b",
    r"\binstead\b",
    r"\bnew question\b",
    r"\bforget that\b",
    r"\bnever mind\b",
    r"\bwhat about\b",
    r"\bseparately\b",
    r"\banother question\b",
)


# ============================================================
# SECTION 06 - CLASSIFIER
# ============================================================

class IntentClassifier:
    """
    Weighted deterministic intent classifier.
    """

    def __init__(
        self,
        *,
        rules: Sequence[IntentRule] | None = None,
        minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
        ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
        maximum_alternatives: int = DEFAULT_MAX_ALTERNATIVES,
        multi_intent_threshold: float = DEFAULT_MULTI_INTENT_THRESHOLD,
        context_weight: float = DEFAULT_CONTEXT_WEIGHT,
        page_context_weight: float = DEFAULT_PAGE_CONTEXT_WEIGHT,
    ) -> None:
        self.rules = tuple(
            rules or INTENT_RULES
        )

        self.minimum_confidence = float(
            minimum_confidence
        )

        self.ambiguity_margin = float(
            ambiguity_margin
        )

        self.maximum_alternatives = max(
            1,
            int(maximum_alternatives),
        )

        self.multi_intent_threshold = float(
            multi_intent_threshold
        )

        self.context_weight = float(
            context_weight
        )

        self.page_context_weight = float(
            page_context_weight
        )

        self._compiled_phrase_patterns: dict[
            tuple[IntentName, str, str],
            re.Pattern[str],
        ] = {}

        self._compiled_regex_patterns: dict[
            tuple[IntentName, str],
            re.Pattern[str],
        ] = {}

        self._compile_patterns()

    # ========================================================
    # SECTION 07 - PUBLIC CLASSIFICATION
    # ========================================================

    def classify(
        self,
        text: str,
        *,
        previous_intent: IntentName | str | None = None,
        page_category: str | None = None,
        conversation_context: Mapping[str, Any] | None = None,
    ) -> IntentResult:
        normalization = normalize_text(text)
        spelling = correct_spelling(
            normalization.normalized_text
        )

        corrected_text = spelling.corrected_text
        lowered = corrected_text.casefold()
        tokens = set(
            TOKEN_PATTERN.findall(lowered)
        )

        previous_intent_value = (
            self._coerce_intent(previous_intent)
        )

        explicit_override = any(
            re.search(
                pattern,
                lowered,
                re.IGNORECASE,
            )
            for pattern
            in EXPLICIT_OVERRIDE_PATTERNS
        )

        candidates = self._score_all_rules(
            text=corrected_text,
            lowered=lowered,
            tokens=tokens,
            previous_intent=previous_intent_value,
            page_category=page_category,
            conversation_context=(
                conversation_context or {}
            ),
            explicit_override=explicit_override,
        )

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.raw_score,
                candidate.precedence_bonus,
                candidate.positive_score,
                candidate.intent.value,
            ),
            reverse=True,
        )

        positive_candidates = [
            candidate
            for candidate in ranked
            if candidate.raw_score > 0
        ]

        if not positive_candidates:
            return IntentResult(
                original_text=text,
                normalized_text=(
                    normalization.normalized_text
                ),
                corrected_text=corrected_text,
                primary_intent=IntentName.UNKNOWN,
                confidence=0.0,
                decision=IntentDecision.UNKNOWN,
                alternatives=(),
                detected_intents=(),
                ambiguity_margin=0.0,
                evidence=(),
                previous_intent_used=False,
                page_context_used=False,
                explicit_override=explicit_override,
            )

        calibrated = self._calibrate_candidates(
            positive_candidates
        )

        best = calibrated[0]
        second = (
            calibrated[1]
            if len(calibrated) > 1
            else None
        )

        margin = (
            best.confidence
            - second.confidence
            if second is not None
            else best.confidence
        )

        detected_intents = self._detect_multi_intents(
            corrected_text,
            calibrated,
        )

        if best.confidence < self.minimum_confidence:
            primary_intent = IntentName.UNKNOWN
            decision = IntentDecision.LOW_CONFIDENCE
        elif (
            second is not None
            and margin < self.ambiguity_margin
        ):
            primary_intent = best.intent
            decision = IntentDecision.AMBIGUOUS
        elif len(detected_intents) > 1:
            primary_intent = best.intent
            decision = IntentDecision.MULTI_INTENT
        else:
            primary_intent = best.intent
            decision = IntentDecision.ACCEPTED

        previous_intent_used = (
            best.context_bonus > 0
        )

        page_context_used = (
            best.page_context_bonus > 0
        )

        return IntentResult(
            original_text=text,
            normalized_text=(
                normalization.normalized_text
            ),
            corrected_text=corrected_text,
            primary_intent=primary_intent,
            confidence=round(
                best.confidence,
                6,
            ),
            decision=decision,
            alternatives=tuple(
                calibrated[
                    : self.maximum_alternatives
                ]
            ),
            detected_intents=detected_intents,
            ambiguity_margin=round(
                margin,
                6,
            ),
            evidence=best.evidence,
            previous_intent_used=(
                previous_intent_used
            ),
            page_context_used=(
                page_context_used
            ),
            explicit_override=explicit_override,
        )

    # ========================================================
    # SECTION 08 - RULE SCORING
    # ========================================================

    def _score_all_rules(
        self,
        *,
        text: str,
        lowered: str,
        tokens: set[str],
        previous_intent: IntentName | None,
        page_category: str | None,
        conversation_context: Mapping[str, Any],
        explicit_override: bool,
    ) -> list[IntentCandidate]:
        candidates: list[IntentCandidate] = []

        normalized_page_category = (
            str(page_category or "")
            .strip()
            .lower()
            .replace("-", "_")
        )

        for rule in self.rules:
            evidence: list[EvidenceMatch] = []

            positive_score = self._score_positive_evidence(
                rule,
                text,
                lowered,
                evidence,
            )

            negative_score = self._score_negative_evidence(
                rule,
                text,
                lowered,
                evidence,
            )

            if (
                rule.required_any
                and not any(
                    required.casefold() in tokens
                    or self._contains_phrase(
                        lowered,
                        required.casefold(),
                    )
                    for required in rule.required_any
                )
            ):
                positive_score *= 0.35

            precedence_bonus = (
                rule.precedence / 1000.0
                if positive_score > 0
                else 0.0
            )

            context_bonus = 0.0

            if (
                previous_intent is not None
                and not explicit_override
                and previous_intent
                in rule.context_compatible
                and positive_score > 0
            ):
                context_bonus = (
                    self.context_weight
                )

                evidence.append(
                    EvidenceMatch(
                        intent=rule.intent,
                        evidence_type="conversation_context",
                        pattern=previous_intent.value,
                        matched_text=previous_intent.value,
                        weight=context_bonus,
                    )
                )

            page_context_bonus = 0.0

            if (
                normalized_page_category
                and normalized_page_category
                in rule.page_categories
                and positive_score > 0
            ):
                page_context_bonus = (
                    self.page_context_weight
                )

                evidence.append(
                    EvidenceMatch(
                        intent=rule.intent,
                        evidence_type="page_context",
                        pattern=normalized_page_category,
                        matched_text=normalized_page_category,
                        weight=page_context_bonus,
                    )
                )

            if conversation_context:
                context_bonus += self._score_structured_context(
                    rule,
                    conversation_context,
                    positive_score,
                    evidence,
                    explicit_override,
                )

            raw_score = (
                positive_score
                - negative_score
                + precedence_bonus
                + context_bonus
                + page_context_bonus
            )

            candidates.append(
                IntentCandidate(
                    intent=rule.intent,
                    raw_score=round(
                        raw_score,
                        6,
                    ),
                    confidence=0.0,
                    positive_score=round(
                        positive_score,
                        6,
                    ),
                    negative_score=round(
                        negative_score,
                        6,
                    ),
                    precedence_bonus=round(
                        precedence_bonus,
                        6,
                    ),
                    context_bonus=round(
                        context_bonus,
                        6,
                    ),
                    page_context_bonus=round(
                        page_context_bonus,
                        6,
                    ),
                    evidence=tuple(evidence),
                )
            )

        return candidates

    def _score_positive_evidence(
        self,
        rule: IntentRule,
        text: str,
        lowered: str,
        evidence: list[EvidenceMatch],
    ) -> float:
        total = 0.0

        for phrase, weight in rule.positive_phrases:
            pattern = self._compiled_phrase_patterns[
                (
                    rule.intent,
                    "positive",
                    phrase,
                )
            ]

            for match in pattern.finditer(lowered):
                total += weight

                evidence.append(
                    EvidenceMatch(
                        intent=rule.intent,
                        evidence_type="positive_phrase",
                        pattern=phrase,
                        matched_text=(
                            text[
                                match.start():
                                match.end()
                            ]
                        ),
                        weight=weight,
                        start_index=match.start(),
                        end_index=match.end(),
                    )
                )

        for regex_pattern, weight in rule.positive_regex:
            pattern = self._compiled_regex_patterns[
                (
                    rule.intent,
                    regex_pattern,
                )
            ]

            for match in pattern.finditer(lowered):
                total += weight

                evidence.append(
                    EvidenceMatch(
                        intent=rule.intent,
                        evidence_type="positive_regex",
                        pattern=regex_pattern,
                        matched_text=(
                            text[
                                match.start():
                                match.end()
                            ]
                        ),
                        weight=weight,
                        start_index=match.start(),
                        end_index=match.end(),
                    )
                )

        return total

    def _score_negative_evidence(
        self,
        rule: IntentRule,
        text: str,
        lowered: str,
        evidence: list[EvidenceMatch],
    ) -> float:
        total = 0.0

        for phrase, weight in rule.negative_phrases:
            pattern = self._compiled_phrase_patterns[
                (
                    rule.intent,
                    "negative",
                    phrase,
                )
            ]

            for match in pattern.finditer(lowered):
                total += weight

                evidence.append(
                    EvidenceMatch(
                        intent=rule.intent,
                        evidence_type="negative_phrase",
                        pattern=phrase,
                        matched_text=(
                            text[
                                match.start():
                                match.end()
                            ]
                        ),
                        weight=-weight,
                        start_index=match.start(),
                        end_index=match.end(),
                    )
                )

        return total

    def _score_structured_context(
        self,
        rule: IntentRule,
        conversation_context: Mapping[str, Any],
        positive_score: float,
        evidence: list[EvidenceMatch],
        explicit_override: bool,
    ) -> float:
        if positive_score <= 0 or explicit_override:
            return 0.0

        active_flow = str(
            conversation_context.get(
                "active_flow",
                ""
            )
        ).strip().lower()

        if (
            active_flow == "private_event"
            and rule.intent in {
                IntentName.PRIVATE_EVENT,
                IntentName.PRIVATE_EVENT_PRICING,
                IntentName.PRIVATE_EVENT_AVAILABILITY,
                IntentName.PRIVATE_EVENT_CONTACT,
            }
        ):
            bonus = self.context_weight

            evidence.append(
                EvidenceMatch(
                    intent=rule.intent,
                    evidence_type="active_flow",
                    pattern="private_event",
                    matched_text="private_event",
                    weight=bonus,
                )
            )

            return bonus

        if (
            active_flow == "reservation"
            and rule.intent in {
                IntentName.RESERVATION,
                IntentName.RESERVATION_CHANGE,
                IntentName.RESERVATION_CANCEL,
            }
        ):
            bonus = self.context_weight

            evidence.append(
                EvidenceMatch(
                    intent=rule.intent,
                    evidence_type="active_flow",
                    pattern="reservation",
                    matched_text="reservation",
                    weight=bonus,
                )
            )

            return bonus

        return 0.0

    # ========================================================
    # SECTION 09 - CONFIDENCE CALIBRATION
    # ========================================================

    def _calibrate_candidates(
        self,
        candidates: Sequence[IntentCandidate],
    ) -> list[IntentCandidate]:
        scores = [
            max(
                candidate.raw_score,
                0.0,
            )
            for candidate in candidates
        ]

        if not scores:
            return []

        maximum = max(scores)

        exponentials = [
            math.exp(
                min(
                    score - maximum,
                    50.0,
                )
            )
            for score in scores
        ]

        denominator = sum(exponentials) or 1.0

        calibrated: list[IntentCandidate] = []

        for candidate, exponential in zip(
            candidates,
            exponentials,
            strict=True,
        ):
            softmax_probability = (
                exponential / denominator
            )

            absolute_strength = (
                1.0
                - math.exp(
                    -max(
                        candidate.raw_score,
                        0.0,
                    )
                )
            )

            confidence = (
                softmax_probability * 0.45
                + absolute_strength * 0.55
            )

            calibrated.append(
                IntentCandidate(
                    intent=candidate.intent,
                    raw_score=candidate.raw_score,
                    confidence=round(
                        min(
                            max(confidence, 0.0),
                            1.0,
                        ),
                        6,
                    ),
                    positive_score=(
                        candidate.positive_score
                    ),
                    negative_score=(
                        candidate.negative_score
                    ),
                    precedence_bonus=(
                        candidate.precedence_bonus
                    ),
                    context_bonus=(
                        candidate.context_bonus
                    ),
                    page_context_bonus=(
                        candidate.page_context_bonus
                    ),
                    evidence=candidate.evidence,
                )
            )

        calibrated.sort(
            key=lambda candidate: (
                candidate.confidence,
                candidate.raw_score,
                candidate.precedence_bonus,
            ),
            reverse=True,
        )

        return calibrated

    # ========================================================
    # SECTION 10 - MULTI-INTENT DETECTION
    # ========================================================

    def _detect_multi_intents(
        self,
        text: str,
        candidates: Sequence[IntentCandidate],
    ) -> tuple[IntentName, ...]:
        if not candidates:
            return ()

        strong_candidates = [
            candidate
            for candidate in candidates
            if (
                candidate.confidence
                >= self.multi_intent_threshold
                or candidate.raw_score >= 1.35
            )
        ]

        if len(strong_candidates) <= 1:
            return (
                strong_candidates[0].intent,
            ) if strong_candidates else (
                candidates[0].intent,
            )

        clauses = [
            clause.strip()
            for clause in CLAUSE_SPLIT_PATTERN.split(
                text
            )
            if clause.strip()
        ]

        if len(clauses) <= 1:
            return (
                candidates[0].intent,
            )

        detected: list[IntentName] = []

        for clause in clauses:
            clause_lower = clause.casefold()
            clause_best: tuple[
                IntentName,
                float,
            ] | None = None

            for rule in self.rules:
                score = 0.0

                for phrase, weight in rule.positive_phrases:
                    if self._contains_phrase(
                        clause_lower,
                        phrase.casefold(),
                    ):
                        score += weight

                for regex_pattern, weight in rule.positive_regex:
                    if re.search(
                        regex_pattern,
                        clause_lower,
                        re.IGNORECASE,
                    ):
                        score += weight

                if (
                    clause_best is None
                    or score > clause_best[1]
                ):
                    clause_best = (
                        rule.intent,
                        score,
                    )

            if (
                clause_best is not None
                and clause_best[1] > 0
                and clause_best[0] not in detected
            ):
                detected.append(
                    clause_best[0]
                )

        return tuple(
            detected[:4]
        ) or (
            candidates[0].intent,
        )

    # ========================================================
    # SECTION 11 - PATTERN COMPILATION
    # ========================================================

    def _compile_patterns(self) -> None:
        for rule in self.rules:
            for phrase, _weight in rule.positive_phrases:
                self._compiled_phrase_patterns[
                    (
                        rule.intent,
                        "positive",
                        phrase,
                    )
                ] = self._compile_phrase(
                    phrase
                )

            for phrase, _weight in rule.negative_phrases:
                self._compiled_phrase_patterns[
                    (
                        rule.intent,
                        "negative",
                        phrase,
                    )
                ] = self._compile_phrase(
                    phrase
                )

            for regex_pattern, _weight in rule.positive_regex:
                self._compiled_regex_patterns[
                    (
                        rule.intent,
                        regex_pattern,
                    )
                ] = re.compile(
                    regex_pattern,
                    re.IGNORECASE,
                )

    @staticmethod
    def _compile_phrase(
        phrase: str,
    ) -> re.Pattern[str]:
        escaped = re.escape(
            phrase.casefold()
        ).replace(
            r"\ ",
            r"\s+",
        )

        return re.compile(
            WORD_BOUNDARY_TEMPLATE.format(
                escaped
            ),
            re.IGNORECASE,
        )

    @staticmethod
    def _contains_phrase(
        text: str,
        phrase: str,
    ) -> bool:
        escaped = re.escape(
            phrase
        ).replace(
            r"\ ",
            r"\s+",
        )

        return bool(
            re.search(
                WORD_BOUNDARY_TEMPLATE.format(
                    escaped
                ),
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _coerce_intent(
        value: IntentName | str | None,
    ) -> IntentName | None:
        if value is None:
            return None

        if isinstance(value, IntentName):
            return value

        candidate = str(value).strip().upper()

        try:
            return IntentName(candidate)
        except ValueError:
            return None


# ============================================================
# SECTION 12 - MODULE-LEVEL CLASSIFIER
# ============================================================

_default_classifier = IntentClassifier()


def classify_intent(
    text: str,
    *,
    previous_intent: IntentName | str | None = None,
    page_category: str | None = None,
    conversation_context: Mapping[str, Any] | None = None,
) -> IntentResult:
    """
    Classify one customer message through the shared intent engine.
    """

    return _default_classifier.classify(
        text,
        previous_intent=previous_intent,
        page_category=page_category,
        conversation_context=conversation_context,
    )


def intent_name(
    text: str,
    *,
    previous_intent: IntentName | str | None = None,
    page_category: str | None = None,
) -> str:
    """
    Return only the primary intent name.
    """

    return classify_intent(
        text,
        previous_intent=previous_intent,
        page_category=page_category,
    ).primary_intent.value


# ============================================================
# SECTION 13 - VALIDATION
# ============================================================

def validate_intent_module() -> dict[str, Any]:
    classifier = IntentClassifier()

    cases = [
        {
            "text": "What time do you close?",
            "expected": IntentName.HOURS_GENERAL,
        },
        {
            "text": "What time does the kitchen close?",
            "expected": IntentName.HOURS_KITCHEN,
        },
        {
            "text": "Show me the menu.",
            "expected": IntentName.MENU_GENERAL,
        },
        {
            "text": "Do you have gluten-free food?",
            "expected": IntentName.MENU_DIETARY,
        },
        {
            "text": "Where can I park?",
            "expected": IntentName.PARKING,
        },
        {
            "text": "I want to book a birthday party for 50 people.",
            "expected": IntentName.PRIVATE_EVENT,
        },
        {
            "text": "How much is a private event package?",
            "expected": IntentName.PRIVATE_EVENT_PRICING,
        },
        {
            "text": "What is happening tonight?",
            "expected": IntentName.EVENTS_TONIGHT,
        },
        {
            "text": "Are you showing the baseball game?",
            "expected": IntentName.SPORTS_VIEWING,
        },
        {
            "text": "I need to speak to a real person.",
            "expected": IntentName.HUMAN_HANDOFF,
        },
        {
            "text": "Can I order food for pickup?",
            "expected": IntentName.TAKEOUT,
        },
        {
            "text": "I need to cancel my reservation.",
            "expected": IntentName.RESERVATION_CANCEL,
        },
    ]

    case_results: list[dict[str, Any]] = []

    for case in cases:
        result = classifier.classify(
            case["text"]
        )

        case_results.append(
            {
                "text": case["text"],
                "expected": (
                    case["expected"].value
                ),
                "actual": (
                    result.primary_intent.value
                ),
                "confidence": result.confidence,
                "decision": result.decision.value,
                "passed": (
                    result.primary_intent
                    == case["expected"]
                ),
            }
        )

    override_result = classifier.classify(
        "Actually, what time does the kitchen close?",
        previous_intent=IntentName.PRIVATE_EVENT,
        conversation_context={
            "active_flow": "private_event",
        },
    )

    context_result = classifier.classify(
        "How much does it cost?",
        previous_intent=(
            IntentName.PRIVATE_EVENT
        ),
        page_category="private_events",
        conversation_context={
            "active_flow": "private_event",
        },
    )

    multi_result = classifier.classify(
        (
            "What time do you close, and what events "
            "are happening tonight?"
        )
    )

    unknown_result = classifier.classify(
        "quantum flux capacitor status"
    )

    checks = {
        "all_cases_passed": all(
            item["passed"]
            for item in case_results
        ),
        "previous_intent_did_not_lock": (
            override_result.primary_intent
            == IntentName.HOURS_KITCHEN
        ),
        "explicit_override_detected": (
            override_result.explicit_override
        ),
        "context_does_not_create_intent_without_evidence": (
            context_result.primary_intent
            in {
                IntentName.UNKNOWN,
                IntentName.PRIVATE_EVENT_PRICING,
            }
        ),
        "multi_intent_detected": (
            multi_result.is_multi_intent
            or multi_result.decision
            == IntentDecision.MULTI_INTENT
        ),
        "unknown_safe": (
            unknown_result.primary_intent
            == IntentName.UNKNOWN
        ),
        "evidence_available": all(
            bool(
                classifier.classify(
                    case["text"]
                ).evidence
            )
            for case in cases
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
        "cases": case_results,
        "override_result": (
            override_result.as_dict()
        ),
        "context_result": (
            context_result.as_dict()
        ),
        "multi_result": (
            multi_result.as_dict()
        ),
        "unknown_result": (
            unknown_result.as_dict()
        ),
    }


# ============================================================
# SECTION 14 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import json

    report = validate_intent_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
