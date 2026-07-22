# ============================================================
# Exact file location: app/nlu/entities.py
# Horseshoe Tavern AI
# Phase 1 Part 1.15
# Restaurant, event, date, time, guest-count, budget, contact,
# menu, sports, reservation, and page-aware entity extraction
# ============================================================

"""
Entity extraction engine for Horseshoe Tavern AI.

Responsibilities:

- Extract dates and relative dates
- Extract times and time ranges
- Extract guest counts
- Extract budgets and price ranges
- Extract phone numbers and email addresses
- Extract event types
- Extract reservation identifiers
- Extract menu, dietary, allergen, and beverage concepts
- Extract sports leagues and teams
- Extract service types such as takeout and delivery
- Extract page-aware and intent-aware entities
- Preserve exact character offsets
- Normalize entity values without destroying original text
- Rank overlapping entities
- Avoid treating ordinary words as unrelated entities
- Support multiple entities of the same type
- Return deterministic confidence and source metadata

This module does not retrieve business facts and does not generate answers.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Sequence

from app.logging_config import get_logger
from app.nlu.intent import IntentName
from app.nlu.normalizer import normalize_text
from app.nlu.spelling import correct_spelling


# ============================================================
# SECTION 01 - LOGGER AND CONSTANTS
# ============================================================

logger = get_logger(__name__)

MAXIMUM_ENTITIES: Final[int] = 64
DEFAULT_REFERENCE_TIMEZONE: Final[str] = "America/New_York"

EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

PHONE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\w)"
    r"(?:\+?1[\s.-]?)?"
    r"(?:\(?\d{3}\)?[\s.-]?)"
    r"\d{3}[\s.-]?\d{4}"
    r"(?!\w)"
)

MONEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\w)"
    r"\$\s?\d+(?:,\d{3})*(?:\.\d{1,2})?"
    r"(?!\w)"
)

MONEY_RANGE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<minimum>\$\s?\d+(?:,\d{3})*(?:\.\d{1,2})?)"
    r"\s*(?:-|to|through|and)\s*"
    r"(?P<maximum>\$\s?\d+(?:,\d{3})*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

PLAIN_BUDGET_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:budget|spend|cost|price)\s*(?:is|of|around|about|up to|under|over)?\s*"
    r"(?P<amount>\d+(?:,\d{3})*(?:\.\d{1,2})?)\b",
    re.IGNORECASE,
)

GUEST_COUNT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?P<count>\d{1,4})\s*"
    r"(?:people|persons|persons?|guests?|attendees?|ppl)\b",
    re.IGNORECASE,
)

PARTY_OF_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:party|table|reservation|booking)\s+(?:of|for)\s+"
    r"(?P<count>\d{1,3})\b",
    re.IGNORECASE,
)

TIME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b"
    r"(?P<hour>0?[1-9]|1[0-2])"
    r"(?::(?P<minute>[0-5]\d))?"
    r"\s*(?P<meridiem>a\.?m\.?|p\.?m\.?)"
    r"\b",
    re.IGNORECASE,
)

TWENTY_FOUR_HOUR_TIME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)\b"
)

TIME_RANGE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b"
    r"(?P<start_hour>0?[1-9]|1[0-2])"
    r"(?::(?P<start_minute>[0-5]\d))?"
    r"\s*(?P<start_meridiem>a\.?m\.?|p\.?m\.?)?"
    r"\s*(?:-|to|until|through)\s*"
    r"(?P<end_hour>0?[1-9]|1[0-2])"
    r"(?::(?P<end_minute>[0-5]\d))?"
    r"\s*(?P<end_meridiem>a\.?m\.?|p\.?m\.?)"
    r"\b",
    re.IGNORECASE,
)

NUMERIC_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b"
    r"(?P<month>\d{1,2})"
    r"[/-]"
    r"(?P<day>\d{1,2})"
    r"(?:[/-](?P<year>\d{2,4}))?"
    r"\b"
)

MONTH_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b"
    r"(?P<month>"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
    r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")"
    r"\s+"
    r"(?P<day>\d{1,2})"
    r"(?:st|nd|rd|th)?"
    r"(?:,\s*(?P<year>\d{4}))?"
    r"\b",
    re.IGNORECASE,
)

RELATIVE_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(today|tonight|tomorrow|this weekend|next weekend)\b",
    re.IGNORECASE,
)

WEEKDAY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b"
    r"(?:(?P<prefix>this|next)\s+)?"
    r"(?P<weekday>"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r")"
    r"\b",
    re.IGNORECASE,
)

RESERVATION_REFERENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:reservation|booking|confirmation|order)"
    r"(?:\s+(?:number|id|code))?"
    r"\s*(?:is|:|#)?\s*"
    r"(?P<reference>[A-Z0-9][A-Z0-9-]{3,31})\b",
    re.IGNORECASE,
)

NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:my name is|name is|i am|i'm)\s+"
    r"(?P<name>[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,2})\b"
)

AGE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?P<age>\d{1,3})\s*(?:years? old|yo)\b",
    re.IGNORECASE,
)


# ============================================================
# SECTION 02 - ENUMERATIONS
# ============================================================

class EntityType(str, Enum):
    DATE = "date"
    RELATIVE_DATE = "relative_date"
    WEEKDAY = "weekday"
    TIME = "time"
    TIME_RANGE = "time_range"

    GUEST_COUNT = "guest_count"
    BUDGET = "budget"
    BUDGET_RANGE = "budget_range"
    PRICE_LIMIT = "price_limit"

    EMAIL = "email"
    PHONE = "phone"
    PERSON_NAME = "person_name"
    AGE = "age"

    EVENT_TYPE = "event_type"
    OCCASION = "occasion"
    SERVICE_TYPE = "service_type"
    RESERVATION_REFERENCE = "reservation_reference"

    MENU_CATEGORY = "menu_category"
    MENU_ITEM = "menu_item"
    DIETARY_REQUIREMENT = "dietary_requirement"
    ALLERGEN = "allergen"
    BEVERAGE_TYPE = "beverage_type"

    SPORTS_LEAGUE = "sports_league"
    SPORTS_TEAM = "sports_team"
    SPORTS_TYPE = "sports_type"

    LOCATION = "location"
    PAGE_CATEGORY = "page_category"
    BUSINESS_NAME = "business_name"

    UNKNOWN = "unknown"


class EntitySource(str, Enum):
    REGEX = "regex"
    DICTIONARY = "dictionary"
    CONTEXT = "context"
    INTENT = "intent"
    NORMALIZER = "normalizer"
    SPELLING = "spelling"


# ============================================================
# SECTION 03 - ENTITY DICTIONARIES
# ============================================================

EVENT_TYPE_TERMS: Final[dict[str, str]] = {
    "birthday": "birthday",
    "birthday party": "birthday",
    "anniversary": "anniversary",
    "engagement party": "engagement",
    "rehearsal dinner": "rehearsal_dinner",
    "bridal shower": "bridal_shower",
    "baby shower": "baby_shower",
    "graduation party": "graduation",
    "retirement party": "retirement",
    "corporate event": "corporate_event",
    "company party": "corporate_event",
    "holiday party": "holiday_party",
    "fundraiser": "fundraiser",
    "memorial": "memorial",
    "celebration of life": "celebration_of_life",
    "wedding reception": "wedding_reception",
    "private event": "private_event",
    "private party": "private_event",
    "group dinner": "group_dinner",
}

SERVICE_TYPE_TERMS: Final[dict[str, str]] = {
    "takeout": "takeout",
    "take out": "takeout",
    "pickup": "pickup",
    "pick up": "pickup",
    "delivery": "delivery",
    "dine-in": "dine_in",
    "dine in": "dine_in",
    "reservation": "reservation",
    "private event": "private_event",
    "catering": "catering",
    "happy hour": "happy_hour",
    "live music": "live_music",
}

MENU_CATEGORY_TERMS: Final[dict[str, str]] = {
    "appetizer": "appetizers",
    "appetizers": "appetizers",
    "starter": "appetizers",
    "starters": "appetizers",
    "salad": "salads",
    "salads": "salads",
    "sandwich": "sandwiches",
    "sandwiches": "sandwiches",
    "burger": "burgers",
    "burgers": "burgers",
    "entree": "entrees",
    "entrees": "entrees",
    "dessert": "desserts",
    "desserts": "desserts",
    "beer": "beer",
    "cocktails": "cocktails",
    "cocktail": "cocktails",
    "wine": "wine",
    "drinks": "beverages",
    "beverages": "beverages",
    "kids menu": "kids_menu",
    "children's menu": "kids_menu",
}

MENU_ITEM_TERMS: Final[dict[str, str]] = {
    "wings": "wings",
    "burger": "burger",
    "cheeseburger": "cheeseburger",
    "pizza": "pizza",
    "fries": "fries",
    "nachos": "nachos",
    "mozzarella sticks": "mozzarella_sticks",
    "chicken sandwich": "chicken_sandwich",
    "steak": "steak",
    "salmon": "salmon",
    "fish and chips": "fish_and_chips",
    "mac and cheese": "mac_and_cheese",
    "caesar salad": "caesar_salad",
    "house salad": "house_salad",
}

DIETARY_TERMS: Final[dict[str, str]] = {
    "vegan": "vegan",
    "vegetarian": "vegetarian",
    "gluten-free": "gluten_free",
    "gluten free": "gluten_free",
    "dairy-free": "dairy_free",
    "dairy free": "dairy_free",
    "low carb": "low_carb",
    "keto": "keto",
    "kosher": "kosher",
    "halal": "halal",
    "nut-free": "nut_free",
    "nut free": "nut_free",
}

ALLERGEN_TERMS: Final[dict[str, str]] = {
    "peanut": "peanut",
    "peanuts": "peanut",
    "tree nut": "tree_nut",
    "tree nuts": "tree_nut",
    "shellfish": "shellfish",
    "fish": "fish",
    "dairy": "dairy",
    "milk": "dairy",
    "egg": "egg",
    "eggs": "egg",
    "soy": "soy",
    "wheat": "wheat",
    "gluten": "gluten",
    "sesame": "sesame",
}

BEVERAGE_TERMS: Final[dict[str, str]] = {
    "beer": "beer",
    "draft beer": "draft_beer",
    "craft beer": "craft_beer",
    "wine": "wine",
    "red wine": "red_wine",
    "white wine": "white_wine",
    "cocktail": "cocktail",
    "cocktails": "cocktail",
    "mocktail": "mocktail",
    "mocktails": "mocktail",
    "whiskey": "whiskey",
    "bourbon": "bourbon",
    "vodka": "vodka",
    "tequila": "tequila",
    "rum": "rum",
    "gin": "gin",
    "soda": "soda",
    "coffee": "coffee",
}

SPORTS_LEAGUES: Final[dict[str, str]] = {
    "nfl": "NFL",
    "nba": "NBA",
    "mlb": "MLB",
    "nhl": "NHL",
    "mls": "MLS",
    "ncaa": "NCAA",
    "ufc": "UFC",
    "fifa": "FIFA",
    "world cup": "FIFA World Cup",
}

SPORTS_TYPES: Final[dict[str, str]] = {
    "football": "football",
    "baseball": "baseball",
    "basketball": "basketball",
    "hockey": "hockey",
    "soccer": "soccer",
    "golf": "golf",
    "boxing": "boxing",
    "mma": "mma",
    "wrestling": "wrestling",
    "tennis": "tennis",
}

SPORTS_TEAMS: Final[dict[str, str]] = {
    "yankees": "New York Yankees",
    "mets": "New York Mets",
    "giants": "New York Giants",
    "jets": "New York Jets",
    "knicks": "New York Knicks",
    "nets": "Brooklyn Nets",
    "rangers": "New York Rangers",
    "islanders": "New York Islanders",
    "devils": "New Jersey Devils",
    "red bulls": "New York Red Bulls",
    "eagles": "Philadelphia Eagles",
    "phillies": "Philadelphia Phillies",
    "76ers": "Philadelphia 76ers",
    "sixers": "Philadelphia 76ers",
    "flyers": "Philadelphia Flyers",
}

LOCATION_TERMS: Final[dict[str, str]] = {
    "morristown": "Morristown",
    "new jersey": "New Jersey",
    "nj": "New Jersey",
    "horseshoe tavern": "Horseshoe Tavern",
    "the horseshoe tavern": "Horseshoe Tavern",
    "horseshoe": "Horseshoe Tavern",
}

BUSINESS_TERMS: Final[dict[str, str]] = {
    "horseshoe tavern": "Horseshoe Tavern",
    "the horseshoe tavern": "Horseshoe Tavern",
    "horseshoe": "Horseshoe Tavern",
}


# ============================================================
# SECTION 04 - DATA CLASSES
# ============================================================

@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    entity_type: EntityType
    original_value: str
    normalized_value: Any
    start_index: int
    end_index: int
    confidence: float
    source: EntitySource
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end_index - self.start_index

    def as_dict(self) -> dict[str, Any]:
        normalized = self.normalized_value

        if isinstance(normalized, (date, datetime, time)):
            normalized = normalized.isoformat()

        if isinstance(normalized, Decimal):
            normalized = str(normalized)

        return {
            "entity_type": self.entity_type.value,
            "original_value": self.original_value,
            "normalized_value": normalized,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "confidence": self.confidence,
            "source": self.source.value,
            "metadata": dict(self.metadata),
            "length": self.length,
        }


@dataclass(frozen=True, slots=True)
class EntityExtractionResult:
    original_text: str
    normalized_text: str
    corrected_text: str
    entities: tuple[ExtractedEntity, ...]
    reference_datetime: datetime
    intent: IntentName | None
    page_category: str | None

    def entities_of_type(
        self,
        entity_type: EntityType,
    ) -> tuple[ExtractedEntity, ...]:
        return tuple(
            entity
            for entity in self.entities
            if entity.entity_type == entity_type
        )

    def first(
        self,
        entity_type: EntityType,
    ) -> ExtractedEntity | None:
        values = self.entities_of_type(
            entity_type
        )

        return values[0] if values else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "corrected_text": self.corrected_text,
            "entities": [
                entity.as_dict()
                for entity in self.entities
            ],
            "reference_datetime": (
                self.reference_datetime.isoformat()
            ),
            "intent": (
                self.intent.value
                if self.intent is not None
                else None
            ),
            "page_category": self.page_category,
            "entity_count": len(self.entities),
        }


# ============================================================
# SECTION 05 - ENTITY EXTRACTOR
# ============================================================

class EntityExtractor:
    """
    Deterministic entity extractor for restaurant operations.
    """

    def __init__(
        self,
        *,
        event_type_terms: Mapping[str, str] | None = None,
        service_type_terms: Mapping[str, str] | None = None,
        menu_category_terms: Mapping[str, str] | None = None,
        menu_item_terms: Mapping[str, str] | None = None,
        dietary_terms: Mapping[str, str] | None = None,
        allergen_terms: Mapping[str, str] | None = None,
        beverage_terms: Mapping[str, str] | None = None,
        sports_leagues: Mapping[str, str] | None = None,
        sports_teams: Mapping[str, str] | None = None,
        sports_types: Mapping[str, str] | None = None,
        location_terms: Mapping[str, str] | None = None,
    ) -> None:
        self.event_type_terms = dict(
            event_type_terms or EVENT_TYPE_TERMS
        )

        self.service_type_terms = dict(
            service_type_terms or SERVICE_TYPE_TERMS
        )

        self.menu_category_terms = dict(
            menu_category_terms or MENU_CATEGORY_TERMS
        )

        self.menu_item_terms = dict(
            menu_item_terms or MENU_ITEM_TERMS
        )

        self.dietary_terms = dict(
            dietary_terms or DIETARY_TERMS
        )

        self.allergen_terms = dict(
            allergen_terms or ALLERGEN_TERMS
        )

        self.beverage_terms = dict(
            beverage_terms or BEVERAGE_TERMS
        )

        self.sports_leagues = dict(
            sports_leagues or SPORTS_LEAGUES
        )

        self.sports_teams = dict(
            sports_teams or SPORTS_TEAMS
        )

        self.sports_types = dict(
            sports_types or SPORTS_TYPES
        )

        self.location_terms = dict(
            location_terms or LOCATION_TERMS
        )

        self.business_terms = dict(
            BUSINESS_TERMS
        )

        self._dictionary_patterns = {
            EntityType.EVENT_TYPE: self._compile_dictionary(
                self.event_type_terms
            ),
            EntityType.SERVICE_TYPE: self._compile_dictionary(
                self.service_type_terms
            ),
            EntityType.MENU_CATEGORY: self._compile_dictionary(
                self.menu_category_terms
            ),
            EntityType.MENU_ITEM: self._compile_dictionary(
                self.menu_item_terms
            ),
            EntityType.DIETARY_REQUIREMENT: self._compile_dictionary(
                self.dietary_terms
            ),
            EntityType.ALLERGEN: self._compile_dictionary(
                self.allergen_terms
            ),
            EntityType.BEVERAGE_TYPE: self._compile_dictionary(
                self.beverage_terms
            ),
            EntityType.SPORTS_LEAGUE: self._compile_dictionary(
                self.sports_leagues
            ),
            EntityType.SPORTS_TEAM: self._compile_dictionary(
                self.sports_teams
            ),
            EntityType.SPORTS_TYPE: self._compile_dictionary(
                self.sports_types
            ),
            EntityType.LOCATION: self._compile_dictionary(
                self.location_terms
            ),
            EntityType.BUSINESS_NAME: self._compile_dictionary(
                self.business_terms
            ),
        }

    # ========================================================
    # SECTION 06 - PUBLIC EXTRACTION
    # ========================================================

    def extract(
        self,
        text: str,
        *,
        intent: IntentName | str | None = None,
        page_category: str | None = None,
        reference_datetime: datetime | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> EntityExtractionResult:
        normalization = normalize_text(text)
        spelling = correct_spelling(
            normalization.normalized_text
        )

        corrected_text = spelling.corrected_text

        resolved_intent = self._coerce_intent(
            intent
        )

        reference = (
            reference_datetime
            or datetime.now().astimezone()
        )

        entities: list[ExtractedEntity] = []

        self._extract_contact_entities(
            corrected_text,
            entities,
        )

        self._extract_money_entities(
            corrected_text,
            entities,
        )

        self._extract_guest_entities(
            corrected_text,
            entities,
        )

        self._extract_time_entities(
            corrected_text,
            entities,
        )

        self._extract_date_entities(
            corrected_text,
            reference,
            entities,
        )

        self._extract_reservation_references(
            corrected_text,
            entities,
        )

        self._extract_names_and_ages(
            corrected_text,
            entities,
        )

        self._extract_dictionary_entities(
            corrected_text,
            entities,
        )

        self._extract_context_entities(
            corrected_text,
            resolved_intent,
            page_category,
            context or {},
            entities,
        )

        entities = self._deduplicate_and_resolve_overlaps(
            entities
        )

        entities.sort(
            key=lambda entity: (
                entity.start_index,
                -entity.length,
                entity.entity_type.value,
            )
        )

        return EntityExtractionResult(
            original_text=text,
            normalized_text=(
                normalization.normalized_text
            ),
            corrected_text=corrected_text,
            entities=tuple(
                entities[:MAXIMUM_ENTITIES]
            ),
            reference_datetime=reference,
            intent=resolved_intent,
            page_category=page_category,
        )

    # ========================================================
    # SECTION 07 - CONTACT ENTITIES
    # ========================================================

    def _extract_contact_entities(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> None:
        for match in EMAIL_PATTERN.finditer(text):
            value = match.group(0)

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.EMAIL,
                    original_value=value,
                    normalized_value=value.casefold(),
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=1.0,
                    source=EntitySource.REGEX,
                )
            )

        for match in PHONE_PATTERN.finditer(text):
            value = match.group(0)
            digits = re.sub(
                r"\D",
                "",
                value,
            )

            if len(digits) == 10:
                normalized = (
                    f"+1{digits}"
                )
            elif len(digits) == 11 and digits.startswith("1"):
                normalized = (
                    f"+{digits}"
                )
            else:
                normalized = digits

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.PHONE,
                    original_value=value,
                    normalized_value=normalized,
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=1.0,
                    source=EntitySource.REGEX,
                )
            )

    # ========================================================
    # SECTION 08 - MONEY ENTITIES
    # ========================================================

    def _extract_money_entities(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> None:
        occupied: list[tuple[int, int]] = []

        for match in MONEY_RANGE_PATTERN.finditer(text):
            minimum = self._parse_money(
                match.group("minimum")
            )

            maximum = self._parse_money(
                match.group("maximum")
            )

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.BUDGET_RANGE,
                    original_value=match.group(0),
                    normalized_value={
                        "minimum": str(minimum),
                        "maximum": str(maximum),
                        "currency": "USD",
                    },
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.99,
                    source=EntitySource.REGEX,
                )
            )

            occupied.append(
                (
                    match.start(),
                    match.end(),
                )
            )

        for match in MONEY_PATTERN.finditer(text):
            if self._range_overlaps(
                match.start(),
                match.end(),
                occupied,
            ):
                continue

            amount = self._parse_money(
                match.group(0)
            )

            nearby = text[
                max(0, match.start() - 30):
                min(len(text), match.end() + 20)
            ].casefold()

            entity_type = (
                EntityType.BUDGET
                if any(
                    word in nearby
                    for word in (
                        "budget",
                        "spend",
                        "party",
                        "event",
                    )
                )
                else EntityType.PRICE_LIMIT
            )

            entities.append(
                ExtractedEntity(
                    entity_type=entity_type,
                    original_value=match.group(0),
                    normalized_value={
                        "amount": str(amount),
                        "currency": "USD",
                    },
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.95,
                    source=EntitySource.REGEX,
                )
            )

        for match in PLAIN_BUDGET_PATTERN.finditer(text):
            if self._range_overlaps(
                match.start(),
                match.end(),
                occupied,
            ):
                continue

            amount_text = match.group(
                "amount"
            )

            amount = self._parse_money(
                amount_text
            )

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.BUDGET,
                    original_value=match.group(0),
                    normalized_value={
                        "amount": str(amount),
                        "currency": "USD",
                    },
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.84,
                    source=EntitySource.REGEX,
                )
            )

    # ========================================================
    # SECTION 09 - GUEST COUNT ENTITIES
    # ========================================================

    def _extract_guest_entities(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> None:
        occupied: list[tuple[int, int]] = []

        for pattern in (
            GUEST_COUNT_PATTERN,
            PARTY_OF_PATTERN,
        ):
            for match in pattern.finditer(text):
                if self._range_overlaps(
                    match.start(),
                    match.end(),
                    occupied,
                ):
                    continue

                count = int(
                    match.group("count")
                )

                if count <= 0:
                    continue

                entities.append(
                    ExtractedEntity(
                        entity_type=EntityType.GUEST_COUNT,
                        original_value=match.group(0),
                        normalized_value=count,
                        start_index=match.start(),
                        end_index=match.end(),
                        confidence=0.98,
                        source=EntitySource.REGEX,
                    )
                )

                occupied.append(
                    (
                        match.start(),
                        match.end(),
                    )
                )

    # ========================================================
    # SECTION 10 - TIME ENTITIES
    # ========================================================

    def _extract_time_entities(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> None:
        occupied: list[tuple[int, int]] = []

        for match in TIME_RANGE_PATTERN.finditer(text):
            start_meridiem = (
                match.group("start_meridiem")
                or match.group("end_meridiem")
            )

            start_time = self._build_time(
                hour=int(
                    match.group("start_hour")
                ),
                minute=int(
                    match.group("start_minute")
                    or 0
                ),
                meridiem=start_meridiem,
            )

            end_time = self._build_time(
                hour=int(
                    match.group("end_hour")
                ),
                minute=int(
                    match.group("end_minute")
                    or 0
                ),
                meridiem=match.group(
                    "end_meridiem"
                ),
            )

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.TIME_RANGE,
                    original_value=match.group(0),
                    normalized_value={
                        "start": start_time.isoformat(
                            timespec="minutes"
                        ),
                        "end": end_time.isoformat(
                            timespec="minutes"
                        ),
                    },
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.99,
                    source=EntitySource.REGEX,
                )
            )

            occupied.append(
                (
                    match.start(),
                    match.end(),
                )
            )

        for match in TIME_PATTERN.finditer(text):
            if self._range_overlaps(
                match.start(),
                match.end(),
                occupied,
            ):
                continue

            normalized = self._build_time(
                hour=int(
                    match.group("hour")
                ),
                minute=int(
                    match.group("minute")
                    or 0
                ),
                meridiem=match.group(
                    "meridiem"
                ),
            )

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.TIME,
                    original_value=match.group(0),
                    normalized_value=normalized,
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.99,
                    source=EntitySource.REGEX,
                )
            )

        for match in TWENTY_FOUR_HOUR_TIME_PATTERN.finditer(text):
            if self._range_overlaps(
                match.start(),
                match.end(),
                occupied,
            ):
                continue

            normalized = time(
                hour=int(
                    match.group("hour")
                ),
                minute=int(
                    match.group("minute")
                ),
            )

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.TIME,
                    original_value=match.group(0),
                    normalized_value=normalized,
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.95,
                    source=EntitySource.REGEX,
                    metadata={
                        "format": "24_hour",
                    },
                )
            )

    # ========================================================
    # SECTION 11 - DATE ENTITIES
    # ========================================================

    def _extract_date_entities(
        self,
        text: str,
        reference: datetime,
        entities: list[ExtractedEntity],
    ) -> None:
        occupied: list[tuple[int, int]] = []

        for match in NUMERIC_DATE_PATTERN.finditer(text):
            month = int(
                match.group("month")
            )

            day = int(
                match.group("day")
            )

            year = self._resolve_year(
                match.group("year"),
                month,
                day,
                reference,
            )

            resolved = self._safe_date(
                year,
                month,
                day,
            )

            if resolved is None:
                continue

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.DATE,
                    original_value=match.group(0),
                    normalized_value=resolved,
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.98,
                    source=EntitySource.REGEX,
                )
            )

            occupied.append(
                (
                    match.start(),
                    match.end(),
                )
            )

        for match in MONTH_DATE_PATTERN.finditer(text):
            if self._range_overlaps(
                match.start(),
                match.end(),
                occupied,
            ):
                continue

            month = self._month_number(
                match.group("month")
            )

            day = int(
                match.group("day")
            )

            year = self._resolve_year(
                match.group("year"),
                month,
                day,
                reference,
            )

            resolved = self._safe_date(
                year,
                month,
                day,
            )

            if resolved is None:
                continue

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.DATE,
                    original_value=match.group(0),
                    normalized_value=resolved,
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.99,
                    source=EntitySource.REGEX,
                )
            )

            occupied.append(
                (
                    match.start(),
                    match.end(),
                )
            )

        for match in RELATIVE_DATE_PATTERN.finditer(text):
            phrase = match.group(0).casefold()

            resolved = self._resolve_relative_date(
                phrase,
                reference,
            )

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.RELATIVE_DATE,
                    original_value=match.group(0),
                    normalized_value=resolved,
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.96,
                    source=EntitySource.REGEX,
                    metadata={
                        "relative_expression": phrase,
                    },
                )
            )

        for match in WEEKDAY_PATTERN.finditer(text):
            if self._range_overlaps(
                match.start(),
                match.end(),
                occupied,
            ):
                continue

            prefix = (
                match.group("prefix")
                or ""
            ).casefold()

            weekday_name = (
                match.group("weekday")
                .casefold()
            )

            resolved = self._resolve_weekday(
                weekday_name,
                prefix,
                reference,
            )

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.WEEKDAY,
                    original_value=match.group(0),
                    normalized_value=resolved,
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.94,
                    source=EntitySource.REGEX,
                    metadata={
                        "weekday": weekday_name,
                        "prefix": prefix or None,
                    },
                )
            )

    # ========================================================
    # SECTION 12 - RESERVATION, NAME, AND AGE
    # ========================================================

    def _extract_reservation_references(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> None:
        for match in RESERVATION_REFERENCE_PATTERN.finditer(text):
            reference = match.group(
                "reference"
            ).upper()

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.RESERVATION_REFERENCE,
                    original_value=match.group(0),
                    normalized_value=reference,
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.92,
                    source=EntitySource.REGEX,
                )
            )

    def _extract_names_and_ages(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> None:
        for match in NAME_PATTERN.finditer(text):
            name = " ".join(
                part.capitalize()
                for part in match.group(
                    "name"
                ).split()
            )

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.PERSON_NAME,
                    original_value=match.group("name"),
                    normalized_value=name,
                    start_index=match.start("name"),
                    end_index=match.end("name"),
                    confidence=0.86,
                    source=EntitySource.REGEX,
                )
            )

        for match in AGE_PATTERN.finditer(text):
            age = int(
                match.group("age")
            )

            if age > 130:
                continue

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.AGE,
                    original_value=match.group(0),
                    normalized_value=age,
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.96,
                    source=EntitySource.REGEX,
                )
            )

    # ========================================================
    # SECTION 13 - DICTIONARY ENTITIES
    # ========================================================

    def _extract_dictionary_entities(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> None:
        mapping_by_type: dict[
            EntityType,
            Mapping[str, str],
        ] = {
            EntityType.EVENT_TYPE: (
                self.event_type_terms
            ),
            EntityType.SERVICE_TYPE: (
                self.service_type_terms
            ),
            EntityType.MENU_CATEGORY: (
                self.menu_category_terms
            ),
            EntityType.MENU_ITEM: (
                self.menu_item_terms
            ),
            EntityType.DIETARY_REQUIREMENT: (
                self.dietary_terms
            ),
            EntityType.ALLERGEN: (
                self.allergen_terms
            ),
            EntityType.BEVERAGE_TYPE: (
                self.beverage_terms
            ),
            EntityType.SPORTS_LEAGUE: (
                self.sports_leagues
            ),
            EntityType.SPORTS_TEAM: (
                self.sports_teams
            ),
            EntityType.SPORTS_TYPE: (
                self.sports_types
            ),
            EntityType.LOCATION: (
                self.location_terms
            ),
            EntityType.BUSINESS_NAME: (
                self.business_terms
            ),
        }

        for entity_type, pattern in (
            self._dictionary_patterns.items()
        ):
            mapping = mapping_by_type[
                entity_type
            ]

            for match in pattern.finditer(text):
                original = match.group(0)
                normalized_key = (
                    original.casefold()
                )

                normalized = mapping.get(
                    normalized_key
                )

                if normalized is None:
                    continue

                confidence = (
                    0.98
                    if entity_type
                    in {
                        EntityType.BUSINESS_NAME,
                        EntityType.SPORTS_TEAM,
                        EntityType.SPORTS_LEAGUE,
                    }
                    else 0.92
                )

                entities.append(
                    ExtractedEntity(
                        entity_type=entity_type,
                        original_value=original,
                        normalized_value=normalized,
                        start_index=match.start(),
                        end_index=match.end(),
                        confidence=confidence,
                        source=EntitySource.DICTIONARY,
                    )
                )

    # ========================================================
    # SECTION 14 - CONTEXT ENTITIES
    # ========================================================

    def _extract_context_entities(
        self,
        text: str,
        intent: IntentName | None,
        page_category: str | None,
        context: Mapping[str, Any],
        entities: list[ExtractedEntity],
    ) -> None:
        if page_category:
            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.PAGE_CATEGORY,
                    original_value=page_category,
                    normalized_value=(
                        page_category
                        .strip()
                        .lower()
                        .replace("-", "_")
                    ),
                    start_index=0,
                    end_index=0,
                    confidence=0.75,
                    source=EntitySource.CONTEXT,
                    metadata={
                        "synthetic": True,
                    },
                )
            )

        if (
            intent
            in {
                IntentName.PRIVATE_EVENT,
                IntentName.PRIVATE_EVENT_PRICING,
                IntentName.PRIVATE_EVENT_AVAILABILITY,
                IntentName.PRIVATE_EVENT_CONTACT,
            }
            and not any(
                entity.entity_type
                == EntityType.SERVICE_TYPE
                and entity.normalized_value
                == "private_event"
                for entity in entities
            )
        ):
            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.SERVICE_TYPE,
                    original_value="private event",
                    normalized_value="private_event",
                    start_index=0,
                    end_index=0,
                    confidence=0.74,
                    source=EntitySource.INTENT,
                    metadata={
                        "synthetic": True,
                    },
                )
            )

        if (
            intent == IntentName.TAKEOUT
            and not any(
                entity.entity_type
                == EntityType.SERVICE_TYPE
                and entity.normalized_value
                == "takeout"
                for entity in entities
            )
        ):
            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.SERVICE_TYPE,
                    original_value="takeout",
                    normalized_value="takeout",
                    start_index=0,
                    end_index=0,
                    confidence=0.78,
                    source=EntitySource.INTENT,
                    metadata={
                        "synthetic": True,
                    },
                )
            )

        if context.get("business_name"):
            business_name = str(
                context["business_name"]
            ).strip()

            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.BUSINESS_NAME,
                    original_value=business_name,
                    normalized_value=business_name,
                    start_index=0,
                    end_index=0,
                    confidence=0.80,
                    source=EntitySource.CONTEXT,
                    metadata={
                        "synthetic": True,
                    },
                )
            )

    # ========================================================
    # SECTION 15 - OVERLAP RESOLUTION
    # ========================================================

    def _deduplicate_and_resolve_overlaps(
        self,
        entities: Sequence[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        unique: dict[
            tuple[
                EntityType,
                int,
                int,
                str,
            ],
            ExtractedEntity,
        ] = {}

        for entity in entities:
            key = (
                entity.entity_type,
                entity.start_index,
                entity.end_index,
                repr(
                    entity.normalized_value
                ),
            )

            existing = unique.get(key)

            if (
                existing is None
                or entity.confidence
                > existing.confidence
            ):
                unique[key] = entity

        candidates = sorted(
            unique.values(),
            key=lambda entity: (
                entity.confidence,
                entity.length,
                self._entity_priority(
                    entity.entity_type
                ),
            ),
            reverse=True,
        )

        accepted: list[
            ExtractedEntity
        ] = []

        for candidate in candidates:
            if (
                candidate.start_index
                == candidate.end_index
            ):
                accepted.append(candidate)
                continue

            conflicting = False

            for existing in accepted:
                if (
                    existing.start_index
                    == existing.end_index
                ):
                    continue

                if not self._overlaps(
                    candidate.start_index,
                    candidate.end_index,
                    existing.start_index,
                    existing.end_index,
                ):
                    continue

                if self._compatible_overlap(
                    candidate,
                    existing,
                ):
                    continue

                conflicting = True
                break

            if not conflicting:
                accepted.append(
                    candidate
                )

        return accepted

    @staticmethod
    def _compatible_overlap(
        left: ExtractedEntity,
        right: ExtractedEntity,
    ) -> bool:
        compatible_pairs = {
            frozenset(
                {
                    EntityType.EVENT_TYPE,
                    EntityType.SERVICE_TYPE,
                }
            ),
            frozenset(
                {
                    EntityType.MENU_CATEGORY,
                    EntityType.BEVERAGE_TYPE,
                }
            ),
            frozenset(
                {
                    EntityType.LOCATION,
                    EntityType.BUSINESS_NAME,
                }
            ),
            frozenset(
                {
                    EntityType.SPORTS_TYPE,
                    EntityType.SPORTS_TEAM,
                }
            ),
        }

        return (
            frozenset(
                {
                    left.entity_type,
                    right.entity_type,
                }
            )
            in compatible_pairs
        )

    @staticmethod
    def _entity_priority(
        entity_type: EntityType,
    ) -> int:
        priorities = {
            EntityType.EMAIL: 100,
            EntityType.PHONE: 100,
            EntityType.RESERVATION_REFERENCE: 98,
            EntityType.BUDGET_RANGE: 96,
            EntityType.TIME_RANGE: 96,
            EntityType.DATE: 94,
            EntityType.RELATIVE_DATE: 93,
            EntityType.WEEKDAY: 92,
            EntityType.TIME: 90,
            EntityType.GUEST_COUNT: 90,
            EntityType.BUDGET: 88,
            EntityType.SPORTS_TEAM: 86,
            EntityType.BUSINESS_NAME: 85,
            EntityType.EVENT_TYPE: 80,
            EntityType.MENU_ITEM: 78,
            EntityType.DIETARY_REQUIREMENT: 76,
            EntityType.ALLERGEN: 76,
        }

        return priorities.get(
            entity_type,
            50,
        )

    # ========================================================
    # SECTION 16 - DATE AND TIME HELPERS
    # ========================================================

    @staticmethod
    def _build_time(
        *,
        hour: int,
        minute: int,
        meridiem: str,
    ) -> time:
        normalized_meridiem = (
            meridiem
            .replace(".", "")
            .casefold()
        )

        if normalized_meridiem == "pm" and hour != 12:
            hour += 12

        if normalized_meridiem == "am" and hour == 12:
            hour = 0

        return time(
            hour=hour,
            minute=minute,
        )

    @staticmethod
    def _resolve_year(
        year_value: str | None,
        month: int,
        day: int,
        reference: datetime,
    ) -> int:
        if year_value:
            year = int(year_value)

            if year < 100:
                year += 2000

            return year

        candidate_year = (
            reference.year
        )

        candidate = EntityExtractor._safe_date(
            candidate_year,
            month,
            day,
        )

        if (
            candidate is not None
            and candidate
            < reference.date()
        ):
            return candidate_year + 1

        return candidate_year

    @staticmethod
    def _safe_date(
        year: int,
        month: int,
        day: int,
    ) -> date | None:
        try:
            return date(
                year,
                month,
                day,
            )
        except ValueError:
            return None

    @staticmethod
    def _month_number(
        value: str,
    ) -> int:
        normalized = value[:3].title()

        for month_number in range(
            1,
            13,
        ):
            if (
                calendar.month_abbr[
                    month_number
                ]
                == normalized
            ):
                return month_number

        raise ValueError(
            f"Unknown month: {value}"
        )

    @staticmethod
    def _resolve_relative_date(
        phrase: str,
        reference: datetime,
    ) -> date:
        current = reference.date()

        if phrase in {
            "today",
            "tonight",
        }:
            return current

        if phrase == "tomorrow":
            return current + timedelta(
                days=1
            )

        if phrase == "this weekend":
            days_until_saturday = (
                5 - current.weekday()
            ) % 7

            return current + timedelta(
                days=days_until_saturday
            )

        if phrase == "next weekend":
            days_until_next_saturday = (
                (5 - current.weekday()) % 7
                + 7
            )

            return current + timedelta(
                days=days_until_next_saturday
            )

        return current

    @staticmethod
    def _resolve_weekday(
        weekday_name: str,
        prefix: str,
        reference: datetime,
    ) -> date:
        weekday_lookup = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        target = weekday_lookup[
            weekday_name
        ]

        current = reference.date()
        delta = (
            target - current.weekday()
        ) % 7

        if prefix == "next":
            if delta == 0:
                delta = 7
            else:
                delta += 7

        return current + timedelta(
            days=delta
        )

    # ========================================================
    # SECTION 17 - GENERAL HELPERS
    # ========================================================

    @staticmethod
    def _parse_money(
        value: str,
    ) -> Decimal:
        cleaned = (
            value
            .replace("$", "")
            .replace(",", "")
            .strip()
        )

        try:
            return Decimal(cleaned)
        except InvalidOperation as exc:
            raise ValueError(
                f"Invalid monetary value: {value}"
            ) from exc

    @staticmethod
    def _compile_dictionary(
        mapping: Mapping[str, str],
    ) -> re.Pattern[str]:
        phrases = sorted(
            mapping.keys(),
            key=len,
            reverse=True,
        )

        if not phrases:
            return re.compile(
                r"(?!x)x"
            )

        pattern = "|".join(
            re.escape(phrase)
            .replace(
                r"\ ",
                r"\s+",
            )
            for phrase in phrases
        )

        return re.compile(
            rf"(?<!\w)(?:{pattern})(?!\w)",
            re.IGNORECASE,
        )

    @staticmethod
    def _coerce_intent(
        value: IntentName | str | None,
    ) -> IntentName | None:
        if value is None:
            return None

        if isinstance(value, IntentName):
            return value

        candidate = str(
            value
        ).strip().upper()

        try:
            return IntentName(
                candidate
            )
        except ValueError:
            return None

    @staticmethod
    def _overlaps(
        start_a: int,
        end_a: int,
        start_b: int,
        end_b: int,
    ) -> bool:
        return (
            start_a < end_b
            and start_b < end_a
        )

    @classmethod
    def _range_overlaps(
        cls,
        start: int,
        end: int,
        ranges: Iterable[
            tuple[int, int]
        ],
    ) -> bool:
        return any(
            cls._overlaps(
                start,
                end,
                range_start,
                range_end,
            )
            for range_start, range_end
            in ranges
        )


# ============================================================
# SECTION 18 - MODULE-LEVEL EXTRACTOR
# ============================================================

_default_extractor = EntityExtractor()


def extract_entities(
    text: str,
    *,
    intent: IntentName | str | None = None,
    page_category: str | None = None,
    reference_datetime: datetime | None = None,
    context: Mapping[str, Any] | None = None,
) -> EntityExtractionResult:
    """
    Extract entities through the shared application extractor.
    """

    return _default_extractor.extract(
        text,
        intent=intent,
        page_category=page_category,
        reference_datetime=reference_datetime,
        context=context,
    )


def entities_as_dicts(
    text: str,
    *,
    intent: IntentName | str | None = None,
    page_category: str | None = None,
    reference_datetime: datetime | None = None,
) -> list[dict[str, Any]]:
    """
    Return entity extraction results as JSON-safe dictionaries.
    """

    return [
        entity.as_dict()
        for entity in extract_entities(
            text,
            intent=intent,
            page_category=page_category,
            reference_datetime=reference_datetime,
        ).entities
    ]


# ============================================================
# SECTION 19 - VALIDATION
# ============================================================

def validate_entities_module() -> dict[str, Any]:
    reference = datetime(
        2026,
        7,
        22,
        12,
        0,
    ).astimezone()

    samples = {
        "private_event": (
            "I need a birthday party for 45 people "
            "on July 25, 2026 from 6pm to 10pm "
            "with a budget of $2,500 to $4,000."
        ),
        "contact": (
            "My email is test@example.com and my phone "
            "is 973-555-1234."
        ),
        "sports": (
            "Are you showing the Yankees game on Friday at 7:30pm?"
        ),
        "menu": (
            "Do you have gluten-free wings and craft beer?"
        ),
        "reservation": (
            "Cancel reservation number ABCD-1234."
        ),
    }

    private_result = extract_entities(
        samples["private_event"],
        intent=IntentName.PRIVATE_EVENT,
        page_category="private_events",
        reference_datetime=reference,
    )

    contact_result = extract_entities(
        samples["contact"],
        reference_datetime=reference,
    )

    sports_result = extract_entities(
        samples["sports"],
        intent=IntentName.SPORTS_VIEWING,
        reference_datetime=reference,
    )

    menu_result = extract_entities(
        samples["menu"],
        intent=IntentName.MENU_DIETARY,
        page_category="menu",
        reference_datetime=reference,
    )

    reservation_result = extract_entities(
        samples["reservation"],
        intent=IntentName.RESERVATION_CANCEL,
        reference_datetime=reference,
    )

    checks = {
        "event_type_detected": (
            private_result.first(
                EntityType.EVENT_TYPE
            )
            is not None
        ),
        "guest_count_detected": (
            private_result.first(
                EntityType.GUEST_COUNT
            ).normalized_value
            == 45
        ),
        "date_detected": (
            private_result.first(
                EntityType.DATE
            ).normalized_value
            == date(
                2026,
                7,
                25,
            )
        ),
        "time_range_detected": (
            private_result.first(
                EntityType.TIME_RANGE
            )
            is not None
        ),
        "budget_range_detected": (
            private_result.first(
                EntityType.BUDGET_RANGE
            )
            is not None
        ),
        "email_detected": (
            contact_result.first(
                EntityType.EMAIL
            ).normalized_value
            == "test@example.com"
        ),
        "phone_detected": (
            contact_result.first(
                EntityType.PHONE
            ).normalized_value
            == "+19735551234"
        ),
        "sports_team_detected": (
            sports_result.first(
                EntityType.SPORTS_TEAM
            ).normalized_value
            == "New York Yankees"
        ),
        "weekday_detected": (
            sports_result.first(
                EntityType.WEEKDAY
            )
            is not None
        ),
        "dietary_detected": (
            menu_result.first(
                EntityType.DIETARY_REQUIREMENT
            ).normalized_value
            == "gluten_free"
        ),
        "menu_item_detected": (
            menu_result.first(
                EntityType.MENU_ITEM
            ).normalized_value
            == "wings"
        ),
        "beverage_detected": (
            menu_result.first(
                EntityType.BEVERAGE_TYPE
            ).normalized_value
            == "craft_beer"
        ),
        "reservation_reference_detected": (
            reservation_result.first(
                EntityType.RESERVATION_REFERENCE
            ).normalized_value
            == "ABCD-1234"
        ),
        "offsets_valid": all(
            entity.start_index <= entity.end_index
            for result in (
                private_result,
                contact_result,
                sports_result,
                menu_result,
                reservation_result,
            )
            for entity in result.entities
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
        "private_event_result": (
            private_result.as_dict()
        ),
        "contact_result": (
            contact_result.as_dict()
        ),
        "sports_result": (
            sports_result.as_dict()
        ),
        "menu_result": (
            menu_result.as_dict()
        ),
        "reservation_result": (
            reservation_result.as_dict()
        ),
    }


# ============================================================
# SECTION 20 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import json

    report = validate_entities_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
