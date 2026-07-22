# ============================================================
# Exact file location: app/nlu/spelling.py
# Horseshoe Tavern AI
# Phase 1 Part 1.13
# Controlled spelling correction, fuzzy matching, protected
# terminology, learned mappings, ambiguity control, and review
# ============================================================

"""
Controlled spelling-correction engine for Horseshoe Tavern AI.

Responsibilities:

- Correct common customer misspellings
- Preserve URLs, emails, phone numbers, dates, times, currency,
  menu item names, venue names, and approved business terminology
- Load verified spelling mappings from the database
- Observe new spelling variants without automatically trusting them
- Rank candidate corrections using edit distance, token similarity,
  phonetic similarity, frequency, and business vocabulary
- Reject ambiguous corrections
- Prevent ordinary words from being changed into unrelated business terms
- Correct multiple misspellings in one message
- Return detailed correction metadata
- Support explicit review and controlled learning
- Never let public input directly overwrite verified mappings

This module receives normalized text from app/nlu/normalizer.py.
It does not perform intent classification.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Sequence

from sqlalchemy.orm import Session

from app.database.repositories import SpellingRepository
from app.logging_config import get_logger
from app.nlu.normalizer import (
    EMAIL_PATTERN,
    PHONE_PATTERN,
    TIME_PATTERN,
    URL_PATTERN,
    CURRENCY_PATTERN,
    DATE_PATTERN,
    NormalizationResult,
    TokenKind,
    normalize_text,
)


# ============================================================
# SECTION 01 - LOGGER AND CONSTANTS
# ============================================================

logger = get_logger(__name__)

MINIMUM_TOKEN_LENGTH: Final[int] = 2
MAXIMUM_TOKEN_LENGTH: Final[int] = 64

DEFAULT_MINIMUM_SCORE: Final[float] = 0.82
DEFAULT_AMBIGUITY_MARGIN: Final[float] = 0.08
DEFAULT_AUTOCORRECT_SCORE: Final[float] = 0.90
DEFAULT_MAXIMUM_EDIT_DISTANCE: Final[int] = 3
DEFAULT_MAXIMUM_CORRECTIONS: Final[int] = 12

WORD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Za-z][A-Za-z'-]*\b"
)

SAFE_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z'-]*$"
)

MULTISPACE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\s+"
)


# ============================================================
# SECTION 02 - VERIFIED BASE VOCABULARY
# ============================================================

VERIFIED_CORRECTIONS: Final[dict[str, str]] = {
    "adress": "address",
    "alchohol": "alcohol",
    "alcoholic": "alcoholic",
    "appetizerz": "appetizers",
    "appitizer": "appetizer",
    "appitizers": "appetizers",
    "availible": "available",
    "bannquet": "banquet",
    "beers": "beers",
    "berthday": "birthday",
    "birtday": "birthday",
    "birthay": "birthday",
    "bookng": "booking",
    "buisness": "business",
    "calender": "calendar",
    "cancelation": "cancellation",
    "catring": "catering",
    "caterng": "catering",
    "coctail": "cocktail",
    "coctails": "cocktails",
    "cocktial": "cocktail",
    "cocktials": "cocktails",
    "contactt": "contact",
    "desser": "dessert",
    "deserts": "desserts",
    "directions": "directions",
    "dinning": "dining",
    "drinkks": "drinks",
    "entertaiment": "entertainment",
    "entertainmant": "entertainment",
    "eventt": "event",
    "eventts": "events",
    "foood": "food",
    "fridayy": "friday",
    "glutenfree": "gluten-free",
    "hapy": "happy",
    "happie": "happy",
    "horsesho": "horseshoe",
    "horseshoee": "horseshoe",
    "horshoe": "horseshoe",
    "hourss": "hours",
    "kichen": "kitchen",
    "kitchn": "kitchen",
    "locaton": "location",
    "locationn": "location",
    "menue": "menu",
    "menuee": "menu",
    "mneu": "menu",
    "morriston": "Morristown",
    "morristownn": "Morristown",
    "musicc": "music",
    "openning": "opening",
    "parkng": "parking",
    "partys": "parties",
    "privat": "private",
    "prvate": "private",
    "reseration": "reservation",
    "reseravtion": "reservation",
    "reservaton": "reservation",
    "reservtion": "reservation",
    "restaraunt": "restaurant",
    "restarant": "restaurant",
    "resturant": "restaurant",
    "saterday": "saturday",
    "satuday": "saturday",
    "schedual": "schedule",
    "scedule": "schedule",
    "specials": "specials",
    "specails": "specials",
    "specals": "specials",
    "thursdayy": "thursday",
    "tomorow": "tomorrow",
    "tommorow": "tomorrow",
    "tuesdayy": "tuesday",
    "vegan": "vegan",
    "vegeterian": "vegetarian",
    "wendsday": "wednesday",
    "wednsday": "wednesday",
    "whats": "what is",
    "wheres": "where is",
}


BUSINESS_VOCABULARY: Final[tuple[str, ...]] = (
    "address",
    "alcohol",
    "appetizer",
    "appetizers",
    "available",
    "banquet",
    "bar",
    "beer",
    "beers",
    "birthday",
    "booking",
    "business",
    "calendar",
    "cancellation",
    "catering",
    "cocktail",
    "cocktails",
    "contact",
    "dessert",
    "desserts",
    "dining",
    "directions",
    "drinks",
    "entertainment",
    "event",
    "events",
    "food",
    "friday",
    "gluten-free",
    "happy",
    "hour",
    "hours",
    "horseshoe",
    "kitchen",
    "live",
    "location",
    "menu",
    "morristown",
    "music",
    "opening",
    "parking",
    "parties",
    "party",
    "private",
    "reservation",
    "restaurant",
    "saturday",
    "schedule",
    "special",
    "specials",
    "sunday",
    "takeout",
    "thursday",
    "today",
    "tomorrow",
    "tuesday",
    "vegan",
    "vegetarian",
    "wednesday",
)


PROTECTED_WORDS: Final[set[str]] = {
    "horseshoe",
    "tavern",
    "morristown",
    "nj",
    "new",
    "jersey",
    "facebook",
    "instagram",
    "google",
    "uber",
    "doordash",
    "grubhub",
    "yelp",
}


COMMON_WORDS: Final[set[str]] = {
    "a",
    "about",
    "after",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "before",
    "but",
    "by",
    "can",
    "close",
    "do",
    "does",
    "for",
    "from",
    "have",
    "hello",
    "help",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "near",
    "of",
    "on",
    "open",
    "or",
    "please",
    "show",
    "tell",
    "that",
    "the",
    "there",
    "they",
    "this",
    "time",
    "to",
    "tonight",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


# ============================================================
# SECTION 03 - ENUMERATIONS
# ============================================================

class CorrectionSource(str, Enum):
    VERIFIED_MAP = "verified_map"
    DATABASE_VERIFIED = "database_verified"
    FUZZY_VOCABULARY = "fuzzy_vocabulary"
    PHONETIC = "phonetic"
    MANUAL = "manual"
    NONE = "none"


class CorrectionDecision(str, Enum):
    APPLIED = "applied"
    SUGGESTED = "suggested"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    PROTECTED = "protected"
    UNCHANGED = "unchanged"


# ============================================================
# SECTION 04 - DATA CLASSES
# ============================================================

@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate: str
    total_score: float
    edit_score: float
    sequence_score: float
    phonetic_score: float
    frequency_score: float
    business_score: float
    edit_distance: int
    source: CorrectionSource

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "total_score": self.total_score,
            "edit_score": self.edit_score,
            "sequence_score": self.sequence_score,
            "phonetic_score": self.phonetic_score,
            "frequency_score": self.frequency_score,
            "business_score": self.business_score,
            "edit_distance": self.edit_distance,
            "source": self.source.value,
        }


@dataclass(frozen=True, slots=True)
class TokenCorrection:
    original: str
    corrected: str
    start_index: int
    end_index: int
    confidence: float
    decision: CorrectionDecision
    source: CorrectionSource
    candidates: tuple[CandidateScore, ...] = ()
    reason: str | None = None
    verified_mapping: bool = False

    @property
    def changed(self) -> bool:
        return (
            self.decision == CorrectionDecision.APPLIED
            and self.original != self.corrected
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "corrected": self.corrected,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "confidence": self.confidence,
            "decision": self.decision.value,
            "source": self.source.value,
            "candidates": [
                item.as_dict()
                for item in self.candidates
            ],
            "reason": self.reason,
            "verified_mapping": self.verified_mapping,
            "changed": self.changed,
        }


@dataclass(frozen=True, slots=True)
class SpellingResult:
    original_text: str
    normalized_text: str
    corrected_text: str
    corrections: tuple[TokenCorrection, ...]
    applied_count: int
    suggested_count: int
    ambiguous_count: int
    protected_count: int
    changed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "corrected_text": self.corrected_text,
            "corrections": [
                correction.as_dict()
                for correction in self.corrections
            ],
            "applied_count": self.applied_count,
            "suggested_count": self.suggested_count,
            "ambiguous_count": self.ambiguous_count,
            "protected_count": self.protected_count,
            "changed": self.changed,
        }


# ============================================================
# SECTION 05 - EDIT DISTANCE
# ============================================================

def levenshtein_distance(
    left: str,
    right: str,
) -> int:
    """
    Compute Levenshtein edit distance using bounded memory.
    """

    if left == right:
        return 0

    if not left:
        return len(right)

    if not right:
        return len(left)

    if len(left) < len(right):
        left, right = right, left

    previous_row = list(
        range(len(right) + 1)
    )

    for left_index, left_character in enumerate(
        left,
        start=1,
    ):
        current_row = [left_index]

        for right_index, right_character in enumerate(
            right,
            start=1,
        ):
            insertion = (
                current_row[right_index - 1]
                + 1
            )

            deletion = (
                previous_row[right_index]
                + 1
            )

            substitution = (
                previous_row[right_index - 1]
                + (
                    0
                    if left_character
                    == right_character
                    else 1
                )
            )

            current_row.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous_row = current_row

    return previous_row[-1]


# ============================================================
# SECTION 06 - PHONETIC SUPPORT
# ============================================================

def simple_phonetic_key(
    value: str,
) -> str:
    """
    Produce a deterministic English-oriented phonetic key.

    This is intentionally lightweight and dependency-free.
    """

    text = re.sub(
        r"[^a-z]",
        "",
        value.casefold(),
    )

    if not text:
        return ""

    replacements = (
        ("ph", "f"),
        ("ght", "t"),
        ("kn", "n"),
        ("wr", "r"),
        ("wh", "w"),
        ("ck", "k"),
        ("qu", "k"),
        ("x", "ks"),
        ("z", "s"),
    )

    for original, replacement in replacements:
        text = text.replace(
            original,
            replacement,
        )

    first_character = text[0]

    groups = {
        "bfpv": "1",
        "cgjkqsxz": "2",
        "dt": "3",
        "l": "4",
        "mn": "5",
        "r": "6",
    }

    encoded: list[str] = []
    previous_code = ""

    for character in text[1:]:
        code = ""

        for letters, mapped_code in groups.items():
            if character in letters:
                code = mapped_code
                break

        if code and code != previous_code:
            encoded.append(code)

        previous_code = code

    return (
        first_character.upper()
        + "".join(encoded)
    )[:6]


# ============================================================
# SECTION 07 - SPELLING ENGINE
# ============================================================

class SpellingEngine:
    """
    Controlled spelling correction with ambiguity safeguards.
    """

    def __init__(
        self,
        *,
        verified_corrections: Mapping[str, str] | None = None,
        vocabulary: Iterable[str] | None = None,
        protected_words: Iterable[str] | None = None,
        minimum_score: float = DEFAULT_MINIMUM_SCORE,
        autocorrect_score: float = DEFAULT_AUTOCORRECT_SCORE,
        ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
        maximum_edit_distance: int = DEFAULT_MAXIMUM_EDIT_DISTANCE,
        maximum_corrections: int = DEFAULT_MAXIMUM_CORRECTIONS,
    ) -> None:
        self.verified_corrections = {
            str(key).casefold(): str(value)
            for key, value in (
                verified_corrections
                or VERIFIED_CORRECTIONS
            ).items()
        }

        base_vocabulary = set(
            item.casefold()
            for item in (
                vocabulary
                or BUSINESS_VOCABULARY
            )
        )

        base_vocabulary.update(
            COMMON_WORDS
        )

        base_vocabulary.update(
            value.casefold()
            for value in self.verified_corrections.values()
            if " " not in value
        )

        self.vocabulary = tuple(
            sorted(base_vocabulary)
        )

        self.protected_words = {
            item.casefold()
            for item in (
                protected_words
                or PROTECTED_WORDS
            )
        }

        self.minimum_score = float(
            minimum_score
        )

        self.autocorrect_score = float(
            autocorrect_score
        )

        self.ambiguity_margin = float(
            ambiguity_margin
        )

        self.maximum_edit_distance = max(
            1,
            int(maximum_edit_distance),
        )

        self.maximum_corrections = max(
            1,
            int(maximum_corrections),
        )

        self.word_frequencies = Counter(
            self.vocabulary
        )

        self.phonetic_index: dict[
            str,
            set[str],
        ] = {}

        for word in self.vocabulary:
            key = simple_phonetic_key(word)

            if key:
                self.phonetic_index.setdefault(
                    key,
                    set(),
                ).add(word)

    # ========================================================
    # SECTION 08 - MAIN CORRECTION FLOW
    # ========================================================

    def correct(
        self,
        text: str,
        *,
        session: Session | None = None,
        autocorrect: bool = True,
        observe_unverified: bool = False,
    ) -> SpellingResult:
        normalization = normalize_text(text)

        database_mappings = (
            self._load_database_mappings(
                normalization,
                session,
            )
            if session is not None
            else {}
        )

        matches = list(
            WORD_PATTERN.finditer(
                normalization.normalized_text
            )
        )

        corrections: list[TokenCorrection] = []

        for match in matches:
            if (
                len(corrections)
                >= self.maximum_corrections
            ):
                break

            token = match.group(0)

            correction = self._correct_token(
                token,
                start_index=match.start(),
                end_index=match.end(),
                database_mappings=database_mappings,
                autocorrect=autocorrect,
            )

            if (
                correction.decision
                != CorrectionDecision.UNCHANGED
            ):
                corrections.append(
                    correction
                )

        corrected_text = self._apply_corrections(
            normalization.normalized_text,
            corrections,
        )

        if (
            session is not None
            and observe_unverified
        ):
            self._observe_candidates(
                session,
                corrections,
            )

        applied_count = sum(
            correction.decision
            == CorrectionDecision.APPLIED
            for correction in corrections
        )

        suggested_count = sum(
            correction.decision
            == CorrectionDecision.SUGGESTED
            for correction in corrections
        )

        ambiguous_count = sum(
            correction.decision
            == CorrectionDecision.AMBIGUOUS
            for correction in corrections
        )

        protected_count = sum(
            correction.decision
            == CorrectionDecision.PROTECTED
            for correction in corrections
        )

        return SpellingResult(
            original_text=text,
            normalized_text=(
                normalization.normalized_text
            ),
            corrected_text=corrected_text,
            corrections=tuple(corrections),
            applied_count=applied_count,
            suggested_count=suggested_count,
            ambiguous_count=ambiguous_count,
            protected_count=protected_count,
            changed=(
                corrected_text
                != normalization.normalized_text
            ),
        )

    # ========================================================
    # SECTION 09 - TOKEN CORRECTION
    # ========================================================

    def _correct_token(
        self,
        token: str,
        *,
        start_index: int,
        end_index: int,
        database_mappings: Mapping[str, str],
        autocorrect: bool,
    ) -> TokenCorrection:
        normalized = token.casefold()

        if self._is_protected_token(token):
            return TokenCorrection(
                original=token,
                corrected=token,
                start_index=start_index,
                end_index=end_index,
                confidence=1.0,
                decision=CorrectionDecision.PROTECTED,
                source=CorrectionSource.NONE,
                reason="Token is protected.",
            )

        if (
            len(normalized)
            < MINIMUM_TOKEN_LENGTH
            or len(normalized)
            > MAXIMUM_TOKEN_LENGTH
        ):
            return TokenCorrection(
                original=token,
                corrected=token,
                start_index=start_index,
                end_index=end_index,
                confidence=1.0,
                decision=CorrectionDecision.UNCHANGED,
                source=CorrectionSource.NONE,
                reason="Token length is outside correction range.",
            )

        if normalized in self.vocabulary:
            return TokenCorrection(
                original=token,
                corrected=token,
                start_index=start_index,
                end_index=end_index,
                confidence=1.0,
                decision=CorrectionDecision.UNCHANGED,
                source=CorrectionSource.NONE,
                reason="Token already exists in vocabulary.",
            )

        if normalized in database_mappings:
            corrected = self._preserve_case(
                token,
                database_mappings[normalized],
            )

            return TokenCorrection(
                original=token,
                corrected=corrected,
                start_index=start_index,
                end_index=end_index,
                confidence=1.0,
                decision=CorrectionDecision.APPLIED,
                source=CorrectionSource.DATABASE_VERIFIED,
                verified_mapping=True,
            )

        if normalized in self.verified_corrections:
            corrected = self._preserve_case(
                token,
                self.verified_corrections[
                    normalized
                ],
            )

            return TokenCorrection(
                original=token,
                corrected=corrected,
                start_index=start_index,
                end_index=end_index,
                confidence=1.0,
                decision=CorrectionDecision.APPLIED,
                source=CorrectionSource.VERIFIED_MAP,
                verified_mapping=True,
            )

        candidates = self._rank_candidates(
            normalized
        )

        if not candidates:
            return TokenCorrection(
                original=token,
                corrected=token,
                start_index=start_index,
                end_index=end_index,
                confidence=0.0,
                decision=CorrectionDecision.REJECTED,
                source=CorrectionSource.NONE,
                reason="No acceptable correction candidate.",
            )

        best = candidates[0]
        second = (
            candidates[1]
            if len(candidates) > 1
            else None
        )

        margin = (
            best.total_score
            - second.total_score
            if second is not None
            else 1.0
        )

        corrected = self._preserve_case(
            token,
            best.candidate,
        )

        if best.total_score < self.minimum_score:
            return TokenCorrection(
                original=token,
                corrected=corrected,
                start_index=start_index,
                end_index=end_index,
                confidence=best.total_score,
                decision=CorrectionDecision.REJECTED,
                source=best.source,
                candidates=tuple(
                    candidates[:5]
                ),
                reason=(
                    "Best candidate did not meet "
                    "the minimum score."
                ),
            )

        if (
            second is not None
            and margin < self.ambiguity_margin
        ):
            return TokenCorrection(
                original=token,
                corrected=corrected,
                start_index=start_index,
                end_index=end_index,
                confidence=best.total_score,
                decision=CorrectionDecision.AMBIGUOUS,
                source=best.source,
                candidates=tuple(
                    candidates[:5]
                ),
                reason=(
                    "Top candidates were too close "
                    "to safely autocorrect."
                ),
            )

        if (
            autocorrect
            and best.total_score
            >= self.autocorrect_score
        ):
            return TokenCorrection(
                original=token,
                corrected=corrected,
                start_index=start_index,
                end_index=end_index,
                confidence=best.total_score,
                decision=CorrectionDecision.APPLIED,
                source=best.source,
                candidates=tuple(
                    candidates[:5]
                ),
            )

        return TokenCorrection(
            original=token,
            corrected=corrected,
            start_index=start_index,
            end_index=end_index,
            confidence=best.total_score,
            decision=CorrectionDecision.SUGGESTED,
            source=best.source,
            candidates=tuple(
                candidates[:5]
            ),
            reason=(
                "Candidate is plausible but below "
                "the automatic correction threshold."
            ),
        )

    # ========================================================
    # SECTION 10 - CANDIDATE RANKING
    # ========================================================

    def _rank_candidates(
        self,
        token: str,
    ) -> list[CandidateScore]:
        candidate_pool = set(
            self._candidate_pool(token)
        )

        scores: list[CandidateScore] = []

        token_phonetic = simple_phonetic_key(
            token
        )

        for candidate in candidate_pool:
            if candidate == token:
                continue

            distance = levenshtein_distance(
                token,
                candidate,
            )

            allowed_distance = self._allowed_distance(
                token,
                candidate,
            )

            if distance > allowed_distance:
                continue

            longest = max(
                len(token),
                len(candidate),
                1,
            )

            edit_score = max(
                0.0,
                1.0 - (
                    distance
                    / longest
                ),
            )

            sequence_score = SequenceMatcher(
                None,
                token,
                candidate,
            ).ratio()

            candidate_phonetic = (
                simple_phonetic_key(
                    candidate
                )
            )

            phonetic_score = (
                1.0
                if (
                    token_phonetic
                    and candidate_phonetic
                    and token_phonetic
                    == candidate_phonetic
                )
                else 0.0
            )

            frequency_score = min(
                1.0,
                math.log1p(
                    self.word_frequencies[
                        candidate
                    ]
                )
                / math.log(3),
            )

            business_score = (
                1.0
                if candidate
                in BUSINESS_VOCABULARY
                else 0.0
            )

            total_score = (
                edit_score * 0.42
                + sequence_score * 0.28
                + phonetic_score * 0.16
                + frequency_score * 0.04
                + business_score * 0.10
            )

            source = (
                CorrectionSource.PHONETIC
                if phonetic_score == 1.0
                else CorrectionSource.FUZZY_VOCABULARY
            )

            scores.append(
                CandidateScore(
                    candidate=candidate,
                    total_score=round(
                        total_score,
                        6,
                    ),
                    edit_score=round(
                        edit_score,
                        6,
                    ),
                    sequence_score=round(
                        sequence_score,
                        6,
                    ),
                    phonetic_score=round(
                        phonetic_score,
                        6,
                    ),
                    frequency_score=round(
                        frequency_score,
                        6,
                    ),
                    business_score=round(
                        business_score,
                        6,
                    ),
                    edit_distance=distance,
                    source=source,
                )
            )

        scores.sort(
            key=lambda item: (
                item.total_score,
                item.business_score,
                -item.edit_distance,
                item.candidate,
            ),
            reverse=True,
        )

        return scores

    def _candidate_pool(
        self,
        token: str,
    ) -> set[str]:
        first_character = token[:1]

        candidates = {
            candidate
            for candidate in self.vocabulary
            if (
                candidate[:1] == first_character
                or abs(
                    len(candidate)
                    - len(token)
                ) <= 1
            )
        }

        phonetic_key = simple_phonetic_key(
            token
        )

        candidates.update(
            self.phonetic_index.get(
                phonetic_key,
                set(),
            )
        )

        return candidates

    def _allowed_distance(
        self,
        token: str,
        candidate: str,
    ) -> int:
        maximum_length = max(
            len(token),
            len(candidate),
        )

        if maximum_length <= 4:
            return 1

        if maximum_length <= 7:
            return min(
                2,
                self.maximum_edit_distance,
            )

        return self.maximum_edit_distance

    # ========================================================
    # SECTION 11 - PROTECTION
    # ========================================================

    def _is_protected_token(
        self,
        token: str,
    ) -> bool:
        normalized = token.casefold()

        if normalized in self.protected_words:
            return True

        if URL_PATTERN.fullmatch(token):
            return True

        if EMAIL_PATTERN.fullmatch(token):
            return True

        if PHONE_PATTERN.fullmatch(token):
            return True

        if DATE_PATTERN.fullmatch(token):
            return True

        if TIME_PATTERN.fullmatch(token):
            return True

        if CURRENCY_PATTERN.fullmatch(token):
            return True

        if any(
            character.isdigit()
            for character in token
        ):
            return True

        if not SAFE_WORD_PATTERN.fullmatch(
            token
        ):
            return True

        return False

    # ========================================================
    # SECTION 12 - DATABASE MAPPINGS
    # ========================================================

    def _load_database_mappings(
        self,
        normalization: NormalizationResult,
        session: Session,
    ) -> dict[str, str]:
        repository = SpellingRepository(
            session
        )

        mappings: dict[str, str] = {}

        observed_words = {
            match.group(0).casefold()
            for match in WORD_PATTERN.finditer(
                normalization.normalized_text
            )
        }

        for word in observed_words:
            matches = repository.lookup(
                word,
                verified_only=True,
            )

            if not matches:
                continue

            best = matches[0]

            mappings[word] = (
                best.corrected_text
            )

        return mappings

    def _observe_candidates(
        self,
        session: Session,
        corrections: Sequence[TokenCorrection],
    ) -> None:
        repository = SpellingRepository(
            session
        )

        for correction in corrections:
            if correction.decision not in {
                CorrectionDecision.APPLIED,
                CorrectionDecision.SUGGESTED,
            }:
                continue

            if correction.verified_mapping:
                continue

            repository.observe(
                incorrect_text=correction.original,
                corrected_text=correction.corrected,
                confidence=Decimal(
                    str(
                        round(
                            correction.confidence,
                            6,
                        )
                    )
                ),
                source_kind="observed_candidate",
            )

    # ========================================================
    # SECTION 13 - APPLY CORRECTIONS
    # ========================================================

    @staticmethod
    def _apply_corrections(
        text: str,
        corrections: Sequence[TokenCorrection],
    ) -> str:
        applied = [
            correction
            for correction in corrections
            if correction.decision
            == CorrectionDecision.APPLIED
        ]

        if not applied:
            return text

        output = text

        for correction in sorted(
            applied,
            key=lambda item: item.start_index,
            reverse=True,
        ):
            output = (
                output[: correction.start_index]
                + correction.corrected
                + output[correction.end_index :]
            )

        return MULTISPACE_PATTERN.sub(
            " ",
            output,
        ).strip()

    @staticmethod
    def _preserve_case(
        original: str,
        corrected: str,
    ) -> str:
        if not corrected:
            return corrected

        if original.isupper():
            return corrected.upper()

        if original[:1].isupper():
            return (
                corrected[:1].upper()
                + corrected[1:]
            )

        return corrected


# ============================================================
# SECTION 14 - MODULE-LEVEL ENGINE
# ============================================================

_default_engine = SpellingEngine()


def correct_spelling(
    text: str,
    *,
    session: Session | None = None,
    autocorrect: bool = True,
    observe_unverified: bool = False,
) -> SpellingResult:
    """
    Correct spelling through the shared application engine.
    """

    return _default_engine.correct(
        text,
        session=session,
        autocorrect=autocorrect,
        observe_unverified=observe_unverified,
    )


def corrected_string(
    text: str,
    *,
    session: Session | None = None,
) -> str:
    """
    Return only the corrected text.
    """

    return correct_spelling(
        text,
        session=session,
    ).corrected_text


# ============================================================
# SECTION 15 - VALIDATION
# ============================================================

def validate_spelling_module() -> dict[str, Any]:
    engine = SpellingEngine()

    cases = [
        {
            "input": "Where is the adress?",
            "expected": "Where is the address?",
        },
        {
            "input": "Can I see the menue?",
            "expected": "Can I see the menu?",
        },
        {
            "input": "Do you have privte event space?",
            "expected": "Do you have private event space?",
        },
        {
            "input": "What is the schedual for fridayy?",
            "expected": "What is the schedule for friday?",
        },
        {
            "input": "I need catring for a birtday",
            "expected": "I need catering for a birthday",
        },
        {
            "input": "Is there parkng near Horseshoe Tavern?",
            "expected": "Is there parking near Horseshoe Tavern?",
        },
    ]

    case_results: list[dict[str, Any]] = []

    for case in cases:
        result = engine.correct(
            case["input"]
        )

        case_results.append(
            {
                "input": case["input"],
                "expected": case["expected"],
                "actual": result.corrected_text,
                "passed": (
                    result.corrected_text
                    == case["expected"]
                ),
                "corrections": [
                    correction.as_dict()
                    for correction
                    in result.corrections
                ],
            }
        )

    protected_result = engine.correct(
        (
            "Email events@example.com or call "
            "973-555-1234 at 7:30pm"
        )
    )

    unknown_result = engine.correct(
        "quantumfluxinator"
    )

    checks = {
        "all_cases_passed": all(
            item["passed"]
            for item in case_results
        ),
        "email_preserved": (
            "events@example.com"
            in protected_result.corrected_text
        ),
        "phone_preserved": (
            "973-555-1234"
            in protected_result.corrected_text
        ),
        "time_preserved": (
            "7:30pm"
            in protected_result.corrected_text
        ),
        "unknown_not_forced": (
            unknown_result.corrected_text
            == "quantumfluxinator"
        ),
        "edit_distance_valid": (
            levenshtein_distance(
                "menu",
                "menue",
            )
            == 1
        ),
        "phonetic_key_available": bool(
            simple_phonetic_key(
                "restaurant"
            )
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
        "protected_result": (
            protected_result.as_dict()
        ),
        "unknown_result": (
            unknown_result.as_dict()
        ),
    }


# ============================================================
# SECTION 16 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import json

    report = validate_spelling_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
