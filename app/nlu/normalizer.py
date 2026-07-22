# ============================================================
# Exact file location: app/nlu/normalizer.py
# Horseshoe Tavern AI
# Phase 1 Part 1.12
# Unicode-safe, business-aware language normalization engine
# ============================================================

"""
Language normalization engine for Horseshoe Tavern AI.

This module prepares public chatbot input for downstream processing while
preserving the original user message.

Responsibilities:

- Unicode normalization
- Control-character removal
- Whitespace normalization
- Smart quote and dash normalization
- Repeated punctuation normalization
- Common contraction expansion
- Common texting shorthand expansion
- Restaurant and event terminology normalization
- Business-name protection
- URL, email, phone, date, time, currency, and identifier protection
- Token extraction with character offsets
- Case-preserving and lowercase normalized forms
- Detection of suspiciously repetitive input
- Safe message-length enforcement
- Deterministic normalization reports
- No direct modification of verified business facts

This module does not decide intent and does not make spelling corrections
based on fuzzy matching. Those responsibilities belong to later NLU stages.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Sequence

from app.config import get_settings


# ============================================================
# SECTION 01 - SETTINGS AND CONSTANTS
# ============================================================

settings = get_settings()

MAXIMUM_INPUT_CHARACTERS: Final[int] = (
    settings.maximum_message_characters
)

MAXIMUM_TOKEN_COUNT: Final[int] = 1000

PROTECTED_TOKEN_PREFIX: Final[str] = "__HS_PROTECTED_"
PROTECTED_TOKEN_SUFFIX: Final[str] = "__"

CONTROL_CHARACTER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)

MULTISPACE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[ \t\f\v]+"
)

MULTINEWLINE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\n{3,}"
)

REPEATED_PUNCTUATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"([!?.,])\1{2,}"
)

REPEATED_CHARACTER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)([a-z])\1{4,}"
)

URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bhttps?://[^\s<>'\"]+",
    re.IGNORECASE,
)

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

TIME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b"
    r"(?:0?[1-9]|1[0-2])"
    r"(?::[0-5]\d)?"
    r"\s?(?:a\.?m\.?|p\.?m\.?)"
    r"\b",
    re.IGNORECASE,
)

DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:"
    r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?"
    r"|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
    r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2}(?:,\s*\d{4})?"
    r")\b",
    re.IGNORECASE,
)

CURRENCY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\w)\$\s?\d+(?:,\d{3})*(?:\.\d{1,2})?(?!\w)"
)

ORDER_REFERENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:order|reservation|confirmation|booking)"
    r"[\s#:.-]*[A-Z0-9-]{4,}\b",
    re.IGNORECASE,
)

TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"""
    https?://[^\s<>'"]+
    |
    [A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}
    |
    (?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}
    |
    \$\s?\d+(?:,\d{3})*(?:\.\d{1,2})?
    |
    [A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*
    |
    [^\w\s]
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ============================================================
# SECTION 02 - CHARACTER TRANSLATION
# ============================================================

CHARACTER_TRANSLATIONS: Final[dict[int, str]] = {
    ord("\u2018"): "'",
    ord("\u2019"): "'",
    ord("\u201A"): "'",
    ord("\u201B"): "'",
    ord("\u2032"): "'",
    ord("\u201C"): '"',
    ord("\u201D"): '"',
    ord("\u201E"): '"',
    ord("\u201F"): '"',
    ord("\u2033"): '"',
    ord("\u2010"): "-",
    ord("\u2011"): "-",
    ord("\u2012"): "-",
    ord("\u2013"): "-",
    ord("\u2014"): "-",
    ord("\u2015"): "-",
    ord("\u2212"): "-",
    ord("\u2026"): "...",
    ord("\u00A0"): " ",
    ord("\u2007"): " ",
    ord("\u202F"): " ",
    ord("\u200B"): "",
    ord("\u200C"): "",
    ord("\u200D"): "",
    ord("\u2060"): "",
    ord("\uFEFF"): "",
}


# ============================================================
# SECTION 03 - CONTRACTION MAP
# ============================================================

CONTRACTION_MAP: Final[dict[str, str]] = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "must've": "must have",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there'd": "there would",
    "there'll": "there will",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'd": "what did",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "which's": "which is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "why'd": "why did",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "y'all": "you all",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}


# ============================================================
# SECTION 04 - SHORTHAND MAP
# ============================================================

SHORTHAND_MAP: Final[dict[str, str]] = {
    "2day": "today",
    "2moro": "tomorrow",
    "2morrow": "tomorrow",
    "4": "for",
    "abt": "about",
    "af": "very",
    "asap": "as soon as possible",
    "atm": "at the moment",
    "b": "be",
    "b4": "before",
    "bc": "because",
    "bday": "birthday",
    "bdayparty": "birthday party",
    "bday party": "birthday party",
    "brb": "be right back",
    "btw": "by the way",
    "c": "see",
    "cuz": "because",
    "dm": "direct message",
    "event space": "private event space",
    "fav": "favorite",
    "fave": "favorite",
    "fr": "for real",
    "fri": "friday",
    "gonna": "going to",
    "gotta": "have to",
    "hmu": "contact me",
    "hrs": "hours",
    "idk": "i do not know",
    "idc": "i do not care",
    "imo": "in my opinion",
    "info": "information",
    "k": "okay",
    "lmk": "let me know",
    "lol": "laughing",
    "menu pls": "menu please",
    "mon": "monday",
    "msg": "message",
    "n": "and",
    "n/a": "not available",
    "ngl": "not going to lie",
    "nite": "night",
    "noon": "12:00 pm",
    "num": "number",
    "party room": "private event space",
    "pls": "please",
    "plz": "please",
    "ppl": "people",
    "private party": "private event",
    "r": "are",
    "res": "reservation",
    "reserv": "reservation",
    "reservation pls": "reservation please",
    "sat": "saturday",
    "sun": "sunday",
    "tbh": "to be honest",
    "tho": "though",
    "thx": "thanks",
    "tmrw": "tomorrow",
    "tues": "tuesday",
    "u": "you",
    "ur": "your",
    "w": "with",
    "w/": "with",
    "w/o": "without",
    "wed": "wednesday",
    "wanna": "want to",
    "wknd": "weekend",
    "ya": "you",
    "yall": "you all",
    "yr": "your",
}


# ============================================================
# SECTION 05 - DOMAIN TERM MAP
# ============================================================

DOMAIN_TERM_MAP: Final[dict[str, str]] = {
    "horseshoe": "Horseshoe Tavern",
    "horse shoe": "Horseshoe Tavern",
    "horseshoe tavern": "Horseshoe Tavern",
    "the horseshoe": "Horseshoe Tavern",
    "the horseshoe tavern": "Horseshoe Tavern",
    "happy hr": "happy hour",
    "happyhrs": "happy hours",
    "happyhours": "happy hours",
    "private dining": "private event",
    "private room": "private event space",
    "banquet": "private event",
    "banquet room": "private event space",
    "function room": "private event space",
    "event room": "private event space",
    "party package": "private event package",
    "drink package": "bar package",
    "open bar": "open-bar package",
    "food package": "catering package",
    "live music": "live music",
    "sports game": "sports broadcast",
    "football game": "football broadcast",
    "baseball game": "baseball broadcast",
    "basketball game": "basketball broadcast",
    "hockey game": "hockey broadcast",
    "take out": "takeout",
    "take-out": "takeout",
    "pickup order": "takeout order",
    "to go": "takeout",
    "to-go": "takeout",
    "dine in": "dine-in",
    "dine-in": "dine-in",
}


# ============================================================
# SECTION 06 - PROTECTED BUSINESS TERMS
# ============================================================

DEFAULT_PROTECTED_TERMS: Final[tuple[str, ...]] = (
    "Horseshoe Tavern",
    "Horseshoe",
    "Morristown",
    "New Jersey",
    "NJ",
    "happy hour",
    "private event",
    "private events",
    "private event space",
    "takeout",
    "dine-in",
    "live music",
    "open bar",
)


# ============================================================
# SECTION 07 - ENUMERATIONS
# ============================================================

class TokenKind(str, Enum):
    WORD = "word"
    NUMBER = "number"
    URL = "url"
    EMAIL = "email"
    PHONE = "phone"
    CURRENCY = "currency"
    DATE = "date"
    TIME = "time"
    PUNCTUATION = "punctuation"
    PROTECTED = "protected"
    UNKNOWN = "unknown"


class NormalizationChangeType(str, Enum):
    UNICODE = "unicode"
    CONTROL_CHARACTER = "control_character"
    WHITESPACE = "whitespace"
    HTML_ENTITY = "html_entity"
    CONTRACTION = "contraction"
    SHORTHAND = "shorthand"
    DOMAIN_TERM = "domain_term"
    PUNCTUATION = "punctuation"
    REPETITION = "repetition"
    TRUNCATION = "truncation"


# ============================================================
# SECTION 08 - DATA STRUCTURES
# ============================================================

@dataclass(frozen=True, slots=True)
class ProtectedValue:
    placeholder: str
    original: str
    kind: TokenKind


@dataclass(frozen=True, slots=True)
class NormalizationChange:
    change_type: NormalizationChangeType
    original: str
    replacement: str
    start_index: int | None = None
    end_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "change_type": self.change_type.value,
            "original": self.original,
            "replacement": self.replacement,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class NormalizedToken:
    text: str
    normalized: str
    kind: TokenKind
    start_index: int
    end_index: int
    is_protected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "normalized": self.normalized,
            "kind": self.kind.value,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "is_protected": self.is_protected,
        }


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    original_text: str
    cleaned_text: str
    normalized_text: str
    lowercase_text: str
    tokens: tuple[NormalizedToken, ...]
    changes: tuple[NormalizationChange, ...]
    protected_values: tuple[ProtectedValue, ...]
    was_truncated: bool
    repeated_character_detected: bool
    repeated_punctuation_detected: bool
    suspicious_repetition_score: float

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def changed(self) -> bool:
        return self.original_text != self.normalized_text

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "normalized_text": self.normalized_text,
            "lowercase_text": self.lowercase_text,
            "tokens": [
                token.as_dict()
                for token in self.tokens
            ],
            "changes": [
                change.as_dict()
                for change in self.changes
            ],
            "protected_values": [
                {
                    "placeholder": item.placeholder,
                    "original": item.original,
                    "kind": item.kind.value,
                }
                for item in self.protected_values
            ],
            "was_truncated": self.was_truncated,
            "repeated_character_detected": (
                self.repeated_character_detected
            ),
            "repeated_punctuation_detected": (
                self.repeated_punctuation_detected
            ),
            "suspicious_repetition_score": (
                self.suspicious_repetition_score
            ),
            "token_count": self.token_count,
            "changed": self.changed,
        }


# ============================================================
# SECTION 09 - NORMALIZER
# ============================================================

class TextNormalizer:
    """
    Deterministic, business-aware text normalizer.
    """

    def __init__(
        self,
        *,
        contraction_map: Mapping[str, str] | None = None,
        shorthand_map: Mapping[str, str] | None = None,
        domain_term_map: Mapping[str, str] | None = None,
        protected_terms: Sequence[str] | None = None,
        maximum_characters: int = MAXIMUM_INPUT_CHARACTERS,
    ) -> None:
        self.contraction_map = {
            key.lower(): value
            for key, value in (
                contraction_map or CONTRACTION_MAP
            ).items()
        }

        self.shorthand_map = {
            key.lower(): value
            for key, value in (
                shorthand_map or SHORTHAND_MAP
            ).items()
        }

        self.domain_term_map = {
            key.lower(): value
            for key, value in (
                domain_term_map or DOMAIN_TERM_MAP
            ).items()
        }

        self.protected_terms = tuple(
            sorted(
                set(
                    protected_terms
                    or DEFAULT_PROTECTED_TERMS
                ),
                key=len,
                reverse=True,
            )
        )

        self.maximum_characters = max(
            1,
            int(maximum_characters),
        )

        self._contraction_pattern = self._build_phrase_pattern(
            self.contraction_map.keys()
        )

        self._shorthand_pattern = self._build_phrase_pattern(
            self.shorthand_map.keys()
        )

        self._domain_pattern = self._build_phrase_pattern(
            self.domain_term_map.keys()
        )

        self._protected_term_pattern = (
            self._build_phrase_pattern(
                self.protected_terms
            )
        )

    # ========================================================
    # SECTION 10 - PUBLIC NORMALIZATION
    # ========================================================

    def normalize(
        self,
        text: str,
        *,
        expand_contractions: bool = True,
        expand_shorthand: bool = True,
        normalize_domain_terms: bool = True,
        preserve_line_breaks: bool = False,
    ) -> NormalizationResult:
        if text is None:
            text = ""

        original_text = str(text)
        changes: list[NormalizationChange] = []

        truncated_text, was_truncated = self._truncate_input(
            original_text,
            changes,
        )

        cleaned_text = self._normalize_unicode(
            truncated_text,
            changes,
        )

        cleaned_text = self._decode_html_entities(
            cleaned_text,
            changes,
        )

        cleaned_text = self._remove_control_characters(
            cleaned_text,
            changes,
        )

        cleaned_text = cleaned_text.translate(
            CHARACTER_TRANSLATIONS
        )

        protected_text, protected_values = self._protect_values(
            cleaned_text
        )

        repeated_character_detected = bool(
            REPEATED_CHARACTER_PATTERN.search(
                protected_text
            )
        )

        repeated_punctuation_detected = bool(
            REPEATED_PUNCTUATION_PATTERN.search(
                protected_text
            )
        )

        protected_text = self._normalize_repetition(
            protected_text,
            changes,
        )

        if expand_contractions:
            protected_text = self._replace_phrases(
                protected_text,
                self._contraction_pattern,
                self.contraction_map,
                NormalizationChangeType.CONTRACTION,
                changes,
            )

        if expand_shorthand:
            protected_text = self._replace_phrases(
                protected_text,
                self._shorthand_pattern,
                self.shorthand_map,
                NormalizationChangeType.SHORTHAND,
                changes,
            )

        if normalize_domain_terms:
            protected_text = self._replace_phrases(
                protected_text,
                self._domain_pattern,
                self.domain_term_map,
                NormalizationChangeType.DOMAIN_TERM,
                changes,
            )

        protected_text = self._normalize_whitespace(
            protected_text,
            preserve_line_breaks=preserve_line_breaks,
            changes=changes,
        )

        restored_text = self._restore_values(
            protected_text,
            protected_values,
        )

        restored_text = self._normalize_spacing_around_punctuation(
            restored_text
        )

        restored_text = restored_text.strip()

        tokens = tuple(
            self._tokenize(restored_text)
        )

        suspicious_score = self._calculate_repetition_score(
            original_text,
            repeated_character_detected=(
                repeated_character_detected
            ),
            repeated_punctuation_detected=(
                repeated_punctuation_detected
            ),
        )

        return NormalizationResult(
            original_text=original_text,
            cleaned_text=cleaned_text,
            normalized_text=restored_text,
            lowercase_text=restored_text.casefold(),
            tokens=tokens,
            changes=tuple(changes),
            protected_values=tuple(protected_values),
            was_truncated=was_truncated,
            repeated_character_detected=(
                repeated_character_detected
            ),
            repeated_punctuation_detected=(
                repeated_punctuation_detected
            ),
            suspicious_repetition_score=suspicious_score,
        )

    # ========================================================
    # SECTION 11 - TRUNCATION
    # ========================================================

    def _truncate_input(
        self,
        text: str,
        changes: list[NormalizationChange],
    ) -> tuple[str, bool]:
        if len(text) <= self.maximum_characters:
            return text, False

        truncated = text[: self.maximum_characters]

        changes.append(
            NormalizationChange(
                change_type=(
                    NormalizationChangeType.TRUNCATION
                ),
                original=text,
                replacement=truncated,
                start_index=self.maximum_characters,
                end_index=len(text),
                metadata={
                    "maximum_characters": (
                        self.maximum_characters
                    ),
                    "removed_characters": (
                        len(text)
                        - self.maximum_characters
                    ),
                },
            )
        )

        return truncated, True

    # ========================================================
    # SECTION 12 - UNICODE AND HTML
    # ========================================================

    def _normalize_unicode(
        self,
        text: str,
        changes: list[NormalizationChange],
    ) -> str:
        normalized = unicodedata.normalize(
            "NFKC",
            text,
        )

        if normalized != text:
            changes.append(
                NormalizationChange(
                    change_type=(
                        NormalizationChangeType.UNICODE
                    ),
                    original=text,
                    replacement=normalized,
                )
            )

        return normalized

    def _decode_html_entities(
        self,
        text: str,
        changes: list[NormalizationChange],
    ) -> str:
        decoded = html.unescape(text)

        if decoded != text:
            changes.append(
                NormalizationChange(
                    change_type=(
                        NormalizationChangeType.HTML_ENTITY
                    ),
                    original=text,
                    replacement=decoded,
                )
            )

        return decoded

    def _remove_control_characters(
        self,
        text: str,
        changes: list[NormalizationChange],
    ) -> str:
        cleaned = CONTROL_CHARACTER_PATTERN.sub(
            "",
            text,
        )

        if cleaned != text:
            changes.append(
                NormalizationChange(
                    change_type=(
                        NormalizationChangeType.CONTROL_CHARACTER
                    ),
                    original=text,
                    replacement=cleaned,
                )
            )

        return cleaned

    # ========================================================
    # SECTION 13 - VALUE PROTECTION
    # ========================================================

    def _protect_values(
        self,
        text: str,
    ) -> tuple[str, list[ProtectedValue]]:
        protected_values: list[ProtectedValue] = []
        protected_text = text

        patterns: tuple[
            tuple[re.Pattern[str], TokenKind],
            ...
        ] = (
            (URL_PATTERN, TokenKind.URL),
            (EMAIL_PATTERN, TokenKind.EMAIL),
            (PHONE_PATTERN, TokenKind.PHONE),
            (ORDER_REFERENCE_PATTERN, TokenKind.PROTECTED),
            (DATE_PATTERN, TokenKind.DATE),
            (TIME_PATTERN, TokenKind.TIME),
            (CURRENCY_PATTERN, TokenKind.CURRENCY),
            (
                self._protected_term_pattern,
                TokenKind.PROTECTED,
            ),
        )

        for pattern, kind in patterns:
            protected_text = pattern.sub(
                lambda match: self._register_protected_value(
                    match.group(0),
                    kind,
                    protected_values,
                ),
                protected_text,
            )

        return protected_text, protected_values

    def _register_protected_value(
        self,
        original: str,
        kind: TokenKind,
        protected_values: list[ProtectedValue],
    ) -> str:
        placeholder = (
            f"{PROTECTED_TOKEN_PREFIX}"
            f"{len(protected_values)}"
            f"{PROTECTED_TOKEN_SUFFIX}"
        )

        protected_values.append(
            ProtectedValue(
                placeholder=placeholder,
                original=original,
                kind=kind,
            )
        )

        return placeholder

    def _restore_values(
        self,
        text: str,
        protected_values: Sequence[ProtectedValue],
    ) -> str:
        restored = text

        for value in protected_values:
            restored = restored.replace(
                value.placeholder,
                value.original,
            )

        return restored

    # ========================================================
    # SECTION 14 - PHRASE REPLACEMENT
    # ========================================================

    @staticmethod
    def _build_phrase_pattern(
        phrases: Iterable[str],
    ) -> re.Pattern[str]:
        escaped = [
            re.escape(str(phrase))
            for phrase in phrases
            if str(phrase).strip()
        ]

        if not escaped:
            return re.compile(r"(?!x)x")

        escaped.sort(
            key=len,
            reverse=True,
        )

        return re.compile(
            r"(?<![\w])(?:"
            + "|".join(escaped)
            + r")(?![\w])",
            re.IGNORECASE,
        )

    def _replace_phrases(
        self,
        text: str,
        pattern: re.Pattern[str],
        replacements: Mapping[str, str],
        change_type: NormalizationChangeType,
        changes: list[NormalizationChange],
    ) -> str:
        def replacement(match: re.Match[str]) -> str:
            original = match.group(0)
            normalized_key = original.casefold()
            replacement_value = replacements.get(
                normalized_key
            )

            if replacement_value is None:
                return original

            replacement_value = self._preserve_initial_case(
                original,
                replacement_value,
            )

            if original != replacement_value:
                changes.append(
                    NormalizationChange(
                        change_type=change_type,
                        original=original,
                        replacement=replacement_value,
                        start_index=match.start(),
                        end_index=match.end(),
                    )
                )

            return replacement_value

        return pattern.sub(
            replacement,
            text,
        )

    @staticmethod
    def _preserve_initial_case(
        original: str,
        replacement: str,
    ) -> str:
        if not replacement:
            return replacement

        if original.isupper():
            return replacement.upper()

        if original[:1].isupper():
            return (
                replacement[:1].upper()
                + replacement[1:]
            )

        return replacement

    # ========================================================
    # SECTION 15 - REPETITION
    # ========================================================

    def _normalize_repetition(
        self,
        text: str,
        changes: list[NormalizationChange],
    ) -> str:
        def character_replacement(
            match: re.Match[str],
        ) -> str:
            original = match.group(0)
            replacement = match.group(1) * 2

            changes.append(
                NormalizationChange(
                    change_type=(
                        NormalizationChangeType.REPETITION
                    ),
                    original=original,
                    replacement=replacement,
                    start_index=match.start(),
                    end_index=match.end(),
                    metadata={
                        "kind": "character",
                    },
                )
            )

            return replacement

        normalized = REPEATED_CHARACTER_PATTERN.sub(
            character_replacement,
            text,
        )

        def punctuation_replacement(
            match: re.Match[str],
        ) -> str:
            original = match.group(0)
            punctuation = match.group(1)

            replacement = (
                punctuation * 2
                if punctuation in {"!", "?"}
                else punctuation
            )

            changes.append(
                NormalizationChange(
                    change_type=(
                        NormalizationChangeType.PUNCTUATION
                    ),
                    original=original,
                    replacement=replacement,
                    start_index=match.start(),
                    end_index=match.end(),
                    metadata={
                        "kind": "repeated_punctuation",
                    },
                )
            )

            return replacement

        return REPEATED_PUNCTUATION_PATTERN.sub(
            punctuation_replacement,
            normalized,
        )

    @staticmethod
    def _calculate_repetition_score(
        text: str,
        *,
        repeated_character_detected: bool,
        repeated_punctuation_detected: bool,
    ) -> float:
        if not text:
            return 0.0

        score = 0.0

        if repeated_character_detected:
            score += 0.35

        if repeated_punctuation_detected:
            score += 0.25

        normalized = text.casefold()
        words = re.findall(
            r"\b\w+\b",
            normalized,
        )

        if words:
            unique_ratio = (
                len(set(words))
                / len(words)
            )

            if len(words) >= 8:
                score += (
                    max(
                        0.0,
                        1.0 - unique_ratio,
                    )
                    * 0.4
                )

        return round(
            min(score, 1.0),
            4,
        )

    # ========================================================
    # SECTION 16 - WHITESPACE AND PUNCTUATION
    # ========================================================

    def _normalize_whitespace(
        self,
        text: str,
        *,
        preserve_line_breaks: bool,
        changes: list[NormalizationChange],
    ) -> str:
        original = text

        if preserve_line_breaks:
            lines = [
                MULTISPACE_PATTERN.sub(
                    " ",
                    line,
                ).strip()
                for line in text.splitlines()
            ]

            normalized = "\n".join(lines)
            normalized = MULTINEWLINE_PATTERN.sub(
                "\n\n",
                normalized,
            )
        else:
            normalized = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

        if normalized != original:
            changes.append(
                NormalizationChange(
                    change_type=(
                        NormalizationChangeType.WHITESPACE
                    ),
                    original=original,
                    replacement=normalized,
                )
            )

        return normalized

    @staticmethod
    def _normalize_spacing_around_punctuation(
        text: str,
    ) -> str:
        normalized = re.sub(
            r"\s+([,.;:!?])",
            r"\1",
            text,
        )

        normalized = re.sub(
            r"([,.;:!?])(?=[A-Za-z])",
            r"\1 ",
            normalized,
        )

        normalized = re.sub(
            r"\(\s+",
            "(",
            normalized,
        )

        normalized = re.sub(
            r"\s+\)",
            ")",
            normalized,
        )

        return re.sub(
            r"[ \t]{2,}",
            " ",
            normalized,
        )

    # ========================================================
    # SECTION 17 - TOKENIZATION
    # ========================================================

    def _tokenize(
        self,
        text: str,
    ) -> list[NormalizedToken]:
        tokens: list[NormalizedToken] = []

        for match in TOKEN_PATTERN.finditer(text):
            token_text = match.group(0)

            tokens.append(
                NormalizedToken(
                    text=token_text,
                    normalized=token_text.casefold(),
                    kind=self._classify_token(
                        token_text
                    ),
                    start_index=match.start(),
                    end_index=match.end(),
                    is_protected=self._is_protected_token(
                        token_text
                    ),
                )
            )

            if len(tokens) >= MAXIMUM_TOKEN_COUNT:
                break

        return tokens

    def _classify_token(
        self,
        token: str,
    ) -> TokenKind:
        if URL_PATTERN.fullmatch(token):
            return TokenKind.URL

        if EMAIL_PATTERN.fullmatch(token):
            return TokenKind.EMAIL

        if PHONE_PATTERN.fullmatch(token):
            return TokenKind.PHONE

        if CURRENCY_PATTERN.fullmatch(token):
            return TokenKind.CURRENCY

        if DATE_PATTERN.fullmatch(token):
            return TokenKind.DATE

        if TIME_PATTERN.fullmatch(token):
            return TokenKind.TIME

        if token.isdigit():
            return TokenKind.NUMBER

        if re.fullmatch(
            r"[^\w\s]+",
            token,
        ):
            return TokenKind.PUNCTUATION

        if self._is_protected_token(token):
            return TokenKind.PROTECTED

        if re.search(
            r"[A-Za-z0-9]",
            token,
        ):
            return TokenKind.WORD

        return TokenKind.UNKNOWN

    def _is_protected_token(
        self,
        token: str,
    ) -> bool:
        token_folded = token.casefold()

        return any(
            token_folded
            == term.casefold()
            for term in self.protected_terms
        )


# ============================================================
# SECTION 18 - MODULE-LEVEL NORMALIZER
# ============================================================

_default_normalizer = TextNormalizer()


def normalize_text(
    text: str,
    *,
    expand_contractions: bool = True,
    expand_shorthand: bool = True,
    normalize_domain_terms: bool = True,
    preserve_line_breaks: bool = False,
) -> NormalizationResult:
    """
    Normalize text through the shared application normalizer.
    """

    return _default_normalizer.normalize(
        text,
        expand_contractions=expand_contractions,
        expand_shorthand=expand_shorthand,
        normalize_domain_terms=normalize_domain_terms,
        preserve_line_breaks=preserve_line_breaks,
    )


def normalized_string(
    text: str,
) -> str:
    """
    Return only the normalized text string.
    """

    return normalize_text(text).normalized_text


def normalized_lowercase_string(
    text: str,
) -> str:
    """
    Return the normalized text using Unicode-aware case folding.
    """

    return normalize_text(text).lowercase_text


# ============================================================
# SECTION 19 - VALIDATION
# ============================================================

def validate_normalizer_module() -> dict[str, Any]:
    cases = [
        {
            "input": "wat time r u open???",
            "expected": "wat time are you open??",
        },
        {
            "input": "I'm looking for a private party room",
            "expected": (
                "I am looking for a private event space"
            ),
        },
        {
            "input": "horseshoe menu pls",
            "expected": "Horseshoe Tavern menu please",
        },
        {
            "input": "Call me at 973-555-1234",
            "expected": "Call me at 973-555-1234",
        },
        {
            "input": (
                "Email test@example.com &amp; send info"
            ),
            "expected": (
                "Email test@example.com & send information"
            ),
        },
        {
            "input": "Party on 7/25 at 7:30pm",
            "expected": "Party on 7/25 at 7:30pm",
        },
    ]

    case_results: list[dict[str, Any]] = []

    for case in cases:
        result = normalize_text(
            case["input"]
        )

        case_results.append(
            {
                "input": case["input"],
                "expected": case["expected"],
                "actual": result.normalized_text,
                "passed": (
                    result.normalized_text
                    == case["expected"]
                ),
            }
        )

    protected_result = normalize_text(
        (
            "Visit https://example.com/menu "
            "or email events@example.com"
        )
    )

    repetition_result = normalize_text(
        "heyyyyyy!!!!!"
    )

    checks = {
        "all_cases_passed": all(
            item["passed"]
            for item in case_results
        ),
        "url_preserved": (
            "https://example.com/menu"
            in protected_result.normalized_text
        ),
        "email_preserved": (
            "events@example.com"
            in protected_result.normalized_text
        ),
        "repetition_detected": (
            repetition_result.repeated_character_detected
            and repetition_result.repeated_punctuation_detected
        ),
        "token_offsets_valid": all(
            token.start_index < token.end_index
            for token in protected_result.tokens
        ),
        "original_preserved": (
            protected_result.original_text
            == (
                "Visit https://example.com/menu "
                "or email events@example.com"
            )
        ),
        "lowercase_available": bool(
            protected_result.lowercase_text
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
        "repetition_result": (
            repetition_result.as_dict()
        ),
    }


# ============================================================
# SECTION 20 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import json

    report = validate_normalizer_module()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
