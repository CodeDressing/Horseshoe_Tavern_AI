# ============================================================
# Exact file location: app\analytics\conversation_analytics.py
# Horseshoe Tavern AI
# Enterprise Conversation Analytics Module
# Phase 1 Part 1.44
# ============================================================

"""
Enterprise-grade conversation analytics for Horseshoe Tavern AI.

This module is intentionally framework-agnostic and dependency-light. It can
consume dictionaries, dataclasses, ORM-derived mappings, API payloads, or
persisted chat-event records and convert them into reliable analytics outputs.

Primary capabilities
--------------------
- Privacy-aware event normalization
- Deterministic session and conversation metrics
- Intent-performance analytics
- Response-quality analytics
- Destination-routing and conversion analytics
- Funnel analysis
- Cohort analysis
- Time-series aggregation
- Rolling-window metrics
- Latency percentiles and service-level indicators
- Error-rate and fallback-rate monitoring
- Human-handoff analytics
- Private-event lead analytics
- Ordering-intent analytics
- Heuristic sentiment and urgency signals
- Data-quality diagnostics
- Baseline anomaly detection
- Export-safe JSON payloads
- Deterministic self-validation

Design principles
-----------------
1. Do not expose raw personally identifiable information in analytics exports.
2. Do not treat user messages as verified business facts.
3. Preserve source lineage and event timestamps.
4. Avoid hidden network calls and hidden persistence.
5. Make every output JSON-safe.
6. Keep calculations deterministic and testable.
7. Degrade safely when optional fields are absent.
8. Favor additive integration with existing Horseshoe Tavern AI services.

The module does not require machine-learning libraries. Its statistical and
behavioral features are suitable as inputs for later ML pipelines, while its
baseline anomaly detector remains deterministic and explainable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import statistics
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Final, Iterable, Mapping, MutableMapping, Sequence


# ============================================================
# SECTION 01 - MODULE METADATA AND CONSTANTS
# ============================================================

CONVERSATION_ANALYTICS_VERSION: Final[str] = "2.0.0"
CONVERSATION_ANALYTICS_PHASE: Final[str] = "Phase 1 Part 1.44"

DEFAULT_BUSINESS_SLUG: Final[str] = "horseshoe-tavern"
DEFAULT_TIMEZONE_NAME: Final[str] = "America/New_York"

MAXIMUM_TEXT_LENGTH: Final[int] = 4000
MAXIMUM_METADATA_DEPTH: Final[int] = 8
MAXIMUM_EXPORT_RECORDS: Final[int] = 100_000

DEFAULT_ROLLING_WINDOW: Final[int] = 7
DEFAULT_ANOMALY_Z_THRESHOLD: Final[float] = 3.0
DEFAULT_MINIMUM_BASELINE_POINTS: Final[int] = 5

EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
PHONE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}(?!\d)"
)
CARD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(?:\d[\s\-]?){13,19}(?!\d)"
)
URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"https?://[^\s<>\"]+",
    re.IGNORECASE,
)
WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")
TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9']+")

PII_REDACTION_TOKEN: Final[str] = "[REDACTED]"
UNKNOWN_VALUE: Final[str] = "unknown"

POSITIVE_TERMS: Final[frozenset[str]] = frozenset(
    {
        "awesome",
        "best",
        "excellent",
        "fantastic",
        "good",
        "great",
        "helpful",
        "love",
        "perfect",
        "thanks",
        "thank",
        "wonderful",
        "yes",
    }
)
NEGATIVE_TERMS: Final[frozenset[str]] = frozenset(
    {
        "angry",
        "awful",
        "bad",
        "broken",
        "complaint",
        "disappointed",
        "hate",
        "horrible",
        "issue",
        "problem",
        "refund",
        "terrible",
        "wrong",
    }
)
URGENCY_TERMS: Final[frozenset[str]] = frozenset(
    {
        "asap",
        "emergency",
        "immediately",
        "now",
        "right away",
        "soon",
        "today",
        "tonight",
        "urgent",
    }
)

ORDERING_INTENTS: Final[frozenset[str]] = frozenset(
    {
        "order",
        "ordering",
        "delivery",
        "pickup",
        "takeout",
        "take_out",
    }
)
PRIVATE_EVENT_INTENTS: Final[frozenset[str]] = frozenset(
    {
        "private_event",
        "private_event_availability",
        "private_event_contact",
        "private_event_pricing",
    }
)
CONTACT_INTENTS: Final[frozenset[str]] = frozenset(
    {
        "contact",
        "human_handoff",
        "complaint",
        "reservation",
        "reservation_change",
        "reservation_cancel",
    }
)
FALLBACK_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        "clarification",
        "safe_fallback",
        "error",
        "failed",
        "unknown",
    }
)


# ============================================================
# SECTION 02 - ENUMERATIONS
# ============================================================

class AnalyticsEventType(str, Enum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    CHAT_COMPLETED = "chat_completed"
    CHAT_FAILED = "chat_failed"
    ACTION_SHOWN = "action_shown"
    ACTION_CLICKED = "action_clicked"
    PAGE_VIEW = "page_view"
    SESSION_STARTED = "session_started"
    SESSION_RESTORED = "session_restored"
    HUMAN_HANDOFF = "human_handoff"
    PRIVATE_EVENT_LEAD = "private_event_lead"
    ORDERING_STARTED = "ordering_started"
    FEEDBACK = "feedback"
    CUSTOM = "custom"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class DataQualitySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AggregationGranularity(str, Enum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class FunnelStep(str, Enum):
    SESSION = "session"
    MESSAGE = "message"
    INTENT_MATCHED = "intent_matched"
    RESPONSE_GENERATED = "response_generated"
    ACTION_SHOWN = "action_shown"
    ACTION_CLICKED = "action_clicked"
    CONVERSION = "conversion"


class ConversionType(str, Enum):
    MENU_VIEW = "menu_view"
    SPECIALS_VIEW = "specials_view"
    EVENTS_VIEW = "events_view"
    GALLERY_VIEW = "gallery_view"
    PRIVATE_EVENT = "private_event"
    CONTACT = "contact"
    DELIVERY_ORDER = "delivery_order"
    PICKUP_ORDER = "pickup_order"
    SOCIAL_VIEW = "social_view"
    HUMAN_HANDOFF = "human_handoff"
    OTHER = "other"


# ============================================================
# SECTION 03 - CORE DATA CLASSES
# ============================================================

@dataclass(frozen=True, slots=True)
class DataQualityIssue:
    code: str
    message: str
    severity: DataQualitySeverity
    event_id: str | None = None
    field_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "event_id": self.event_id,
            "field_name": self.field_name,
            "metadata": make_json_safe(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ConversationAnalyticsEvent:
    event_id: str
    event_type: AnalyticsEventType
    occurred_at: datetime

    business_slug: str = DEFAULT_BUSINESS_SLUG
    session_id: str | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    request_id: str | None = None

    role: str | None = None
    message_text: str | None = None
    intent: str | None = None
    nlu_decision: str | None = None
    response_decision: str | None = None

    intent_confidence: float | None = None
    answer_confidence: float | None = None
    retrieval_confidence: float | None = None
    factuality_confidence: float | None = None

    latency_ms: float | None = None
    source_count: int = 0
    action_count: int = 0
    destination_match_count: int = 0
    verified_fact_count: int = 0
    unsupported_claim_count: int = 0
    stale_source_count: int = 0

    page_url: str | None = None
    page_category: str | None = None
    destination_key: str | None = None
    destination_url: str | None = None
    analytics_event: str | None = None

    human_handoff_required: bool = False
    private_event_draft_present: bool = False
    feedback_score: float | None = None
    success: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(
        self,
        *,
        include_message_text: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at.isoformat(),
            "business_slug": self.business_slug,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "request_id": self.request_id,
            "role": self.role,
            "intent": self.intent,
            "nlu_decision": self.nlu_decision,
            "response_decision": self.response_decision,
            "intent_confidence": self.intent_confidence,
            "answer_confidence": self.answer_confidence,
            "retrieval_confidence": self.retrieval_confidence,
            "factuality_confidence": self.factuality_confidence,
            "latency_ms": self.latency_ms,
            "source_count": self.source_count,
            "action_count": self.action_count,
            "destination_match_count": self.destination_match_count,
            "verified_fact_count": self.verified_fact_count,
            "unsupported_claim_count": self.unsupported_claim_count,
            "stale_source_count": self.stale_source_count,
            "page_url": self.page_url,
            "page_category": self.page_category,
            "destination_key": self.destination_key,
            "destination_url": self.destination_url,
            "analytics_event": self.analytics_event,
            "human_handoff_required": self.human_handoff_required,
            "private_event_draft_present": self.private_event_draft_present,
            "feedback_score": self.feedback_score,
            "success": self.success,
            "metadata": make_json_safe(self.metadata),
        }

        if include_message_text:
            payload["message_text"] = redact_sensitive_text(
                self.message_text
            )

        return payload


@dataclass(frozen=True, slots=True)
class IntentMetric:
    intent: str
    event_count: int
    conversation_count: int
    average_intent_confidence: float
    average_answer_confidence: float
    fallback_rate: float
    handoff_rate: float
    action_rate: float
    conversion_rate: float
    average_latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return make_json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class TimeSeriesPoint:
    period_start: datetime
    period_label: str
    event_count: int
    conversation_count: int
    success_rate: float
    fallback_rate: float
    action_click_rate: float
    average_latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["period_start"] = self.period_start.isoformat()
        return make_json_safe(payload)


@dataclass(frozen=True, slots=True)
class FunnelMetric:
    step: FunnelStep
    count: int
    step_conversion_rate: float
    overall_conversion_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step.value,
            "count": self.count,
            "step_conversion_rate": self.step_conversion_rate,
            "overall_conversion_rate": self.overall_conversion_rate,
        }


@dataclass(frozen=True, slots=True)
class DestinationMetric:
    destination_key: str
    shown_count: int
    click_count: int
    unique_conversations: int
    click_through_rate: float
    conversion_type: ConversionType

    def as_dict(self) -> dict[str, Any]:
        return {
            "destination_key": self.destination_key,
            "shown_count": self.shown_count,
            "click_count": self.click_count,
            "unique_conversations": self.unique_conversations,
            "click_through_rate": self.click_through_rate,
            "conversion_type": self.conversion_type.value,
        }


@dataclass(frozen=True, slots=True)
class ConversationSummaryMetric:
    conversation_id: str
    session_id: str | None
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    event_count: int
    user_message_count: int
    assistant_message_count: int
    unique_intents: tuple[str, ...]
    fallback_count: int
    action_click_count: int
    conversion_count: int
    human_handoff_required: bool
    private_event_lead: bool
    average_latency_ms: float
    sentiment: SentimentLabel

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["started_at"] = self.started_at.isoformat()
        payload["ended_at"] = self.ended_at.isoformat()
        payload["unique_intents"] = list(self.unique_intents)
        payload["sentiment"] = self.sentiment.value
        return make_json_safe(payload)


@dataclass(frozen=True, slots=True)
class AnomalySignal:
    metric_name: str
    period_label: str
    observed_value: float
    baseline_mean: float
    baseline_standard_deviation: float
    z_score: float
    direction: str
    severity: str
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return make_json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class ConversationAnalyticsReport:
    generated_at: datetime
    business_slug: str
    event_count: int
    session_count: int
    conversation_count: int
    user_message_count: int
    assistant_message_count: int

    success_rate: float
    fallback_rate: float
    handoff_rate: float
    private_event_lead_rate: float
    ordering_intent_rate: float
    action_show_rate: float
    action_click_rate: float
    destination_click_through_rate: float

    average_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    average_intent_confidence: float
    average_answer_confidence: float
    average_retrieval_confidence: float
    average_factuality_confidence: float

    verified_fact_count: int
    unsupported_claim_count: int
    stale_source_count: int

    sentiment_distribution: dict[str, int]
    top_intents: tuple[tuple[str, int], ...]
    top_pages: tuple[tuple[str, int], ...]

    intent_metrics: tuple[IntentMetric, ...]
    destination_metrics: tuple[DestinationMetric, ...]
    funnel_metrics: tuple[FunnelMetric, ...]
    time_series: tuple[TimeSeriesPoint, ...]
    conversation_summaries: tuple[ConversationSummaryMetric, ...]
    anomaly_signals: tuple[AnomalySignal, ...]
    data_quality_issues: tuple[DataQualityIssue, ...]

    metadata: dict[str, Any]
    service_version: str
    service_phase: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "business_slug": self.business_slug,
            "event_count": self.event_count,
            "session_count": self.session_count,
            "conversation_count": self.conversation_count,
            "user_message_count": self.user_message_count,
            "assistant_message_count": self.assistant_message_count,
            "success_rate": self.success_rate,
            "fallback_rate": self.fallback_rate,
            "handoff_rate": self.handoff_rate,
            "private_event_lead_rate": self.private_event_lead_rate,
            "ordering_intent_rate": self.ordering_intent_rate,
            "action_show_rate": self.action_show_rate,
            "action_click_rate": self.action_click_rate,
            "destination_click_through_rate": (
                self.destination_click_through_rate
            ),
            "average_latency_ms": self.average_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "p90_latency_ms": self.p90_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "average_intent_confidence": self.average_intent_confidence,
            "average_answer_confidence": self.average_answer_confidence,
            "average_retrieval_confidence": (
                self.average_retrieval_confidence
            ),
            "average_factuality_confidence": (
                self.average_factuality_confidence
            ),
            "verified_fact_count": self.verified_fact_count,
            "unsupported_claim_count": self.unsupported_claim_count,
            "stale_source_count": self.stale_source_count,
            "sentiment_distribution": copy.deepcopy(
                self.sentiment_distribution
            ),
            "top_intents": [
                {"intent": intent, "count": count}
                for intent, count in self.top_intents
            ],
            "top_pages": [
                {"page": page, "count": count}
                for page, count in self.top_pages
            ],
            "intent_metrics": [
                metric.as_dict()
                for metric in self.intent_metrics
            ],
            "destination_metrics": [
                metric.as_dict()
                for metric in self.destination_metrics
            ],
            "funnel_metrics": [
                metric.as_dict()
                for metric in self.funnel_metrics
            ],
            "time_series": [
                point.as_dict()
                for point in self.time_series
            ],
            "conversation_summaries": [
                summary.as_dict()
                for summary in self.conversation_summaries
            ],
            "anomaly_signals": [
                signal.as_dict()
                for signal in self.anomaly_signals
            ],
            "data_quality_issues": [
                issue.as_dict()
                for issue in self.data_quality_issues
            ],
            "metadata": make_json_safe(self.metadata),
            "service_version": self.service_version,
            "service_phase": self.service_phase,
        }


# ============================================================
# SECTION 04 - GENERAL UTILITY FUNCTIONS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_json_safe(
    value: Any,
    *,
    _depth: int = 0,
) -> Any:
    if _depth > MAXIMUM_METADATA_DEPTH:
        return str(value)

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if is_dataclass(value):
        return make_json_safe(
            asdict(value),
            _depth=_depth + 1,
        )

    if isinstance(value, Mapping):
        return {
            str(key): make_json_safe(
                item,
                _depth=_depth + 1,
            )
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            make_json_safe(
                item,
                _depth=_depth + 1,
            )
            for item in value
        ]

    return str(value)


def normalize_identifier(
    value: Any,
) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip()

    return normalized or None


def normalize_label(
    value: Any,
) -> str:
    normalized = str(value or "").strip().casefold()
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_") or UNKNOWN_VALUE


def coerce_bool(
    value: Any,
    *,
    default: bool = False,
) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value or "").strip().casefold()

    if normalized in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    return default


def coerce_int(
    value: Any,
    *,
    default: int = 0,
    minimum: int | None = None,
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        result = default

    if minimum is not None:
        result = max(result, minimum)

    return result


def coerce_float(
    value: Any,
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if not math.isfinite(result):
        return default

    if minimum is not None:
        result = max(result, minimum)

    if maximum is not None:
        result = min(result, maximum)

    return result


def coerce_datetime(
    value: Any,
    *,
    default: datetime | None = None,
) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(
            value,
            datetime.min.time(),
        )
    elif isinstance(value, (int, float)):
        result = datetime.fromtimestamp(
            float(value),
            tz=timezone.utc,
        )
    elif isinstance(value, str):
        candidate = value.strip()

        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"

        try:
            result = datetime.fromisoformat(candidate)
        except ValueError:
            result = default or utc_now()
    else:
        result = default or utc_now()

    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)

    return result.astimezone(timezone.utc)


def clamp_probability(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    result = coerce_float(
        value,
        default=default,
        minimum=0.0,
        maximum=1.0,
    )
    return float(result if result is not None else default)


def safe_divide(
    numerator: float | int,
    denominator: float | int,
    *,
    default: float = 0.0,
    precision: int = 6,
) -> float:
    if denominator == 0:
        return default

    return round(
        float(numerator) / float(denominator),
        precision,
    )


def safe_mean(
    values: Iterable[float | int | None],
    *,
    precision: int = 6,
) -> float:
    numeric = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(float(value))
    ]

    if not numeric:
        return 0.0

    return round(
        statistics.fmean(numeric),
        precision,
    )


def percentile(
    values: Iterable[float | int | None],
    percentile_value: float,
    *,
    precision: int = 3,
) -> float:
    numeric = sorted(
        float(value)
        for value in values
        if value is not None
        and math.isfinite(float(value))
    )

    if not numeric:
        return 0.0

    percentile_value = min(
        max(float(percentile_value), 0.0),
        100.0,
    )

    if len(numeric) == 1:
        return round(numeric[0], precision)

    rank = (
        percentile_value
        / 100.0
        * (len(numeric) - 1)
    )
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))

    if lower_index == upper_index:
        return round(
            numeric[lower_index],
            precision,
        )

    fraction = rank - lower_index

    interpolated = (
        numeric[lower_index]
        + (
            numeric[upper_index]
            - numeric[lower_index]
        )
        * fraction
    )

    return round(interpolated, precision)


def stable_hash(
    value: Any,
    *,
    namespace: str = "horseshoe-analytics",
    length: int = 24,
) -> str:
    digest = hashlib.sha256(
        f"{namespace}:{value}".encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()

    return digest[: max(8, int(length))]


def redact_sensitive_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    text = str(value).replace("\x00", " ")
    text = EMAIL_PATTERN.sub(
        PII_REDACTION_TOKEN,
        text,
    )
    text = PHONE_PATTERN.sub(
        PII_REDACTION_TOKEN,
        text,
    )
    text = CARD_PATTERN.sub(
        PII_REDACTION_TOKEN,
        text,
    )
    text = WHITESPACE_PATTERN.sub(
        " ",
        text,
    ).strip()

    return text[:MAXIMUM_TEXT_LENGTH]


def tokenize(
    value: Any,
) -> tuple[str, ...]:
    text = str(value or "").casefold()
    return tuple(
        TOKEN_PATTERN.findall(text)
    )


def classify_sentiment(
    value: Any,
) -> SentimentLabel:
    tokens = tokenize(value)

    if not tokens:
        return SentimentLabel.UNKNOWN

    positive = sum(
        1
        for token in tokens
        if token in POSITIVE_TERMS
    )
    negative = sum(
        1
        for token in tokens
        if token in NEGATIVE_TERMS
    )

    if positive and negative:
        return SentimentLabel.MIXED

    if positive:
        return SentimentLabel.POSITIVE

    if negative:
        return SentimentLabel.NEGATIVE

    return SentimentLabel.NEUTRAL


def urgency_score(
    value: Any,
) -> float:
    text = str(value or "").casefold()
    tokens = set(tokenize(text))

    matches = 0

    for term in URGENCY_TERMS:
        if " " in term:
            if term in text:
                matches += 1
        elif term in tokens:
            matches += 1

    return round(
        min(matches / 3.0, 1.0),
        6,
    )


# ============================================================
# SECTION 05 - EVENT NORMALIZATION
# ============================================================

def _mapping_from_record(
    record: Any,
) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)

    if is_dataclass(record):
        return asdict(record)

    model_dump = getattr(
        record,
        "model_dump",
        None,
    )

    if callable(model_dump):
        dumped = model_dump(mode="python")

        if isinstance(dumped, Mapping):
            return dict(dumped)

    as_dict_method = getattr(
        record,
        "as_dict",
        None,
    )

    if callable(as_dict_method):
        dumped = as_dict_method()

        if isinstance(dumped, Mapping):
            return dict(dumped)

    if hasattr(record, "__dict__"):
        return {
            key: value
            for key, value
            in vars(record).items()
            if not key.startswith("_")
        }

    raise TypeError(
        "Analytics records must be mappings, dataclasses, "
        "Pydantic-style models, or objects exposing as_dict()."
    )


def _first_present(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]

    return default


def _normalize_event_type(
    value: Any,
) -> AnalyticsEventType:
    normalized = normalize_label(value)

    aliases = {
        "message_user": AnalyticsEventType.USER_MESSAGE,
        "user": AnalyticsEventType.USER_MESSAGE,
        "assistant": AnalyticsEventType.ASSISTANT_MESSAGE,
        "message_assistant": AnalyticsEventType.ASSISTANT_MESSAGE,
        "chat_message_completed": AnalyticsEventType.CHAT_COMPLETED,
        "completed": AnalyticsEventType.CHAT_COMPLETED,
        "failed": AnalyticsEventType.CHAT_FAILED,
        "action_displayed": AnalyticsEventType.ACTION_SHOWN,
        "destination_shown": AnalyticsEventType.ACTION_SHOWN,
        "link_clicked": AnalyticsEventType.ACTION_CLICKED,
        "destination_clicked": AnalyticsEventType.ACTION_CLICKED,
        "click": AnalyticsEventType.ACTION_CLICKED,
        "private_event": AnalyticsEventType.PRIVATE_EVENT_LEAD,
        "ordering": AnalyticsEventType.ORDERING_STARTED,
    }

    if normalized in aliases:
        return aliases[normalized]

    try:
        return AnalyticsEventType(normalized)
    except ValueError:
        return AnalyticsEventType.CUSTOM


def normalize_analytics_event(
    record: Any,
    *,
    business_slug: str = DEFAULT_BUSINESS_SLUG,
    default_occurred_at: datetime | None = None,
) -> ConversationAnalyticsEvent:
    payload = _mapping_from_record(record)

    metadata = _first_present(
        payload,
        (
            "metadata",
            "metadata_json",
            "processing_metadata",
            "event_metadata",
        ),
        {},
    )

    if not isinstance(metadata, Mapping):
        metadata = {"raw_metadata": str(metadata)}

    event_id = normalize_identifier(
        _first_present(
            payload,
            ("event_id", "id", "analytics_id"),
        )
    ) or f"analytics_{uuid.uuid4().hex}"

    event_type = _normalize_event_type(
        _first_present(
            payload,
            (
                "event_type",
                "event_name",
                "type",
                "name",
            ),
            AnalyticsEventType.CUSTOM.value,
        )
    )

    occurred_at = coerce_datetime(
        _first_present(
            payload,
            (
                "occurred_at",
                "created_at",
                "timestamp",
                "event_time",
                "server_time",
            ),
            default_occurred_at or utc_now(),
        ),
        default=default_occurred_at or utc_now(),
    )

    role = normalize_identifier(
        _first_present(
            payload,
            ("role", "message_role"),
        )
    )

    message_text = _first_present(
        payload,
        (
            "message_text",
            "message",
            "text",
            "original_text",
        ),
    )

    event = ConversationAnalyticsEvent(
        event_id=event_id,
        event_type=event_type,
        occurred_at=occurred_at,
        business_slug=normalize_identifier(
            _first_present(
                payload,
                ("business_slug",),
                business_slug,
            )
        ) or business_slug,
        session_id=normalize_identifier(
            _first_present(
                payload,
                (
                    "session_id",
                    "browser_session_id",
                ),
            )
        ),
        conversation_id=normalize_identifier(
            _first_present(
                payload,
                ("conversation_id",),
            )
        ),
        message_id=normalize_identifier(
            _first_present(
                payload,
                ("message_id",),
            )
        ),
        request_id=normalize_identifier(
            _first_present(
                payload,
                ("request_id",),
            )
        ),
        role=normalize_label(role) if role else None,
        message_text=(
            redact_sensitive_text(message_text)
            if message_text is not None
            else None
        ),
        intent=normalize_label(
            _first_present(
                payload,
                (
                    "intent",
                    "detected_intent",
                    "primary_intent",
                ),
                UNKNOWN_VALUE,
            )
        ),
        nlu_decision=normalize_label(
            _first_present(
                payload,
                ("nlu_decision",),
                UNKNOWN_VALUE,
            )
        ),
        response_decision=normalize_label(
            _first_present(
                payload,
                (
                    "response_decision",
                    "grounded_response_decision",
                    "decision",
                ),
                UNKNOWN_VALUE,
            )
        ),
        intent_confidence=coerce_float(
            _first_present(
                payload,
                (
                    "intent_confidence",
                    "nlu_confidence",
                ),
            ),
            minimum=0.0,
            maximum=1.0,
        ),
        answer_confidence=coerce_float(
            _first_present(
                payload,
                (
                    "answer_confidence",
                    "response_confidence",
                ),
            ),
            minimum=0.0,
            maximum=1.0,
        ),
        retrieval_confidence=coerce_float(
            _first_present(
                payload,
                ("retrieval_confidence",),
            ),
            minimum=0.0,
            maximum=1.0,
        ),
        factuality_confidence=coerce_float(
            _first_present(
                payload,
                (
                    "factuality_confidence",
                    "factuality",
                ),
            ),
            minimum=0.0,
            maximum=1.0,
        ),
        latency_ms=coerce_float(
            _first_present(
                payload,
                (
                    "latency_ms",
                    "processing_time_ms",
                    "duration_ms",
                ),
            ),
            minimum=0.0,
        ),
        source_count=coerce_int(
            _first_present(
                payload,
                (
                    "source_count",
                    "response_source_count",
                ),
                0,
            ),
            minimum=0,
        ),
        action_count=coerce_int(
            _first_present(
                payload,
                (
                    "action_count",
                    "response_action_count",
                ),
                0,
            ),
            minimum=0,
        ),
        destination_match_count=coerce_int(
            _first_present(
                payload,
                ("destination_match_count",),
                0,
            ),
            minimum=0,
        ),
        verified_fact_count=coerce_int(
            _first_present(
                payload,
                ("verified_fact_count",),
                0,
            ),
            minimum=0,
        ),
        unsupported_claim_count=coerce_int(
            _first_present(
                payload,
                ("unsupported_claim_count",),
                0,
            ),
            minimum=0,
        ),
        stale_source_count=coerce_int(
            _first_present(
                payload,
                ("stale_source_count",),
                0,
            ),
            minimum=0,
        ),
        page_url=normalize_identifier(
            _first_present(
                payload,
                ("page_url",),
            )
        ),
        page_category=normalize_label(
            _first_present(
                payload,
                ("page_category",),
                UNKNOWN_VALUE,
            )
        ),
        destination_key=normalize_identifier(
            _first_present(
                payload,
                (
                    "destination_key",
                    "destination",
                ),
            )
        ),
        destination_url=normalize_identifier(
            _first_present(
                payload,
                ("destination_url", "url"),
            )
        ),
        analytics_event=normalize_identifier(
            _first_present(
                payload,
                ("analytics_event",),
            )
        ),
        human_handoff_required=coerce_bool(
            _first_present(
                payload,
                (
                    "human_handoff_required",
                    "handoff_required",
                ),
                False,
            )
        ),
        private_event_draft_present=coerce_bool(
            _first_present(
                payload,
                (
                    "private_event_draft_present",
                    "private_event_lead",
                ),
                False,
            )
        ),
        feedback_score=coerce_float(
            _first_present(
                payload,
                (
                    "feedback_score",
                    "rating",
                ),
            ),
            minimum=-1.0,
            maximum=5.0,
        ),
        success=coerce_bool(
            _first_present(
                payload,
                ("success", "completed"),
                True,
            ),
            default=True,
        ),
        metadata=make_json_safe(dict(metadata)),
    )

    return event


def normalize_analytics_events(
    records: Iterable[Any],
    *,
    business_slug: str = DEFAULT_BUSINESS_SLUG,
) -> tuple[ConversationAnalyticsEvent, ...]:
    normalized = [
        normalize_analytics_event(
            record,
            business_slug=business_slug,
        )
        for record in records
    ]

    normalized.sort(
        key=lambda event: (
            event.occurred_at,
            event.event_id,
        )
    )

    return tuple(normalized)


# ============================================================
# SECTION 06 - DATA QUALITY VALIDATION
# ============================================================

def validate_event(
    event: ConversationAnalyticsEvent,
) -> tuple[DataQualityIssue, ...]:
    issues: list[DataQualityIssue] = []

    if not event.event_id:
        issues.append(
            DataQualityIssue(
                code="missing_event_id",
                message="Event is missing an event identifier.",
                severity=DataQualitySeverity.ERROR,
                field_name="event_id",
            )
        )

    if not event.business_slug:
        issues.append(
            DataQualityIssue(
                code="missing_business_slug",
                message="Event is missing a business slug.",
                severity=DataQualitySeverity.ERROR,
                event_id=event.event_id,
                field_name="business_slug",
            )
        )

    if event.occurred_at > utc_now() + timedelta(minutes=5):
        issues.append(
            DataQualityIssue(
                code="future_timestamp",
                message="Event timestamp is unexpectedly in the future.",
                severity=DataQualitySeverity.WARNING,
                event_id=event.event_id,
                field_name="occurred_at",
            )
        )

    if (
        event.event_type
        in {
            AnalyticsEventType.USER_MESSAGE,
            AnalyticsEventType.ASSISTANT_MESSAGE,
            AnalyticsEventType.CHAT_COMPLETED,
        }
        and not event.conversation_id
    ):
        issues.append(
            DataQualityIssue(
                code="missing_conversation_id",
                message=(
                    "Conversation-oriented event is missing "
                    "conversation_id."
                ),
                severity=DataQualitySeverity.WARNING,
                event_id=event.event_id,
                field_name="conversation_id",
            )
        )

    if (
        event.intent_confidence is not None
        and not 0.0 <= event.intent_confidence <= 1.0
    ):
        issues.append(
            DataQualityIssue(
                code="invalid_intent_confidence",
                message="Intent confidence is outside [0, 1].",
                severity=DataQualitySeverity.ERROR,
                event_id=event.event_id,
                field_name="intent_confidence",
            )
        )

    if (
        event.answer_confidence is not None
        and not 0.0 <= event.answer_confidence <= 1.0
    ):
        issues.append(
            DataQualityIssue(
                code="invalid_answer_confidence",
                message="Answer confidence is outside [0, 1].",
                severity=DataQualitySeverity.ERROR,
                event_id=event.event_id,
                field_name="answer_confidence",
            )
        )

    if event.latency_ms is not None and event.latency_ms < 0:
        issues.append(
            DataQualityIssue(
                code="negative_latency",
                message="Latency cannot be negative.",
                severity=DataQualitySeverity.ERROR,
                event_id=event.event_id,
                field_name="latency_ms",
            )
        )

    if event.unsupported_claim_count > 0:
        issues.append(
            DataQualityIssue(
                code="unsupported_claims_detected",
                message=(
                    "Response metadata reports one or more "
                    "unsupported claims."
                ),
                severity=DataQualitySeverity.WARNING,
                event_id=event.event_id,
                field_name="unsupported_claim_count",
                metadata={
                    "count": event.unsupported_claim_count,
                },
            )
        )

    return tuple(issues)


def validate_events(
    events: Sequence[ConversationAnalyticsEvent],
) -> tuple[DataQualityIssue, ...]:
    issues: list[DataQualityIssue] = []

    seen_ids: set[str] = set()

    for event in events:
        issues.extend(
            validate_event(event)
        )

        if event.event_id in seen_ids:
            issues.append(
                DataQualityIssue(
                    code="duplicate_event_id",
                    message="Duplicate event identifier detected.",
                    severity=DataQualitySeverity.WARNING,
                    event_id=event.event_id,
                    field_name="event_id",
                )
            )

        seen_ids.add(event.event_id)

    return tuple(issues)


# ============================================================
# SECTION 07 - EVENT CLASSIFICATION HELPERS
# ============================================================

def is_fallback_event(
    event: ConversationAnalyticsEvent,
) -> bool:
    return (
        event.response_decision in FALLBACK_DECISIONS
        or event.nlu_decision in FALLBACK_DECISIONS
        or event.intent == UNKNOWN_VALUE
        or event.event_type == AnalyticsEventType.CHAT_FAILED
        or not event.success
    )


def is_action_shown(
    event: ConversationAnalyticsEvent,
) -> bool:
    return (
        event.event_type == AnalyticsEventType.ACTION_SHOWN
        or (
            event.event_type
            in {
                AnalyticsEventType.ASSISTANT_MESSAGE,
                AnalyticsEventType.CHAT_COMPLETED,
            }
            and event.action_count > 0
        )
    )


def is_action_clicked(
    event: ConversationAnalyticsEvent,
) -> bool:
    return (
        event.event_type == AnalyticsEventType.ACTION_CLICKED
        or bool(
            event.analytics_event
            and event.analytics_event.endswith(
                (
                    "_clicked",
                    "_opened",
                    "_started",
                )
            )
        )
    )


def conversion_type_for_event(
    event: ConversationAnalyticsEvent,
) -> ConversionType:
    key = normalize_label(
        event.destination_key
        or event.analytics_event
        or event.intent
    )

    if "menu" in key:
        return ConversionType.MENU_VIEW

    if "special" in key:
        return ConversionType.SPECIALS_VIEW

    if "event" in key and "private" not in key:
        return ConversionType.EVENTS_VIEW

    if "gallery" in key or "photo" in key:
        return ConversionType.GALLERY_VIEW

    if "private" in key:
        return ConversionType.PRIVATE_EVENT

    if "delivery" in key:
        return ConversionType.DELIVERY_ORDER

    if (
        "pickup" in key
        or "takeout" in key
        or "spoton" in key
    ):
        return ConversionType.PICKUP_ORDER

    if (
        "facebook" in key
        or "instagram" in key
        or "social" in key
    ):
        return ConversionType.SOCIAL_VIEW

    if "handoff" in key:
        return ConversionType.HUMAN_HANDOFF

    if (
        "contact" in key
        or "phone" in key
        or "email" in key
    ):
        return ConversionType.CONTACT

    return ConversionType.OTHER


def is_conversion_event(
    event: ConversationAnalyticsEvent,
) -> bool:
    return (
        event.event_type
        in {
            AnalyticsEventType.PRIVATE_EVENT_LEAD,
            AnalyticsEventType.ORDERING_STARTED,
            AnalyticsEventType.HUMAN_HANDOFF,
        }
        or is_action_clicked(event)
    )


# ============================================================
# SECTION 08 - CONVERSATION SUMMARIZATION
# ============================================================

def summarize_conversations(
    events: Sequence[ConversationAnalyticsEvent],
) -> tuple[ConversationSummaryMetric, ...]:
    grouped: dict[
        str,
        list[ConversationAnalyticsEvent],
    ] = defaultdict(list)

    for event in events:
        if event.conversation_id:
            grouped[event.conversation_id].append(
                event
            )

    summaries: list[ConversationSummaryMetric] = []

    for conversation_id, conversation_events in grouped.items():
        conversation_events.sort(
            key=lambda event: (
                event.occurred_at,
                event.event_id,
            )
        )

        started_at = conversation_events[0].occurred_at
        ended_at = conversation_events[-1].occurred_at

        user_messages = [
            event
            for event in conversation_events
            if (
                event.event_type
                == AnalyticsEventType.USER_MESSAGE
                or event.role == "user"
            )
        ]
        assistant_messages = [
            event
            for event in conversation_events
            if (
                event.event_type
                == AnalyticsEventType.ASSISTANT_MESSAGE
                or event.role == "assistant"
            )
        ]

        intents = tuple(
            dict.fromkeys(
                event.intent
                for event in conversation_events
                if event.intent
                and event.intent != UNKNOWN_VALUE
            )
        )

        sentiment_votes = Counter(
            classify_sentiment(event.message_text).value
            for event in user_messages
            if event.message_text
        )

        if sentiment_votes:
            sentiment = SentimentLabel(
                sentiment_votes.most_common(1)[0][0]
            )
        else:
            sentiment = SentimentLabel.UNKNOWN

        latencies = [
            event.latency_ms
            for event in conversation_events
            if event.latency_ms is not None
        ]

        summaries.append(
            ConversationSummaryMetric(
                conversation_id=conversation_id,
                session_id=next(
                    (
                        event.session_id
                        for event in conversation_events
                        if event.session_id
                    ),
                    None,
                ),
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=round(
                    max(
                        (
                            ended_at
                            - started_at
                        ).total_seconds(),
                        0.0,
                    ),
                    3,
                ),
                event_count=len(conversation_events),
                user_message_count=len(user_messages),
                assistant_message_count=len(
                    assistant_messages
                ),
                unique_intents=intents,
                fallback_count=sum(
                    1
                    for event in conversation_events
                    if is_fallback_event(event)
                ),
                action_click_count=sum(
                    1
                    for event in conversation_events
                    if is_action_clicked(event)
                ),
                conversion_count=sum(
                    1
                    for event in conversation_events
                    if is_conversion_event(event)
                ),
                human_handoff_required=any(
                    event.human_handoff_required
                    or event.event_type
                    == AnalyticsEventType.HUMAN_HANDOFF
                    for event in conversation_events
                ),
                private_event_lead=any(
                    event.private_event_draft_present
                    or event.event_type
                    == AnalyticsEventType.PRIVATE_EVENT_LEAD
                    for event in conversation_events
                ),
                average_latency_ms=safe_mean(
                    latencies,
                    precision=3,
                ),
                sentiment=sentiment,
            )
        )

    summaries.sort(
        key=lambda summary: (
            summary.started_at,
            summary.conversation_id,
        )
    )

    return tuple(summaries)


# ============================================================
# SECTION 09 - INTENT ANALYTICS
# ============================================================

def build_intent_metrics(
    events: Sequence[ConversationAnalyticsEvent],
) -> tuple[IntentMetric, ...]:
    grouped: dict[
        str,
        list[ConversationAnalyticsEvent],
    ] = defaultdict(list)

    for event in events:
        if event.intent:
            grouped[event.intent].append(event)

    metrics: list[IntentMetric] = []

    for intent, intent_events in grouped.items():
        conversation_ids = {
            event.conversation_id
            for event in intent_events
            if event.conversation_id
        }

        fallback_count = sum(
            1
            for event in intent_events
            if is_fallback_event(event)
        )

        handoff_count = sum(
            1
            for event in intent_events
            if event.human_handoff_required
            or event.event_type
            == AnalyticsEventType.HUMAN_HANDOFF
        )

        action_count = sum(
            1
            for event in intent_events
            if is_action_shown(event)
        )

        conversion_count = sum(
            1
            for event in intent_events
            if is_conversion_event(event)
        )

        metrics.append(
            IntentMetric(
                intent=intent,
                event_count=len(intent_events),
                conversation_count=len(
                    conversation_ids
                ),
                average_intent_confidence=safe_mean(
                    event.intent_confidence
                    for event in intent_events
                ),
                average_answer_confidence=safe_mean(
                    event.answer_confidence
                    for event in intent_events
                ),
                fallback_rate=safe_divide(
                    fallback_count,
                    len(intent_events),
                ),
                handoff_rate=safe_divide(
                    handoff_count,
                    len(intent_events),
                ),
                action_rate=safe_divide(
                    action_count,
                    len(intent_events),
                ),
                conversion_rate=safe_divide(
                    conversion_count,
                    len(intent_events),
                ),
                average_latency_ms=safe_mean(
                    event.latency_ms
                    for event in intent_events
                ),
            )
        )

    metrics.sort(
        key=lambda metric: (
            -metric.event_count,
            metric.intent,
        )
    )

    return tuple(metrics)


# ============================================================
# SECTION 10 - DESTINATION ANALYTICS
# ============================================================

def build_destination_metrics(
    events: Sequence[ConversationAnalyticsEvent],
) -> tuple[DestinationMetric, ...]:
    shown_counts: Counter[str] = Counter()
    click_counts: Counter[str] = Counter()
    conversations: dict[str, set[str]] = defaultdict(set)
    conversion_types: dict[str, ConversionType] = {}

    for event in events:
        key = normalize_label(
            event.destination_key
            or event.analytics_event
            or ""
        )

        if not key or key == UNKNOWN_VALUE:
            continue

        if is_action_shown(event):
            shown_counts[key] += 1

        if is_action_clicked(event):
            click_counts[key] += 1

        if event.conversation_id:
            conversations[key].add(
                event.conversation_id
            )

        conversion_types[key] = conversion_type_for_event(
            event
        )

    all_keys = set(shown_counts) | set(click_counts)

    metrics = [
        DestinationMetric(
            destination_key=key,
            shown_count=shown_counts[key],
            click_count=click_counts[key],
            unique_conversations=len(
                conversations[key]
            ),
            click_through_rate=safe_divide(
                click_counts[key],
                shown_counts[key],
            ),
            conversion_type=conversion_types.get(
                key,
                ConversionType.OTHER,
            ),
        )
        for key in all_keys
    ]

    metrics.sort(
        key=lambda metric: (
            -metric.click_count,
            -metric.shown_count,
            metric.destination_key,
        )
    )

    return tuple(metrics)


# ============================================================
# SECTION 11 - FUNNEL ANALYTICS
# ============================================================

def build_funnel_metrics(
    events: Sequence[ConversationAnalyticsEvent],
) -> tuple[FunnelMetric, ...]:
    session_ids = {
        event.session_id
        for event in events
        if event.session_id
    }

    message_conversations = {
        event.conversation_id
        for event in events
        if event.conversation_id
        and event.event_type
        in {
            AnalyticsEventType.USER_MESSAGE,
            AnalyticsEventType.ASSISTANT_MESSAGE,
            AnalyticsEventType.CHAT_COMPLETED,
        }
    }

    matched_conversations = {
        event.conversation_id
        for event in events
        if event.conversation_id
        and event.intent
        and event.intent != UNKNOWN_VALUE
    }

    response_conversations = {
        event.conversation_id
        for event in events
        if event.conversation_id
        and event.event_type
        in {
            AnalyticsEventType.ASSISTANT_MESSAGE,
            AnalyticsEventType.CHAT_COMPLETED,
        }
        and event.success
    }

    action_shown_conversations = {
        event.conversation_id
        for event in events
        if event.conversation_id
        and is_action_shown(event)
    }

    action_clicked_conversations = {
        event.conversation_id
        for event in events
        if event.conversation_id
        and is_action_clicked(event)
    }

    converted_conversations = {
        event.conversation_id
        for event in events
        if event.conversation_id
        and is_conversion_event(event)
    }

    counts = [
        (
            FunnelStep.SESSION,
            len(session_ids),
        ),
        (
            FunnelStep.MESSAGE,
            len(message_conversations),
        ),
        (
            FunnelStep.INTENT_MATCHED,
            len(matched_conversations),
        ),
        (
            FunnelStep.RESPONSE_GENERATED,
            len(response_conversations),
        ),
        (
            FunnelStep.ACTION_SHOWN,
            len(action_shown_conversations),
        ),
        (
            FunnelStep.ACTION_CLICKED,
            len(action_clicked_conversations),
        ),
        (
            FunnelStep.CONVERSION,
            len(converted_conversations),
        ),
    ]

    initial_count = counts[0][1]
    previous_count = initial_count

    metrics: list[FunnelMetric] = []

    for step, count in counts:
        metrics.append(
            FunnelMetric(
                step=step,
                count=count,
                step_conversion_rate=safe_divide(
                    count,
                    previous_count,
                )
                if metrics
                else 1.0,
                overall_conversion_rate=safe_divide(
                    count,
                    initial_count,
                ),
            )
        )

        previous_count = count

    return tuple(metrics)


# ============================================================
# SECTION 12 - TIME-SERIES ANALYTICS
# ============================================================

def period_start_for(
    value: datetime,
    granularity: AggregationGranularity,
) -> datetime:
    value = coerce_datetime(value)

    if granularity == AggregationGranularity.HOUR:
        return value.replace(
            minute=0,
            second=0,
            microsecond=0,
        )

    if granularity == AggregationGranularity.DAY:
        return value.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

    if granularity == AggregationGranularity.WEEK:
        start = value - timedelta(
            days=value.weekday()
        )
        return start.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

    return value.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )


def period_label_for(
    value: datetime,
    granularity: AggregationGranularity,
) -> str:
    if granularity == AggregationGranularity.HOUR:
        return value.strftime("%Y-%m-%d %H:00 UTC")

    if granularity == AggregationGranularity.DAY:
        return value.strftime("%Y-%m-%d")

    if granularity == AggregationGranularity.WEEK:
        year, week, _ = value.isocalendar()
        return f"{year}-W{week:02d}"

    return value.strftime("%Y-%m")


def build_time_series(
    events: Sequence[ConversationAnalyticsEvent],
    *,
    granularity: AggregationGranularity = AggregationGranularity.DAY,
) -> tuple[TimeSeriesPoint, ...]:
    grouped: dict[
        datetime,
        list[ConversationAnalyticsEvent],
    ] = defaultdict(list)

    for event in events:
        grouped[
            period_start_for(
                event.occurred_at,
                granularity,
            )
        ].append(event)

    points: list[TimeSeriesPoint] = []

    for period_start, period_events in sorted(
        grouped.items()
    ):
        conversations = {
            event.conversation_id
            for event in period_events
            if event.conversation_id
        }

        successes = sum(
            1
            for event in period_events
            if event.success
        )

        fallbacks = sum(
            1
            for event in period_events
            if is_fallback_event(event)
        )

        shown = sum(
            1
            for event in period_events
            if is_action_shown(event)
        )

        clicked = sum(
            1
            for event in period_events
            if is_action_clicked(event)
        )

        points.append(
            TimeSeriesPoint(
                period_start=period_start,
                period_label=period_label_for(
                    period_start,
                    granularity,
                ),
                event_count=len(period_events),
                conversation_count=len(
                    conversations
                ),
                success_rate=safe_divide(
                    successes,
                    len(period_events),
                ),
                fallback_rate=safe_divide(
                    fallbacks,
                    len(period_events),
                ),
                action_click_rate=safe_divide(
                    clicked,
                    shown,
                ),
                average_latency_ms=safe_mean(
                    event.latency_ms
                    for event in period_events
                ),
            )
        )

    return tuple(points)


# ============================================================
# SECTION 13 - ROLLING METRICS
# ============================================================

def rolling_average(
    values: Sequence[float | int],
    *,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> tuple[float, ...]:
    window = max(int(window), 1)
    output: list[float] = []

    for index in range(len(values)):
        start = max(
            0,
            index - window + 1,
        )
        output.append(
            safe_mean(
                values[start : index + 1]
            )
        )

    return tuple(output)


def rolling_fallback_rate(
    points: Sequence[TimeSeriesPoint],
    *,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> tuple[float, ...]:
    return rolling_average(
        [
            point.fallback_rate
            for point in points
        ],
        window=window,
    )


def rolling_latency(
    points: Sequence[TimeSeriesPoint],
    *,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> tuple[float, ...]:
    return rolling_average(
        [
            point.average_latency_ms
            for point in points
        ],
        window=window,
    )


# ============================================================
# SECTION 14 - BASELINE ANOMALY DETECTION
# ============================================================

def detect_series_anomalies(
    points: Sequence[TimeSeriesPoint],
    *,
    metric_name: str,
    z_threshold: float = DEFAULT_ANOMALY_Z_THRESHOLD,
    minimum_baseline_points: int = DEFAULT_MINIMUM_BASELINE_POINTS,
) -> tuple[AnomalySignal, ...]:
    supported_metrics = {
        "event_count",
        "conversation_count",
        "success_rate",
        "fallback_rate",
        "action_click_rate",
        "average_latency_ms",
    }

    if metric_name not in supported_metrics:
        raise ValueError(
            f"Unsupported anomaly metric: {metric_name}"
        )

    signals: list[AnomalySignal] = []

    for index, point in enumerate(points):
        if index < minimum_baseline_points:
            continue

        baseline_points = points[:index]
        baseline_values = [
            float(
                getattr(
                    baseline_point,
                    metric_name,
                )
            )
            for baseline_point
            in baseline_points
        ]

        if len(baseline_values) < minimum_baseline_points:
            continue

        baseline_mean = statistics.fmean(
            baseline_values
        )

        baseline_standard_deviation = (
            statistics.pstdev(
                baseline_values
            )
        )

        if baseline_standard_deviation == 0:
            continue

        observed = float(
            getattr(
                point,
                metric_name,
            )
        )

        z_score = (
            observed - baseline_mean
        ) / baseline_standard_deviation

        if abs(z_score) < z_threshold:
            continue

        direction = (
            "above"
            if z_score > 0
            else "below"
        )

        absolute_z = abs(z_score)

        severity = (
            "critical"
            if absolute_z >= z_threshold * 2
            else "warning"
        )

        signals.append(
            AnomalySignal(
                metric_name=metric_name,
                period_label=point.period_label,
                observed_value=round(
                    observed,
                    6,
                ),
                baseline_mean=round(
                    baseline_mean,
                    6,
                ),
                baseline_standard_deviation=round(
                    baseline_standard_deviation,
                    6,
                ),
                z_score=round(
                    z_score,
                    6,
                ),
                direction=direction,
                severity=severity,
                explanation=(
                    f"{metric_name} was {direction} its prior baseline "
                    f"by {absolute_z:.2f} standard deviations."
                ),
            )
        )

    return tuple(signals)


def detect_default_anomalies(
    points: Sequence[TimeSeriesPoint],
) -> tuple[AnomalySignal, ...]:
    signals: list[AnomalySignal] = []

    for metric_name in (
        "event_count",
        "fallback_rate",
        "action_click_rate",
        "average_latency_ms",
    ):
        signals.extend(
            detect_series_anomalies(
                points,
                metric_name=metric_name,
            )
        )

    signals.sort(
        key=lambda signal: (
            signal.period_label,
            -abs(signal.z_score),
            signal.metric_name,
        )
    )

    return tuple(signals)


# ============================================================
# SECTION 15 - COHORT ANALYTICS
# ============================================================

def cohort_key_for_event(
    event: ConversationAnalyticsEvent,
    *,
    granularity: AggregationGranularity = AggregationGranularity.WEEK,
) -> str:
    start = period_start_for(
        event.occurred_at,
        granularity,
    )
    return period_label_for(
        start,
        granularity,
    )


def build_session_cohorts(
    events: Sequence[ConversationAnalyticsEvent],
    *,
    granularity: AggregationGranularity = AggregationGranularity.WEEK,
) -> dict[str, dict[str, Any]]:
    first_event_by_session: dict[
        str,
        ConversationAnalyticsEvent,
    ] = {}

    events_by_session: dict[
        str,
        list[ConversationAnalyticsEvent],
    ] = defaultdict(list)

    for event in events:
        if not event.session_id:
            continue

        events_by_session[
            event.session_id
        ].append(event)

        existing = first_event_by_session.get(
            event.session_id
        )

        if (
            existing is None
            or event.occurred_at
            < existing.occurred_at
        ):
            first_event_by_session[
                event.session_id
            ] = event

    cohort_sessions: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for session_id, first_event in first_event_by_session.items():
        cohort_sessions[
            cohort_key_for_event(
                first_event,
                granularity=granularity,
            )
        ].add(session_id)

    output: dict[str, dict[str, Any]] = {}

    for cohort, session_ids in sorted(
        cohort_sessions.items()
    ):
        cohort_events = [
            event
            for session_id in session_ids
            for event in events_by_session[
                session_id
            ]
        ]

        returning_sessions = sum(
            1
            for session_id in session_ids
            if len(
                {
                    period_start_for(
                        event.occurred_at,
                        AggregationGranularity.DAY,
                    )
                    for event in events_by_session[
                        session_id
                    ]
                }
            )
            > 1
        )

        conversions = sum(
            1
            for event in cohort_events
            if is_conversion_event(event)
        )

        output[cohort] = {
            "session_count": len(session_ids),
            "event_count": len(cohort_events),
            "returning_session_count": (
                returning_sessions
            ),
            "return_rate": safe_divide(
                returning_sessions,
                len(session_ids),
            ),
            "conversion_count": conversions,
            "conversion_rate": safe_divide(
                conversions,
                len(session_ids),
            ),
        }

    return output


# ============================================================
# SECTION 16 - REPORT BUILDER
# ============================================================

class ConversationAnalyticsEngine:
    """
    Deterministic analytics engine for normalized conversation events.
    """

    def __init__(
        self,
        *,
        business_slug: str = DEFAULT_BUSINESS_SLUG,
        granularity: AggregationGranularity = (
            AggregationGranularity.DAY
        ),
    ) -> None:
        self.business_slug = (
            normalize_identifier(
                business_slug
            )
            or DEFAULT_BUSINESS_SLUG
        )
        self.granularity = granularity

    def normalize(
        self,
        records: Iterable[Any],
    ) -> tuple[ConversationAnalyticsEvent, ...]:
        return normalize_analytics_events(
            records,
            business_slug=self.business_slug,
        )

    def build_report(
        self,
        records: Iterable[Any],
        *,
        generated_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConversationAnalyticsReport:
        events = self.normalize(records)

        if len(events) > MAXIMUM_EXPORT_RECORDS:
            raise ValueError(
                "The analytics report exceeds the configured "
                "maximum event count."
            )

        quality_issues = validate_events(
            events
        )

        sessions = {
            event.session_id
            for event in events
            if event.session_id
        }

        conversations = {
            event.conversation_id
            for event in events
            if event.conversation_id
        }

        user_messages = [
            event
            for event in events
            if (
                event.event_type
                == AnalyticsEventType.USER_MESSAGE
                or event.role == "user"
            )
        ]

        assistant_messages = [
            event
            for event in events
            if (
                event.event_type
                == AnalyticsEventType.ASSISTANT_MESSAGE
                or event.role == "assistant"
            )
        ]

        fallback_count = sum(
            1
            for event in events
            if is_fallback_event(event)
        )

        handoff_count = sum(
            1
            for event in events
            if event.human_handoff_required
            or event.event_type
            == AnalyticsEventType.HUMAN_HANDOFF
        )

        private_event_count = sum(
            1
            for event in events
            if event.private_event_draft_present
            or event.event_type
            == AnalyticsEventType.PRIVATE_EVENT_LEAD
            or event.intent in PRIVATE_EVENT_INTENTS
        )

        ordering_intent_count = sum(
            1
            for event in events
            if event.intent in ORDERING_INTENTS
            or event.event_type
            == AnalyticsEventType.ORDERING_STARTED
        )

        action_shown_count = sum(
            1
            for event in events
            if is_action_shown(event)
        )

        action_clicked_count = sum(
            1
            for event in events
            if is_action_clicked(event)
        )

        successful_count = sum(
            1
            for event in events
            if event.success
        )

        latencies = [
            event.latency_ms
            for event in events
            if event.latency_ms is not None
        ]

        sentiment_distribution = Counter(
            classify_sentiment(
                event.message_text
            ).value
            for event in user_messages
        )

        intent_counter = Counter(
            event.intent
            for event in events
            if event.intent
        )

        page_counter = Counter(
            event.page_category
            for event in events
            if event.page_category
        )

        time_series = build_time_series(
            events,
            granularity=self.granularity,
        )

        report = ConversationAnalyticsReport(
            generated_at=coerce_datetime(
                generated_at or utc_now()
            ),
            business_slug=self.business_slug,
            event_count=len(events),
            session_count=len(sessions),
            conversation_count=len(
                conversations
            ),
            user_message_count=len(
                user_messages
            ),
            assistant_message_count=len(
                assistant_messages
            ),
            success_rate=safe_divide(
                successful_count,
                len(events),
            ),
            fallback_rate=safe_divide(
                fallback_count,
                len(events),
            ),
            handoff_rate=safe_divide(
                handoff_count,
                len(conversations),
            ),
            private_event_lead_rate=safe_divide(
                private_event_count,
                len(conversations),
            ),
            ordering_intent_rate=safe_divide(
                ordering_intent_count,
                len(conversations),
            ),
            action_show_rate=safe_divide(
                action_shown_count,
                len(events),
            ),
            action_click_rate=safe_divide(
                action_clicked_count,
                len(events),
            ),
            destination_click_through_rate=safe_divide(
                action_clicked_count,
                action_shown_count,
            ),
            average_latency_ms=safe_mean(
                latencies,
                precision=3,
            ),
            median_latency_ms=percentile(
                latencies,
                50,
            ),
            p90_latency_ms=percentile(
                latencies,
                90,
            ),
            p95_latency_ms=percentile(
                latencies,
                95,
            ),
            p99_latency_ms=percentile(
                latencies,
                99,
            ),
            average_intent_confidence=safe_mean(
                event.intent_confidence
                for event in events
            ),
            average_answer_confidence=safe_mean(
                event.answer_confidence
                for event in events
            ),
            average_retrieval_confidence=safe_mean(
                event.retrieval_confidence
                for event in events
            ),
            average_factuality_confidence=safe_mean(
                event.factuality_confidence
                for event in events
            ),
            verified_fact_count=sum(
                event.verified_fact_count
                for event in events
            ),
            unsupported_claim_count=sum(
                event.unsupported_claim_count
                for event in events
            ),
            stale_source_count=sum(
                event.stale_source_count
                for event in events
            ),
            sentiment_distribution=dict(
                sentiment_distribution
            ),
            top_intents=tuple(
                intent_counter.most_common(20)
            ),
            top_pages=tuple(
                page_counter.most_common(20)
            ),
            intent_metrics=build_intent_metrics(
                events
            ),
            destination_metrics=(
                build_destination_metrics(
                    events
                )
            ),
            funnel_metrics=build_funnel_metrics(
                events
            ),
            time_series=time_series,
            conversation_summaries=(
                summarize_conversations(
                    events
                )
            ),
            anomaly_signals=(
                detect_default_anomalies(
                    time_series
                )
            ),
            data_quality_issues=(
                quality_issues
            ),
            metadata={
                **make_json_safe(
                    dict(metadata or {})
                ),
                "cohorts": (
                    build_session_cohorts(
                        events
                    )
                ),
                "rolling_fallback_rate": list(
                    rolling_fallback_rate(
                        time_series
                    )
                ),
                "rolling_latency_ms": list(
                    rolling_latency(
                        time_series
                    )
                ),
                "privacy": {
                    "message_text_exported": False,
                    "pii_redaction_enabled": True,
                    "raw_user_messages_are_not_facts": True,
                },
            },
            service_version=(
                CONVERSATION_ANALYTICS_VERSION
            ),
            service_phase=(
                CONVERSATION_ANALYTICS_PHASE
            ),
        )

        return report


# ============================================================
# SECTION 17 - CONVENIENCE API
# ============================================================

def analyze_conversations(
    records: Iterable[Any],
    *,
    business_slug: str = DEFAULT_BUSINESS_SLUG,
    granularity: AggregationGranularity = AggregationGranularity.DAY,
    generated_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ConversationAnalyticsReport:
    return ConversationAnalyticsEngine(
        business_slug=business_slug,
        granularity=granularity,
    ).build_report(
        records,
        generated_at=generated_at,
        metadata=metadata,
    )


def analytics_report_json(
    records: Iterable[Any],
    *,
    business_slug: str = DEFAULT_BUSINESS_SLUG,
    granularity: AggregationGranularity = AggregationGranularity.DAY,
    indent: int = 2,
) -> str:
    report = analyze_conversations(
        records,
        business_slug=business_slug,
        granularity=granularity,
    )

    return json.dumps(
        report.as_dict(),
        indent=indent,
        sort_keys=True,
        ensure_ascii=False,
    )


def analytics_health_summary(
    report: ConversationAnalyticsReport,
) -> dict[str, Any]:
    status = "healthy"
    reasons: list[str] = []

    if report.success_rate < 0.95:
        status = "warning"
        reasons.append(
            "Success rate is below 95%."
        )

    if report.fallback_rate > 0.15:
        status = "warning"
        reasons.append(
            "Fallback rate exceeds 15%."
        )

    if report.p95_latency_ms > 2000:
        status = "warning"
        reasons.append(
            "P95 latency exceeds 2000 ms."
        )

    if report.unsupported_claim_count > 0:
        status = "critical"
        reasons.append(
            "Unsupported claims were reported."
        )

    if any(
        issue.severity == DataQualitySeverity.ERROR
        for issue in report.data_quality_issues
    ):
        status = "critical"
        reasons.append(
            "Analytics data-quality errors were detected."
        )

    return {
        "status": status,
        "reasons": reasons,
        "event_count": report.event_count,
        "conversation_count": report.conversation_count,
        "success_rate": report.success_rate,
        "fallback_rate": report.fallback_rate,
        "p95_latency_ms": report.p95_latency_ms,
        "unsupported_claim_count": (
            report.unsupported_claim_count
        ),
        "generated_at": report.generated_at.isoformat(),
        "service_version": report.service_version,
        "service_phase": report.service_phase,
    }


# ============================================================
# SECTION 18 - SELF-VALIDATION
# ============================================================

def _sample_events() -> list[dict[str, Any]]:
    base = datetime(
        2026,
        7,
        22,
        16,
        0,
        tzinfo=timezone.utc,
    )

    return [
        {
            "event_id": "evt_001",
            "event_type": "session_started",
            "occurred_at": base.isoformat(),
            "session_id": "session_001",
            "conversation_id": "conversation_001",
            "success": True,
        },
        {
            "event_id": "evt_002",
            "event_type": "user_message",
            "occurred_at": (
                base + timedelta(seconds=2)
            ).isoformat(),
            "session_id": "session_001",
            "conversation_id": "conversation_001",
            "message_id": "message_001",
            "role": "user",
            "message": (
                "Show me the menu tonight. "
                "My email is test@example.com."
            ),
            "intent": "menu_general",
            "intent_confidence": 0.98,
            "success": True,
        },
        {
            "event_id": "evt_003",
            "event_type": "chat_completed",
            "occurred_at": (
                base + timedelta(seconds=3)
            ).isoformat(),
            "session_id": "session_001",
            "conversation_id": "conversation_001",
            "message_id": "message_002",
            "role": "assistant",
            "intent": "menu_general",
            "nlu_decision": "matched",
            "response_decision": "grounded",
            "intent_confidence": 0.98,
            "answer_confidence": 0.97,
            "retrieval_confidence": 0.95,
            "factuality_confidence": 1.0,
            "processing_time_ms": 184.2,
            "response_source_count": 1,
            "response_action_count": 1,
            "destination_match_count": 1,
            "verified_fact_count": 1,
            "unsupported_claim_count": 0,
            "destination_key": "menu",
            "success": True,
        },
        {
            "event_id": "evt_004",
            "event_type": "action_shown",
            "occurred_at": (
                base + timedelta(seconds=4)
            ).isoformat(),
            "session_id": "session_001",
            "conversation_id": "conversation_001",
            "destination_key": "menu",
            "analytics_event": "official_menu_opened",
            "success": True,
        },
        {
            "event_id": "evt_005",
            "event_type": "action_clicked",
            "occurred_at": (
                base + timedelta(seconds=8)
            ).isoformat(),
            "session_id": "session_001",
            "conversation_id": "conversation_001",
            "destination_key": "menu",
            "destination_url": (
                "https://www.thehorseshoetavern.com/menu"
            ),
            "analytics_event": "official_menu_opened",
            "success": True,
        },
        {
            "event_id": "evt_006",
            "event_type": "user_message",
            "occurred_at": (
                base + timedelta(days=1)
            ).isoformat(),
            "session_id": "session_002",
            "conversation_id": "conversation_002",
            "message_id": "message_003",
            "role": "user",
            "message": (
                "I need a private party for 50 people."
            ),
            "intent": "private_event",
            "intent_confidence": 0.96,
            "success": True,
        },
        {
            "event_id": "evt_007",
            "event_type": "private_event_lead",
            "occurred_at": (
                base
                + timedelta(days=1, seconds=3)
            ).isoformat(),
            "session_id": "session_002",
            "conversation_id": "conversation_002",
            "intent": "private_event",
            "private_event_draft_present": True,
            "destination_key": "private_events",
            "success": True,
        },
    ]


def validate_conversation_analytics_module() -> dict[str, Any]:
    sample = _sample_events()

    engine = ConversationAnalyticsEngine()

    events = engine.normalize(sample)
    report = engine.build_report(
        sample,
        generated_at=datetime(
            2026,
            7,
            24,
            tzinfo=timezone.utc,
        ),
        metadata={
            "validation": True,
        },
    )

    redacted_event = next(
        event
        for event in events
        if event.message_text
        and "menu tonight" in event.message_text
    )

    checks = {
        "normalized_event_count": (
            len(events) == len(sample)
        ),
        "report_event_count": (
            report.event_count == len(sample)
        ),
        "session_count": (
            report.session_count == 2
        ),
        "conversation_count": (
            report.conversation_count == 2
        ),
        "menu_click_detected": any(
            metric.destination_key == "menu"
            and metric.click_count >= 1
            for metric in report.destination_metrics
        ),
        "private_event_detected": (
            report.private_event_lead_rate > 0
        ),
        "latency_computed": (
            report.average_latency_ms > 0
        ),
        "intent_metrics_present": bool(
            report.intent_metrics
        ),
        "funnel_present": bool(
            report.funnel_metrics
        ),
        "time_series_present": bool(
            report.time_series
        ),
        "conversation_summaries_present": bool(
            report.conversation_summaries
        ),
        "pii_redacted": (
            "test@example.com"
            not in (
                redacted_event.message_text
                or ""
            )
        ),
        "json_safe": bool(
            json.dumps(
                report.as_dict()
            )
        ),
        "health_summary_present": bool(
            analytics_health_summary(
                report
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
        "event_count": report.event_count,
        "conversation_count": (
            report.conversation_count
        ),
        "service_version": (
            CONVERSATION_ANALYTICS_VERSION
        ),
        "service_phase": (
            CONVERSATION_ANALYTICS_PHASE
        ),
    }


# ============================================================
# SECTION 19 - PUBLIC EXPORTS
# ============================================================

__all__ = [
    "AggregationGranularity",
    "AnalyticsEventType",
    "AnomalySignal",
    "CONVERSATION_ANALYTICS_PHASE",
    "CONVERSATION_ANALYTICS_VERSION",
    "ConversationAnalyticsEngine",
    "ConversationAnalyticsEvent",
    "ConversationAnalyticsReport",
    "ConversationSummaryMetric",
    "ConversionType",
    "DataQualityIssue",
    "DataQualitySeverity",
    "DestinationMetric",
    "FunnelMetric",
    "FunnelStep",
    "IntentMetric",
    "SentimentLabel",
    "TimeSeriesPoint",
    "analytics_health_summary",
    "analytics_report_json",
    "analyze_conversations",
    "build_destination_metrics",
    "build_funnel_metrics",
    "build_intent_metrics",
    "build_session_cohorts",
    "build_time_series",
    "classify_sentiment",
    "conversion_type_for_event",
    "detect_default_anomalies",
    "detect_series_anomalies",
    "make_json_safe",
    "normalize_analytics_event",
    "normalize_analytics_events",
    "percentile",
    "redact_sensitive_text",
    "rolling_average",
    "rolling_fallback_rate",
    "rolling_latency",
    "safe_divide",
    "safe_mean",
    "stable_hash",
    "summarize_conversations",
    "urgency_score",
    "validate_conversation_analytics_module",
    "validate_event",
    "validate_events",
]


# ============================================================
# SECTION 20 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    validation_report = (
        validate_conversation_analytics_module()
    )

    print(
        json.dumps(
            validation_report,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    if validation_report["status"] != "ok":
        raise SystemExit(1)
