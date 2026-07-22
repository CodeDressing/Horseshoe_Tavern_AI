# ============================================================
# Exact file location: app\analytics\failure_analytics.py
# Horseshoe Tavern AI
# Enterprise Failure Analytics Module
# Phase 1 Part 1.45
# ============================================================

"""
Enterprise failure analytics for Horseshoe Tavern AI.

This module provides deterministic, explainable, privacy-aware analysis for:

- Application exceptions
- API failures
- Chat-processing failures
- NLU failures
- Knowledge-retrieval failures
- Response-generation failures
- Destination-routing failures
- Persistence and transaction failures
- Render deployment and startup failures
- Database failures
- Configuration failures
- Validation failures
- Authentication and authorization failures
- External-provider failures
- User-facing fallback and degraded-mode events
- Repeated and correlated failure clusters
- Failure severity scoring
- Error-budget and reliability metrics
- Mean time between failures
- Mean time to recovery
- Failure-rate time series
- Baseline anomaly detection
- Root-cause candidate extraction
- Remediation recommendations
- JSON-safe reporting
- Deterministic self-validation

The module is dependency-light, framework-agnostic, and suitable for direct
integration with persisted analytics events, structured logs, repository rows,
API payloads, or exception records.

Privacy and safety principles
-----------------------------
- Raw secrets are never intentionally emitted in reports.
- Email addresses, phone numbers, card-like values, tokens, passwords,
  connection strings, and authorization headers are redacted.
- User messages are not treated as verified business facts.
- Recommendations remain advisory and explainable.
- The module performs no network calls and no hidden persistence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import statistics
import traceback
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Sequence


# ============================================================
# SECTION 01 - MODULE METADATA AND CONSTANTS
# ============================================================

FAILURE_ANALYTICS_VERSION: Final[str] = "2.0.0"
FAILURE_ANALYTICS_PHASE: Final[str] = "Phase 1 Part 1.45"

DEFAULT_BUSINESS_SLUG: Final[str] = "horseshoe-tavern"
DEFAULT_SERVICE_NAME: Final[str] = "horseshoe-tavern-ai"
UNKNOWN_VALUE: Final[str] = "unknown"

MAXIMUM_TEXT_LENGTH: Final[int] = 6000
MAXIMUM_STACKTRACE_LENGTH: Final[int] = 24000
MAXIMUM_METADATA_DEPTH: Final[int] = 8
MAXIMUM_REPORT_FAILURES: Final[int] = 100_000

DEFAULT_Z_THRESHOLD: Final[float] = 3.0
DEFAULT_MINIMUM_BASELINE_POINTS: Final[int] = 5
DEFAULT_RECOVERY_GAP_SECONDS: Final[int] = 300

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
BEARER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b",
    re.IGNORECASE,
)
SECRET_ASSIGNMENT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(api[_-]?key|secret|password|passwd|token|authorization|database_url|session_signing_key)\b\s*[:=]\s*([^\s,;]+)"
)
POSTGRES_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\bpostgres(?:ql)?(?:\+\w+)?://[^\s]+"
)
GENERIC_URL_CREDENTIAL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b([a-z][a-z0-9+.-]*://)([^/\s:@]+):([^@\s/]+)@"
)
WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")
TRACEBACK_FILE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'File\s+"(?P<path>[^"]+)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<function>[^\n]+)'
)
PYTHON_EXCEPTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<class>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning)):\s*(?P<message>.+)"
)


# ============================================================
# SECTION 02 - ENUMERATIONS
# ============================================================

class FailureCategory(str, Enum):
    APPLICATION_STARTUP = "application_startup"
    CONFIGURATION = "configuration"
    IMPORT = "import"
    SYNTAX = "syntax"
    VALIDATION = "validation"
    DATABASE = "database"
    TRANSACTION = "transaction"
    PERSISTENCE = "persistence"
    API = "api"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NLU = "nlu"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    RESPONSE_GENERATION = "response_generation"
    DESTINATION_ROUTING = "destination_routing"
    EXTERNAL_PROVIDER = "external_provider"
    DEPLOYMENT = "deployment"
    HEALTH_CHECK = "health_check"
    USER_INPUT = "user_input"
    DATA_QUALITY = "data_quality"
    UNKNOWN = "unknown"


class FailureSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FailureStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class FailureSource(str, Enum):
    EXCEPTION = "exception"
    LOG = "log"
    API_RESPONSE = "api_response"
    HEALTH_CHECK = "health_check"
    DEPLOYMENT_LOG = "deployment_log"
    ANALYTICS_EVENT = "analytics_event"
    DATABASE_RECORD = "database_record"
    USER_REPORT = "user_report"
    CUSTOM = "custom"


class FailureTrend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


class AggregationGranularity(str, Enum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


# ============================================================
# SECTION 03 - DATA CLASSES
# ============================================================

@dataclass(frozen=True, slots=True)
class FailureFrame:
    file_path: str | None
    line_number: int | None
    function_name: str | None
    code_line: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "function_name": self.function_name,
            "code_line": self.code_line,
        }


@dataclass(frozen=True, slots=True)
class FailureRecord:
    failure_id: str
    occurred_at: datetime
    category: FailureCategory
    severity: FailureSeverity
    source: FailureSource

    business_slug: str = DEFAULT_BUSINESS_SLUG
    service_name: str = DEFAULT_SERVICE_NAME
    environment: str = UNKNOWN_VALUE

    status: FailureStatus = FailureStatus.OPEN
    success: bool = False

    exception_type: str | None = None
    message: str = ""
    normalized_message: str = ""
    fingerprint: str = ""

    stacktrace: str | None = None
    frames: tuple[FailureFrame, ...] = ()

    request_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    deployment_id: str | None = None
    commit_sha: str | None = None

    route: str | None = None
    method: str | None = None
    status_code: int | None = None
    latency_ms: float | None = None

    component: str | None = None
    operation: str | None = None
    provider: str | None = None

    retry_count: int = 0
    recoverable: bool = False
    user_visible: bool = False
    human_handoff_required: bool = False

    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(
        self,
        *,
        include_stacktrace: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "failure_id": self.failure_id,
            "occurred_at": self.occurred_at.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "source": self.source.value,
            "business_slug": self.business_slug,
            "service_name": self.service_name,
            "environment": self.environment,
            "status": self.status.value,
            "success": self.success,
            "exception_type": self.exception_type,
            "message": self.message,
            "normalized_message": self.normalized_message,
            "fingerprint": self.fingerprint,
            "frames": [
                frame.as_dict()
                for frame in self.frames
            ],
            "request_id": self.request_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "deployment_id": self.deployment_id,
            "commit_sha": self.commit_sha,
            "route": self.route,
            "method": self.method,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "component": self.component,
            "operation": self.operation,
            "provider": self.provider,
            "retry_count": self.retry_count,
            "recoverable": self.recoverable,
            "user_visible": self.user_visible,
            "human_handoff_required": self.human_handoff_required,
            "resolved_at": (
                self.resolved_at.isoformat()
                if self.resolved_at
                else None
            ),
            "metadata": make_json_safe(
                self.metadata
            ),
        }

        if include_stacktrace:
            payload["stacktrace"] = redact_sensitive_text(
                self.stacktrace,
                maximum_length=MAXIMUM_STACKTRACE_LENGTH,
            )

        return payload


@dataclass(frozen=True, slots=True)
class FailureCluster:
    fingerprint: str
    category: FailureCategory
    severity: FailureSeverity
    exception_type: str | None
    normalized_message: str
    first_seen_at: datetime
    last_seen_at: datetime
    occurrence_count: int
    unique_sessions: int
    unique_conversations: int
    unique_requests: int
    affected_routes: tuple[str, ...]
    affected_components: tuple[str, ...]
    recoverable_rate: float
    user_visible_rate: float
    resolved_rate: float
    representative_failure_ids: tuple[str, ...]
    root_cause_candidates: tuple[str, ...]
    recommendations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "category": self.category.value,
            "severity": self.severity.value,
            "exception_type": self.exception_type,
            "normalized_message": self.normalized_message,
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "occurrence_count": self.occurrence_count,
            "unique_sessions": self.unique_sessions,
            "unique_conversations": self.unique_conversations,
            "unique_requests": self.unique_requests,
            "affected_routes": list(self.affected_routes),
            "affected_components": list(self.affected_components),
            "recoverable_rate": self.recoverable_rate,
            "user_visible_rate": self.user_visible_rate,
            "resolved_rate": self.resolved_rate,
            "representative_failure_ids": list(
                self.representative_failure_ids
            ),
            "root_cause_candidates": list(
                self.root_cause_candidates
            ),
            "recommendations": list(
                self.recommendations
            ),
        }


@dataclass(frozen=True, slots=True)
class FailureTimeSeriesPoint:
    period_start: datetime
    period_label: str
    failure_count: int
    critical_count: int
    high_count: int
    open_count: int
    resolved_count: int
    unique_fingerprints: int
    unique_conversations: int
    user_visible_rate: float
    recovery_rate: float
    average_latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["period_start"] = self.period_start.isoformat()
        return make_json_safe(payload)


@dataclass(frozen=True, slots=True)
class FailureAnomaly:
    metric_name: str
    period_label: str
    observed_value: float
    baseline_mean: float
    baseline_standard_deviation: float
    z_score: float
    severity: FailureSeverity
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return make_json_safe(payload)


@dataclass(frozen=True, slots=True)
class FailureReliabilityMetrics:
    total_failures: int
    open_failures: int
    resolved_failures: int
    critical_failures: int
    high_failures: int
    user_visible_failures: int
    recoverable_failures: int

    failure_rate: float
    open_rate: float
    resolution_rate: float
    recoverable_rate: float
    user_visible_rate: float

    mean_time_between_failures_seconds: float
    median_time_between_failures_seconds: float
    mean_time_to_recovery_seconds: float
    median_time_to_recovery_seconds: float

    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    error_budget_consumed: float
    availability_estimate: float

    def as_dict(self) -> dict[str, Any]:
        return make_json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class FailureAnalyticsReport:
    generated_at: datetime
    business_slug: str
    service_name: str
    environment: str

    reliability: FailureReliabilityMetrics
    category_distribution: dict[str, int]
    severity_distribution: dict[str, int]
    status_distribution: dict[str, int]
    source_distribution: dict[str, int]

    top_exception_types: tuple[tuple[str, int], ...]
    top_routes: tuple[tuple[str, int], ...]
    top_components: tuple[tuple[str, int], ...]
    top_providers: tuple[tuple[str, int], ...]

    clusters: tuple[FailureCluster, ...]
    time_series: tuple[FailureTimeSeriesPoint, ...]
    anomalies: tuple[FailureAnomaly, ...]
    recommendations: tuple[str, ...]
    metadata: dict[str, Any]

    service_version: str
    service_phase: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "business_slug": self.business_slug,
            "service_name": self.service_name,
            "environment": self.environment,
            "reliability": self.reliability.as_dict(),
            "category_distribution": copy.deepcopy(
                self.category_distribution
            ),
            "severity_distribution": copy.deepcopy(
                self.severity_distribution
            ),
            "status_distribution": copy.deepcopy(
                self.status_distribution
            ),
            "source_distribution": copy.deepcopy(
                self.source_distribution
            ),
            "top_exception_types": [
                {
                    "exception_type": name,
                    "count": count,
                }
                for name, count in self.top_exception_types
            ],
            "top_routes": [
                {
                    "route": route,
                    "count": count,
                }
                for route, count in self.top_routes
            ],
            "top_components": [
                {
                    "component": component,
                    "count": count,
                }
                for component, count in self.top_components
            ],
            "top_providers": [
                {
                    "provider": provider,
                    "count": count,
                }
                for provider, count in self.top_providers
            ],
            "clusters": [
                cluster.as_dict()
                for cluster in self.clusters
            ],
            "time_series": [
                point.as_dict()
                for point in self.time_series
            ],
            "anomalies": [
                anomaly.as_dict()
                for anomaly in self.anomalies
            ],
            "recommendations": list(
                self.recommendations
            ),
            "metadata": make_json_safe(
                self.metadata
            ),
            "service_version": self.service_version,
            "service_phase": self.service_phase,
        }


# ============================================================
# SECTION 04 - GENERAL UTILITIES
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
    text = str(value or "").strip().casefold()
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or UNKNOWN_VALUE


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
    numeric: list[float] = []

    for value in values:
        if value is None:
            continue

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue

        if math.isfinite(number):
            numeric.append(number)

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
    numeric: list[float] = []

    for value in values:
        if value is None:
            continue

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue

        if math.isfinite(number):
            numeric.append(number)

    numeric.sort()

    if not numeric:
        return 0.0

    if len(numeric) == 1:
        return round(numeric[0], precision)

    percentile_value = min(
        max(float(percentile_value), 0.0),
        100.0,
    )

    rank = (
        percentile_value
        / 100.0
        * (len(numeric) - 1)
    )

    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))

    if lower == upper:
        return round(numeric[lower], precision)

    weight = rank - lower

    value = (
        numeric[lower]
        + (
            numeric[upper]
            - numeric[lower]
        )
        * weight
    )

    return round(value, precision)


def stable_hash(
    value: Any,
    *,
    namespace: str = "horseshoe-failure",
    length: int = 32,
) -> str:
    payload = f"{namespace}:{value}".encode(
        "utf-8",
        errors="replace",
    )

    return hashlib.sha256(payload).hexdigest()[
        :max(8, int(length))
    ]


def redact_sensitive_text(
    value: Any,
    *,
    maximum_length: int = MAXIMUM_TEXT_LENGTH,
) -> str | None:
    if value is None:
        return None

    text = str(value).replace("\x00", " ")
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
    text = CARD_PATTERN.sub("[REDACTED_CARD]", text)
    text = BEARER_PATTERN.sub("Bearer [REDACTED_TOKEN]", text)
    text = POSTGRES_URL_PATTERN.sub(
        "[REDACTED_DATABASE_URL]",
        text,
    )
    text = GENERIC_URL_CREDENTIAL_PATTERN.sub(
        r"\1[REDACTED_USER]:[REDACTED_PASSWORD]@",
        text,
    )
    text = SECRET_ASSIGNMENT_PATTERN.sub(
        lambda match: (
            f"{match.group(1)}=[REDACTED_SECRET]"
        ),
        text,
    )

    return text[:maximum_length]


def normalize_failure_message(
    value: Any,
) -> str:
    text = redact_sensitive_text(value) or ""
    text = text.casefold()

    text = re.sub(
        r"\b[0-9a-f]{7,64}\b",
        "<hash>",
        text,
    )
    text = re.sub(
        r"\b\d{4}-\d{2}-\d{2}[t ][0-9:.+\-z]+\b",
        "<timestamp>",
        text,
    )
    text = re.sub(
        r"\b\d+\b",
        "<number>",
        text,
    )
    text = re.sub(
        r"[A-Za-z]:\\[^\s]+",
        "<windows_path>",
        text,
    )
    text = re.sub(
        r"/(?:[^/\s]+/)+[^/\s]+",
        "<path>",
        text,
    )
    text = WHITESPACE_PATTERN.sub(
        " ",
        text,
    ).strip()

    return text[:MAXIMUM_TEXT_LENGTH]


# ============================================================
# SECTION 05 - RECORD COERCION
# ============================================================

def _mapping_from_record(
    record: Any,
) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)

    if isinstance(record, BaseException):
        return {
            "exception": record,
            "exception_type": (
                record.__class__.__name__
            ),
            "message": str(record),
            "stacktrace": "".join(
                traceback.format_exception(
                    type(record),
                    record,
                    record.__traceback__,
                )
            ),
        }

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
            for key, value in vars(record).items()
            if not key.startswith("_")
        }

    return {
        "message": str(record),
    }


def _first_present(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]

    return default


# ============================================================
# SECTION 06 - FRAME AND EXCEPTION PARSING
# ============================================================

def parse_traceback_frames(
    stacktrace: Any,
) -> tuple[FailureFrame, ...]:
    if not stacktrace:
        return ()

    text = str(stacktrace)
    lines = text.splitlines()

    frames: list[FailureFrame] = []

    for index, line in enumerate(lines):
        match = TRACEBACK_FILE_PATTERN.search(line)

        if not match:
            continue

        code_line = None

        if index + 1 < len(lines):
            candidate = lines[index + 1].strip()

            if candidate and not candidate.startswith(
                "File "
            ):
                code_line = redact_sensitive_text(
                    candidate
                )

        frames.append(
            FailureFrame(
                file_path=redact_sensitive_text(
                    match.group("path")
                ),
                line_number=coerce_int(
                    match.group("line"),
                    minimum=0,
                ),
                function_name=(
                    match.group("function")
                    .strip()
                ),
                code_line=code_line,
            )
        )

    return tuple(frames)


def infer_exception_type(
    payload: Mapping[str, Any],
    message: str,
    stacktrace: str | None,
) -> str | None:
    explicit = normalize_identifier(
        _first_present(
            payload,
            (
                "exception_type",
                "error_type",
                "exception_class",
                "class_name",
            ),
        )
    )

    if explicit:
        return explicit

    exception = payload.get("exception")

    if isinstance(exception, BaseException):
        return exception.__class__.__name__

    for candidate in (
        stacktrace,
        message,
    ):
        if not candidate:
            continue

        matches = list(
            PYTHON_EXCEPTION_PATTERN.finditer(
                str(candidate)
            )
        )

        if matches:
            return matches[-1].group(
                "class"
            ).split(".")[-1]

    return None


# ============================================================
# SECTION 07 - CATEGORY INFERENCE
# ============================================================

def infer_failure_category(
    *,
    exception_type: str | None,
    message: str,
    stacktrace: str | None,
    component: str | None,
    operation: str | None,
    source: FailureSource,
) -> FailureCategory:
    text = " ".join(
        value
        for value in (
            exception_type,
            message,
            stacktrace,
            component,
            operation,
        )
        if value
    ).casefold()

    if (
        "syntaxerror" in text
        or "invalid syntax" in text
        or "unterminated string" in text
    ):
        return FailureCategory.SYNTAX

    if (
        "modulenotfounderror" in text
        or "importerror" in text
        or "cannot import name" in text
        or "no module named" in text
    ):
        return FailureCategory.IMPORT

    if (
        "settingsError".casefold() in text
        or "validationerror" in text
        or "environment variable" in text
        or "allowed_hosts" in text
        or "admin_password" in text
    ):
        return FailureCategory.CONFIGURATION

    if (
        "application startup" in text
        or "startup failed" in text
        or "uvicorn" in text
        and "exited with status" in text
    ):
        return FailureCategory.APPLICATION_STARTUP

    if (
        "sqlalchemy" in text
        or "database" in text
        or "psycopg" in text
        or "sqlite" in text
        or "relation does not exist" in text
        or "operationalerror" in text
    ):
        return FailureCategory.DATABASE

    if (
        "rollback" in text
        or "commit failed" in text
        or "transaction" in text
    ):
        return FailureCategory.TRANSACTION

    if (
        "repository" in text
        or "persist" in text
        or "storage" in text
    ):
        return FailureCategory.PERSISTENCE

    if (
        "timeout" in text
        or "timed out" in text
        or "readtimeout" in text
    ):
        return FailureCategory.TIMEOUT

    if (
        "429" in text
        or "rate limit" in text
        or "too many requests" in text
    ):
        return FailureCategory.RATE_LIMIT

    if (
        "401" in text
        or "authentication" in text
        or "invalid credentials" in text
    ):
        return FailureCategory.AUTHENTICATION

    if (
        "403" in text
        or "authorization" in text
        or "permission denied" in text
    ):
        return FailureCategory.AUTHORIZATION

    if (
        "nlu" in text
        or "intent detection" in text
        or "entity detection" in text
    ):
        return FailureCategory.NLU

    if (
        "knowledge_service" in text
        or "knowledge retrieval" in text
        or "verified knowledge" in text
    ):
        return FailureCategory.KNOWLEDGE_RETRIEVAL

    if (
        "response_service" in text
        or "response generation" in text
        or "grounded response" in text
    ):
        return FailureCategory.RESPONSE_GENERATION

    if (
        "destination_routing" in text
        or "destination routing" in text
        or "official destination" in text
    ):
        return FailureCategory.DESTINATION_ROUTING

    if (
        "render" in text
        or "deploy" in text
        or "build failed" in text
        or source == FailureSource.DEPLOYMENT_LOG
    ):
        return FailureCategory.DEPLOYMENT

    if (
        "health" in text
        or source == FailureSource.HEALTH_CHECK
    ):
        return FailureCategory.HEALTH_CHECK

    if (
        "chownow" in text
        or "spoton" in text
        or "facebook" in text
        or "instagram" in text
        or "external provider" in text
    ):
        return FailureCategory.EXTERNAL_PROVIDER

    if (
        "422" in text
        or "bad request" in text
        or "invalid input" in text
    ):
        return FailureCategory.USER_INPUT

    if (
        "json" in text
        or "schema" in text
        or "data quality" in text
    ):
        return FailureCategory.DATA_QUALITY

    if (
        "http" in text
        or "connection" in text
        or "network" in text
    ):
        return FailureCategory.NETWORK

    if (
        "api" in text
        or "endpoint" in text
        or "route" in text
    ):
        return FailureCategory.API

    return FailureCategory.UNKNOWN


# ============================================================
# SECTION 08 - SEVERITY SCORING
# ============================================================

SEVERITY_ORDER: Final[dict[FailureSeverity, int]] = {
    FailureSeverity.INFO: 0,
    FailureSeverity.LOW: 1,
    FailureSeverity.MEDIUM: 2,
    FailureSeverity.HIGH: 3,
    FailureSeverity.CRITICAL: 4,
}


def max_severity(
    values: Iterable[FailureSeverity],
) -> FailureSeverity:
    return max(
        values,
        key=lambda value: SEVERITY_ORDER[value],
        default=FailureSeverity.INFO,
    )


def infer_failure_severity(
    *,
    category: FailureCategory,
    status_code: int | None,
    user_visible: bool,
    human_handoff_required: bool,
    recoverable: bool,
    retry_count: int,
    explicit: Any = None,
) -> FailureSeverity:
    normalized_explicit = normalize_label(
        explicit
    )

    try:
        if normalized_explicit != UNKNOWN_VALUE:
            return FailureSeverity(
                normalized_explicit
            )
    except ValueError:
        pass

    score = 0

    if category in {
        FailureCategory.APPLICATION_STARTUP,
        FailureCategory.DEPLOYMENT,
        FailureCategory.DATABASE,
        FailureCategory.IMPORT,
        FailureCategory.SYNTAX,
        FailureCategory.CONFIGURATION,
    }:
        score += 3

    if category in {
        FailureCategory.TRANSACTION,
        FailureCategory.AUTHENTICATION,
        FailureCategory.AUTHORIZATION,
        FailureCategory.HEALTH_CHECK,
    }:
        score += 2

    if status_code is not None:
        if status_code >= 500:
            score += 2
        elif status_code >= 400:
            score += 1

    if user_visible:
        score += 1

    if human_handoff_required:
        score += 1

    if not recoverable:
        score += 1

    if retry_count >= 3:
        score += 1

    if score >= 6:
        return FailureSeverity.CRITICAL

    if score >= 4:
        return FailureSeverity.HIGH

    if score >= 2:
        return FailureSeverity.MEDIUM

    if score >= 1:
        return FailureSeverity.LOW

    return FailureSeverity.INFO


# ============================================================
# SECTION 09 - FINGERPRINTING
# ============================================================

def build_failure_fingerprint(
    *,
    category: FailureCategory,
    exception_type: str | None,
    normalized_message: str,
    component: str | None,
    operation: str | None,
    route: str | None,
    frames: Sequence[FailureFrame],
) -> str:
    first_application_frame = next(
        (
            frame
            for frame in frames
            if frame.file_path
            and (
                "/app/" in frame.file_path.replace("\\", "/")
                or "app/" in frame.file_path.replace("\\", "/")
            )
        ),
        frames[-1] if frames else None,
    )

    frame_signature = ""

    if first_application_frame:
        frame_signature = "|".join(
            [
                first_application_frame.file_path or "",
                str(
                    first_application_frame.line_number
                    or ""
                ),
                first_application_frame.function_name or "",
            ]
        )

    payload = "|".join(
        [
            category.value,
            exception_type or "",
            normalized_message,
            normalize_label(component),
            normalize_label(operation),
            normalize_label(route),
            frame_signature,
        ]
    )

    return stable_hash(
        payload,
        namespace="failure-fingerprint",
        length=40,
    )


# ============================================================
# SECTION 10 - NORMALIZATION
# ============================================================

def normalize_failure_record(
    record: Any,
    *,
    business_slug: str = DEFAULT_BUSINESS_SLUG,
    service_name: str = DEFAULT_SERVICE_NAME,
    default_occurred_at: datetime | None = None,
) -> FailureRecord:
    payload = _mapping_from_record(record)

    stacktrace_raw = _first_present(
        payload,
        (
            "stacktrace",
            "traceback",
            "stack",
            "exception_trace",
        ),
    )

    stacktrace = redact_sensitive_text(
        stacktrace_raw,
        maximum_length=MAXIMUM_STACKTRACE_LENGTH,
    )

    message = redact_sensitive_text(
        _first_present(
            payload,
            (
                "message",
                "error",
                "detail",
                "reason",
                "exception_message",
            ),
            "",
        )
    ) or ""

    exception_type = infer_exception_type(
        payload,
        message,
        stacktrace,
    )

    frames = parse_traceback_frames(
        stacktrace
    )

    source_value = normalize_label(
        _first_present(
            payload,
            (
                "source",
                "failure_source",
                "record_source",
            ),
            FailureSource.CUSTOM.value,
        )
    )

    try:
        source = FailureSource(source_value)
    except ValueError:
        source = FailureSource.CUSTOM

    component = normalize_identifier(
        _first_present(
            payload,
            (
                "component",
                "module",
                "service",
            ),
        )
    )

    operation = normalize_identifier(
        _first_present(
            payload,
            (
                "operation",
                "action",
                "function",
            ),
        )
    )

    category_value = normalize_label(
        _first_present(
            payload,
            ("category", "failure_category"),
            UNKNOWN_VALUE,
        )
    )

    if category_value != UNKNOWN_VALUE:
        try:
            category = FailureCategory(
                category_value
            )
        except ValueError:
            category = infer_failure_category(
                exception_type=exception_type,
                message=message,
                stacktrace=stacktrace,
                component=component,
                operation=operation,
                source=source,
            )
    else:
        category = infer_failure_category(
            exception_type=exception_type,
            message=message,
            stacktrace=stacktrace,
            component=component,
            operation=operation,
            source=source,
        )

    status_code = coerce_int(
        _first_present(
            payload,
            (
                "status_code",
                "http_status",
                "response_status",
            ),
            0,
        ),
        default=0,
        minimum=0,
    )

    if status_code == 0:
        status_code = None

    retry_count = coerce_int(
        _first_present(
            payload,
            (
                "retry_count",
                "retries",
                "attempt",
            ),
            0,
        ),
        minimum=0,
    )

    recoverable = coerce_bool(
        _first_present(
            payload,
            (
                "recoverable",
                "retryable",
                "transient",
            ),
            False,
        )
    )

    user_visible = coerce_bool(
        _first_present(
            payload,
            (
                "user_visible",
                "shown_to_user",
                "public_error",
            ),
            False,
        )
    )

    human_handoff_required = coerce_bool(
        _first_present(
            payload,
            (
                "human_handoff_required",
                "handoff_required",
            ),
            False,
        )
    )

    severity = infer_failure_severity(
        category=category,
        status_code=status_code,
        user_visible=user_visible,
        human_handoff_required=(
            human_handoff_required
        ),
        recoverable=recoverable,
        retry_count=retry_count,
        explicit=_first_present(
            payload,
            ("severity", "level"),
        ),
    )

    status_value = normalize_label(
        _first_present(
            payload,
            ("status", "failure_status"),
            FailureStatus.OPEN.value,
        )
    )

    try:
        status = FailureStatus(
            status_value
        )
    except ValueError:
        status = FailureStatus.OPEN

    occurred_at = coerce_datetime(
        _first_present(
            payload,
            (
                "occurred_at",
                "created_at",
                "timestamp",
                "time",
            ),
            default_occurred_at or utc_now(),
        ),
        default=default_occurred_at or utc_now(),
    )

    resolved_raw = _first_present(
        payload,
        (
            "resolved_at",
            "recovered_at",
            "closed_at",
        ),
    )

    resolved_at = (
        coerce_datetime(
            resolved_raw
        )
        if resolved_raw is not None
        else None
    )

    normalized_message = (
        normalize_failure_message(
            message
        )
    )

    route = normalize_identifier(
        _first_present(
            payload,
            (
                "route",
                "path",
                "endpoint",
            ),
        )
    )

    fingerprint = normalize_identifier(
        _first_present(
            payload,
            ("fingerprint",),
        )
    ) or build_failure_fingerprint(
        category=category,
        exception_type=exception_type,
        normalized_message=normalized_message,
        component=component,
        operation=operation,
        route=route,
        frames=frames,
    )

    metadata = _first_present(
        payload,
        (
            "metadata",
            "metadata_json",
            "context",
        ),
        {},
    )

    if not isinstance(metadata, Mapping):
        metadata = {
            "raw_metadata": str(metadata),
        }

    return FailureRecord(
        failure_id=normalize_identifier(
            _first_present(
                payload,
                (
                    "failure_id",
                    "event_id",
                    "id",
                ),
            )
        )
        or f"failure_{uuid.uuid4().hex}",
        occurred_at=occurred_at,
        category=category,
        severity=severity,
        source=source,
        business_slug=normalize_identifier(
            _first_present(
                payload,
                ("business_slug",),
                business_slug,
            )
        )
        or business_slug,
        service_name=normalize_identifier(
            _first_present(
                payload,
                ("service_name",),
                service_name,
            )
        )
        or service_name,
        environment=normalize_label(
            _first_present(
                payload,
                (
                    "environment",
                    "env",
                ),
                UNKNOWN_VALUE,
            )
        ),
        status=status,
        success=coerce_bool(
            _first_present(
                payload,
                ("success",),
                False,
            )
        ),
        exception_type=exception_type,
        message=message,
        normalized_message=normalized_message,
        fingerprint=fingerprint,
        stacktrace=stacktrace,
        frames=frames,
        request_id=normalize_identifier(
            _first_present(
                payload,
                ("request_id",),
            )
        ),
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
        deployment_id=normalize_identifier(
            _first_present(
                payload,
                (
                    "deployment_id",
                    "deploy_id",
                ),
            )
        ),
        commit_sha=normalize_identifier(
            _first_present(
                payload,
                (
                    "commit_sha",
                    "commit",
                    "git_sha",
                ),
            )
        ),
        route=route,
        method=normalize_identifier(
            _first_present(
                payload,
                (
                    "method",
                    "http_method",
                ),
            )
        ),
        status_code=status_code,
        latency_ms=coerce_float(
            _first_present(
                payload,
                (
                    "latency_ms",
                    "duration_ms",
                    "processing_time_ms",
                ),
            ),
            minimum=0.0,
        ),
        component=component,
        operation=operation,
        provider=normalize_identifier(
            _first_present(
                payload,
                (
                    "provider",
                    "external_provider",
                ),
            )
        ),
        retry_count=retry_count,
        recoverable=recoverable,
        user_visible=user_visible,
        human_handoff_required=(
            human_handoff_required
        ),
        resolved_at=resolved_at,
        metadata=make_json_safe(
            dict(metadata)
        ),
    )


def normalize_failure_records(
    records: Iterable[Any],
    *,
    business_slug: str = DEFAULT_BUSINESS_SLUG,
    service_name: str = DEFAULT_SERVICE_NAME,
) -> tuple[FailureRecord, ...]:
    failures = [
        normalize_failure_record(
            record,
            business_slug=business_slug,
            service_name=service_name,
        )
        for record in records
    ]

    failures.sort(
        key=lambda failure: (
            failure.occurred_at,
            failure.failure_id,
        )
    )

    return tuple(failures)


# ============================================================
# SECTION 11 - ROOT CAUSE CANDIDATES
# ============================================================

def root_cause_candidates_for_cluster(
    failures: Sequence[FailureRecord],
) -> tuple[str, ...]:
    candidates: list[str] = []

    categories = Counter(
        failure.category.value
        for failure in failures
    )

    top_category = (
        categories.most_common(1)[0][0]
        if categories
        else UNKNOWN_VALUE
    )

    candidates.append(
        f"Primary category: {top_category}."
    )

    exception_types = Counter(
        failure.exception_type
        for failure in failures
        if failure.exception_type
    )

    if exception_types:
        name, count = exception_types.most_common(1)[0]
        candidates.append(
            f"Most frequent exception type is {name} "
            f"({count} occurrences)."
        )

    components = Counter(
        failure.component
        for failure in failures
        if failure.component
    )

    if components:
        component, count = components.most_common(1)[0]
        candidates.append(
            f"Component {component} appears in "
            f"{count} occurrences."
        )

    routes = Counter(
        failure.route
        for failure in failures
        if failure.route
    )

    if routes:
        route, count = routes.most_common(1)[0]
        candidates.append(
            f"Route {route} appears in {count} occurrences."
        )

    first_frames = Counter(
        (
            failure.frames[-1].file_path,
            failure.frames[-1].line_number,
            failure.frames[-1].function_name,
        )
        for failure in failures
        if failure.frames
    )

    if first_frames:
        (
            file_path,
            line_number,
            function_name,
        ), count = first_frames.most_common(1)[0]

        candidates.append(
            "Most common terminal frame: "
            f"{file_path}:{line_number} in {function_name} "
            f"({count} occurrences)."
        )

    return tuple(
        dict.fromkeys(candidates)
    )


# ============================================================
# SECTION 12 - REMEDIATION RECOMMENDATIONS
# ============================================================

def recommendations_for_failure(
    failure: FailureRecord,
) -> tuple[str, ...]:
    recommendations: list[str] = []

    category = failure.category

    if category == FailureCategory.SYNTAX:
        recommendations.extend(
            [
                "Run python -m py_compile on the affected file before committing.",
                "Inspect the first syntax-error line and the line immediately above it.",
                "Prevent generated separators or shell output from being written as Python source.",
            ]
        )

    elif category == FailureCategory.IMPORT:
        recommendations.extend(
            [
                "Confirm the imported module exists in the Git commit deployed to production.",
                "Run a full application import locally before push.",
                "Verify package __init__.py exports do not create circular imports.",
            ]
        )

    elif category == FailureCategory.CONFIGURATION:
        recommendations.extend(
            [
                "Validate required environment variables during a dedicated preflight check.",
                "Use JSON syntax for list-valued Pydantic settings.",
                "Keep production secret validation enabled and update Render environment values instead of weakening code.",
            ]
        )

    elif category == FailureCategory.APPLICATION_STARTUP:
        recommendations.extend(
            [
                "Run the exact production Uvicorn command locally before deployment.",
                "Inspect the first application traceback rather than only the final exit status.",
                "Add startup-health diagnostics that identify the failing subsystem.",
            ]
        )

    elif category == FailureCategory.DATABASE:
        recommendations.extend(
            [
                "Validate database connectivity and schema compatibility before accepting traffic.",
                "Use managed transactions with rollback on failure.",
                "Confirm production uses durable PostgreSQL rather than ephemeral local SQLite.",
            ]
        )

    elif category == FailureCategory.TRANSACTION:
        recommendations.extend(
            [
                "Rollback the active session after any persistence exception.",
                "Keep one transaction boundary around each chat request.",
                "Record commit and rollback outcomes in structured analytics.",
            ]
        )

    elif category == FailureCategory.TIMEOUT:
        recommendations.extend(
            [
                "Add bounded timeouts and exponential backoff.",
                "Separate transient provider failures from permanent validation failures.",
                "Track retry count and provider latency by operation.",
            ]
        )

    elif category == FailureCategory.RATE_LIMIT:
        recommendations.extend(
            [
                "Honor Retry-After when present.",
                "Apply bounded exponential backoff with jitter.",
                "Cache verified static destination data to reduce unnecessary calls.",
            ]
        )

    elif category == FailureCategory.DESTINATION_ROUTING:
        recommendations.extend(
            [
                "Validate every destination URL against the centralized registry.",
                "Test menu, specials, events, gallery, private-events, contact, delivery, pickup, and social intents.",
                "Ensure quick-action buttons submit messages or open official destinations explicitly.",
            ]
        )

    elif category == FailureCategory.NLU:
        recommendations.extend(
            [
                "Capture the normalized input, detected intent, confidence, and clarification decision.",
                "Add regression cases for misspellings and multi-intent requests.",
                "Do not let previous conversation intent lock a new user request.",
            ]
        )

    elif category == FailureCategory.KNOWLEDGE_RETRIEVAL:
        recommendations.extend(
            [
                "Distinguish no verified record from retrieval failure.",
                "Expose source freshness and trust level in diagnostics.",
                "Use safe fallbacks rather than fabricating business facts.",
            ]
        )

    elif category == FailureCategory.RESPONSE_GENERATION:
        recommendations.extend(
            [
                "Validate response actions and metadata before persistence.",
                "Keep unsupported-claim count at zero.",
                "Ensure internal analytics and storage details are not exposed in user-facing copy.",
            ]
        )

    elif category == FailureCategory.DEPLOYMENT:
        recommendations.extend(
            [
                "Verify local and origin/main commit hashes match before deployment.",
                "Confirm Render deployed the intended commit SHA.",
                "Treat green build status separately from successful application startup.",
            ]
        )

    elif category == FailureCategory.EXTERNAL_PROVIDER:
        recommendations.extend(
            [
                "Retain the official Horseshoe Tavern webpage as a safe fallback.",
                "Validate provider URLs periodically.",
                "Track provider failures independently from internal application failures.",
            ]
        )

    elif category == FailureCategory.AUTHENTICATION:
        recommendations.extend(
            [
                "Verify credential presence without logging secret values.",
                "Rotate compromised or uncertain credentials.",
                "Use constant-time comparison where applicable.",
            ]
        )

    elif category == FailureCategory.AUTHORIZATION:
        recommendations.extend(
            [
                "Confirm the authenticated principal has the required permission.",
                "Keep authorization checks server-side.",
                "Log permission decisions without exposing sensitive policy data.",
            ]
        )

    elif category == FailureCategory.DATA_QUALITY:
        recommendations.extend(
            [
                "Validate schema shape before analytics aggregation.",
                "Reject malformed records or quarantine them for review.",
                "Preserve source lineage and timestamps.",
            ]
        )

    else:
        recommendations.extend(
            [
                "Capture a complete structured traceback and operation context.",
                "Reproduce the failure with the smallest deterministic test.",
                "Add a regression test before closing the incident.",
            ]
        )

    if failure.user_visible:
        recommendations.append(
            "Provide a concise user-facing fallback while preserving detailed diagnostics internally."
        )

    if failure.human_handoff_required:
        recommendations.append(
            "Ensure the human-handoff action reaches a monitored contact channel."
        )

    if failure.retry_count >= 3:
        recommendations.append(
            "Stop unbounded retries and escalate after the configured retry limit."
        )

    return tuple(
        dict.fromkeys(recommendations)
    )


# ============================================================
# SECTION 13 - CLUSTERING
# ============================================================

def build_failure_clusters(
    failures: Sequence[FailureRecord],
) -> tuple[FailureCluster, ...]:
    grouped: dict[
        str,
        list[FailureRecord],
    ] = defaultdict(list)

    for failure in failures:
        grouped[failure.fingerprint].append(
            failure
        )

    clusters: list[FailureCluster] = []

    for fingerprint, cluster_failures in grouped.items():
        cluster_failures.sort(
            key=lambda failure: (
                failure.occurred_at,
                failure.failure_id,
            )
        )

        representative = cluster_failures[0]

        resolved_count = sum(
            1
            for failure in cluster_failures
            if (
                failure.status
                == FailureStatus.RESOLVED
                or failure.resolved_at is not None
            )
        )

        recommendations: list[str] = []

        for failure in cluster_failures[:5]:
            recommendations.extend(
                recommendations_for_failure(
                    failure
                )
            )

        clusters.append(
            FailureCluster(
                fingerprint=fingerprint,
                category=representative.category,
                severity=max_severity(
                    failure.severity
                    for failure in cluster_failures
                ),
                exception_type=(
                    representative.exception_type
                ),
                normalized_message=(
                    representative.normalized_message
                ),
                first_seen_at=(
                    cluster_failures[0].occurred_at
                ),
                last_seen_at=(
                    cluster_failures[-1].occurred_at
                ),
                occurrence_count=len(
                    cluster_failures
                ),
                unique_sessions=len(
                    {
                        failure.session_id
                        for failure in cluster_failures
                        if failure.session_id
                    }
                ),
                unique_conversations=len(
                    {
                        failure.conversation_id
                        for failure in cluster_failures
                        if failure.conversation_id
                    }
                ),
                unique_requests=len(
                    {
                        failure.request_id
                        for failure in cluster_failures
                        if failure.request_id
                    }
                ),
                affected_routes=tuple(
                    sorted(
                        {
                            failure.route
                            for failure in cluster_failures
                            if failure.route
                        }
                    )
                ),
                affected_components=tuple(
                    sorted(
                        {
                            failure.component
                            for failure in cluster_failures
                            if failure.component
                        }
                    )
                ),
                recoverable_rate=safe_divide(
                    sum(
                        1
                        for failure in cluster_failures
                        if failure.recoverable
                    ),
                    len(cluster_failures),
                ),
                user_visible_rate=safe_divide(
                    sum(
                        1
                        for failure in cluster_failures
                        if failure.user_visible
                    ),
                    len(cluster_failures),
                ),
                resolved_rate=safe_divide(
                    resolved_count,
                    len(cluster_failures),
                ),
                representative_failure_ids=tuple(
                    failure.failure_id
                    for failure in cluster_failures[:5]
                ),
                root_cause_candidates=(
                    root_cause_candidates_for_cluster(
                        cluster_failures
                    )
                ),
                recommendations=tuple(
                    dict.fromkeys(
                        recommendations
                    )
                ),
            )
        )

    clusters.sort(
        key=lambda cluster: (
            -SEVERITY_ORDER[
                cluster.severity
            ],
            -cluster.occurrence_count,
            cluster.first_seen_at,
        )
    )

    return tuple(clusters)


# ============================================================
# SECTION 14 - TIME-SERIES
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
        return value.strftime(
            "%Y-%m-%d %H:00 UTC"
        )

    if granularity == AggregationGranularity.DAY:
        return value.strftime("%Y-%m-%d")

    if granularity == AggregationGranularity.WEEK:
        year, week, _ = value.isocalendar()
        return f"{year}-W{week:02d}"

    return value.strftime("%Y-%m")


def build_failure_time_series(
    failures: Sequence[FailureRecord],
    *,
    granularity: AggregationGranularity = (
        AggregationGranularity.DAY
    ),
) -> tuple[FailureTimeSeriesPoint, ...]:
    grouped: dict[
        datetime,
        list[FailureRecord],
    ] = defaultdict(list)

    for failure in failures:
        grouped[
            period_start_for(
                failure.occurred_at,
                granularity,
            )
        ].append(failure)

    points: list[FailureTimeSeriesPoint] = []

    for period_start, period_failures in sorted(
        grouped.items()
    ):
        resolved_count = sum(
            1
            for failure in period_failures
            if (
                failure.status
                == FailureStatus.RESOLVED
                or failure.resolved_at
                is not None
            )
        )

        points.append(
            FailureTimeSeriesPoint(
                period_start=period_start,
                period_label=period_label_for(
                    period_start,
                    granularity,
                ),
                failure_count=len(
                    period_failures
                ),
                critical_count=sum(
                    1
                    for failure in period_failures
                    if failure.severity
                    == FailureSeverity.CRITICAL
                ),
                high_count=sum(
                    1
                    for failure in period_failures
                    if failure.severity
                    == FailureSeverity.HIGH
                ),
                open_count=sum(
                    1
                    for failure in period_failures
                    if failure.status
                    in {
                        FailureStatus.OPEN,
                        FailureStatus.INVESTIGATING,
                    }
                ),
                resolved_count=resolved_count,
                unique_fingerprints=len(
                    {
                        failure.fingerprint
                        for failure in period_failures
                    }
                ),
                unique_conversations=len(
                    {
                        failure.conversation_id
                        for failure in period_failures
                        if failure.conversation_id
                    }
                ),
                user_visible_rate=safe_divide(
                    sum(
                        1
                        for failure in period_failures
                        if failure.user_visible
                    ),
                    len(period_failures),
                ),
                recovery_rate=safe_divide(
                    resolved_count,
                    len(period_failures),
                ),
                average_latency_ms=safe_mean(
                    failure.latency_ms
                    for failure in period_failures
                ),
            )
        )

    return tuple(points)


# ============================================================
# SECTION 15 - RELIABILITY METRICS
# ============================================================

def calculate_time_between_failures(
    failures: Sequence[FailureRecord],
) -> tuple[float, ...]:
    timestamps = sorted(
        failure.occurred_at
        for failure in failures
    )

    return tuple(
        max(
            (
                timestamps[index]
                - timestamps[index - 1]
            ).total_seconds(),
            0.0,
        )
        for index in range(
            1,
            len(timestamps),
        )
    )


def calculate_recovery_durations(
    failures: Sequence[FailureRecord],
) -> tuple[float, ...]:
    durations: list[float] = []

    for failure in failures:
        if failure.resolved_at is None:
            continue

        durations.append(
            max(
                (
                    failure.resolved_at
                    - failure.occurred_at
                ).total_seconds(),
                0.0,
            )
        )

    return tuple(durations)


def build_reliability_metrics(
    failures: Sequence[FailureRecord],
    *,
    total_operation_count: int | None = None,
    target_availability: float = 0.999,
) -> FailureReliabilityMetrics:
    total_failures = len(failures)

    open_failures = sum(
        1
        for failure in failures
        if failure.status
        in {
            FailureStatus.OPEN,
            FailureStatus.INVESTIGATING,
        }
    )

    resolved_failures = sum(
        1
        for failure in failures
        if (
            failure.status
            == FailureStatus.RESOLVED
            or failure.resolved_at is not None
        )
    )

    critical_failures = sum(
        1
        for failure in failures
        if failure.severity
        == FailureSeverity.CRITICAL
    )

    high_failures = sum(
        1
        for failure in failures
        if failure.severity
        == FailureSeverity.HIGH
    )

    user_visible_failures = sum(
        1
        for failure in failures
        if failure.user_visible
    )

    recoverable_failures = sum(
        1
        for failure in failures
        if failure.recoverable
    )

    operations = (
        max(
            int(total_operation_count),
            total_failures,
        )
        if total_operation_count
        is not None
        else total_failures
    )

    failure_rate = safe_divide(
        total_failures,
        operations,
    )

    availability_estimate = round(
        max(
            0.0,
            min(
                1.0,
                1.0 - failure_rate,
            ),
        ),
        6,
    )

    allowed_error_rate = max(
        1.0 - target_availability,
        1e-9,
    )

    error_budget_consumed = round(
        failure_rate / allowed_error_rate,
        6,
    )

    time_between = calculate_time_between_failures(
        failures
    )

    recovery_durations = (
        calculate_recovery_durations(
            failures
        )
    )

    latencies = [
        failure.latency_ms
        for failure in failures
        if failure.latency_ms is not None
    ]

    return FailureReliabilityMetrics(
        total_failures=total_failures,
        open_failures=open_failures,
        resolved_failures=resolved_failures,
        critical_failures=critical_failures,
        high_failures=high_failures,
        user_visible_failures=(
            user_visible_failures
        ),
        recoverable_failures=(
            recoverable_failures
        ),
        failure_rate=failure_rate,
        open_rate=safe_divide(
            open_failures,
            total_failures,
        ),
        resolution_rate=safe_divide(
            resolved_failures,
            total_failures,
        ),
        recoverable_rate=safe_divide(
            recoverable_failures,
            total_failures,
        ),
        user_visible_rate=safe_divide(
            user_visible_failures,
            total_failures,
        ),
        mean_time_between_failures_seconds=(
            safe_mean(
                time_between,
                precision=3,
            )
        ),
        median_time_between_failures_seconds=(
            percentile(
                time_between,
                50,
            )
        ),
        mean_time_to_recovery_seconds=(
            safe_mean(
                recovery_durations,
                precision=3,
            )
        ),
        median_time_to_recovery_seconds=(
            percentile(
                recovery_durations,
                50,
            )
        ),
        average_latency_ms=safe_mean(
            latencies,
            precision=3,
        ),
        p95_latency_ms=percentile(
            latencies,
            95,
        ),
        p99_latency_ms=percentile(
            latencies,
            99,
        ),
        error_budget_consumed=(
            error_budget_consumed
        ),
        availability_estimate=(
            availability_estimate
        ),
    )


# ============================================================
# SECTION 16 - ANOMALY DETECTION
# ============================================================

def detect_failure_anomalies(
    points: Sequence[FailureTimeSeriesPoint],
    *,
    metric_name: str = "failure_count",
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    minimum_baseline_points: int = (
        DEFAULT_MINIMUM_BASELINE_POINTS
    ),
) -> tuple[FailureAnomaly, ...]:
    supported = {
        "failure_count",
        "critical_count",
        "high_count",
        "open_count",
        "resolved_count",
        "unique_fingerprints",
        "unique_conversations",
        "user_visible_rate",
        "recovery_rate",
        "average_latency_ms",
    }

    if metric_name not in supported:
        raise ValueError(
            f"Unsupported failure anomaly metric: {metric_name}"
        )

    anomalies: list[FailureAnomaly] = []

    for index, point in enumerate(points):
        if index < minimum_baseline_points:
            continue

        baseline_values = [
            float(
                getattr(
                    baseline_point,
                    metric_name,
                )
            )
            for baseline_point in points[:index]
        ]

        if len(baseline_values) < minimum_baseline_points:
            continue

        baseline_mean = statistics.fmean(
            baseline_values
        )

        baseline_std = statistics.pstdev(
            baseline_values
        )

        if baseline_std == 0:
            continue

        observed = float(
            getattr(
                point,
                metric_name,
            )
        )

        z_score = (
            observed - baseline_mean
        ) / baseline_std

        if abs(z_score) < z_threshold:
            continue

        severity = (
            FailureSeverity.CRITICAL
            if abs(z_score)
            >= z_threshold * 2
            else FailureSeverity.HIGH
        )

        direction = (
            "above"
            if z_score > 0
            else "below"
        )

        anomalies.append(
            FailureAnomaly(
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
                    baseline_std,
                    6,
                ),
                z_score=round(
                    z_score,
                    6,
                ),
                severity=severity,
                explanation=(
                    f"{metric_name} was {direction} the historical "
                    f"baseline by {abs(z_score):.2f} standard deviations."
                ),
            )
        )

    return tuple(anomalies)


def detect_default_failure_anomalies(
    points: Sequence[FailureTimeSeriesPoint],
) -> tuple[FailureAnomaly, ...]:
    anomalies: list[FailureAnomaly] = []

    for metric_name in (
        "failure_count",
        "critical_count",
        "open_count",
        "user_visible_rate",
        "average_latency_ms",
    ):
        anomalies.extend(
            detect_failure_anomalies(
                points,
                metric_name=metric_name,
            )
        )

    anomalies.sort(
        key=lambda anomaly: (
            anomaly.period_label,
            -abs(anomaly.z_score),
            anomaly.metric_name,
        )
    )

    return tuple(anomalies)


# ============================================================
# SECTION 17 - GLOBAL RECOMMENDATIONS
# ============================================================

def build_global_recommendations(
    failures: Sequence[FailureRecord],
    clusters: Sequence[FailureCluster],
    reliability: FailureReliabilityMetrics,
) -> tuple[str, ...]:
    recommendations: list[str] = []

    for cluster in clusters[:10]:
        recommendations.extend(
            cluster.recommendations[:3]
        )

    if reliability.open_failures > 0:
        recommendations.append(
            "Review and assign all open failure clusters before the next release."
        )

    if reliability.user_visible_rate > 0.10:
        recommendations.append(
            "Reduce user-visible failures by adding safe fallbacks and preflight validation."
        )

    if reliability.recoverable_rate < 0.50:
        recommendations.append(
            "Classify transient failures explicitly and add bounded retry handling where safe."
        )

    if reliability.p95_latency_ms > 2000:
        recommendations.append(
            "Profile slow failing operations and separate provider latency from internal processing latency."
        )

    if reliability.error_budget_consumed > 1.0:
        recommendations.append(
            "The estimated error budget is exceeded; prioritize reliability work before feature expansion."
        )

    critical_categories = {
        failure.category
        for failure in failures
        if failure.severity
        == FailureSeverity.CRITICAL
    }

    if FailureCategory.DEPLOYMENT in critical_categories:
        recommendations.append(
            "Require application import and exact production-start-command verification before every deployment."
        )

    if FailureCategory.CONFIGURATION in critical_categories:
        recommendations.append(
            "Add a deployment preflight command that validates all required Render environment variables."
        )

    return tuple(
        dict.fromkeys(
            recommendations
        )
    )


# ============================================================
# SECTION 18 - ENGINE
# ============================================================

class FailureAnalyticsEngine:
    """
    Deterministic failure analytics engine.
    """

    def __init__(
        self,
        *,
        business_slug: str = DEFAULT_BUSINESS_SLUG,
        service_name: str = DEFAULT_SERVICE_NAME,
        environment: str = UNKNOWN_VALUE,
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
        self.service_name = (
            normalize_identifier(
                service_name
            )
            or DEFAULT_SERVICE_NAME
        )
        self.environment = normalize_label(
            environment
        )
        self.granularity = granularity

    def normalize(
        self,
        records: Iterable[Any],
    ) -> tuple[FailureRecord, ...]:
        return normalize_failure_records(
            records,
            business_slug=self.business_slug,
            service_name=self.service_name,
        )

    def build_report(
        self,
        records: Iterable[Any],
        *,
        generated_at: datetime | None = None,
        total_operation_count: int | None = None,
        target_availability: float = 0.999,
        metadata: Mapping[str, Any] | None = None,
    ) -> FailureAnalyticsReport:
        failures = self.normalize(records)

        if len(failures) > MAXIMUM_REPORT_FAILURES:
            raise ValueError(
                "Failure report exceeds the configured maximum record count."
            )

        reliability = build_reliability_metrics(
            failures,
            total_operation_count=(
                total_operation_count
            ),
            target_availability=(
                target_availability
            ),
        )

        clusters = build_failure_clusters(
            failures
        )

        time_series = build_failure_time_series(
            failures,
            granularity=self.granularity,
        )

        anomalies = (
            detect_default_failure_anomalies(
                time_series
            )
        )

        category_distribution = Counter(
            failure.category.value
            for failure in failures
        )

        severity_distribution = Counter(
            failure.severity.value
            for failure in failures
        )

        status_distribution = Counter(
            failure.status.value
            for failure in failures
        )

        source_distribution = Counter(
            failure.source.value
            for failure in failures
        )

        exception_types = Counter(
            failure.exception_type
            for failure in failures
            if failure.exception_type
        )

        routes = Counter(
            failure.route
            for failure in failures
            if failure.route
        )

        components = Counter(
            failure.component
            for failure in failures
            if failure.component
        )

        providers = Counter(
            failure.provider
            for failure in failures
            if failure.provider
        )

        recommendations = (
            build_global_recommendations(
                failures,
                clusters,
                reliability,
            )
        )

        return FailureAnalyticsReport(
            generated_at=coerce_datetime(
                generated_at or utc_now()
            ),
            business_slug=self.business_slug,
            service_name=self.service_name,
            environment=self.environment,
            reliability=reliability,
            category_distribution=dict(
                category_distribution
            ),
            severity_distribution=dict(
                severity_distribution
            ),
            status_distribution=dict(
                status_distribution
            ),
            source_distribution=dict(
                source_distribution
            ),
            top_exception_types=tuple(
                exception_types.most_common(20)
            ),
            top_routes=tuple(
                routes.most_common(20)
            ),
            top_components=tuple(
                components.most_common(20)
            ),
            top_providers=tuple(
                providers.most_common(20)
            ),
            clusters=clusters,
            time_series=time_series,
            anomalies=anomalies,
            recommendations=recommendations,
            metadata={
                **make_json_safe(
                    dict(metadata or {})
                ),
                "privacy": {
                    "secrets_redacted": True,
                    "stacktraces_excluded_by_default": True,
                    "raw_user_messages_are_not_business_facts": True,
                },
                "target_availability": (
                    target_availability
                ),
                "total_operation_count": (
                    total_operation_count
                ),
            },
            service_version=(
                FAILURE_ANALYTICS_VERSION
            ),
            service_phase=(
                FAILURE_ANALYTICS_PHASE
            ),
        )


# ============================================================
# SECTION 19 - CONVENIENCE API
# ============================================================

def analyze_failures(
    records: Iterable[Any],
    *,
    business_slug: str = DEFAULT_BUSINESS_SLUG,
    service_name: str = DEFAULT_SERVICE_NAME,
    environment: str = UNKNOWN_VALUE,
    granularity: AggregationGranularity = (
        AggregationGranularity.DAY
    ),
    generated_at: datetime | None = None,
    total_operation_count: int | None = None,
    target_availability: float = 0.999,
    metadata: Mapping[str, Any] | None = None,
) -> FailureAnalyticsReport:
    return FailureAnalyticsEngine(
        business_slug=business_slug,
        service_name=service_name,
        environment=environment,
        granularity=granularity,
    ).build_report(
        records,
        generated_at=generated_at,
        total_operation_count=(
            total_operation_count
        ),
        target_availability=(
            target_availability
        ),
        metadata=metadata,
    )


def failure_report_json(
    records: Iterable[Any],
    *,
    business_slug: str = DEFAULT_BUSINESS_SLUG,
    service_name: str = DEFAULT_SERVICE_NAME,
    environment: str = UNKNOWN_VALUE,
    indent: int = 2,
) -> str:
    report = analyze_failures(
        records,
        business_slug=business_slug,
        service_name=service_name,
        environment=environment,
    )

    return json.dumps(
        report.as_dict(),
        indent=indent,
        sort_keys=True,
        ensure_ascii=False,
    )


def failure_health_summary(
    report: FailureAnalyticsReport,
) -> dict[str, Any]:
    status = "healthy"
    reasons: list[str] = []

    reliability = report.reliability

    if reliability.critical_failures > 0:
        status = "critical"
        reasons.append(
            "Critical failures are present."
        )

    elif reliability.open_failures > 0:
        status = "warning"
        reasons.append(
            "Open failures require review."
        )

    if reliability.user_visible_rate > 0.10:
        if status == "healthy":
            status = "warning"
        reasons.append(
            "More than 10% of failures were user-visible."
        )

    if reliability.error_budget_consumed > 1.0:
        status = "critical"
        reasons.append(
            "Estimated error budget is exceeded."
        )

    return {
        "status": status,
        "reasons": reasons,
        "total_failures": (
            reliability.total_failures
        ),
        "open_failures": (
            reliability.open_failures
        ),
        "critical_failures": (
            reliability.critical_failures
        ),
        "availability_estimate": (
            reliability.availability_estimate
        ),
        "error_budget_consumed": (
            reliability.error_budget_consumed
        ),
        "generated_at": (
            report.generated_at.isoformat()
        ),
        "service_version": (
            report.service_version
        ),
        "service_phase": (
            report.service_phase
        ),
    }


# ============================================================
# SECTION 20 - EXCEPTION CAPTURE HELPERS
# ============================================================

def failure_from_exception(
    exception: BaseException,
    *,
    occurred_at: datetime | None = None,
    business_slug: str = DEFAULT_BUSINESS_SLUG,
    service_name: str = DEFAULT_SERVICE_NAME,
    environment: str = UNKNOWN_VALUE,
    component: str | None = None,
    operation: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    conversation_id: str | None = None,
    message_id: str | None = None,
    route: str | None = None,
    method: str | None = None,
    status_code: int | None = None,
    latency_ms: float | None = None,
    recoverable: bool = False,
    user_visible: bool = False,
    human_handoff_required: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> FailureRecord:
    stacktrace = "".join(
        traceback.format_exception(
            type(exception),
            exception,
            exception.__traceback__,
        )
    )

    return normalize_failure_record(
        {
            "failure_id": (
                f"failure_{uuid.uuid4().hex}"
            ),
            "occurred_at": (
                occurred_at or utc_now()
            ),
            "source": (
                FailureSource.EXCEPTION.value
            ),
            "exception": exception,
            "exception_type": (
                exception.__class__.__name__
            ),
            "message": str(exception),
            "stacktrace": stacktrace,
            "business_slug": business_slug,
            "service_name": service_name,
            "environment": environment,
            "component": component,
            "operation": operation,
            "request_id": request_id,
            "session_id": session_id,
            "conversation_id": (
                conversation_id
            ),
            "message_id": message_id,
            "route": route,
            "method": method,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "recoverable": recoverable,
            "user_visible": user_visible,
            "human_handoff_required": (
                human_handoff_required
            ),
            "metadata": dict(
                metadata or {}
            ),
        },
        business_slug=business_slug,
        service_name=service_name,
    )


# ============================================================
# SECTION 21 - SELF-VALIDATION
# ============================================================

def _sample_failures() -> list[dict[str, Any]]:
    base = datetime(
        2026,
        7,
        22,
        20,
        0,
        tzinfo=timezone.utc,
    )

    import_stack = """
Traceback (most recent call last):
  File "/opt/render/project/src/app/main.py", line 54, in <module>
    from app.services.destination_routing import match_destinations
ModuleNotFoundError: No module named 'app.services.destination_routing'
""".strip()

    syntax_stack = """
Traceback (most recent call last):
  File "/opt/render/project/src/app/services/response_service.py", line 2
    ==========================================================
    ^^
SyntaxError: invalid syntax
""".strip()

    return [
        {
            "failure_id": "failure_001",
            "occurred_at": base.isoformat(),
            "source": "deployment_log",
            "message": (
                "ModuleNotFoundError: No module named "
                "'app.services.destination_routing'"
            ),
            "stacktrace": import_stack,
            "service_name": DEFAULT_SERVICE_NAME,
            "environment": "production",
            "component": "app.main",
            "operation": "startup",
            "deployment_id": "deploy_001",
            "commit_sha": "abc1234",
            "recoverable": False,
            "user_visible": True,
        },
        {
            "failure_id": "failure_002",
            "occurred_at": (
                base + timedelta(minutes=10)
            ).isoformat(),
            "source": "deployment_log",
            "message": (
                "ModuleNotFoundError: No module named "
                "'app.services.destination_routing'"
            ),
            "stacktrace": import_stack,
            "service_name": DEFAULT_SERVICE_NAME,
            "environment": "production",
            "component": "app.main",
            "operation": "startup",
            "deployment_id": "deploy_002",
            "commit_sha": "def5678",
            "recoverable": False,
            "user_visible": True,
        },
        {
            "failure_id": "failure_003",
            "occurred_at": (
                base + timedelta(hours=1)
            ).isoformat(),
            "source": "deployment_log",
            "message": "SyntaxError: invalid syntax",
            "stacktrace": syntax_stack,
            "service_name": DEFAULT_SERVICE_NAME,
            "environment": "production",
            "component": (
                "app.services.response_service"
            ),
            "operation": "import",
            "deployment_id": "deploy_003",
            "commit_sha": "987abcd",
            "recoverable": False,
            "user_visible": True,
            "resolved_at": (
                base
                + timedelta(hours=1, minutes=20)
            ).isoformat(),
            "status": "resolved",
        },
        {
            "failure_id": "failure_004",
            "occurred_at": (
                base + timedelta(days=1)
            ).isoformat(),
            "source": "api_response",
            "message": (
                "Upstream provider timed out after 5 seconds"
            ),
            "exception_type": "ReadTimeout",
            "service_name": DEFAULT_SERVICE_NAME,
            "environment": "production",
            "component": "knowledge_service",
            "operation": "retrieve",
            "route": "/api/chat/message",
            "status_code": 504,
            "latency_ms": 5000,
            "recoverable": True,
            "retry_count": 2,
            "user_visible": False,
        },
        {
            "failure_id": "failure_005",
            "occurred_at": (
                base + timedelta(days=1, minutes=3)
            ).isoformat(),
            "source": "api_response",
            "message": (
                "Upstream provider timed out after 5 seconds"
            ),
            "exception_type": "ReadTimeout",
            "service_name": DEFAULT_SERVICE_NAME,
            "environment": "production",
            "component": "knowledge_service",
            "operation": "retrieve",
            "route": "/api/chat/message",
            "status_code": 504,
            "latency_ms": 5000,
            "recoverable": True,
            "retry_count": 2,
            "user_visible": False,
        },
    ]


def validate_failure_analytics_module() -> dict[str, Any]:
    sample = _sample_failures()

    engine = FailureAnalyticsEngine(
        environment="production"
    )

    failures = engine.normalize(sample)

    report = engine.build_report(
        sample,
        generated_at=datetime(
            2026,
            7,
            24,
            tzinfo=timezone.utc,
        ),
        total_operation_count=1000,
        metadata={
            "validation": True,
        },
    )

    import_failures = [
        failure
        for failure in failures
        if failure.category
        == FailureCategory.IMPORT
    ]

    syntax_failures = [
        failure
        for failure in failures
        if failure.category
        == FailureCategory.SYNTAX
    ]

    timeout_failures = [
        failure
        for failure in failures
        if failure.category
        == FailureCategory.TIMEOUT
    ]

    secret_text = redact_sensitive_text(
        "password=supersecret "
        "Authorization: Bearer abc.def.ghi "
        "email@example.com"
    ) or ""

    checks = {
        "normalized_failure_count": (
            len(failures) == len(sample)
        ),
        "report_failure_count": (
            report.reliability.total_failures
            == len(sample)
        ),
        "import_category_detected": (
            len(import_failures) == 2
        ),
        "syntax_category_detected": (
            len(syntax_failures) == 1
        ),
        "timeout_category_detected": (
            len(timeout_failures) == 2
        ),
        "clusters_created": (
            len(report.clusters) >= 3
        ),
        "duplicate_import_clustered": any(
            cluster.category
            == FailureCategory.IMPORT
            and cluster.occurrence_count == 2
            for cluster in report.clusters
        ),
        "time_series_present": bool(
            report.time_series
        ),
        "recommendations_present": bool(
            report.recommendations
        ),
        "reliability_metrics_present": (
            report.reliability
            .availability_estimate
            >= 0.0
        ),
        "secret_redacted": (
            "supersecret"
            not in secret_text
            and "abc.def.ghi"
            not in secret_text
            and "email@example.com"
            not in secret_text
        ),
        "json_safe": bool(
            json.dumps(
                report.as_dict()
            )
        ),
        "health_summary_present": bool(
            failure_health_summary(
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
        "failure_count": (
            report.reliability
            .total_failures
        ),
        "cluster_count": len(
            report.clusters
        ),
        "service_version": (
            FAILURE_ANALYTICS_VERSION
        ),
        "service_phase": (
            FAILURE_ANALYTICS_PHASE
        ),
    }


# ============================================================
# SECTION 22 - PUBLIC EXPORTS
# ============================================================

__all__ = [
    "AggregationGranularity",
    "FAILURE_ANALYTICS_PHASE",
    "FAILURE_ANALYTICS_VERSION",
    "FailureAnalyticsEngine",
    "FailureAnalyticsReport",
    "FailureAnomaly",
    "FailureCategory",
    "FailureCluster",
    "FailureFrame",
    "FailureRecord",
    "FailureReliabilityMetrics",
    "FailureSeverity",
    "FailureSource",
    "FailureStatus",
    "FailureTimeSeriesPoint",
    "FailureTrend",
    "analyze_failures",
    "build_failure_clusters",
    "build_failure_fingerprint",
    "build_failure_time_series",
    "build_global_recommendations",
    "build_reliability_metrics",
    "detect_default_failure_anomalies",
    "detect_failure_anomalies",
    "failure_from_exception",
    "failure_health_summary",
    "failure_report_json",
    "infer_failure_category",
    "infer_failure_severity",
    "make_json_safe",
    "normalize_failure_message",
    "normalize_failure_record",
    "normalize_failure_records",
    "parse_traceback_frames",
    "percentile",
    "recommendations_for_failure",
    "redact_sensitive_text",
    "root_cause_candidates_for_cluster",
    "safe_divide",
    "safe_mean",
    "stable_hash",
    "validate_failure_analytics_module",
]


# ============================================================
# SECTION 23 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    validation_report = (
        validate_failure_analytics_module()
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
