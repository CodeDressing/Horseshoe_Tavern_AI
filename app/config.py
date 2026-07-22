# ============================================================
# Exact file location: app/config.py
# Horseshoe Tavern AI
# Phase 1 Part 1.3
# Centralized application configuration and environment control
# ============================================================

"""
Central configuration module for Horseshoe Tavern AI.

This module provides:

- Environment-variable loading
- Development, testing, staging, and production configuration
- Database URL normalization
- CORS and website-origin validation
- Chatbot behavior settings
- Conversation-storage controls
- Controlled-learning settings
- Model-training safeguards
- SEO and structured-data settings
- Security and privacy settings
- Logging and observability configuration
- Render deployment compatibility
- Configuration validation
- Safe public configuration reporting

Business facts such as current hours, menu items, events, pricing,
specials, and private-event packages must not be hard-coded here.
Those facts belong in the verified knowledge and database layers.
"""

from __future__ import annotations

import json
import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urlparse

from pydantic import (
    AliasChoices,
    AnyHttpUrl,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# SECTION 01 - PATH CONSTANTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIRECTORY = PROJECT_ROOT / "app"
DATA_DIRECTORY = PROJECT_ROOT / "data"
LOG_DIRECTORY = PROJECT_ROOT / "logs"
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts"
MODEL_DIRECTORY = APP_DIRECTORY / "models"
TEMPLATE_DIRECTORY = APP_DIRECTORY / "templates"
STATIC_DIRECTORY = APP_DIRECTORY / "static"

DEFAULT_SQLITE_PATH = PROJECT_ROOT / "horseshoe_tavern_ai.db"
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"


# ============================================================
# SECTION 02 - TYPE ALIASES
# ============================================================

EnvironmentName = Literal["development", "testing", "staging", "production"]
LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LearningModeName = Literal["disabled", "collect_only", "reviewed", "automatic"]
ResponseVariationMode = Literal[
    "disabled",
    "deterministic",
    "weighted",
    "adaptive",
]
DatabaseMode = Literal["sync", "async"]
PrivacyMode = Literal["strict", "balanced", "analytics"]


# ============================================================
# SECTION 03 - HELPER FUNCTIONS
# ============================================================

def _split_delimited_value(value: Any) -> list[str]:
    """
    Convert environment-friendly list formats into a clean list.

    Supported input forms:

    - Python list or tuple
    - JSON list
    - Comma-separated string
    - Semicolon-separated string
    - Newline-separated string
    """

    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    raw = str(value).strip()

    if not raw:
        return []

    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    normalized = raw.replace(";", ",").replace("\n", ",")

    return [
        item.strip()
        for item in normalized.split(",")
        if item.strip()
    ]


def _normalize_origin(origin: str) -> str:
    """
    Normalize a browser origin.

    Origins must contain only a scheme and network location.
    Paths and trailing slashes are removed.
    """

    candidate = origin.strip()

    if not candidate:
        raise ValueError("Origin values cannot be empty.")

    parsed = urlparse(candidate)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"Invalid origin scheme for {candidate!r}. "
            "Only http and https are supported."
        )

    if not parsed.netloc:
        raise ValueError(
            f"Invalid origin {candidate!r}: hostname is missing."
        )

    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _normalize_database_url(database_url: str) -> str:
    """
    Normalize database URLs supplied by hosting providers.

    Some providers still emit postgres:// URLs. SQLAlchemy expects
    postgresql:// or an explicitly selected PostgreSQL driver.
    """

    value = database_url.strip()

    if not value:
        return DEFAULT_DATABASE_URL

    if value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://"):]

    return value


def _ensure_runtime_directories() -> None:
    """
    Ensure non-secret runtime directories exist locally.

    Render and similar hosts provide ephemeral filesystems, so these
    directories are used for temporary artifacts, logs, and models only.
    Durable business and conversation data belongs in PostgreSQL.
    """

    for directory in (
        DATA_DIRECTORY,
        LOG_DIRECTORY,
        ARTIFACT_DIRECTORY,
        MODEL_DIRECTORY,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# SECTION 04 - APPLICATION SETTINGS
# ============================================================

class Settings(BaseSettings):
    """
    Validated Horseshoe Tavern AI configuration.

    Environment variables are case-insensitive and may be loaded from
    the project-root .env file during local development.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
        populate_by_name=True,
    )

    # --------------------------------------------------------
    # Application identity
    # --------------------------------------------------------

    app_name: str = Field(
        default="Horseshoe Tavern AI",
        validation_alias=AliasChoices("APP_NAME", "PROJECT_NAME"),
    )

    app_slug: str = Field(
        default="horseshoe-tavern-ai",
        validation_alias=AliasChoices("APP_SLUG", "PROJECT_SLUG"),
    )

    app_version: str = Field(
        default="0.1.0",
        validation_alias=AliasChoices("APP_VERSION", "VERSION"),
    )

    app_description: str = Field(
        default=(
            "Verified AI concierge, private-event lead system, "
            "customer-conversion platform, and learning architecture "
            "for Horseshoe Tavern."
        ),
        validation_alias="APP_DESCRIPTION",
    )

    environment: EnvironmentName = Field(
        default="development",
        validation_alias=AliasChoices(
            "APP_ENV",
            "ENVIRONMENT",
            "ENV",
        ),
    )

    debug: bool = Field(
        default=False,
        validation_alias="DEBUG",
    )

    testing: bool = Field(
        default=False,
        validation_alias="TESTING",
    )

    host: str = Field(
        default="0.0.0.0",
        validation_alias="HOST",
    )

    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("PORT", "APP_PORT"),
    )

    public_base_url: AnyHttpUrl = Field(
        default="http://localhost:8000",
        validation_alias=AliasChoices(
            "PUBLIC_BASE_URL",
            "APP_BASE_URL",
            "RENDER_EXTERNAL_URL",
        ),
    )

    timezone: str = Field(
        default="America/New_York",
        validation_alias=AliasChoices(
            "APP_TIMEZONE",
            "TIMEZONE",
            "TZ",
        ),
    )

    # --------------------------------------------------------
    # Horseshoe Tavern website identity
    # --------------------------------------------------------

    business_name: str = Field(
        default="Horseshoe Tavern",
        validation_alias="BUSINESS_NAME",
    )

    business_slug: str = Field(
        default="horseshoe-tavern",
        validation_alias="BUSINESS_SLUG",
    )

    business_website_url: AnyHttpUrl = Field(
        default="https://www.thehorseshoetavern.com/",
        validation_alias="BUSINESS_WEBSITE_URL",
    )

    canonical_domain: str = Field(
        default="www.thehorseshoetavern.com",
        validation_alias="CANONICAL_DOMAIN",
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    database_url: str = Field(
        default=DEFAULT_DATABASE_URL,
        validation_alias=AliasChoices(
            "DATABASE_URL",
            "POSTGRES_URL",
            "POSTGRESQL_URL",
        ),
    )

    database_mode: DatabaseMode = Field(
        default="sync",
        validation_alias="DATABASE_MODE",
    )

    database_echo: bool = Field(
        default=False,
        validation_alias="DATABASE_ECHO",
    )

    database_pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        validation_alias="DATABASE_POOL_SIZE",
    )

    database_max_overflow: int = Field(
        default=10,
        ge=0,
        le=200,
        validation_alias="DATABASE_MAX_OVERFLOW",
    )

    database_pool_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        validation_alias="DATABASE_POOL_TIMEOUT_SECONDS",
    )

    database_pool_recycle_seconds: int = Field(
        default=1800,
        ge=60,
        le=86400,
        validation_alias="DATABASE_POOL_RECYCLE_SECONDS",
    )

    # --------------------------------------------------------
    # Security
    # --------------------------------------------------------

    secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(48)),
        validation_alias="SECRET_KEY",
    )

    widget_signing_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(48)),
        validation_alias="WIDGET_SIGNING_KEY",
    )

    admin_session_secret: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(48)),
        validation_alias="ADMIN_SESSION_SECRET",
    )

    admin_username: str = Field(
        default="admin",
        min_length=3,
        max_length=128,
        validation_alias="ADMIN_USERNAME",
    )

    admin_password: SecretStr = Field(
        default=SecretStr("replace-me-before-production"),
        validation_alias="ADMIN_PASSWORD",
    )

    access_token_expire_minutes: int = Field(
        default=60,
        ge=5,
        le=10080,
        validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES",
    )

    allowed_hosts: list[str] = Field(
        default_factory=lambda: [
            "localhost",
            "127.0.0.1",
            "testserver",
            "www.thehorseshoetavern.com",
            "thehorseshoetavern.com",
        ],
        validation_alias="ALLOWED_HOSTS",
    )

    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "https://www.thehorseshoetavern.com",
            "https://thehorseshoetavern.com",
        ],
        validation_alias="ALLOWED_ORIGINS",
    )

    cors_allow_credentials: bool = Field(
        default=True,
        validation_alias="CORS_ALLOW_CREDENTIALS",
    )

    cors_allowed_methods: list[str] = Field(
        default_factory=lambda: [
            "GET",
            "POST",
            "OPTIONS",
        ],
        validation_alias="CORS_ALLOWED_METHODS",
    )

    cors_allowed_headers: list[str] = Field(
        default_factory=lambda: [
            "Accept",
            "Authorization",
            "Content-Type",
            "Origin",
            "X-Requested-With",
            "X-Request-ID",
            "X-Widget-Signature",
        ],
        validation_alias="CORS_ALLOWED_HEADERS",
    )

    rate_limit_enabled: bool = Field(
        default=True,
        validation_alias="RATE_LIMIT_ENABLED",
    )

    chat_rate_limit: str = Field(
        default="30/minute",
        validation_alias="CHAT_RATE_LIMIT",
    )

    lead_rate_limit: str = Field(
        default="8/hour",
        validation_alias="LEAD_RATE_LIMIT",
    )

    maximum_message_characters: int = Field(
        default=3000,
        ge=100,
        le=20000,
        validation_alias="MAXIMUM_MESSAGE_CHARACTERS",
    )

    maximum_conversation_messages: int = Field(
        default=100,
        ge=5,
        le=1000,
        validation_alias="MAXIMUM_CONVERSATION_MESSAGES",
    )

    # --------------------------------------------------------
    # Conversation storage and memory
    # --------------------------------------------------------

    store_conversations: bool = Field(
        default=True,
        validation_alias="STORE_CONVERSATIONS",
    )

    store_original_messages: bool = Field(
        default=True,
        validation_alias="STORE_ORIGINAL_MESSAGES",
    )

    store_normalized_messages: bool = Field(
        default=True,
        validation_alias="STORE_NORMALIZED_MESSAGES",
    )

    store_model_metadata: bool = Field(
        default=True,
        validation_alias="STORE_MODEL_METADATA",
    )

    store_retrieval_sources: bool = Field(
        default=True,
        validation_alias="STORE_RETRIEVAL_SOURCES",
    )

    session_memory_enabled: bool = Field(
        default=True,
        validation_alias="SESSION_MEMORY_ENABLED",
    )

    customer_memory_enabled: bool = Field(
        default=False,
        validation_alias="CUSTOMER_MEMORY_ENABLED",
    )

    conversation_retention_days: int = Field(
        default=365,
        ge=1,
        le=3650,
        validation_alias="CONVERSATION_RETENTION_DAYS",
    )

    anonymous_session_retention_days: int = Field(
        default=90,
        ge=1,
        le=730,
        validation_alias="ANONYMOUS_SESSION_RETENTION_DAYS",
    )

    redact_sensitive_data: bool = Field(
        default=True,
        validation_alias="REDACT_SENSITIVE_DATA",
    )

    privacy_mode: PrivacyMode = Field(
        default="strict",
        validation_alias="PRIVACY_MODE",
    )

    # --------------------------------------------------------
    # Natural-language understanding
    # --------------------------------------------------------

    nlu_enabled: bool = Field(
        default=True,
        validation_alias="NLU_ENABLED",
    )

    spelling_correction_enabled: bool = Field(
        default=True,
        validation_alias="SPELLING_CORRECTION_ENABLED",
    )

    fuzzy_matching_enabled: bool = Field(
        default=True,
        validation_alias="FUZZY_MATCHING_ENABLED",
    )

    phonetic_matching_enabled: bool = Field(
        default=True,
        validation_alias="PHONETIC_MATCHING_ENABLED",
    )

    semantic_matching_enabled: bool = Field(
        default=False,
        validation_alias="SEMANTIC_MATCHING_ENABLED",
    )

    fuzzy_match_threshold: float = Field(
        default=0.82,
        ge=0.0,
        le=1.0,
        validation_alias="FUZZY_MATCH_THRESHOLD",
    )

    fuzzy_ambiguity_margin: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        validation_alias="FUZZY_AMBIGUITY_MARGIN",
    )

    minimum_intent_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        validation_alias="MINIMUM_INTENT_CONFIDENCE",
    )

    minimum_answer_confidence: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
        validation_alias="MINIMUM_ANSWER_CONFIDENCE",
    )

    default_language: str = Field(
        default="en",
        min_length=2,
        max_length=12,
        validation_alias="DEFAULT_LANGUAGE",
    )

    # --------------------------------------------------------
    # Response behavior
    # --------------------------------------------------------

    response_variation_mode: ResponseVariationMode = Field(
        default="weighted",
        validation_alias="RESPONSE_VARIATION_MODE",
    )

    avoid_repeated_responses: bool = Field(
        default=True,
        validation_alias="AVOID_REPEATED_RESPONSES",
    )

    response_history_window: int = Field(
        default=10,
        ge=1,
        le=100,
        validation_alias="RESPONSE_HISTORY_WINDOW",
    )

    maximum_response_characters: int = Field(
        default=4000,
        ge=100,
        le=20000,
        validation_alias="MAXIMUM_RESPONSE_CHARACTERS",
    )

    include_source_metadata: bool = Field(
        default=True,
        validation_alias="INCLUDE_SOURCE_METADATA",
    )

    require_verified_business_facts: bool = Field(
        default=True,
        validation_alias="REQUIRE_VERIFIED_BUSINESS_FACTS",
    )

    allow_unverified_business_claims: bool = Field(
        default=False,
        validation_alias="ALLOW_UNVERIFIED_BUSINESS_CLAIMS",
    )

    human_handoff_enabled: bool = Field(
        default=True,
        validation_alias="HUMAN_HANDOFF_ENABLED",
    )

    # --------------------------------------------------------
    # Controlled learning
    # --------------------------------------------------------

    learning_mode: LearningModeName = Field(
        default="reviewed",
        validation_alias="LEARNING_MODE",
    )

    collect_training_candidates: bool = Field(
        default=True,
        validation_alias="COLLECT_TRAINING_CANDIDATES",
    )

    require_human_training_review: bool = Field(
        default=True,
        validation_alias="REQUIRE_HUMAN_TRAINING_REVIEW",
    )

    allow_raw_input_model_training: bool = Field(
        default=False,
        validation_alias="ALLOW_RAW_INPUT_MODEL_TRAINING",
    )

    allow_customer_messages_to_update_business_facts: bool = Field(
        default=False,
        validation_alias=(
            "ALLOW_CUSTOMER_MESSAGES_TO_UPDATE_BUSINESS_FACTS"
        ),
    )

    minimum_training_examples_per_intent: int = Field(
        default=20,
        ge=2,
        le=100000,
        validation_alias="MINIMUM_TRAINING_EXAMPLES_PER_INTENT",
    )

    candidate_model_minimum_improvement: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        validation_alias="CANDIDATE_MODEL_MINIMUM_IMPROVEMENT",
    )

    candidate_model_maximum_regression: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        validation_alias="CANDIDATE_MODEL_MAXIMUM_REGRESSION",
    )

    model_promotion_requires_approval: bool = Field(
        default=True,
        validation_alias="MODEL_PROMOTION_REQUIRES_APPROVAL",
    )

    poisoning_detection_enabled: bool = Field(
        default=True,
        validation_alias="POISONING_DETECTION_ENABLED",
    )

    drift_detection_enabled: bool = Field(
        default=True,
        validation_alias="DRIFT_DETECTION_ENABLED",
    )

    # --------------------------------------------------------
    # Optional generative AI
    # --------------------------------------------------------

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )

    generative_ai_enabled: bool = Field(
        default=False,
        validation_alias="GENERATIVE_AI_ENABLED",
    )

    generative_model_name: str = Field(
        default="",
        validation_alias="GENERATIVE_MODEL_NAME",
    )

    generative_temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        validation_alias="GENERATIVE_TEMPERATURE",
    )

    generative_max_output_tokens: int = Field(
        default=500,
        ge=50,
        le=8000,
        validation_alias="GENERATIVE_MAX_OUTPUT_TOKENS",
    )

    # --------------------------------------------------------
    # SEO and schema
    # --------------------------------------------------------

    seo_enabled: bool = Field(
        default=True,
        validation_alias="SEO_ENABLED",
    )

    schema_org_enabled: bool = Field(
        default=True,
        validation_alias="SCHEMA_ORG_ENABLED",
    )

    sitemap_enabled: bool = Field(
        default=True,
        validation_alias="SITEMAP_ENABLED",
    )

    robots_enabled: bool = Field(
        default=True,
        validation_alias="ROBOTS_ENABLED",
    )

    seo_indexing_enabled: bool = Field(
        default=False,
        validation_alias="SEO_INDEXING_ENABLED",
    )

    canonical_base_url: AnyHttpUrl = Field(
        default="https://www.thehorseshoetavern.com/",
        validation_alias="CANONICAL_BASE_URL",
    )

    # --------------------------------------------------------
    # Analytics
    # --------------------------------------------------------

    analytics_enabled: bool = Field(
        default=True,
        validation_alias="ANALYTICS_ENABLED",
    )

    conversion_tracking_enabled: bool = Field(
        default=True,
        validation_alias="CONVERSION_TRACKING_ENABLED",
    )

    attribution_tracking_enabled: bool = Field(
        default=True,
        validation_alias="ATTRIBUTION_TRACKING_ENABLED",
    )

    google_analytics_measurement_id: str | None = Field(
        default=None,
        validation_alias="GOOGLE_ANALYTICS_MEASUREMENT_ID",
    )

    # --------------------------------------------------------
    # Logging and monitoring
    # --------------------------------------------------------

    log_level: LogLevelName = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
    )

    json_logging: bool = Field(
        default=False,
        validation_alias="JSON_LOGGING",
    )

    log_requests: bool = Field(
        default=True,
        validation_alias="LOG_REQUESTS",
    )

    log_responses: bool = Field(
        default=False,
        validation_alias="LOG_RESPONSES",
    )

    expose_metrics: bool = Field(
        default=True,
        validation_alias="EXPOSE_METRICS",
    )

    health_check_database: bool = Field(
        default=True,
        validation_alias="HEALTH_CHECK_DATABASE",
    )

    # --------------------------------------------------------
    # Static constants
    # --------------------------------------------------------

    project_root: ClassVar[Path] = PROJECT_ROOT
    app_directory: ClassVar[Path] = APP_DIRECTORY
    data_directory: ClassVar[Path] = DATA_DIRECTORY
    log_directory: ClassVar[Path] = LOG_DIRECTORY
    artifact_directory: ClassVar[Path] = ARTIFACT_DIRECTORY
    model_directory: ClassVar[Path] = MODEL_DIRECTORY
    template_directory: ClassVar[Path] = TEMPLATE_DIRECTORY
    static_directory: ClassVar[Path] = STATIC_DIRECTORY

    # ========================================================
    # SECTION 05 - FIELD NORMALIZERS
    # ========================================================

    @field_validator(
        "allowed_hosts",
        "cors_allowed_methods",
        "cors_allowed_headers",
        mode="before",
    )
    @classmethod
    def parse_string_lists(cls, value: Any) -> list[str]:
        return _split_delimited_value(value)

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: Any) -> list[str]:
        origins = _split_delimited_value(value)

        normalized: list[str] = []

        for origin in origins:
            candidate = _normalize_origin(origin)

            if candidate not in normalized:
                normalized.append(candidate)

        return normalized

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: Any) -> str:
        return _normalize_database_url(
            str(value) if value is not None else ""
        )

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: Any) -> str:
        candidate = str(value or "development").strip().lower()

        aliases = {
            "dev": "development",
            "test": "testing",
            "stage": "staging",
            "prod": "production",
        }

        return aliases.get(candidate, candidate)

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: Any) -> str:
        return str(value or "INFO").strip().upper()

    @field_validator(
        "app_slug",
        "business_slug",
        mode="before",
    )
    @classmethod
    def normalize_slugs(cls, value: Any) -> str:
        candidate = str(value or "").strip().lower()
        candidate = candidate.replace("_", "-").replace(" ", "-")

        while "--" in candidate:
            candidate = candidate.replace("--", "-")

        candidate = candidate.strip("-")

        if not candidate:
            raise ValueError("Slug values cannot be empty.")

        return candidate

    @field_validator("canonical_domain", mode="before")
    @classmethod
    def normalize_domain(cls, value: Any) -> str:
        candidate = str(value or "").strip().lower()

        if "://" in candidate:
            candidate = urlparse(candidate).netloc

        candidate = candidate.strip("/")

        if not candidate:
            raise ValueError("Canonical domain cannot be empty.")

        return candidate

    # ========================================================
    # SECTION 06 - CROSS-FIELD VALIDATION
    # ========================================================

    @model_validator(mode="after")
    def validate_environment_rules(self) -> "Settings":
        if self.testing:
            self.environment = "testing"

        if self.environment == "development" and os.getenv("DEBUG") is None:
            self.debug = True

        if self.environment in {"staging", "production"}:
            if self.debug:
                raise ValueError(
                    "DEBUG must be false in staging and production."
                )

            if (
                self.admin_password.get_secret_value()
                == "replace-me-before-production"
            ):
                raise ValueError(
                    "ADMIN_PASSWORD must be replaced before staging "
                    "or production startup."
                )

            if len(self.secret_key.get_secret_value()) < 32:
                raise ValueError(
                    "SECRET_KEY must contain at least 32 characters "
                    "in staging and production."
                )

            if len(self.widget_signing_key.get_secret_value()) < 32:
                raise ValueError(
                    "WIDGET_SIGNING_KEY must contain at least "
                    "32 characters in staging and production."
                )

        if (
            self.allow_customer_messages_to_update_business_facts
            and self.require_verified_business_facts
        ):
            raise ValueError(
                "Customer messages cannot update verified business facts."
            )

        if self.allow_raw_input_model_training:
            raise ValueError(
                "Raw public inputs may not be used directly for model "
                "training. Build reviewed training examples instead."
            )

        if self.learning_mode == "automatic":
            if self.require_human_training_review:
                raise ValueError(
                    "Automatic learning mode conflicts with required "
                    "human training review."
                )

            if self.model_promotion_requires_approval:
                raise ValueError(
                    "Automatic learning mode conflicts with required "
                    "model-promotion approval."
                )

        if self.generative_ai_enabled:
            if self.openai_api_key is None:
                raise ValueError(
                    "OPENAI_API_KEY is required when generative AI "
                    "is enabled."
                )

            if not self.generative_model_name.strip():
                raise ValueError(
                    "GENERATIVE_MODEL_NAME is required when "
                    "generative AI is enabled."
                )

        if (
            self.candidate_model_maximum_regression
            > self.candidate_model_minimum_improvement
        ):
            raise ValueError(
                "Maximum regression cannot exceed the required "
                "candidate-model improvement."
            )

        return self

    # ========================================================
    # SECTION 07 - DERIVED PROPERTIES
    # ========================================================

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_testing(self) -> bool:
        return self.environment == "testing"

    @property
    def is_staging(self) -> bool:
        return self.environment == "staging"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def uses_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def uses_postgresql(self) -> bool:
        return self.database_url.startswith(
            ("postgresql://", "postgresql+")
        )

    @property
    def public_base_url_string(self) -> str:
        return str(self.public_base_url).rstrip("/")

    @property
    def business_website_url_string(self) -> str:
        return str(self.business_website_url).rstrip("/")

    @property
    def canonical_base_url_string(self) -> str:
        return str(self.canonical_base_url).rstrip("/")

    @property
    def widget_script_url(self) -> str:
        return (
            f"{self.public_base_url_string}"
            "/static/widget/horseshoe-widget.js"
        )

    @property
    def widget_stylesheet_url(self) -> str:
        return (
            f"{self.public_base_url_string}"
            "/static/widget/horseshoe-widget.css"
        )

    @property
    def learning_is_enabled(self) -> bool:
        return self.learning_mode != "disabled"

    @property
    def training_requires_review(self) -> bool:
        return (
            self.learning_is_enabled
            and self.require_human_training_review
        )

    # ========================================================
    # SECTION 08 - SAFE PUBLIC REPORTING
    # ========================================================

    def public_summary(self) -> dict[str, Any]:
        """
        Return non-secret settings suitable for diagnostics.

        Secret values, credentials, and API keys are deliberately excluded.
        """

        return {
            "app_name": self.app_name,
            "app_slug": self.app_slug,
            "app_version": self.app_version,
            "environment": self.environment,
            "debug": self.debug,
            "testing": self.testing,
            "host": self.host,
            "port": self.port,
            "public_base_url": self.public_base_url_string,
            "timezone": self.timezone,
            "business_name": self.business_name,
            "business_slug": self.business_slug,
            "business_website_url": self.business_website_url_string,
            "database_backend": (
                "sqlite"
                if self.uses_sqlite
                else "postgresql"
                if self.uses_postgresql
                else "other"
            ),
            "allowed_origins": list(self.allowed_origins),
            "allowed_hosts": list(self.allowed_hosts),
            "store_conversations": self.store_conversations,
            "session_memory_enabled": self.session_memory_enabled,
            "customer_memory_enabled": self.customer_memory_enabled,
            "privacy_mode": self.privacy_mode,
            "spelling_correction_enabled": (
                self.spelling_correction_enabled
            ),
            "fuzzy_matching_enabled": self.fuzzy_matching_enabled,
            "phonetic_matching_enabled": (
                self.phonetic_matching_enabled
            ),
            "semantic_matching_enabled": (
                self.semantic_matching_enabled
            ),
            "response_variation_mode": self.response_variation_mode,
            "learning_mode": self.learning_mode,
            "require_human_training_review": (
                self.require_human_training_review
            ),
            "model_promotion_requires_approval": (
                self.model_promotion_requires_approval
            ),
            "require_verified_business_facts": (
                self.require_verified_business_facts
            ),
            "generative_ai_enabled": self.generative_ai_enabled,
            "seo_enabled": self.seo_enabled,
            "schema_org_enabled": self.schema_org_enabled,
            "analytics_enabled": self.analytics_enabled,
            "log_level": self.log_level,
        }

    def startup_warnings(self) -> list[str]:
        """
        Return non-fatal configuration warnings.
        """

        warnings: list[str] = []

        if self.is_development:
            if (
                self.admin_password.get_secret_value()
                == "replace-me-before-production"
            ):
                warnings.append(
                    "Development admin password is still using the "
                    "placeholder value."
                )

            if self.uses_sqlite:
                warnings.append(
                    "SQLite is enabled. Use PostgreSQL for durable "
                    "production conversation and lead storage."
                )

        if not self.allowed_origins:
            warnings.append(
                "No CORS origins are configured; the external website "
                "widget will not be able to call the API."
            )

        if not self.store_conversations:
            warnings.append(
                "Conversation storage is disabled, so reviewed learning "
                "and historical analytics will be unavailable."
            )

        if self.customer_memory_enabled:
            warnings.append(
                "Customer memory is enabled. Confirm consent, privacy, "
                "access-control, and retention policies before launch."
            )

        if self.seo_indexing_enabled and not self.is_production:
            warnings.append(
                "SEO indexing is enabled outside production."
            )

        if self.generative_ai_enabled:
            warnings.append(
                "Generative AI is enabled. All business-fact answers "
                "must pass retrieval and factual validation."
            )

        return warnings


# ============================================================
# SECTION 09 - SETTINGS FACTORY
# ============================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return one cached Settings instance for the process.
    """

    _ensure_runtime_directories()
    return Settings()


def reload_settings() -> Settings:
    """
    Clear the cached settings instance and reload environment values.

    Intended for tests and controlled administrative operations.
    """

    get_settings.cache_clear()
    return get_settings()


# ============================================================
# SECTION 10 - MODULE-LEVEL SETTINGS INSTANCE
# ============================================================

settings = get_settings()


# ============================================================
# SECTION 11 - CONFIGURATION VALIDATION REPORT
# ============================================================

def build_configuration_report() -> dict[str, Any]:
    """
    Produce a safe configuration report for local verification.
    """

    return {
        "status": "ok",
        "configuration": settings.public_summary(),
        "warnings": settings.startup_warnings(),
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "app_directory": str(APP_DIRECTORY),
            "data_directory": str(DATA_DIRECTORY),
            "log_directory": str(LOG_DIRECTORY),
            "artifact_directory": str(ARTIFACT_DIRECTORY),
            "model_directory": str(MODEL_DIRECTORY),
            "template_directory": str(TEMPLATE_DIRECTORY),
            "static_directory": str(STATIC_DIRECTORY),
        },
    }


# ============================================================
# SECTION 12 - DIRECT EXECUTION SUPPORT
# ============================================================

if __name__ == "__main__":
    print(
        json.dumps(
            build_configuration_report(),
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
