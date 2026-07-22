# ============================================================
# Exact file location: app/database/base.py
# Horseshoe Tavern AI
# Phase 1 Part 1.5
# SQLAlchemy declarative foundation, model mixins, metadata,
# identifiers, timestamps, serialization, and inspection helpers
# ============================================================

"""
Shared SQLAlchemy database foundation for Horseshoe Tavern AI.

This module provides:

- One centralized SQLAlchemy declarative base
- Deterministic database naming conventions
- UUID primary-key support
- Integer primary-key support
- Created and updated timestamps
- Soft-deletion support
- Source provenance fields
- Verification and freshness fields
- Versioning and optimistic-update support
- Safe model serialization
- Model inspection helpers
- UTC-aware datetime utilities
- JSON-compatible value conversion
- Table-name generation
- Repr helpers that avoid exposing sensitive values
- Declarative model validation support

No business-specific tables are declared here. Concrete models belong in
app/database/models.py and should inherit from the mixins defined here.
"""

from __future__ import annotations

import enum
import json
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Any, ClassVar, Final, TypeVar
from urllib.parse import ParseResult
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    MetaData,
    String,
    Text,
    event,
    inspect,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Mapper,
    declared_attr,
    mapped_column,
)


# ============================================================
# SECTION 01 - SQLALCHEMY NAMING CONVENTION
# ============================================================

NAMING_CONVENTION: Final[dict[str, str]] = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": (
        "fk_%(table_name)s_%(column_0_name)s_"
        "%(referred_table_name)s"
    ),
    "pk": "pk_%(table_name)s",
}

MODEL_METADATA = MetaData(
    naming_convention=NAMING_CONVENTION,
)


# ============================================================
# SECTION 02 - MODULE CONSTANTS
# ============================================================

DEFAULT_STRING_LENGTH: Final[int] = 255
DEFAULT_NAME_LENGTH: Final[int] = 200
DEFAULT_SLUG_LENGTH: Final[int] = 200
DEFAULT_URL_LENGTH: Final[int] = 2048
DEFAULT_EMAIL_LENGTH: Final[int] = 320
DEFAULT_PHONE_LENGTH: Final[int] = 64
DEFAULT_STATUS_LENGTH: Final[int] = 64
DEFAULT_SOURCE_LENGTH: Final[int] = 255
DEFAULT_VERSION_LENGTH: Final[int] = 100

SENSITIVE_ATTRIBUTE_FRAGMENTS: Final[tuple[str, ...]] = (
    "password",
    "passwd",
    "secret",
    "token",
    "authorization",
    "api_key",
    "apikey",
    "private_key",
    "session_key",
    "widget_signing_key",
    "admin_password",
    "credit_card",
    "card_number",
    "cvv",
    "cvc",
)

CAMEL_CASE_BOUNDARY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!^)(?=[A-Z])"
)

NON_IDENTIFIER_CHARACTER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[^a-zA-Z0-9_]+"
)


# ============================================================
# SECTION 03 - TYPE VARIABLES
# ============================================================

ModelType = TypeVar(
    "ModelType",
    bound="Base",
)


# ============================================================
# SECTION 04 - DATETIME UTILITIES
# ============================================================

def utc_now() -> datetime:
    """
    Return a timezone-aware current UTC datetime.
    """

    return datetime.now(UTC)


def ensure_utc(value: datetime | None) -> datetime | None:
    """
    Normalize a datetime to timezone-aware UTC.

    Naive datetimes are interpreted as UTC because application and database
    timestamps are defined as UTC internally.
    """

    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)

    return value.astimezone(UTC)


def isoformat_utc(value: datetime | None) -> str | None:
    """
    Convert a datetime to a normalized ISO-8601 UTC string.
    """

    normalized = ensure_utc(value)

    if normalized is None:
        return None

    return normalized.isoformat().replace("+00:00", "Z")


# ============================================================
# SECTION 05 - IDENTIFIER UTILITIES
# ============================================================

def new_uuid() -> UUID:
    """
    Generate a version-4 UUID.
    """

    return uuid.uuid4()


def new_uuid_string() -> str:
    """
    Generate a version-4 UUID as a canonical string.
    """

    return str(new_uuid())


def normalize_uuid(
    value: UUID | str | None,
) -> UUID | None:
    """
    Normalize UUID-compatible input.
    """

    if value is None:
        return None

    if isinstance(value, UUID):
        return value

    return UUID(str(value).strip())


# ============================================================
# SECTION 06 - NAME AND SLUG UTILITIES
# ============================================================

def class_name_to_table_name(class_name: str) -> str:
    """
    Convert a Python class name to a plural snake_case table name.

    Examples:
        ConversationMessage -> conversation_messages
        FAQEntry -> faq_entries
        Business -> businesses
    """

    if not class_name or not class_name.strip():
        raise ValueError("Class name cannot be empty.")

    cleaned = NON_IDENTIFIER_CHARACTER_PATTERN.sub(
        "",
        class_name.strip(),
    )

    snake_case = CAMEL_CASE_BOUNDARY_PATTERN.sub(
        "_",
        cleaned,
    ).lower()

    if snake_case.endswith("y") and not snake_case.endswith(
        ("ay", "ey", "iy", "oy", "uy")
    ):
        return f"{snake_case[:-1]}ies"

    if snake_case.endswith(("s", "x", "z", "ch", "sh")):
        return f"{snake_case}es"

    return f"{snake_case}s"


def normalize_slug(value: str) -> str:
    """
    Normalize text into a URL-safe lowercase slug.
    """

    candidate = value.strip().lower()
    candidate = re.sub(r"[^a-z0-9]+", "-", candidate)
    candidate = re.sub(r"-{2,}", "-", candidate)

    return candidate.strip("-")


# ============================================================
# SECTION 07 - SERIALIZATION UTILITIES
# ============================================================

def is_sensitive_attribute(name: str) -> bool:
    """
    Determine whether an attribute name likely contains sensitive data.
    """

    normalized = name.strip().lower().replace("-", "_")

    return any(
        fragment in normalized
        for fragment in SENSITIVE_ATTRIBUTE_FRAGMENTS
    )


def to_json_compatible(
    value: Any,
    *,
    maximum_depth: int = 8,
    _depth: int = 0,
) -> Any:
    """
    Convert common Python and SQLAlchemy values to JSON-compatible forms.
    """

    if _depth >= maximum_depth:
        return "[MAX_DEPTH_REACHED]"

    if value is None:
        return None

    if isinstance(value, bool | int | float | str):
        return value

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, datetime):
        return isoformat_utc(value)

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, time):
        return value.isoformat()

    if isinstance(value, enum.Enum):
        return to_json_compatible(
            value.value,
            maximum_depth=maximum_depth,
            _depth=_depth + 1,
        )

    if isinstance(value, ParseResult):
        return value.geturl()

    if isinstance(value, Mapping):
        return {
            str(key): to_json_compatible(
                nested_value,
                maximum_depth=maximum_depth,
                _depth=_depth + 1,
            )
            for key, nested_value in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            to_json_compatible(
                item,
                maximum_depth=maximum_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    if isinstance(value, Iterable) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            to_json_compatible(
                item,
                maximum_depth=maximum_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    if hasattr(value, "model_dump"):
        return to_json_compatible(
            value.model_dump(),
            maximum_depth=maximum_depth,
            _depth=_depth + 1,
        )

    if hasattr(value, "as_dict"):
        try:
            return to_json_compatible(
                value.as_dict(),
                maximum_depth=maximum_depth,
                _depth=_depth + 1,
            )
        except Exception:
            pass

    return str(value)


# ============================================================
# SECTION 08 - DECLARATIVE BASE
# ============================================================

class Base(DeclarativeBase):
    """
    Shared SQLAlchemy declarative base.
    """

    metadata = MODEL_METADATA

    __abstract__ = True

    __repr_exclude__: ClassVar[set[str]] = set()
    __serialize_exclude__: ClassVar[set[str]] = set()
    __serialize_include_relationships__: ClassVar[set[str]] = set()

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return class_name_to_table_name(cls.__name__)

    def __repr__(self) -> str:
        """
        Provide a safe, compact model representation.
        """

        values = self._repr_values()
        rendered = ", ".join(
            f"{key}={value!r}"
            for key, value in values.items()
        )

        return f"{self.__class__.__name__}({rendered})"

    def _repr_values(self) -> dict[str, Any]:
        """
        Return safe fields for __repr__.
        """

        mapper = inspect(self.__class__)
        preferred_names = (
            "id",
            "uuid",
            "slug",
            "name",
            "title",
            "status",
            "created_at",
        )

        values: dict[str, Any] = {}

        for name in preferred_names:
            if name in self.__repr_exclude__:
                continue

            if is_sensitive_attribute(name):
                continue

            if name not in mapper.columns:
                continue

            try:
                value = getattr(self, name)
            except Exception:
                continue

            if value is not None:
                values[name] = to_json_compatible(value)

            if len(values) >= 4:
                break

        if not values:
            for column in mapper.columns:
                name = column.key

                if name in self.__repr_exclude__:
                    continue

                if is_sensitive_attribute(name):
                    continue

                try:
                    value = getattr(self, name)
                except Exception:
                    continue

                if value is not None:
                    values[name] = to_json_compatible(value)

                if len(values) >= 3:
                    break

        return values

    def as_dict(
        self,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        include_relationships: bool = False,
        redact_sensitive: bool = True,
        maximum_relationship_depth: int = 1,
        _current_depth: int = 0,
    ) -> dict[str, Any]:
        """
        Serialize mapped columns and optionally selected relationships.

        Relationships are excluded by default to prevent recursive payloads
        and unintentional database loading.
        """

        mapper = inspect(self.__class__)

        include_names = set(include or ())
        exclude_names = set(exclude or ())
        exclude_names.update(self.__serialize_exclude__)

        output: dict[str, Any] = {}

        for column in mapper.columns:
            name = column.key

            if include_names and name not in include_names:
                continue

            if name in exclude_names:
                continue

            if redact_sensitive and is_sensitive_attribute(name):
                output[name] = "[REDACTED]"
                continue

            try:
                value = getattr(self, name)
            except Exception:
                continue

            output[name] = to_json_compatible(value)

        should_include_relationships = (
            include_relationships
            and _current_depth < maximum_relationship_depth
        )

        if should_include_relationships:
            permitted_relationships = (
                self.__serialize_include_relationships__
            )

            for relationship in mapper.relationships:
                name = relationship.key

                if permitted_relationships and (
                    name not in permitted_relationships
                ):
                    continue

                if include_names and name not in include_names:
                    continue

                if name in exclude_names:
                    continue

                try:
                    value = getattr(self, name)
                except Exception:
                    continue

                if value is None:
                    output[name] = None
                    continue

                if relationship.uselist:
                    output[name] = [
                        item.as_dict(
                            include_relationships=True,
                            redact_sensitive=redact_sensitive,
                            maximum_relationship_depth=(
                                maximum_relationship_depth
                            ),
                            _current_depth=_current_depth + 1,
                        )
                        if isinstance(item, Base)
                        else to_json_compatible(item)
                        for item in value
                    ]
                elif isinstance(value, Base):
                    output[name] = value.as_dict(
                        include_relationships=True,
                        redact_sensitive=redact_sensitive,
                        maximum_relationship_depth=(
                            maximum_relationship_depth
                        ),
                        _current_depth=_current_depth + 1,
                    )
                else:
                    output[name] = to_json_compatible(value)

        return output

    def to_json(
        self,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        include_relationships: bool = False,
        redact_sensitive: bool = True,
        indent: int | None = None,
    ) -> str:
        """
        Serialize the model to JSON.
        """

        return json.dumps(
            self.as_dict(
                include=include,
                exclude=exclude,
                include_relationships=include_relationships,
                redact_sensitive=redact_sensitive,
            ),
            ensure_ascii=False,
            default=str,
            indent=indent,
        )

    @classmethod
    def column_names(cls) -> tuple[str, ...]:
        """
        Return mapped column names.
        """

        mapper = inspect(cls)

        return tuple(
            column.key
            for column in mapper.columns
        )

    @classmethod
    def relationship_names(cls) -> tuple[str, ...]:
        """
        Return mapped relationship names.
        """

        mapper = inspect(cls)

        return tuple(
            relationship.key
            for relationship in mapper.relationships
        )

    @classmethod
    def primary_key_names(cls) -> tuple[str, ...]:
        """
        Return primary-key column names.
        """

        mapper = inspect(cls)

        return tuple(
            column.key
            for column in mapper.primary_key
        )

    @classmethod
    def mapped_table_name(cls) -> str:
        """
        Return the mapped table name.
        """

        return cls.__table__.name


# ============================================================
# SECTION 09 - PRIMARY KEY MIXINS
# ============================================================

class IntegerPrimaryKeyMixin:
    """
    Autoincrementing integer primary key.
    """

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )


class UUIDPrimaryKeyMixin:
    """
    UUID primary key stored as a canonical 36-character string.

    String storage provides consistent SQLite and PostgreSQL behavior during
    early development. A native PostgreSQL UUID type may be introduced later
    through an explicit migration if desired.
    """

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid_string,
    )


# ============================================================
# SECTION 10 - TIMESTAMP MIXINS
# ============================================================

class CreatedAtMixin:
    """
    Add a UTC creation timestamp.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )


class UpdatedAtMixin:
    """
    Add a UTC update timestamp.
    """

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        index=True,
    )


class TimestampMixin(
    CreatedAtMixin,
    UpdatedAtMixin,
):
    """
    Add created_at and updated_at timestamps.
    """


# ============================================================
# SECTION 11 - SOFT DELETE MIXIN
# ============================================================

class SoftDeleteMixin:
    """
    Add soft-deletion fields and helpers.
    """

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    deleted_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    def soft_delete(
        self,
        *,
        reason: str | None = None,
        deleted_at: datetime | None = None,
    ) -> None:
        self.is_deleted = True
        self.deleted_at = ensure_utc(deleted_at) or utc_now()
        self.deleted_reason = (
            reason.strip()
            if reason and reason.strip()
            else None
        )

    def restore(self) -> None:
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_reason = None


# ============================================================
# SECTION 12 - ACTIVE STATUS MIXIN
# ============================================================

class ActiveStatusMixin:
    """
    Add a simple active/inactive status.
    """

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    def activate(self) -> None:
        self.is_active = True

    def deactivate(self) -> None:
        self.is_active = False


# ============================================================
# SECTION 13 - SOURCE PROVENANCE MIXIN
# ============================================================

class SourceProvenanceMixin:
    """
    Track where a business fact or training object originated.
    """

    source_type: Mapped[str | None] = mapped_column(
        String(DEFAULT_SOURCE_LENGTH),
        nullable=True,
        index=True,
    )

    source_name: Mapped[str | None] = mapped_column(
        String(DEFAULT_SOURCE_LENGTH),
        nullable=True,
    )

    source_url: Mapped[str | None] = mapped_column(
        String(DEFAULT_URL_LENGTH),
        nullable=True,
    )

    source_reference: Mapped[str | None] = mapped_column(
        String(DEFAULT_SOURCE_LENGTH),
        nullable=True,
    )

    source_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    source_retrieved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    source_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )


# ============================================================
# SECTION 14 - VERIFICATION MIXIN
# ============================================================

class VerificationMixin:
    """
    Track whether a record has been reviewed as trusted business truth.
    """

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    verified_by: Mapped[str | None] = mapped_column(
        String(DEFAULT_NAME_LENGTH),
        nullable=True,
    )

    verification_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    def mark_verified(
        self,
        *,
        verified_by: str,
        notes: str | None = None,
        verified_at: datetime | None = None,
    ) -> None:
        verifier = verified_by.strip()

        if not verifier:
            raise ValueError(
                "verified_by cannot be empty."
            )

        self.is_verified = True
        self.verified_at = ensure_utc(verified_at) or utc_now()
        self.verified_by = verifier
        self.verification_notes = (
            notes.strip()
            if notes and notes.strip()
            else None
        )

    def mark_unverified(
        self,
        *,
        notes: str | None = None,
    ) -> None:
        self.is_verified = False
        self.verified_at = None
        self.verified_by = None
        self.verification_notes = (
            notes.strip()
            if notes and notes.strip()
            else None
        )


# ============================================================
# SECTION 15 - FRESHNESS MIXIN
# ============================================================

class FreshnessMixin:
    """
    Track validity windows and last freshness checks.
    """

    effective_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    effective_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    freshness_checked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    freshness_status: Mapped[str | None] = mapped_column(
        String(DEFAULT_STATUS_LENGTH),
        nullable=True,
        index=True,
    )

    @property
    def is_currently_effective(self) -> bool:
        now = utc_now()

        starts_on_time = (
            self.effective_from is None
            or ensure_utc(self.effective_from) <= now
        )

        has_not_expired = (
            self.effective_until is None
            or ensure_utc(self.effective_until) >= now
        )

        return starts_on_time and has_not_expired

    @property
    def is_expired(self) -> bool:
        if self.effective_until is None:
            return False

        return ensure_utc(self.effective_until) < utc_now()


# ============================================================
# SECTION 16 - VERSIONING MIXIN
# ============================================================

class VersionedMixin:
    """
    Add integer version tracking for update auditing.
    """

    record_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    model_version: Mapped[str | None] = mapped_column(
        String(DEFAULT_VERSION_LENGTH),
        nullable=True,
        index=True,
    )

    knowledge_version: Mapped[str | None] = mapped_column(
        String(DEFAULT_VERSION_LENGTH),
        nullable=True,
        index=True,
    )

    def increment_record_version(self) -> int:
        self.record_version = (
            int(self.record_version or 0) + 1
        )

        return self.record_version


# ============================================================
# SECTION 17 - PUBLIC IDENTIFIER MIXIN
# ============================================================

class PublicIdentifierMixin:
    """
    Add a stable externally safe public identifier.
    """

    public_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        index=True,
        default=new_uuid_string,
    )


# ============================================================
# SECTION 18 - COMMON COMPOSITE MIXINS
# ============================================================

class StandardRecordMixin(
    UUIDPrimaryKeyMixin,
    PublicIdentifierMixin,
    TimestampMixin,
):
    """
    Standard record identity and timestamps.
    """


class GovernedRecordMixin(
    StandardRecordMixin,
    ActiveStatusMixin,
    SoftDeleteMixin,
    SourceProvenanceMixin,
    VerificationMixin,
    FreshnessMixin,
    VersionedMixin,
):
    """
    Full governance support for knowledge and business-fact records.
    """


# ============================================================
# SECTION 19 - MODEL INSPECTION HELPERS
# ============================================================

def get_mapped_model_classes() -> tuple[type[Base], ...]:
    """
    Return all currently registered mapped model classes.
    """

    classes: list[type[Base]] = []

    for mapper in Base.registry.mappers:
        mapped_class = mapper.class_

        if (
            isinstance(mapped_class, type)
            and issubclass(mapped_class, Base)
        ):
            classes.append(mapped_class)

    return tuple(
        sorted(
            classes,
            key=lambda item: item.__name__,
        )
    )


def collect_model_inventory() -> list[dict[str, Any]]:
    """
    Return safe metadata describing registered SQLAlchemy models.
    """

    inventory: list[dict[str, Any]] = []

    for model_class in get_mapped_model_classes():
        mapper = inspect(model_class)

        inventory.append(
            {
                "model": model_class.__name__,
                "table": model_class.__table__.name,
                "columns": [
                    column.key
                    for column in mapper.columns
                ],
                "primary_keys": [
                    column.key
                    for column in mapper.primary_key
                ],
                "relationships": [
                    relationship.key
                    for relationship in mapper.relationships
                ],
            }
        )

    return inventory


def validate_model_mapping(
    model_class: type[Base],
) -> dict[str, Any]:
    """
    Validate basic SQLAlchemy model mapping requirements.
    """

    errors: list[str] = []
    warnings: list[str] = []

    try:
        mapper = inspect(model_class)
    except Exception as exc:
        return {
            "status": "failed",
            "model": getattr(
                model_class,
                "__name__",
                str(model_class),
            ),
            "errors": [str(exc)],
            "warnings": [],
        }

    table_name = model_class.__table__.name

    if not table_name:
        errors.append(
            "Mapped table name is empty."
        )

    if not mapper.primary_key:
        errors.append(
            "Model has no primary key."
        )

    column_names = {
        column.key
        for column in mapper.columns
    }

    if "created_at" not in column_names:
        warnings.append(
            "Model does not include created_at."
        )

    if "updated_at" not in column_names:
        warnings.append(
            "Model does not include updated_at."
        )

    sensitive_columns = sorted(
        name
        for name in column_names
        if is_sensitive_attribute(name)
    )

    if sensitive_columns:
        warnings.append(
            "Model contains sensitive columns that must be "
            f"redacted during serialization: {sensitive_columns}"
        )

    return {
        "status": "ok" if not errors else "failed",
        "model": model_class.__name__,
        "table": table_name,
        "errors": errors,
        "warnings": warnings,
        "columns": sorted(column_names),
        "primary_keys": [
            column.key
            for column in mapper.primary_key
        ],
    }


# ============================================================
# SECTION 20 - SQLALCHEMY EVENTS
# ============================================================

@event.listens_for(Mapper, "before_update")
def _increment_version_before_update(
    mapper: Mapper[Any],
    connection: Any,
    target: Any,
) -> None:
    """
    Increment record_version automatically before updates.

    The mapper and connection arguments are required by SQLAlchemy's event
    contract even though this implementation only needs the target object.
    """

    del mapper
    del connection

    if isinstance(target, VersionedMixin):
        target.increment_record_version()


# ============================================================
# SECTION 21 - INTERNAL VALIDATION MODEL
# ============================================================

class _FoundationVerificationModel(
    StandardRecordMixin,
    Base,
):
    """
    Internal model used only to validate the declarative foundation.

    This table is never intended for application migrations.
    """

    __tablename__ = "_foundation_verification_models"

    name: Mapped[str] = mapped_column(
        String(DEFAULT_NAME_LENGTH),
        nullable=False,
    )

    password_hash: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    __repr_exclude__ = {
        "password_hash",
    }

    __serialize_exclude__ = set()


# ============================================================
# SECTION 22 - FOUNDATION SELF-TEST
# ============================================================

def validate_database_foundation() -> dict[str, Any]:
    """
    Run deterministic foundation checks.
    """

    model = _FoundationVerificationModel(
        name="Horseshoe Tavern",
        password_hash="must-not-be-exposed",
    )

    serialized = model.as_dict()
    representation = repr(model)

    checks = {
        "table_name_generation": (
            class_name_to_table_name("ConversationMessage")
            == "conversation_messages"
        ),
        "business_pluralization": (
            class_name_to_table_name("Business")
            == "businesses"
        ),
        "uuid_generation": (
            len(new_uuid_string()) == 36
        ),
        "utc_timestamp_is_aware": (
            utc_now().tzinfo is not None
        ),
        "mapped_table_name": (
            model.mapped_table_name()
            == "_foundation_verification_models"
        ),
        "primary_key_detected": (
            model.primary_key_names() == ("id",)
        ),
        "sensitive_serialization_redacted": (
            serialized.get("password_hash")
            == "[REDACTED]"
        ),
        "sensitive_repr_excluded": (
            "must-not-be-exposed" not in representation
        ),
        "name_serialized": (
            serialized.get("name")
            == "Horseshoe Tavern"
        ),
        "public_id_generated": (
            isinstance(model.public_id, str)
            and len(model.public_id) == 36
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
        "model_validation": validate_model_mapping(
            _FoundationVerificationModel
        ),
        "serialized_sample": serialized,
        "repr_sample": representation,
        "metadata_naming_convention": dict(
            NAMING_CONVENTION
        ),
    }


# ============================================================
# SECTION 23 - DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    report = validate_database_foundation()

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    if report["status"] != "ok":
        raise SystemExit(1)
