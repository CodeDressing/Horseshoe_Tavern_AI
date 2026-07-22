"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

Exact file location:
app/database/migrations/versions/${up_revision}_${slug}.py
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}


# Revision identifiers used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: str | Sequence[str] | None = ${repr(down_revision)}
branch_labels: str | Sequence[str] | None = ${repr(branch_labels)}
depends_on: str | Sequence[str] | None = ${repr(depends_on)}


def upgrade() -> None:
    """Apply this database revision."""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """Reverse this database revision."""
    ${downgrades if downgrades else "pass"}
