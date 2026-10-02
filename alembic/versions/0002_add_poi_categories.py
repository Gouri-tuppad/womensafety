"""add police and hospital to poi_category enum

Revision ID: 0002
Revises: 0001
Create Date: 2024-01-02 00:00:00.000000

Adds two new POI categories:
  - police  : police stations (safe_haven=True)
  - hospital : hospitals and clinics (safe_haven=True)

PostgreSQL 12+ allows ALTER TYPE ... ADD VALUE inside a transaction.
This cluster uses PostgreSQL 16, so no special transaction isolation is needed.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ADD VALUE is idempotent with IF NOT EXISTS (PostgreSQL 9.3+)
    op.execute("ALTER TYPE poi_category ADD VALUE IF NOT EXISTS 'police'")
    op.execute("ALTER TYPE poi_category ADD VALUE IF NOT EXISTS 'hospital'")


def downgrade() -> None:
    # PostgreSQL does not support DROP VALUE from an enum; the safest downgrade
    # is to migrate existing rows back to 'shop' and leave the enum values in
    # place (removing enum values requires recreating the entire type).
    op.execute(
        "UPDATE pois SET category = 'shop' WHERE category IN ('police', 'hospital')"
    )
    # NOTE: The enum values 'police' and 'hospital' remain in the type after
    # downgrade. To fully remove them you would need to recreate poi_category.
