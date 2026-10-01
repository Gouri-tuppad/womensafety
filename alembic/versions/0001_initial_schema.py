"""initial schema – all tables

Revision ID: 0001
Revises: None
Create Date: 2024-01-01 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import geoalchemy2


# revision identifiers
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable PostGIS extension
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    # ── users ────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("phone", sa.String(20), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("role", sa.Enum("woman", "child", "parent", "volunteer", "authority", name="user_role"), nullable=False),
        sa.Column("trust_score", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("profile_type", sa.String(60), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── trusted_contacts ─────────────────────────────────────────────────
    op.create_table(
        "trusted_contacts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("contact_name", sa.String(120), nullable=False),
        sa.Column("contact_phone", sa.String(20), nullable=False),
        sa.Column("relationship_label", sa.String(60), nullable=False),
        sa.Column("priority", sa.Integer(), server_default="1", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── road_segments ────────────────────────────────────────────────────
    op.create_table(
        "road_segments",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("osm_id", sa.BigInteger(), nullable=False, index=True),
        sa.Column("geometry", geoalchemy2.Geometry(geometry_type="LINESTRING", srid=4326), nullable=False),
        sa.Column("road_type", sa.String(60), nullable=False),
        sa.Column("name", sa.String(200), nullable=True),
        sa.Column("length_m", sa.Float(), nullable=False),
        sa.Column("lit_tag", sa.String(10), nullable=True),
        sa.Column("lamp_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("lit_ratio", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("zone", sa.String(30), nullable=False, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_road_zone_osmid", "road_segments", ["zone", "osm_id"])
    op.execute("CREATE INDEX ix_road_segments_geom ON road_segments USING GIST (geometry)")

    # ── pois ─────────────────────────────────────────────────────────────
    op.create_table(
        "pois",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("category", sa.Enum("lamp", "shop", "atm", "petrol", "bus_stop", "metro", "pharmacy", "factory_gate", name="poi_category"), nullable=False),
        sa.Column("geometry", geoalchemy2.Geometry(geometry_type="POINT", srid=4326), nullable=False),
        sa.Column("safe_haven", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("zone", sa.String(30), nullable=False, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_poi_zone_category", "pois", ["zone", "category"])
    op.execute("CREATE INDEX ix_pois_geom ON pois USING GIST (geometry)")

    # ── incidents ────────────────────────────────────────────────────────
    op.create_table(
        "incidents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("geometry", geoalchemy2.Geometry(geometry_type="POINT", srid=4326), nullable=False),
        sa.Column("incident_type", sa.String(60), nullable=False),
        sa.Column("severity", sa.Integer(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.Enum("official", "news", "user", "synthetic", name="incident_source"), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("zone", sa.String(30), nullable=False, index=True),
        sa.Column("verified", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_incident_zone_type", "incidents", ["zone", "incident_type"])
    op.create_index("ix_incident_occurred", "incidents", ["occurred_at"])
    op.execute("CREATE INDEX ix_incidents_geom ON incidents USING GIST (geometry)")

    # ── reports ──────────────────────────────────────────────────────────
    op.create_table(
        "reports",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("incident_id", sa.Uuid(), sa.ForeignKey("incidents.id"), nullable=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("geometry", geoalchemy2.Geometry(geometry_type="POINT", srid=4326, from_text="ST_GeomFromEWKT", name="geometry"), nullable=True),
        sa.Column("status", sa.Enum("pending", "reviewed", "verified", "rejected", name="report_status"), server_default="pending", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── trips ────────────────────────────────────────────────────────────
    op.create_table(
        "trips",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("status", sa.Enum("planned", "active", "completed", "sos_triggered", "deviated", "stale", name="trip_status"), server_default="planned", nullable=False),
        sa.Column("origin", geoalchemy2.Geometry(geometry_type="POINT", srid=4326), nullable=False),
        sa.Column("destination", geoalchemy2.Geometry(geometry_type="POINT", srid=4326), nullable=False),
        sa.Column("route_geojson", postgresql.JSONB(), nullable=True),
        sa.Column("safety_score", sa.Float(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── trip_locations ───────────────────────────────────────────────────
    op.create_table(
        "trip_locations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("trip_id", sa.Uuid(), sa.ForeignKey("trips.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("geometry", geoalchemy2.Geometry(geometry_type="POINT", srid=4326), nullable=False),
        sa.Column("accuracy_m", sa.Float(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.execute("CREATE INDEX ix_trip_locations_geom ON trip_locations USING GIST (geometry)")

    # ── sos_events ───────────────────────────────────────────────────────
    op.create_table(
        "sos_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("trip_id", sa.Uuid(), sa.ForeignKey("trips.id"), nullable=True),
        sa.Column("status", sa.Enum("countdown", "active", "cancelled", "resolved", name="sos_status"), server_default="countdown", nullable=False),
        sa.Column("trigger_type", sa.String(30), nullable=False),
        sa.Column("confidence", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── sos_locations ────────────────────────────────────────────────────
    op.create_table(
        "sos_locations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("sos_event_id", sa.Uuid(), sa.ForeignKey("sos_events.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("geometry", geoalchemy2.Geometry(geometry_type="POINT", srid=4326), nullable=False),
        sa.Column("accuracy_m", sa.Float(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.execute("CREATE INDEX ix_sos_locations_geom ON sos_locations USING GIST (geometry)")

    # ── evidence_files ───────────────────────────────────────────────────
    op.create_table(
        "evidence_files",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("sos_event_id", sa.Uuid(), sa.ForeignKey("sos_events.id", ondelete="SET NULL"), nullable=True),
        sa.Column("report_id", sa.Uuid(), sa.ForeignKey("reports.id", ondelete="SET NULL"), nullable=True),
        sa.Column("file_type", sa.String(20), nullable=False),
        sa.Column("file_path", sa.String(500), nullable=False),
        sa.Column("duration_s", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── geofences ────────────────────────────────────────────────────────
    op.create_table(
        "geofences",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("geometry", geoalchemy2.Geometry(geometry_type="POLYGON", srid=4326), nullable=False),
        sa.Column("alert_on_exit", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.execute("CREATE INDEX ix_geofences_geom ON geofences USING GIST (geometry)")

    # ── volunteers ───────────────────────────────────────────────────────
    op.create_table(
        "volunteers",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True),
        sa.Column("is_available", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("last_location", geoalchemy2.Geometry(geometry_type="POINT", srid=4326), nullable=True),
        sa.Column("radius_m", sa.Float(), server_default="1000.0", nullable=False),
        sa.Column("verified", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.execute("CREATE INDEX ix_volunteers_location ON volunteers USING GIST (last_location)")

    # ── audit_log ────────────────────────────────────────────────────────
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("action", sa.String(60), nullable=False),
        sa.Column("resource_type", sa.String(60), nullable=False),
        sa.Column("resource_id", sa.String(60), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── feedback ─────────────────────────────────────────────────────────
    op.create_table(
        "feedback",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("trip_id", sa.Uuid(), sa.ForeignKey("trips.id", ondelete="SET NULL"), nullable=True),
        sa.Column("route_rating", sa.Integer(), nullable=False),
        sa.Column("safety_rating", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_table("audit_log")
    op.drop_table("volunteers")
    op.drop_table("geofences")
    op.drop_table("evidence_files")
    op.drop_table("sos_locations")
    op.drop_table("sos_events")
    op.drop_table("trip_locations")
    op.drop_table("trips")
    op.drop_table("reports")
    op.drop_table("incidents")
    op.drop_table("pois")
    op.drop_table("road_segments")
    op.drop_table("trusted_contacts")
    op.drop_table("users")

    for enum_name in ["user_role", "poi_category", "incident_source", "report_status", "trip_status", "sos_status"]:
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")
