"""
RoadSegment model – per-edge metadata overlaid on the OSMnx graph.

Geometry: PostGIS LineString (SRID 4326) with spatial index.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geometry

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class RoadSegment(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "road_segments"

    osm_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)

    # PostGIS LineString geometry (WGS-84)
    geometry: Mapped[str] = mapped_column(
        Geometry(geometry_type="LINESTRING", srid=4326, spatial_index=True),
        nullable=False,
    )

    road_type: Mapped[str] = mapped_column(
        String(60), nullable=False, doc="OSM highway tag: residential, tertiary, …",
    )
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    length_m: Mapped[float] = mapped_column(Float, nullable=False)

    # Lighting metadata (SYNTHETIC unless stated otherwise)
    lit_tag: Mapped[str | None] = mapped_column(
        String(10), nullable=True, doc="OSM lit=yes/no/limited",
    )
    lamp_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    lit_ratio: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False,
        doc="Fraction 0-1 of segment length that is well-lit. SYNTHETIC.",
    )

    zone: Mapped[str] = mapped_column(String(30), index=True, nullable=False)

    __table_args__ = (
        Index("ix_road_zone_osmid", "zone", "osm_id"),
    )

    def __repr__(self) -> str:
        return f"<RoadSegment osm_id={self.osm_id} zone={self.zone}>"
