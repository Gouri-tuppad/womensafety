"""
POI model – Points of Interest relevant to safety.

Geometry: PostGIS Point (SRID 4326) with spatial index.
"""

from __future__ import annotations

from enum import Enum as PyEnum

from sqlalchemy import Boolean, Enum, Index, String
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geometry

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class POICategory(str, PyEnum):
    LAMP = "lamp"
    SHOP = "shop"
    ATM = "atm"
    PETROL = "petrol"
    BUS_STOP = "bus_stop"
    METRO = "metro"
    PHARMACY = "pharmacy"
    FACTORY_GATE = "factory_gate"


class POI(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "pois"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[POICategory] = mapped_column(
        Enum(POICategory, name="poi_category", create_constraint=True),
        nullable=False,
    )

    # PostGIS Point (WGS-84)
    geometry: Mapped[str] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=True),
        nullable=False,
    )

    safe_haven: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
        doc="True if this POI can serve as a safe refuge (e.g. 24h pharmacy, metro station).",
    )
    zone: Mapped[str] = mapped_column(String(30), index=True, nullable=False)

    __table_args__ = (
        Index("ix_poi_zone_category", "zone", "category"),
    )

    def __repr__(self) -> str:
        return f"<POI {self.name!r} cat={self.category.value}>"
