"""
exceptions.py – Domain exception hierarchy for SafeWay.

Each exception carries a machine-readable ``code`` and human-readable
``detail`` so the API layer can produce consistent error responses.
"""

from __future__ import annotations


class SafeWayError(Exception):
    """Base for all SafeWay domain errors."""

    def __init__(self, detail: str = "An unexpected error occurred.", code: str = "SAFEWAY_ERROR"):
        self.detail = detail
        self.code = code
        super().__init__(detail)


class RouteNotFoundError(SafeWayError):
    def __init__(self, detail: str = "No safe route found between the given points."):
        super().__init__(detail=detail, code="ROUTE_NOT_FOUND")


class ZoneOutOfBoundsError(SafeWayError):
    def __init__(self, detail: str = "Coordinate is outside the active zone boundary."):
        super().__init__(detail=detail, code="ZONE_OUT_OF_BOUNDS")


class SOSCooldownError(SafeWayError):
    def __init__(self, detail: str = "SOS cooldown period has not elapsed."):
        super().__init__(detail=detail, code="SOS_COOLDOWN")


class TripNotFoundError(SafeWayError):
    def __init__(self, detail: str = "Trip not found."):
        super().__init__(detail=detail, code="TRIP_NOT_FOUND")


class TripAlreadyEndedError(SafeWayError):
    def __init__(self, detail: str = "This trip has already ended."):
        super().__init__(detail=detail, code="TRIP_ALREADY_ENDED")


class DuplicateError(SafeWayError):
    def __init__(self, detail: str = "Resource already exists."):
        super().__init__(detail=detail, code="DUPLICATE")
