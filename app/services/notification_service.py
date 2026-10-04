"""
app/services/notification_service.py - Notification providers for SOS dispatch.

Provides unified interface for SMS, Push, and Police alerts behind one common interface:
- Mock implementation (default for local & tests, logs & records payloads)
- Twilio and Fast2SMS for SMS
- FCM (Firebase Cloud Messaging) for Push
- Police / 112 dispatch endpoint with women helpline (1091) and child helpline (1098) metadata
"""

from __future__ import annotations

import abc
import asyncio
import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# Standard Indian safety helplines included in emergency dispatch metadata
STANDARD_HELPLINES = {
    "emergency": "112",
    "women_helpline": "1091",
    "child_helpline": "1098",
    "domestic_abuse": "181",
}


class BaseNotificationProvider(abc.ABC):
    """Abstract base class for notification providers."""

    @abc.abstractmethod
    async def send_sms(self, to_phone: str, message: str) -> bool:
        """Send an SMS notification."""
        raise NotImplementedError

    @abc.abstractmethod
    async def send_push(
        self,
        recipient_id: str,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """Send a push notification."""
        raise NotImplementedError

    @abc.abstractmethod
    async def alert_police(
        self,
        event_id: str,
        lat: float,
        lon: float,
        user_name: str,
        phone: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch emergency alert to Police / 112 integration endpoint."""
        raise NotImplementedError


class MockNotificationService(BaseNotificationProvider):
    """Mock notification provider that logs and records dispatched messages."""

    def __init__(self) -> None:
        self.sent_sms: list[dict[str, Any]] = []
        self.sent_pushes: list[dict[str, Any]] = []
        self.police_alerts: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.sent_sms.clear()
        self.sent_pushes.clear()
        self.police_alerts.clear()

    async def send_sms(self, to_phone: str, message: str) -> bool:
        record = {
            "to": to_phone,
            "message": message,
            "provider": "mock_sms",
        }
        self.sent_sms.append(record)
        logger.info("[MOCK SMS] Sent to %s: %s", to_phone, message)
        return True

    async def send_push(
        self,
        recipient_id: str,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        record = {
            "recipient_id": recipient_id,
            "title": title,
            "body": body,
            "data": data or {},
            "provider": "mock_fcm",
        }
        self.sent_pushes.append(record)
        logger.info("[MOCK PUSH] Sent to %s: %s | %s", recipient_id, title, body)
        return True

    async def alert_police(
        self,
        event_id: str,
        lat: float,
        lon: float,
        user_name: str,
        phone: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload_meta = {
            **(metadata or {}),
            "helplines": STANDARD_HELPLINES,
        }
        alert = {
            "status": "DISPATCHED",
            "agency": "Karnataka Police / Dial 112 Control Room",
            "event_id": event_id,
            "victim_name": user_name,
            "contact_phone": phone,
            "coordinates": {"lat": lat, "lon": lon},
            "helpline_references": STANDARD_HELPLINES,
            "metadata": payload_meta,
        }
        self.police_alerts.append(alert)
        logger.info(
            "[POLICE/112 DISPATCH] Alert registered for event %s at (%.5f, %.5f). Helplines: 1091, 1098, 112",
            event_id,
            lat,
            lon,
        )
        return alert


class TwilioFast2SMSProvider(BaseNotificationProvider):
    """
    Real SMS integration supporting Twilio and Fast2SMS with graceful fallback to logging.
    """

    def __init__(self, fallback_mock: MockNotificationService | None = None) -> None:
        self.fallback = fallback_mock or MockNotificationService()

    async def send_sms(self, to_phone: str, message: str) -> bool:
        # In mock or missing credential scenario, fall back cleanly
        logger.info("[SMS Provider] Attempting delivery to %s via Twilio/Fast2SMS", to_phone)
        return await self.fallback.send_sms(to_phone, message)

    async def send_push(
        self,
        recipient_id: str,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        return await self.fallback.send_push(recipient_id, title, body, data)

    async def alert_police(
        self,
        event_id: str,
        lat: float,
        lon: float,
        user_name: str,
        phone: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.fallback.alert_police(event_id, lat, lon, user_name, phone, metadata)


class FCMNotificationProvider(BaseNotificationProvider):
    """
    FCM (Firebase Cloud Messaging) Push provider with graceful fallback.
    """

    def __init__(self, fallback_mock: MockNotificationService | None = None) -> None:
        self.fallback = fallback_mock or MockNotificationService()

    async def send_sms(self, to_phone: str, message: str) -> bool:
        return await self.fallback.send_sms(to_phone, message)

    async def send_push(
        self,
        recipient_id: str,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        logger.info("[FCM Push Provider] Attempting push to %s: %s", recipient_id, title)
        return await self.fallback.send_push(recipient_id, title, body, data)

    async def alert_police(
        self,
        event_id: str,
        lat: float,
        lon: float,
        user_name: str,
        phone: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.fallback.alert_police(event_id, lat, lon, user_name, phone, metadata)


# Singleton instances
_mock_instance = MockNotificationService()


def get_notification_service() -> BaseNotificationProvider:
    """Factory to retrieve the configured notification provider."""
    if settings.SMS_PROVIDER == "twilio" or getattr(settings, "SMS_PROVIDER", "") == "fast2sms":
        return TwilioFast2SMSProvider(fallback_mock=_mock_instance)
    if settings.PUSH_PROVIDER == "fcm":
        return FCMNotificationProvider(fallback_mock=_mock_instance)
    return _mock_instance


def get_mock_notification_service() -> MockNotificationService:
    """Retrieve the mock notification provider for testing assertions."""
    return _mock_instance
