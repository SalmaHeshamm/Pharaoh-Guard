"""
Low-level notification senders. These never decide *whether* to notify —
that decision belongs to the Dispatch Agent. These functions only know
*how* to deliver a message once asked.
"""
from __future__ import annotations

import logging

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)


def send_slack_alert(message: str) -> tuple[bool, str]:
    settings = get_settings()

    if settings.dispatch_dry_run or not settings.slack_webhook_url:
        logger.info("[DRY-RUN] Slack alert: %s", message)
        return True, "dry_run"

    try:
        resp = httpx.post(settings.slack_webhook_url, json={"text": message}, timeout=5.0)
        resp.raise_for_status()
        return True, f"slack_status_{resp.status_code}"
    except httpx.HTTPError as exc:
        logger.error("Slack webhook failed: %s", exc)
        return False, str(exc)


def send_sms_alert(message: str, phone_numbers: list[str]) -> tuple[bool, str]:
    settings = get_settings()

    if settings.dispatch_dry_run or not settings.sms_webhook_url:
        logger.info("[DRY-RUN] SMS to %s: %s", phone_numbers, message)
        return True, "dry_run"

    try:
        resp = httpx.post(
            settings.sms_webhook_url,
            json={"message": message, "recipients": phone_numbers},
            timeout=5.0,
        )
        resp.raise_for_status()
        return True, f"sms_status_{resp.status_code}"
    except httpx.HTTPError as exc:
        logger.error("SMS webhook failed: %s", exc)
        return False, str(exc)
