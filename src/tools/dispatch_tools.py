"""
The concrete "hands" of the system. Each function here is a real,
callable operational action. In production these would call the venue's
actual staffing/access-control APIs — for now they log + notify via
tools/notification_tools.py, gated by RRS_DISPATCH_DRY_RUN.

Every function returns (success: bool, detail: str) so the Dispatch
Agent can build a DispatchResult without caring about internals.
"""
from __future__ import annotations

import logging

from config.site_master_data import get_site_profile
from tools.notification_tools import send_slack_alert, send_sms_alert

logger = logging.getLogger(__name__)


def reallocate_security(site_name: str, additional_staff: int) -> tuple[bool, str]:
    profile = get_site_profile(site_name)
    message = (
        f":rotating_light: *{site_name}* — request +{additional_staff} security staff "
        f"(site has {profile.entry_gates} entry / {profile.exit_gates} exit gates)."
    )
    ok, detail = send_slack_alert(message)
    return ok, f"reallocate_security: {detail}"


def open_exit_gate(site_name: str, gates_to_open: int) -> tuple[bool, str]:
    profile = get_site_profile(site_name)
    if gates_to_open > profile.exit_gates:
        gates_to_open = profile.exit_gates
        logger.warning(
            "Requested more exit gates than exist at %s; capped at %d",
            site_name, gates_to_open,
        )
    message = f":door: *{site_name}* — open {gates_to_open} additional exit gate(s)."
    ok, detail = send_slack_alert(message)
    return ok, f"open_exit_gate: {detail}"


def throttle_entry(site_name: str, target_rate_pct: int) -> tuple[bool, str]:
    message = (
        f":stop_sign: *{site_name}* — throttle entry to {target_rate_pct}% "
        f"of normal admission rate."
    )
    ok, detail = send_slack_alert(message)
    return ok, f"throttle_entry: {detail}"


def notify_medical_team(site_name: str, note: str = "") -> tuple[bool, str]:
    profile = get_site_profile(site_name)
    message = f":ambulance: *{site_name}* — medical team alert. {note}".strip()
    ok, detail = send_slack_alert(message)
    _ = profile  # reserved for future: route to nearest hospital by region
    return ok, f"notify_medical_team: {detail}"


def trigger_emergency_protocol(site_name: str, emergency_type: str, note: str = "") -> tuple[bool, str]:
    message = f":warning: *{site_name}* — EMERGENCY PROTOCOL ({emergency_type}). {note}".strip()
    ok, detail = send_slack_alert(message)
    return ok, f"trigger_emergency_protocol: {detail}"


def notify_operations_channel(site_name: str, note: str) -> tuple[bool, str]:
    message = f":bell: *{site_name}* — {note}"
    ok, detail = send_slack_alert(message)
    return ok, f"notify_operations_channel: {detail}"


# Registry used by the Dispatch Agent to resolve an ActionItem.action_type
# to a callable without a long if/elif chain.
ACTION_REGISTRY = {
    "reallocate_security": reallocate_security,
    "open_exit_gate": open_exit_gate,
    "throttle_entry": throttle_entry,
    "notify_medical_team": notify_medical_team,
    "trigger_emergency_protocol": trigger_emergency_protocol,
    "notify_operations_channel": notify_operations_channel,
}
