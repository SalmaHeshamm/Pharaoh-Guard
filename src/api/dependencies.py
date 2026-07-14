"""
FastAPI dependency-style helpers for the /predict route.

Thin wrapper around core.risk_model — kept as its own module (rather than
inlining into the route) so routes stay focused on HTTP concerns and the
scoring logic stays independently testable/importable (e.g. from a
Streamlit dashboard or a batch script) without spinning up FastAPI.
"""
from __future__ import annotations

from typing import Any

from config.site_master_data import get_site_profile
from core import history_store
from core.risk_model import score as score_situation


def predict_risk(raw: dict[str, Any]) -> dict[str, Any]:
    """
    raw must contain: timestamp, site_name, current_visitors, security_staff,
    medical_team, police_units, temperature, humidity, wind_speed, visibility.
    Optional: queue_length, queue_time, special_events, vip_visits,
    school_trips, emergency_active, holiday.

    Looks up the site profile, pulls the previous hour's visitor count from
    history_store for the momentum features, scores via the trained model,
    then records this reading as the new "previous" value for next time.

    Returns the full dict from core.risk_model.score(): risk_level,
    probabilities, proba_critical, risk_score, occupancy_rate, khamsin,
    crowd_density, weather_score, security_score, operational_load.
    """
    site = get_site_profile(raw["site_name"])
    prev_visitors = history_store.get_previous_visitors(raw["site_name"])

    result = score_situation(raw, site, prev_visitors)

    history_store.update(raw["site_name"], raw["timestamp"], raw["current_visitors"])
    return result
