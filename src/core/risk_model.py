"""
Trained risk model wrapper: load once, score single situations.

Ported from `pharaoh-guard-min/pharaoh_guard/core/model.py` — the reference
implementation already verified (100% agreement with the batch model on
500 reconstructed rows, and re-checked here against the Giza use-case
scenario from the pitch deck: 17,250 visitors -> Critical, P(Critical)=0.9997).

A "situation" is a natural operational snapshot — the values an operations
room actually observes (site, time, visitor count, staffing, weather) —
and `build_features()` re-derives the 39 engineered columns the model was
trained on (Notebook 04) before scoring. Composite descriptive scores
(Crowd_Density / Weather_Score / Security_Score / Operational_Load) are
also computed here, from the exact formulas in Notebook 01 — these are
NEVER fed to the model (they're excluded as leaky in feature_manifest.json)
but are kept on SiteSnapshot for the Reasoning Agent's explanations.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.settings import get_settings

RISK_ORDER = ["Low", "Medium", "High", "Critical"]

# Egyptian public holidays actually observed 2023-2024 (Notebook 01).
HOLIDAYS_2023_2024 = [
    "2023-01-01", "2023-01-07", "2023-04-17", "2023-04-21", "2023-04-22", "2023-04-23",
    "2023-04-25", "2023-05-01", "2023-06-28", "2023-06-29", "2023-06-30",
    "2023-07-23", "2023-10-06",
    "2024-01-01", "2024-01-07", "2024-04-10", "2024-04-11", "2024-04-12",
    "2024-04-25", "2024-05-01", "2024-05-06", "2024-06-16", "2024-06-17", "2024-06-18",
    "2024-07-23", "2024-10-06",
]
HOLIDAY_SET = set(pd.to_datetime(HOLIDAYS_2023_2024).date)

HEAT_BINS = [-10, 18, 26, 34, 42, 60]  # identical to Notebook 04
GATE_THROUGHPUT_PER_HOUR = 350  # visitors/gate/hour — Notebook 01 queue model


def get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


@lru_cache
def get_risk_model():
    """Load the trained sklearn Pipeline (preprocessing included) once."""
    settings = get_settings()
    path = settings.risk_model_path
    if not path.exists():
        raise FileNotFoundError(
            f"Risk model artifact not found at {path}. "
            "Check RRS_RISK_MODEL_PATH / settings.risk_model_path."
        )
    return joblib.load(path)


@lru_cache
def get_feature_order() -> list[str]:
    """The exact 39-column order the model was trained on (Notebook 04)."""
    settings = get_settings()
    with open(settings.feature_manifest_path) as f:
        manifest = json.load(f)
    return manifest["features"]


def model_classes() -> list[str]:
    return list(get_risk_model().named_steps["model"].classes_)


def build_features(raw: dict[str, Any], site, prev_visitors: float | None) -> pd.DataFrame:
    """Turn a natural operational snapshot into the model's engineered feature row.

    `raw` needs: timestamp, current_visitors, security_staff, medical_team,
    police_units, temperature, humidity, wind_speed, visibility. Everything
    else (queue, event flags) defaults sensibly. `site` is a SiteProfile
    (from config.site_master_data) and `prev_visitors` is the last known
    visitor count for this site (None => momentum defaults to 0).
    """
    ts = pd.to_datetime(raw["timestamp"])
    month, hour = ts.month, ts.hour
    dow = ts.strftime("%A")

    capacity = site.capacity
    entry_gates = site.entry_gates

    visitors = float(raw["current_visitors"])
    lag = float(prev_visitors) if prev_visitors is not None else visitors
    queue_len = float(raw.get("queue_length", 0))
    queue_time = float(raw.get("queue_time", 0))

    sec = float(raw["security_staff"])
    medical = float(raw["medical_team"])
    police = float(raw["police_units"])
    temp = float(raw["temperature"])
    humidity = float(raw["humidity"])
    wind = float(raw["wind_speed"])
    vis = float(raw["visibility"])

    special = int(raw.get("special_events", 0))
    vip = int(raw.get("vip_visits", 0))
    school = int(raw.get("school_trips", 0))
    emergency_active = int(raw.get("emergency_active", 0))
    holiday = int(raw["holiday"]) if "holiday" in raw else int(ts.date() in HOLIDAY_SET)

    row = {
        # calendar
        "Hour": hour,
        "Hour_Sin": np.sin(2 * np.pi * hour / 24),
        "Hour_Cos": np.cos(2 * np.pi * hour / 24),
        "Month_Sin": np.sin(2 * np.pi * month / 12),
        "Month_Cos": np.cos(2 * np.pi * month / 12),
        "Weekend": int(dow in ("Friday", "Saturday")),
        "Holiday": holiday,
        "Is_Peak_Hour": int(10 <= hour <= 12),
        "Is_Evening_Show": int(hour >= 19),
        "Season": get_season(month),
        # site identity / static
        "Site_Name": site.name,
        "Site_Capacity": capacity,
        "Site_Sensitivity": site.sensitivity,
        "Entry_Gates": entry_gates,
        "Exit_Gates": site.exit_gates,
        # live crowd
        "Current_Visitors": visitors,
        "Occupancy_Rate": round(visitors / capacity, 4),
        "Visitors_per_Gate": round(visitors / entry_gates, 1),
        "Visitors_Lag_1h": lag,
        "Visitors_Delta_1h": visitors - lag,
        "Queue_Length": queue_len,
        "Queue_Time": queue_time,
        "Queue_per_Gate": round(queue_len / entry_gates, 1),
        # staffing
        "Security_Staff": sec,
        "Medical_Team": medical,
        "Police_Units": police,
        "Staff_per_1000": round(sec / max(visitors, 1) * 1000, 2),
        "Medical_per_1000": round(medical / max(visitors, 1) * 1000, 2),
        # weather
        "Temperature": temp,
        "Humidity": humidity,
        "Wind_Speed": wind,
        "Visibility": vis,
        "Khamsin_Flag": int((wind >= 20) and (vis <= 7.5)),
        "Heat_Level": int(pd.cut([temp], bins=HEAT_BINS, labels=[0, 1, 2, 3, 4])[0]),
        # demand context
        "Special_Events": special,
        "VIP_Visits": vip,
        "School_Trips": school,
        "Event_Pressure": special + vip + school,
        "Emergency_Active": emergency_active,
    }
    return pd.DataFrame([row])[get_feature_order()]


def compute_composite_scores(
    occupancy_rate: float,
    site_sensitivity: float,
    temperature: float,
    wind_speed: float,
    visibility: float,
    security_staff: float,
    current_visitors: float,
    entry_gates: int,
    emergency_active: bool,
) -> dict[str, float]:
    """Descriptive-only composite scores (Notebook 01 formulas).

    These are excluded from the model's inputs as leaky (they're built FROM
    the same signals the model already sees / from Risk_Score itself), but
    the Reasoning Agent's explanations still reference them, so we compute
    them here from raw telemetry instead of asking the API caller to supply
    already-derived numbers.
    """
    crowd_density = float(np.clip(occupancy_rate * (0.7 + 0.3 * site_sensitivity), 0, 1.3))

    temp_comfort = 1 - min(abs(temperature - 24) / 20, 1)
    vis_comfort = visibility / 10
    wind_comfort = 1 - min(wind_speed / 60, 1)
    weather_score = float(np.clip(temp_comfort * 0.5 + vis_comfort * 0.3 + wind_comfort * 0.2, 0, 1))

    staff_ratio = security_staff / max(current_visitors, 1) * 1000
    security_score = float(np.clip(staff_ratio / 25, 0, 1))

    gate_capacity_per_hour = entry_gates * GATE_THROUGHPUT_PER_HOUR
    queue_pressure_raw = max(0.0, (current_visitors - gate_capacity_per_hour) / gate_capacity_per_hour)

    operational_load = float(np.clip(
        0.45 * occupancy_rate
        + 0.30 * min(queue_pressure_raw, 1)
        + 0.15 * (1 - security_score)
        + 0.10 * (1 if emergency_active else 0),
        0, 1,
    ))

    return {
        "crowd_density": round(crowd_density, 4),
        "weather_score": round(weather_score, 4),
        "security_score": round(security_score, 4),
        "operational_load": round(operational_load, 4),
    }


def score(raw: dict[str, Any], site, prev_visitors: float | None) -> dict[str, Any]:
    """Natural operational snapshot -> full risk assessment.

    Returns risk_level (the model's own classification), all four class
    probabilities, proba_critical, occupancy_rate/khamsin, and the
    descriptive composite scores for downstream use.
    """
    X = build_features(raw, site, prev_visitors)
    model = get_risk_model()
    level = str(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    prob_by_class = {cls: round(float(p), 4) for cls, p in zip(model_classes(), proba)}

    emergency_active = bool(int(raw.get("emergency_active", 0)))
    composites = compute_composite_scores(
        occupancy_rate=float(X["Occupancy_Rate"].iloc[0]),
        site_sensitivity=site.sensitivity,
        temperature=float(raw["temperature"]),
        wind_speed=float(raw["wind_speed"]),
        visibility=float(raw["visibility"]),
        security_staff=float(raw["security_staff"]),
        current_visitors=float(raw["current_visitors"]),
        entry_gates=site.entry_gates,
        emergency_active=emergency_active,
    )

    return {
        "risk_level": level,
        "probabilities": {c: prob_by_class[c] for c in RISK_ORDER},
        "proba_critical": prob_by_class["Critical"],
        # risk_score kept as a 0-100 display number for the API/UI —
        # driven by P(Critical), the metric the deck's use-case actually
        # gates the emergency protocol on, NOT reverse-engineered from
        # the deprecated risk_score_* buckets.
        "risk_score": round(prob_by_class["Critical"] * 100, 2),
        "occupancy_rate": float(X["Occupancy_Rate"].iloc[0]),
        "khamsin": bool(X["Khamsin_Flag"].iloc[0]),
        **composites,
    }
