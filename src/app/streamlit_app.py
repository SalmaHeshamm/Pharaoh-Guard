"""
PHARAOH GUARD — Heritage Site Risk Response System
Live monitoring command-center dashboard (card-grid layout).

Two things happen here, both tucked away below the 8 monitoring cards so
they stay the focus of the main screen:
  1. Full-day simulator — generates one synthetic telemetry reading per
     site, per hour, for a full 24-hour day, and scores all 8 sites
     together in a single request each hour (POST /predict/batch), so
     every card refreshes at once, hour by hour, driven by the real ML
     model + full agent pipeline. (Manual entry / /predict/manual has
     been removed — /predict is the trusted, model-backed path now.)
  2. Admin chat assistant — a chat box backed by POST /chat/admin, for
     asking about any site's live status, emergency protocols, or the
     daily report, and for dispatching real operational actions on
     request (e.g. "ابعت فريق أمن لسقارة").

Run with:
    streamlit run streamlit_app.py
(from the project root, with the FastAPI server already running —
default expected at http://127.0.0.1:8000)
"""
from __future__ import annotations

import base64
import math
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from config.site_master_data import SITE_MASTER_DATA

# --------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------- #
ASSETS_DIR = Path(__file__).parent / "assets"


def _b64(path: Path) -> str | None:
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return None


LOGO_B64 = _b64(ASSETS_DIR / "pharaoh_guard_logo.png")
ICON_B64 = _b64(ASSETS_DIR / "pharaoh_guard_icon.png")

# --------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------- #
st.set_page_config(
    page_title="PHARAOH GUARD — Command Center",
    page_icon=(f"data:image/png;base64,{ICON_B64}" if ICON_B64 else "🛡️"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------- #
# Global theme / CSS
# --------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-0: #06070A;
        --bg-1: #0C0F14;
        --panel: #14181F;
        --panel-border: rgba(212,175,55,0.16);
        --gold: #D9AE47;
        --gold-bright: #F0C24E;
        --text-hi: #F4EEE1;
        --text-mid: #ADA79A;
        --text-dim: #726F66;
        --green: #34C77B;
        --green-deep: #0E2B1B;
        --red: #E5423F;
        --red-deep: #2E0D0D;
        --amber: #E0A83F;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(42,78,143,0.10), transparent 45%),
            radial-gradient(circle at 85% 5%, rgba(217,174,71,0.07), transparent 40%),
            linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 45%, var(--bg-1) 100%);
        color: var(--text-hi);
    }

    .block-container { padding-top: 1rem; padding-bottom: 2.5rem; max-width: 1550px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08090C 0%, #0D1015 100%);
        border-right: 1px solid var(--panel-border);
    }
    section[data-testid="stSidebar"] * { color: var(--text-hi) !important; }

    h1, h2, h3 { font-family: 'Cinzel', serif !important; letter-spacing: 0.02em; }

    /* ---- Header ---- */
    .cc-header {
        display: flex; align-items: center; justify-content: space-between;
        background: linear-gradient(90deg, var(--panel) 0%, #10141B 100%);
        border: 1px solid var(--panel-border);
        border-radius: 14px;
        padding: 14px 26px;
        margin-bottom: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .cc-header-left { display: flex; align-items: center; gap: 14px; flex: 1; }
    .cc-logo-img { height: 42px; border-radius: 8px; }
    .cc-header-center { flex: 2; text-align: center; }
    .cc-title { font-family: 'Cinzel', serif; font-size: 21px; font-weight: 700;
        letter-spacing: 0.08em; color: var(--gold-bright); margin: 0; line-height: 1.15; }
    .cc-subtitle { font-size: 12px; color: var(--text-mid); letter-spacing: 0.04em; margin-top: 1px; }
    .cc-header-right { flex: 1; display: flex; flex-direction: column; align-items: flex-end; gap: 6px; }
    .cc-clock-time { font-size: 16px; font-weight: 600; color: var(--text-hi); font-variant-numeric: tabular-nums; }

    .status-pill { display: flex; align-items: center; gap: 8px; padding: 5px 13px;
        border-radius: 999px; font-size: 12px; font-weight: 600; letter-spacing: 0.03em; }
    .status-pill.on { background: rgba(52,199,123,0.12); color: var(--green); border: 1px solid rgba(52,199,123,0.35); }
    .status-pill.off { background: rgba(229,66,63,0.12); color: var(--red); border: 1px solid rgba(229,66,63,0.35); }

    .pulse-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .pulse-dot.on { background: var(--green); animation: pulseGreen 1.8s infinite; }
    .pulse-dot.off { background: var(--red); animation: pulseRed 1.8s infinite; }
    @keyframes pulseGreen { 0%{box-shadow:0 0 0 0 rgba(52,199,123,0.55);} 70%{box-shadow:0 0 0 8px rgba(52,199,123,0);} 100%{box-shadow:0 0 0 0 rgba(52,199,123,0);} }
    @keyframes pulseRed { 0%{box-shadow:0 0 0 0 rgba(229,66,63,0.55);} 70%{box-shadow:0 0 0 8px rgba(229,66,63,0);} 100%{box-shadow:0 0 0 0 rgba(229,66,63,0);} }

    /* ---- Latest event bar ---- */
    .event-bar { display: flex; align-items: center; gap: 10px; padding: 10px 18px;
        border-radius: 10px; margin-bottom: 12px; font-size: 13.5px; font-weight: 500; }
    .event-bar.alert { background: rgba(229,66,63,0.10); border: 1px solid rgba(229,66,63,0.35); color: #FFD9D6; }
    .event-bar.calm { background: rgba(52,199,123,0.08); border: 1px solid rgba(52,199,123,0.28); color: #CFEEDD; }

    /* ---- Status summary bar ---- */
    .status-bar { display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }
    .status-chip { flex: 1; min-width: 150px; background: var(--panel); border: 1px solid var(--panel-border);
        border-radius: 10px; padding: 10px 16px; display: flex; align-items: center; justify-content: space-between; }
    .status-chip .label { font-size: 12px; color: var(--text-mid); letter-spacing: 0.02em; }
    .status-chip .value { font-size: 19px; font-weight: 700; font-family: 'Cinzel', serif; }
    .status-chip .value.green { color: var(--green); }
    .status-chip .value.red { color: var(--red); }
    .status-chip .value.gold { color: var(--gold-bright); }

    /* ---- Site monitoring cards ---- */
    .section-title { font-size: 12px; text-transform: uppercase; letter-spacing: 0.14em;
        color: var(--gold); font-weight: 700; margin: 6px 0 10px 2px; display: flex; align-items: center; gap: 8px; }
    .section-title::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, var(--panel-border), transparent); }

    .site-card { border-radius: 16px; padding: 16px 18px 14px 18px; margin-bottom: 4px;
        position: relative; overflow: hidden; border: 1px solid transparent; min-height: 168px; }

    .site-card.normal {
        background: linear-gradient(160deg, var(--green-deep) 0%, #0A2016 100%);
        border-color: rgba(52,199,123,0.35);
        animation: cardBreathe 4.5s ease-in-out infinite;
    }
    .site-card.risk {
        background: linear-gradient(160deg, var(--red-deep) 0%, #22090A 100%);
        border-color: rgba(229,66,63,0.55);
        animation: cardRiskPulse 1.6s ease-in-out infinite;
    }
    @keyframes cardBreathe {
        0%, 100% { box-shadow: 0 0 16px rgba(52,199,123,0.18); }
        50%      { box-shadow: 0 0 28px rgba(52,199,123,0.36); }
    }
    @keyframes cardRiskPulse {
        0%, 100% { box-shadow: 0 0 20px rgba(229,66,63,0.45), 0 0 0 0 rgba(229,66,63,0.45); }
        50%      { box-shadow: 0 0 40px rgba(229,66,63,0.85), 0 0 0 8px rgba(229,66,63,0); }
    }

    .site-card-top { display: flex; align-items: center; justify-content: space-between; }
    .site-card-name { font-size: 14.5px; font-weight: 700; color: var(--text-hi); display: flex; align-items: center; gap: 8px; }
    .site-card-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .site-card-dot.normal { background: var(--green); }
    .site-card-dot.risk { background: var(--red); animation: pulseRed 1.4s infinite; }

    .site-card-status { font-size: 19px; font-weight: 800; margin-top: 14px; letter-spacing: 0.01em; }
    .site-card-status.normal { color: var(--green); }
    .site-card-status.risk { color: #FFEDEC; }

    .site-card-sub { font-size: 12px; color: var(--text-mid); margin-top: 3px; }
    .site-card-sub.risk { color: #F5C6C4; }

    .site-card-footer { margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.08);
        font-size: 11px; color: var(--text-dim); display: flex; justify-content: space-between; }

    /* "View report" control under each card, styled as a slim action bar */
    div[data-testid="stButton"] button {
        background: var(--bg-1) !important; color: var(--gold) !important;
        border: 1px solid var(--panel-border) !important; border-radius: 8px !important;
        font-size: 11.5px !important; padding: 4px 8px !important; letter-spacing: 0.03em;
        margin-top: -2px;
    }
    div[data-testid="stButton"] button:hover {
        border-color: var(--gold) !important; box-shadow: 0 0 12px rgba(217,174,71,0.25);
    }

    /* ---- Risk badges / chips (used in dialog + run-analysis feed) ---- */
    .risk-badge { display: inline-block; padding: 4px 14px; border-radius: 999px; font-weight: 700; font-size: 13px; letter-spacing: 0.02em; }
    .risk-badge.Critical { animation: cardRiskPulse 1.6s infinite; }
    .driver-chip { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 11.5px;
        margin-right: 6px; margin-bottom: 6px; background: rgba(201,168,118,0.12); color: #C9A876; border: 1px solid rgba(201,168,118,0.25); }
    .action-card { border: 1px solid var(--panel-border); background: var(--bg-1); border-radius: 10px;
        padding: 12px 16px; margin-bottom: 8px; }
    .stat-chip { display: inline-block; background: var(--bg-1); border: 1px solid var(--panel-border);
        border-radius: 8px; padding: 6px 10px; margin: 3px; font-size: 12px; color: var(--text-mid); }
    .stat-chip b { color: var(--text-hi); }

    div[data-testid="stForm"] { background: var(--panel); border: 1px solid var(--panel-border); border-radius: 14px; padding: 10px 6px; }
    .streamlit-expanderHeader { background: var(--bg-1) !important; border-radius: 8px !important; }
    .map-frame { border: 1px solid var(--panel-border); border-radius: 14px; overflow: hidden;
        background: var(--panel); padding: 6px; }

    @media (max-width: 1100px) {
        .status-bar { flex-direction: column; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------- #
# Risk styling / site metadata
# --------------------------------------------------------------------- #
RISK_STYLE = {
    "Low":      {"text": "#34C77B", "bg": "rgba(52,199,123,0.14)", "border": "rgba(52,199,123,0.35)", "label": "Low",      "map": "#34C77B"},
    "Medium":   {"text": "#E0A83F", "bg": "rgba(224,168,63,0.14)", "border": "rgba(224,168,63,0.35)", "label": "Medium",   "map": "#E0A83F"},
    "High":     {"text": "#E5423F", "bg": "rgba(229,66,63,0.16)",  "border": "rgba(229,66,63,0.40)",  "label": "High",     "map": "#E5423F"},
    "Critical": {"text": "#FFFFFF", "bg": "#7A1414",               "border": "#E5423F",               "label": "Critical", "map": "#E5423F"},
}
RISK_STATES = ("High", "Critical")  # levels that flip a card into RISK mode

SITE_ICONS = {
    "Giza Pyramids": "🔺",
    "Grand Egyptian Museum": "🏛️",
    "Saqqara": "🪨",
    "Luxor Temple": "🛕",
    "Karnak Temple": "⛩️",
    "Abu Simbel": "🗿",
    "Valley of the Kings": "⚱️",
    "Citadel of Cairo": "🏰",
}

SITE_COORDS = {
    "Giza Pyramids":         (29.9792, 31.1342),
    "Grand Egyptian Museum": (29.9938, 31.1108),
    "Saqqara":               (29.8710, 31.2165),
    "Luxor Temple":          (25.6997, 32.6396),
    "Karnak Temple":         (25.7188, 32.6573),
    "Abu Simbel":            (22.3372, 31.6258),
    "Valley of the Kings":   (25.7402, 32.6014),
    "Citadel of Cairo":      (30.0287, 31.2601),
}

SITE_NAMES = list(SITE_MASTER_DATA.keys())

if "history" not in st.session_state:
    st.session_state.history = []  # list[dict]

if "site_status" not in st.session_state:
    st.session_state.site_status = {}  # site_name -> {risk_score, risk_level, escalated, updated, response, payload}

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # list[{"role": "user"/"assistant", "content": str}]


# --------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------- #
def check_health(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def call_predict_batch(base_url: str, payloads: list[dict]) -> list[dict] | None:
    """One round trip scoring all 8 sites' readings for one hour at once."""
    try:
        r = requests.post(f"{base_url}/predict/batch", json={"readings": payloads}, timeout=180)
        r.raise_for_status()
        return r.json()["results"]
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return None


def call_chat_admin(base_url: str, message: str, session_id: str = "streamlit_admin") -> dict | None:
    try:
        r = requests.post(
            f"{base_url}/chat/admin", json={"message": message, "session_id": session_id}, timeout=90
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the chat assistant: {exc}")
        return None


# --------------------------------------------------------------------- #
# Synthetic reading generator (Simulator mode) — unchanged
# --------------------------------------------------------------------- #
def synth_reading(site_name: str, hour: int, sim_dt: datetime, shortage_today: bool) -> dict:
    profile = SITE_MASTER_DATA[site_name]

    if 6 <= hour <= 20:
        curve = max(0.05, math.sin((hour - 6) / 14 * math.pi))
    else:
        curve = 0.03
    occupancy_rate = min(0.98, round(curve * profile.popularity * random.uniform(0.75, 1.15), 4))
    current_visitors = int(profile.capacity * occupancy_rate)
    crowd_density = min(1.0, round(occupancy_rate * random.uniform(0.85, 1.1), 4))

    queue_length = int(current_visitors * random.uniform(0.05, 0.18))
    queue_time = round(min(90.0, queue_length / max(1, profile.entry_gates) / 1.1), 1)

    temperature = round(22 + profile.base_temp_offset + random.uniform(-4, 4), 1)
    humidity = round(random.uniform(30, 70), 1)
    wind_speed = round(random.uniform(2, 25), 1)
    visibility = round(random.uniform(6, 10), 1)
    weather_score = round(
        max(0.0, min(1.0,
            1 - abs(temperature - 24) / 40 - wind_speed / 120 - max(0, 10 - visibility) / 20
            + random.uniform(-0.05, 0.05)
        )), 4
    )

    security_staff = max(2, round(profile.capacity / 470))
    medical_team = max(1, round(profile.capacity / 2140))
    police_units = max(1, round(profile.capacity / 3750))

    demand_pressure = occupancy_rate * profile.capacity / 220 + 5
    security_score = round(max(0.05, min(1.0, security_staff / demand_pressure)), 4)
    if shortage_today:
        security_score = round(security_score * random.uniform(0.2, 0.45), 4)

    operational_load = round(
        max(0.0, min(1.0, 0.4 * occupancy_rate + 0.3 * (queue_time / 60) + 0.3 * random.uniform(0.3, 0.9))), 4
    )

    school_trips = bool(9 <= hour <= 13 and random.random() < 0.12)
    special_events = bool(random.random() < 0.04)
    vip_visits = bool(random.random() < 0.03)

    roll = random.random()
    if roll < 0.02:
        emergency_type = "Medical"
    elif roll < 0.035:
        emergency_type = "Security"
    elif roll < 0.05:
        emergency_type = "Lost Person"
    else:
        emergency_type = "No_Emergency"

    return {
        "timestamp": sim_dt.isoformat(),
        "site_name": site_name,
        "hour": hour,
        "current_visitors": current_visitors,
        "site_capacity": profile.capacity,
        "entry_gates": profile.entry_gates,
        "exit_gates": profile.exit_gates,
        "security_staff": security_staff,
        "medical_team": medical_team,
        "police_units": police_units,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "visibility": visibility,
        "special_events": special_events,
        "vip_visits": vip_visits,
        "school_trips": school_trips,
        "emergency_type": emergency_type,
        "queue_length": queue_length,
        "queue_time": queue_time,
        "occupancy_rate": occupancy_rate,
        "crowd_density": crowd_density,
        "weather_score": weather_score,
        "security_score": security_score,
        "site_sensitivity": profile.sensitivity,
        "operational_load": operational_load,
    }


# --------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------- #
def risk_badge_html(level: str) -> str:
    style = RISK_STYLE.get(level, RISK_STYLE["Medium"])
    return (
        f'<span class="risk-badge {level}" style="background:{style["bg"]};color:{style["text"]};'
        f'border:1px solid {style["border"]}">{style["label"]}</span>'
    )


def _update_site_status(site_name: str, risk_score: float, risk_level: str, escalated: bool,
                         response: dict | None = None, payload: dict | None = None) -> None:
    # Always overwrite with the latest reading — this is what makes a card
    # "sticky" red across consecutive risky readings, and flip back to green
    # the moment a normal reading comes in for that site.
    st.session_state.site_status[site_name] = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "escalated": escalated,
        "updated": datetime.now(),
        "response": response,
        "payload": payload,
    }


def render_response(response: dict, site_name: str, risk_score: float, risk_level: str, payload: dict | None = None) -> None:
    escalated = response.get("escalated", False)
    result = response.get("result")
    _update_site_status(site_name, risk_score, risk_level, escalated, response=response, payload=payload)

    top = st.container()
    with top:
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.markdown(f"### {site_name}")
        c2.markdown(
            '<p style="font-size:12px;color:#ADA79A;margin:0;letter-spacing:0.05em;text-transform:uppercase;">Risk score</p>'
            f'<p style="font-size:24px;font-weight:700;margin:0;color:#F4EEE1;">{risk_score:.1f} / 100</p>',
            unsafe_allow_html=True,
        )
        c3.markdown(risk_badge_html(risk_level), unsafe_allow_html=True)

    if not escalated or result is None:
        st.success("Situation normal — not escalated to the agent pipeline.")
        _append_history(site_name, risk_score, risk_level, "None", "—")
        return

    explanation = result["explanation"]
    recommendations = result["recommendations"]
    dispatch_results = result.get("dispatch_results", [])

    st.markdown(explanation["summary_en"])
    chips = "".join(f'<span class="driver-chip">{d}</span>' for d in explanation.get("key_drivers", []))
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("#### Recommended actions")
    top_action_desc = "—"
    any_real_dispatch = False
    for action, dispatch in zip(recommendations["actions"], dispatch_results or [{}] * len(recommendations["actions"])):
        dry_run = dispatch.get("dry_run", True)
        dispatched = dispatch.get("dispatched", False)
        if dispatched and not dry_run:
            status = "🟢 Sent to the operations team"
            any_real_dispatch = True
        elif dispatched and dry_run:
            status = "🟡 Logged only (dry run)"
        else:
            status = "⚪ Not executed"
        st.markdown(
            f'<div class="action-card"><b>{action["description"]}</b>'
            f'<br><span style="font-size:12.5px;color:#ADA79A;">Priority {action["priority"]} — {status}</span></div>',
            unsafe_allow_html=True,
        )
        if top_action_desc == "—":
            top_action_desc = action["description"]

    show_json = st.toggle(
        "Show technical details (JSON)",
        key=f"response_json_{site_name}_{datetime.now().timestamp()}"
    )
    if show_json:
        st.json(response)

    dispatch_note = "Dry run" if not any_real_dispatch else "Sent for real"
    _append_history(site_name, risk_score, risk_level, top_action_desc, dispatch_note)


def _append_history(site_name, risk_score, risk_level, action, dispatch_note):
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Site": site_name,
        "Risk score": round(risk_score, 1),
        "Level": RISK_STYLE.get(risk_level, {}).get("label", risk_level),
        "Top action": action,
        "Dispatch status": dispatch_note,
    })


def render_history_log():
    """Compact session log."""

    if not st.session_state.history:
        return

    show_history = st.toggle(
        "🗂️ Show Session Activity Log",
        key="show_session_history"
    )

    if show_history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------------------------- #
# Site detail dialog (drawer/modal)
# --------------------------------------------------------------------- #
SENSOR_FIELDS = [
    ("current_visitors", "Current visitors"),
    ("site_capacity", "Capacity"),
    ("occupancy_rate", "Occupancy rate"),
    ("crowd_density", "Crowd density"),
    ("queue_length", "Queue length"),
    ("queue_time", "Queue time (min)"),
    ("temperature", "Temperature °C"),
    ("humidity", "Humidity %"),
    ("wind_speed", "Wind speed"),
    ("visibility", "Visibility"),
    ("security_staff", "Security staff"),
    ("medical_team", "Medical team"),
    ("police_units", "Police units"),
    ("emergency_type", "Emergency type"),
]


@st.dialog("Site Report", width="large")
def open_site_dialog(site_name: str):
    icon = SITE_ICONS.get(site_name, "🏛️")
    info = st.session_state.site_status.get(site_name)

    st.markdown(f"### {icon} {site_name}")

    if not info:
        st.info("No telemetry received yet for this site in this session. Run the simulator or "
                 "manual entry (below the cards) to populate live data for this site.")
        return

    level = info["risk_level"]
    score = info["risk_score"]

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.markdown(risk_badge_html(level), unsafe_allow_html=True)
    c2.metric("Risk score", f"{score:.1f} / 100")
    c3.metric("Last reading", info["updated"].strftime("%H:%M:%S"))
    st.caption(f"Escalated to agent pipeline: {'Yes' if info['escalated'] else 'No'}")

    payload = info.get("payload")
    if payload:
        st.markdown("#### 📡 Sensor readings")
        chips = "".join(
            f'<span class="stat-chip">{label}: <b>{payload[key]}</b></span>'
            for key, label in SENSOR_FIELDS if key in payload
        )
        st.markdown(chips, unsafe_allow_html=True)
        st.caption(f"Reading timestamp: {payload.get('timestamp', '—')}")

    response = info.get("response")
    result = response.get("result") if response else None

    if response and response.get("escalated") and result:
        explanation = result["explanation"]
        recommendations = result["recommendations"]
        dispatch_results = result.get("dispatch_results", [])

        st.markdown("#### 🧠 AI explanation")
        st.markdown(explanation["summary_en"])
        chips = "".join(f'<span class="driver-chip">{d}</span>' for d in explanation.get("key_drivers", []))
        st.markdown(chips, unsafe_allow_html=True)

        st.markdown("#### ✅ Recommended actions & dispatch status")
        for action, dispatch in zip(recommendations["actions"], dispatch_results or [{}] * len(recommendations["actions"])):
            dry_run = dispatch.get("dry_run", True)
            dispatched = dispatch.get("dispatched", False)
            if dispatched and not dry_run:
                status = "🟢 Sent to the operations team"
            elif dispatched and dry_run:
                status = "🟡 Logged only (dry run)"
            else:
                status = "⚪ Not executed"
            st.markdown(
                f'<div class="action-card"><b>{action["description"]}</b>'
                f'<br><span style="font-size:12.5px;color:#ADA79A;">Priority {action["priority"]} — {status}</span></div>',
                unsafe_allow_html=True,
            )

        show_json = st.toggle(
            "🔧 Technical details (raw JSON)",
            key=f"dialog_json_{site_name}_{datetime.now().timestamp()}"
        )
        if show_json:
            st.json(response)
    else:
        st.success("Situation normal — not escalated to the agent pipeline.")
        if response:
            with st.expander("🔧 Technical details (raw JSON)"):
                st.json(response)


# --------------------------------------------------------------------- #
# Sidebar — connection + mode selection
# --------------------------------------------------------------------- #
with st.sidebar:
    if ICON_B64:
        st.markdown(
            f'<div style="text-align:center;margin-bottom:6px;">'
            f'<img src="data:image/png;base64,{ICON_B64}" style="width:60px;border-radius:14px;" /></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="text-align:center;font-family:Cinzel,serif;font-size:16px;font-weight:700;'
        'color:#F0C24E;letter-spacing:0.08em;">PHARAOH GUARD</div>'
        '<div style="text-align:center;font-size:10.5px;color:#ADA79A;margin-bottom:14px;">HERITAGE OPS CONTROL</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    base_url = st.text_input("API base URL", value="http://127.0.0.1:8000")
    healthy = check_health(base_url)
    st.markdown(
        (
            '<div class="status-pill on"><span class="pulse-dot on"></span>Connected to the API</div>'
            if healthy else
            '<div class="status-pill off"><span class="pulse-dot off"></span>Can\'t reach the API</div>'
        ),
        unsafe_allow_html=True,
    )
    if not healthy:
        st.caption("Make sure uvicorn is running: `uvicorn api.main:app --reload`")

    st.markdown("---")
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.site_status = {}
        st.rerun()


# --------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------- #
now = datetime.now()
logo_html = (
    f'<img class="cc-logo-img" src="data:image/png;base64,{LOGO_B64}" />'
    if LOGO_B64 else '<span style="font-size:30px;">🛡️</span>'
)
status_html = (
    '<div class="status-pill on"><span class="pulse-dot on"></span>Online</div>'
    if healthy else
    '<div class="status-pill off"><span class="pulse-dot off"></span>Offline</div>'
)

st.markdown(
    f"""
    <div class="cc-header">
        <div class="cc-header-left">{logo_html}</div>
        <div class="cc-header-center">
            <p class="cc-title">PHARAOH GUARD</p>
            <p class="cc-subtitle">Heritage Monitoring Command Center</p>
        </div>
        <div class="cc-header-right">
            <div class="cc-clock-time">{now.strftime('%H:%M:%S')}</div>
            {status_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------- #
# Latest event bar
# --------------------------------------------------------------------- #
statuses = st.session_state.site_status
latest_risk_event = None
for h in reversed(st.session_state.history):
    if h["Level"] in RISK_STATES:
        latest_risk_event = h
        break

dashboard_placeholder = st.empty()


def render_dashboard_top() -> None:
    """
    Renders the latest-event bar, status summary bar, and the 8 site
    cards, always reading fresh from st.session_state.site_status. Called
    once on initial page load AND again after every simulated hour, via
    dashboard_placeholder, so the cards actually flip color live during
    the run instead of only updating once the whole 24-hour loop ends.
    """
    statuses = st.session_state.site_status

    latest_risk_event = None
    for h in reversed(st.session_state.history):
        if h["Level"] in RISK_STATES:
            latest_risk_event = h
            break

    if latest_risk_event:
        st.markdown(
            f'<div class="event-bar alert">🚨 <b>Latest Event</b>&nbsp;&nbsp;'
            f'{latest_risk_event["Time"]} • {latest_risk_event["Level"]} Risk detected at '
            f'{latest_risk_event["Site"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="event-bar calm">✅ All monitored sites are operating normally.</div>',
            unsafe_allow_html=True,
        )

    # ---- Status summary bar ---- #
    total_sites = len(SITE_NAMES)
    active_risks = sum(1 for s in statuses.values() if s["risk_level"] in RISK_STATES)
    safe_sites = total_sites - active_risks
    latest_update = max((s["updated"] for s in statuses.values()), default=None)
    latest_update_str = latest_update.strftime("%H:%M:%S") if latest_update else "—"

    st.markdown(
        f"""
        <div class="status-bar">
            <div class="status-chip"><span class="label">🟢 Safe Sites</span><span class="value green">{safe_sites}</span></div>
            <div class="status-chip"><span class="label">🔴 Active Risks</span><span class="value red">{active_risks}</span></div>
            <div class="status-chip"><span class="label">🏛 Total Sites</span><span class="value gold">{total_sites}</span></div>
            <div class="status-chip"><span class="label">🕒 Last Update</span><span class="value">{latest_update_str}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- 8 monitoring cards — 2 rows x 4 columns ---- #
    st.markdown('<div class="section-title">🛰️ Site Monitoring</div>', unsafe_allow_html=True)

    for row_start in (0, 4):
        cols = st.columns(4)
        for i, site in enumerate(SITE_NAMES[row_start:row_start + 4]):
            info = statuses.get(site)
            level = info["risk_level"] if info else "Low"
            is_risk = level in RISK_STATES
            icon = SITE_ICONS.get(site, "🏛️")
            last_reading = info["updated"].strftime("%H:%M:%S") if info else "No data yet"

            with cols[i]:
                if is_risk:
                    st.markdown(
                        f"""
                        <div class="site-card risk">
                            <div class="site-card-top">
                                <div class="site-card-name">{icon} {site}</div>
                                <div class="site-card-dot risk"></div>
                            </div>
                            <div class="site-card-status risk">🚨 RISK DETECTED</div>
                            <div class="site-card-sub risk">Risk Level: {level} · Detected {last_reading}</div>
                            <div class="site-card-footer"><span>Last Reading</span><span>{last_reading}</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="site-card normal">
                            <div class="site-card-top">
                                <div class="site-card-name">{icon} {site}</div>
                                <div class="site-card-dot normal"></div>
                            </div>
                            <div class="site-card-status normal">✔ NORMAL</div>
                            <div class="site-card-sub">Monitoring...</div>
                            <div class="site-card-footer"><span>Last Reading</span><span>{last_reading}</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)


with dashboard_placeholder.container():
    render_dashboard_top()

# ---- View a site's full report — one-time selectbox, not tied to the
# refreshing placeholder above, so its widget isn't recreated every hour
# tick during the simulation loop (which would raise duplicate-key errors). ---- #
col_sel, col_btn = st.columns([4, 1])
selected_site_for_report = col_sel.selectbox("📋 View a site's full report", SITE_NAMES, label_visibility="collapsed")
if col_btn.button("🔍 View Report", use_container_width=True):
    open_site_dialog(selected_site_for_report)

if not healthy:
    st.warning("The FastAPI server isn't running. Start it first with: `uvicorn api.main:app --reload`")

# --------------------------------------------------------------------- #
# Live Map (secondary section, below the cards)
# --------------------------------------------------------------------- #
with st.expander("🗺️ Live Map", expanded=False):
    lats, lons, colors, sizes, hover_texts, customdata = [], [], [], [], [], []
    halo_lats, halo_lons, halo_colors, halo_sizes = [], [], [], []

    for site in SITE_NAMES:
        lat, lon = SITE_COORDS.get(site, (26.8, 30.8))
        st_info = statuses.get(site)
        level = st_info["risk_level"] if st_info else "Low"
        score = st_info["risk_score"] if st_info else None
        style = RISK_STYLE.get(level, RISK_STYLE["Low"])

        lats.append(lat); lons.append(lon)
        colors.append(style["map"])
        sizes.append(22 if level in RISK_STATES else 15)
        customdata.append(site)
        score_txt = f"{score:.1f}/100" if score is not None else "no reading yet"
        hover_texts.append(f"{site}<br>Status: {level}<br>Risk score: {score_txt}")

        if level in RISK_STATES:
            halo_lats.append(lat); halo_lons.append(lon)
            halo_colors.append(style["map"])
            halo_sizes.append(46 if level == "Critical" else 36)

    fig = go.Figure()
    if halo_lats:
        fig.add_trace(go.Scattermap(
            lat=halo_lats, lon=halo_lons, mode="markers",
            marker=dict(size=halo_sizes, color=halo_colors, opacity=0.30),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scattermap(
        lat=lats, lon=lons, mode="markers+text",
        marker=dict(size=sizes, color=colors, opacity=0.95),
        text=[s.split()[0] for s in SITE_NAMES],
        textposition="top center",
        textfont=dict(color="#F4EEE1", size=11),
        customdata=customdata,
        hovertext=hover_texts, hoverinfo="text",
        showlegend=False,
    ))
    fig.update_layout(
        map=dict(style="dark", center=dict(lat=26.8, lon=30.9), zoom=4.6),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
    )
    st.markdown('<div class="map-frame">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key="site_map")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------- #
# Run Analysis — one button = a full simulated day (24 hourly readings)
# sent for ALL 8 sites at once, one round trip per hour via /predict/batch.
# --------------------------------------------------------------------- #
with st.expander("⚙️ Run Full-Day Simulation (all 8 sites)", expanded=False):
    st.caption(
        "Generates one synthetic reading per site, per hour, for a full 24-hour day, and "
        "scores all 8 sites together in a single request each hour (/predict/batch) — so "
        "the cards above refresh for every site at once, hour by hour."
    )

    speed = st.slider("Simulation speed (seconds/hour)", 0.1, 3.0, 0.5, step=0.1)
    shortage_today = st.checkbox(
        "Simulate a temporary security staff shortage today, at every site (to test High/Critical scenarios)",
        value=False,
    )

    run = st.button("▶️ Start full-day simulation (24 hours × 8 sites)", type="primary", disabled=not healthy)

    if run:
        sim_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        progress = st.progress(0.0)
        hour_slot = st.container()
        hours = list(range(24))

        for i, hour in enumerate(hours):
            sim_dt = sim_date.replace(hour=hour)
            payloads = [synth_reading(site, hour, sim_dt, shortage_today) for site in SITE_NAMES]

            results = call_predict_batch(base_url, payloads)

            with hour_slot:
                st.markdown(f"**⏱ Hour {hour:02d}:00 — all 8 sites**")
                if results:
                    row_cols = st.columns(4)
                    for j, (site, payload, result) in enumerate(zip(SITE_NAMES, payloads, results)):
                        risk_score = result["risk_score"]
                        risk_level = result["risk_level"]
                        _update_site_status(site, risk_score, risk_level, result.get("escalated", False),
                                             response=result, payload=payload)
                        top_action = "—"
                        if result.get("escalated") and result.get("result"):
                            actions = result["result"]["recommendations"]["actions"]
                            top_action = actions[0]["description"] if actions else "—"
                        _append_history(site, risk_score, risk_level, top_action,
                                         "Dry run" if result.get("escalated") else "—")

                        with row_cols[j % 4]:
                            st.markdown(
                                f'<div style="font-size:12.5px;"><b>{SITE_ICONS.get(site, "🏛️")} {site}</b><br>'
                                f'{risk_badge_html(risk_level)} <span style="color:#ADA79A;">{risk_score:.1f}/100</span></div>',
                                unsafe_allow_html=True,
                            )
                        if result.get("escalated") and result.get("result"):
                            show_details = st.toggle(
                                f"🚨 {site} — escalated, view details",
                                key=f"sim_detail_{site}_{hour}",
                            )
                            if show_details:
                                st.markdown(result["result"]["explanation"]["summary_en"])
                                for action in result["result"]["recommendations"]["actions"]:
                                    st.markdown(f"- **{action['description']}** (priority {action['priority']})")
                st.markdown("---")

            progress.progress((i + 1) / len(hours))
            with dashboard_placeholder.container():
                render_dashboard_top()
            time.sleep(speed)

        st.success("Full-day simulation finished ✅ — cards above reflect the latest (23:00) reading for every site.")
        st.rerun()

    render_history_log()

# --------------------------------------------------------------------- #
# Admin Chat Assistant — ask about any site, protocols, today's report,
# or ask it to dispatch a real action (e.g. "ابعت فريق أمن لسقارة").
# --------------------------------------------------------------------- #
st.markdown('<div class="section-title">🤖 Admin Assistant</div>', unsafe_allow_html=True)

with st.container():
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for action in msg.get("actions_taken", []) or []:
                status_icon = "🟢" if action.get("dispatched") and not action.get("dry_run") else "🟡"
                st.caption(
                    f"{status_icon} {action.get('action_type', '?')} @ {action.get('target_site', '?')} — "
                    f"{action.get('detail', '')}"
                )

    col_input, col_clear = st.columns([6, 1])
    with col_clear:
        if st.button("🗑️ Clear chat", use_container_width=True, disabled=not healthy):
            st.session_state.chat_messages = []
            try:
                requests.post(f"{base_url}/chat/admin/clear", params={"session_id": "streamlit_admin"}, timeout=5)
            except requests.RequestException:
                pass
            st.rerun()

    user_msg = st.chat_input(
        "اسأل عن حالة أي موقع، بروتوكول الطوارئ، تقرير النهاردة، أو اطلب إجراء (مثلاً: ابعت فريق أمن لسقارة)...",
        disabled=not healthy,
    )
    if user_msg:
        st.session_state.chat_messages.append({"role": "user", "content": user_msg})
        with st.spinner("بيفكر..."):
            outcome = call_chat_admin(base_url, user_msg)
        if outcome:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": outcome["reply"],
                "actions_taken": outcome.get("actions_taken", []),
            })
        st.rerun()