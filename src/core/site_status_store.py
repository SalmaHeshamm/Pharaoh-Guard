"""
Thread-safe in-memory cache of the *latest* reading for every site,
escalated or not. `run_store` only remembers escalated AgentRunResults
(for the daily report), and `history_store` only remembers the previous
visitor count (for momentum features) — neither is enough to answer
"what's the status of Karnak right now", which is exactly what the admin
chat agent needs. This store fills that gap.

In-memory by design (matches the rest of the project's stores) — resets
on restart.
"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

_lock = threading.Lock()
_latest: dict[str, dict[str, Any]] = {}


def update(
    site_name: str,
    *,
    timestamp: datetime,
    risk_score: float,
    risk_level: str,
    escalated: bool,
    emergency_type: str = "No_Emergency",
    occupancy_rate: float | None = None,
    current_visitors: int | None = None,
    explanation_summary: str | None = None,
    top_action: str | None = None,
) -> None:
    with _lock:
        _latest[site_name] = {
            "site_name": site_name,
            "timestamp": timestamp.isoformat(),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "escalated": escalated,
            "emergency_type": emergency_type,
            "occupancy_rate": occupancy_rate,
            "current_visitors": current_visitors,
            "explanation_summary": explanation_summary,
            "top_action": top_action,
        }


def get(site_name: str) -> dict[str, Any] | None:
    with _lock:
        return _latest.get(site_name)


def get_all() -> dict[str, dict[str, Any]]:
    with _lock:
        return dict(_latest)


def clear() -> None:
    with _lock:
        _latest.clear()
