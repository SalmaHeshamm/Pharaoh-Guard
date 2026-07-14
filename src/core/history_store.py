"""
Minimal thread-safe per-site "last observed visitor count" cache.

The model needs Visitors_Lag_1h / Visitors_Delta_1h (momentum features),
which require knowing the previous hour's reading for the same site.
A single RawTelemetry snapshot doesn't carry that by itself, so this
in-memory store remembers the last reading per site between /predict
calls. Swap for Redis in production if the API runs across multiple
processes/workers — this exists so a single-process deployment (or the
demo) works correctly out of the box.
"""
from __future__ import annotations

import threading
from datetime import datetime

_lock = threading.Lock()
_last_reading: dict[str, tuple[datetime, float]] = {}


def get_previous_visitors(site_name: str) -> float | None:
    """Last known visitor count for this site, or None on first sighting."""
    with _lock:
        entry = _last_reading.get(site_name)
        return entry[1] if entry else None


def update(site_name: str, timestamp: datetime, current_visitors: float) -> None:
    """Record this reading as the new 'previous' value for the site."""
    with _lock:
        _last_reading[site_name] = (timestamp, current_visitors)


def clear() -> None:
    """Test/reset hook."""
    with _lock:
        _last_reading.clear()
