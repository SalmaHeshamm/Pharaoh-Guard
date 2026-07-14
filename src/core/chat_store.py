"""
In-memory conversation history for the admin chat agent, keyed by
session_id. Deliberately not persisted (matches the rest of the
project's in-memory stores) — resets on server restart or when the
admin clears the chat.
"""
from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_sessions: dict[str, list[dict[str, Any]]] = {}

MAX_TURNS_KEPT = 20  # user+assistant pairs; keeps prompts bounded


def get_history(session_id: str) -> list[dict[str, Any]]:
    with _lock:
        return list(_sessions.get(session_id, []))


def append(session_id: str, message: dict[str, Any]) -> None:
    with _lock:
        history = _sessions.setdefault(session_id, [])
        history.append(message)
        # Trim from the front, keeping the list roughly bounded.
        max_messages = MAX_TURNS_KEPT * 2
        if len(history) > max_messages:
            del history[: len(history) - max_messages]


def clear(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)
