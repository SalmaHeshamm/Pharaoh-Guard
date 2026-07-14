"""
Minimal in-memory store of today's escalated AgentRunResults, used to
feed the Report Agent. Swap for Redis/Postgres in production — this
exists so the reporting flow is demonstrable without extra infra.
"""
from __future__ import annotations

from datetime import date as date_cls

from core.schemas import AgentRunResult

_store: dict[str, list[AgentRunResult]] = {}


def record(result: AgentRunResult, run_date: str | None = None) -> None:
    key = run_date or date_cls.today().isoformat()
    _store.setdefault(key, []).append(result)


def get_results(run_date: str | None = None) -> list[AgentRunResult]:
    key = run_date or date_cls.today().isoformat()
    return _store.get(key, [])


def clear(run_date: str | None = None) -> None:
    key = run_date or date_cls.today().isoformat()
    _store.pop(key, None)
