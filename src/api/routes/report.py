"""
GET /report/daily?date=YYYY-MM-DD

Aggregates every escalated snapshot recorded today (or a given date) into
one narrative report via the Report Agent.
"""
from __future__ import annotations

from datetime import date as date_cls

from fastapi import APIRouter, HTTPException

from agents.report_agent import ReportAgent
from core.run_store import get_results
from core.schemas import DailyReport

router = APIRouter(prefix="/report", tags=["report"])
_report_agent = ReportAgent()


@router.get("/daily", response_model=DailyReport)
def daily_report(date: str | None = None) -> DailyReport:
    target_date = date or date_cls.today().isoformat()
    results = get_results(target_date)

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No escalated snapshots recorded for {target_date} yet.",
        )

    return _report_agent.run(target_date, results)
