"""
Report Agent: runs separately from the per-snapshot pipeline (see
orchestration/graph.py — this is invoked by api/routes/report.py on a
schedule or on demand), aggregating a batch of AgentRunResult objects
into one narrative daily report.
"""
from __future__ import annotations

from collections import Counter

from core.schemas import AgentRunResult, DailyReport
from tools.llm_client import complete

_SYSTEM_PROMPT = """\
You are writing a daily operations summary for site managers overseeing \
Egyptian heritage sites. Write 4-6 sentences in Egyptian Arabic colloquial, \
professional but not overly formal. Focus on which sites need attention \
tomorrow and why, based only on the data given. Do not invent numbers.
"""


class ReportAgent:
    name = "report_agent"

    def run(self, date: str, run_results: list[AgentRunResult]) -> DailyReport:
        risk_counter = Counter(r.snapshot.risk_level.value for r in run_results)
        critical_sites = sorted({
            r.snapshot.site_name for r in run_results
            if r.snapshot.risk_level.value == "Critical"
        })

        top_recs = [
            action.description
            for r in run_results
            for action in r.recommendations.actions
            if action.priority <= 2
        ][:10]

        narrative = complete(_SYSTEM_PROMPT, self._build_prompt(date, risk_counter, critical_sites, top_recs))

        return DailyReport(
            date=date,
            total_snapshots=len(run_results),
            risk_level_breakdown=dict(risk_counter),
            critical_sites=critical_sites,
            top_recommendations=top_recs,
            narrative_ar=narrative,
        )

    @staticmethod
    def _build_prompt(
        date: str, risk_counter: Counter, critical_sites: list[str], top_recs: list[str]
    ) -> str:
        recs_block = "\n".join(f"- {r}" for r in top_recs) or "(none)"
        return f"""\
Date: {date}
Risk level breakdown: {dict(risk_counter)}
Sites that hit Critical today: {critical_sites or "none"}
Top priority recommendations issued today:
{recs_block}
"""
