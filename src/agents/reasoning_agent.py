"""
Reasoning Agent: turns raw numeric features into a grounded natural-
language explanation. Pulls relevant protocol excerpts via RAG so the
explanation cites real operational guidance instead of hallucinating it.
"""
from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from core.schemas import Explanation, SiteSnapshot
from rag.retriever import retrieve_protocol
from tools.llm_client import complete_json

_SYSTEM_PROMPT = """\
You are a risk-explanation engine for Egyptian heritage-site crowd and \
safety management. You will be given structured site telemetry and \
relevant protocol excerpts. Respond with ONLY a JSON object, no prose, \
no markdown fences, matching exactly this schema:

{
  "summary_ar": "<2-3 sentence explanation in Egyptian Arabic colloquial>",
  "summary_en": "<2-3 sentence explanation in English>",
  "key_drivers": ["<short driver 1>", "<short driver 2>", "..."]
}

Ground every claim in the numbers given. Do not invent staffing numbers \
or protocol content that isn't in the provided excerpts.
"""


class ReasoningAgent(BaseAgent):
    name = "reasoning_agent"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        snapshot: SiteSnapshot = state["snapshot"]

        protocol_excerpts = retrieve_protocol(
            query=f"risk level {snapshot.risk_level.value} at {snapshot.site_name}",
            emergency_type=snapshot.emergency_type,
        )

        user_prompt = self._build_user_prompt(snapshot, protocol_excerpts)
        raw = complete_json(_SYSTEM_PROMPT, user_prompt)

        explanation = Explanation(
            site_name=snapshot.site_name,
            risk_level=snapshot.risk_level,
            risk_score=snapshot.risk_score,
            summary_ar=raw.get("summary_ar", ""),
            summary_en=raw.get("summary_en", ""),
            key_drivers=raw.get("key_drivers", []),
        )
        return {"explanation": explanation, "protocol_excerpts": protocol_excerpts}

    @staticmethod
    def _build_user_prompt(snapshot: SiteSnapshot, excerpts: list[str]) -> str:
        excerpts_block = "\n---\n".join(excerpts) if excerpts else "(no matching protocol excerpts)"
        return f"""\
Site telemetry:
- Site: {snapshot.site_name}
- Risk level: {snapshot.risk_level.value} (score={snapshot.risk_score:.1f}/100)
- Emergency type: {snapshot.emergency_type.value}
- Occupancy rate: {snapshot.occupancy_rate:.2f}
- Crowd density: {snapshot.crowd_density:.2f}
- Weather score: {snapshot.weather_score:.2f}
- Security score: {snapshot.security_score:.2f}
- Site sensitivity: {snapshot.site_sensitivity:.2f}
- Operational load: {snapshot.operational_load:.2f}
- Queue length / time: {snapshot.queue_length} / {snapshot.queue_time:.1f} min
- Current visitors / capacity: {snapshot.current_visitors} / {snapshot.site_capacity}
- Security staff / medical team / police units: \
{snapshot.security_staff} / {snapshot.medical_team} / {snapshot.police_units}
- Special event: {snapshot.special_events}, VIP visit: {snapshot.vip_visits}, \
School trip: {snapshot.school_trips}

Relevant protocol excerpts:
{excerpts_block}

Explain why the risk level is what it is and what an operator should know.
"""
