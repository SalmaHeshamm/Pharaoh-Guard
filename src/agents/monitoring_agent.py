"""
Monitoring Agent: the entry point of the pipeline. Decides whether a
SiteSnapshot deserves the rest of the agent pipeline at all. This keeps
LLM calls (expensive) off the ~85% of records that are Low/Medium risk
with no emergency (see risk_level_distribution in dataset_summary.json).
"""
from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from config.settings import get_settings
from core.schemas import EmergencyType, RiskLevel, SiteSnapshot

_LEVEL_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}


class MonitoringAgent(BaseAgent):
    name = "monitoring_agent"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        snapshot: SiteSnapshot = state["snapshot"]
        settings = get_settings()

        trigger_level = _LEVEL_ORDER[settings.monitoring_trigger_level]
        should_escalate = (
            _LEVEL_ORDER[snapshot.risk_level] >= trigger_level
            or snapshot.emergency_type != EmergencyType.NONE
        )

        return {
            "should_escalate": should_escalate,
            "escalation_reason": (
                "emergency_active" if snapshot.emergency_type != EmergencyType.NONE
                else "risk_threshold_exceeded" if should_escalate
                else "within_normal_range"
            ),
        }
