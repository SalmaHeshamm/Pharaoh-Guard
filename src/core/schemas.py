"""
Shared data contracts. Every agent, tool, and API route imports its
input/output shapes from here — nothing is redefined ad hoc in a file.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------- #
# Static reference data
# ---------------------------------------------------------------------- #
class SiteProfile(BaseModel):
    name: str
    capacity: int
    popularity: float
    sensitivity: float
    entry_gates: int
    exit_gates: int
    indoor: bool
    base_temp_offset: float
    night_show: bool
    region: str


# ---------------------------------------------------------------------- #
# Risk levels / emergency types (matches dataset_summary.json distributions)
# ---------------------------------------------------------------------- #
class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class EmergencyType(str, Enum):
    NONE = "No_Emergency"
    MEDICAL = "Medical"
    LOST_PERSON = "Lost Person"
    SECURITY = "Security"


# ---------------------------------------------------------------------- #
# Input record: one row coming out of the feature-engineered dataset,
# after notebook 04/05 have already produced Risk_Score / Risk_Level.
# This is the hand-off contract from the ML side to this system.
# ---------------------------------------------------------------------- #
class SiteSnapshot(BaseModel):
    timestamp: datetime
    site_name: str
    hour: int
    current_visitors: int
    site_capacity: int
    entry_gates: int
    exit_gates: int
    security_staff: int
    medical_team: int
    police_units: int
    temperature: float
    humidity: float
    wind_speed: float
    visibility: float
    special_events: bool
    vip_visits: bool
    school_trips: bool
    emergency_type: EmergencyType = EmergencyType.NONE
    queue_length: int
    queue_time: float
    occupancy_rate: float
    crowd_density: float
    weather_score: float
    security_score: float
    site_sensitivity: float
    operational_load: float
    risk_score: float
    risk_level: RiskLevel
    recommendation: str = ""  # raw rule-based text from notebook 05, if present


# ---------------------------------------------------------------------- #
# Agent outputs
# ---------------------------------------------------------------------- #
class Explanation(BaseModel):
    """Output of the Reasoning Agent."""
    site_name: str
    risk_level: RiskLevel
    risk_score: float
    summary_ar: str
    summary_en: str
    key_drivers: list[str] = Field(default_factory=list)


class ActionItem(BaseModel):
    """A single concrete, numeric action proposed by the Recommendation Agent."""
    action_type: str  # e.g. "reallocate_security", "open_exit_gate", "notify_medical"
    description: str
    target_site: str
    payload: dict = Field(default_factory=dict)
    priority: int = Field(ge=1, le=5, description="1 = most urgent")


class RecommendationBundle(BaseModel):
    site_name: str
    risk_level: RiskLevel
    actions: list[ActionItem]


class DispatchResult(BaseModel):
    action: ActionItem
    dispatched: bool
    dry_run: bool
    detail: str
    dispatched_at: datetime = Field(default_factory=datetime.utcnow)


class AgentRunResult(BaseModel):
    """Final aggregate result returned by the orchestration graph for one snapshot."""
    snapshot: SiteSnapshot
    explanation: Explanation
    recommendations: RecommendationBundle
    dispatch_results: list[DispatchResult] = Field(default_factory=list)


class DailyReport(BaseModel):
    date: str
    total_snapshots: int
    risk_level_breakdown: dict[str, int]
    critical_sites: list[str]
    top_recommendations: list[str]
    narrative_ar: str
