"""
The Recommendation Agent's action_type / target_site / payload / priority
are fully rule-based and never touched by an LLM. Only the human-readable
`description` goes through an LLM phrasing pass at the end, and it falls
back to the deterministic template text if that call fails (e.g. no API
key in this test environment) — so these tests stay fast and offline.
Run with: pytest tests/
"""
from datetime import datetime

from src.agents.recommendation_agent import RecommendationAgent
from src.core.schemas import EmergencyType, RiskLevel, SiteSnapshot


def _base_snapshot(**overrides) -> SiteSnapshot:
    defaults = dict(
        timestamp=datetime(2026, 7, 4, 12, 0),
        site_name="Valley of the Kings",
        hour=12,
        current_visitors=5000,
        site_capacity=6000,
        entry_gates=3,
        exit_gates=2,
        security_staff=10,
        medical_team=2,
        police_units=2,
        temperature=38.0,
        humidity=0.2,
        wind_speed=10.0,
        visibility=9.0,
        special_events=False,
        vip_visits=False,
        school_trips=False,
        emergency_type=EmergencyType.NONE,
        queue_length=500,
        queue_time=20.0,
        occupancy_rate=0.5,
        crowd_density=0.5,
        weather_score=0.6,
        security_score=0.3,
        site_sensitivity=0.9,
        operational_load=0.5,
        risk_score=40.0,
        risk_level=RiskLevel.MEDIUM,
    )
    defaults.update(overrides)
    return SiteSnapshot(**defaults)


def test_high_occupancy_triggers_throttle_and_gate_opening():
    snapshot = _base_snapshot(occupancy_rate=0.95, risk_level=RiskLevel.HIGH)
    agent = RecommendationAgent()

    result = agent.run({"snapshot": snapshot})
    action_types = {a.action_type for a in result["recommendations"].actions}

    assert "throttle_entry" in action_types
    assert "open_exit_gate" in action_types


def test_medical_emergency_triggers_notify_medical_team():
    snapshot = _base_snapshot(emergency_type=EmergencyType.MEDICAL)
    agent = RecommendationAgent()

    result = agent.run({"snapshot": snapshot})
    action_types = {a.action_type for a in result["recommendations"].actions}

    assert "notify_medical_team" in action_types
    assert result["recommendations"].actions[0].priority == 1


def test_no_risk_falls_back_to_log_only():
    snapshot = _base_snapshot(occupancy_rate=0.3, risk_level=RiskLevel.LOW, site_sensitivity=0.5)
    agent = RecommendationAgent()

    result = agent.run({"snapshot": snapshot})
    action_types = {a.action_type for a in result["recommendations"].actions}

    assert action_types == {"log_only"}


def test_description_never_empty_when_llm_unavailable():
    """
    No Groq API key is configured in this test environment, so the phrasing
    call in RecommendationAgent must fail internally and fall back to the
    original template description — the pipeline must never crash or
    return a blank description.
    """
    snapshot = _base_snapshot(occupancy_rate=0.95, risk_level=RiskLevel.HIGH)
    agent = RecommendationAgent()

    result = agent.run({"snapshot": snapshot})

    for action in result["recommendations"].actions:
        assert isinstance(action.description, str)
        assert action.description.strip() != ""