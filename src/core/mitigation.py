"""
"What-if" mitigation search: given a situation the trained model already
scored as High/Critical, find the smallest staffing increase that brings
the predicted risk back down to a target level — by actually re-querying
the model, not by guessing a fixed percentage.

Search strategy is deliberately SEQUENTIAL, not a full grid search:
  1. Try increasing security_staff alone first (the fastest/cheapest lever
     to deploy operationally — reallocating existing staff).
  2. If maxed out and still not enough, keep security at its max and start
     adding medical_team on top.
  3. If still not enough, keep both at their max and start adding
     police_units on top.

This mirrors the same "deterministic, auditable, not a black box"
preference already stated in agents/recommendation_agent.py, and it's
O(steps) instead of O(steps^3) for a full grid search — the trained
sklearn Pipeline is fast, but there's no reason to pay for an exhaustive
search when a simple priority order already matches how sites actually
reallocate staff.
"""
from __future__ import annotations

from typing import Any

from core.risk_model import RISK_ORDER, score as score_situation
from core.schemas import MitigationResult, RiskLevel, SiteProfile


def _risk_index(level: str) -> int:
    return RISK_ORDER.index(level)


def find_minimal_staffing(
    raw: dict[str, Any],
    site: SiteProfile,
    prev_visitors: float | None,
    target_max_risk: RiskLevel | str = RiskLevel.MEDIUM,
    max_extra_security: int = 60,
    max_extra_medical: int = 20,
    max_extra_police: int = 20,
    step: int = 5,
) -> MitigationResult:
    """
    raw: same shape core.risk_model.build_features expects (timestamp,
         current_visitors, security_staff, medical_team, police_units,
         temperature, humidity, wind_speed, visibility, queue_length,
         queue_time, special_events, vip_visits, school_trips,
         emergency_active, optional holiday). NOT mutated.
    site: the SiteProfile for raw["site_name"] (caller already has it via
          config.site_master_data.get_site_profile — kept as a required
          param here rather than looked up internally, so this stays a
          pure function that's easy to unit test).
    prev_visitors: same momentum input core.risk_model.score expects.
                   Pass None if unknown; matches the model's own fallback
                   (lag defaults to current visitors, i.e. zero delta).
    """
    target_str = target_max_risk.value if isinstance(target_max_risk, RiskLevel) else target_max_risk
    target_index = _risk_index(target_str)

    base_security = int(raw["security_staff"])
    base_medical = int(raw["medical_team"])
    base_police = int(raw["police_units"])

    def _try(extra_security: int, extra_medical: int, extra_police: int) -> dict[str, Any]:
        candidate = dict(raw)
        candidate["security_staff"] = base_security + extra_security
        candidate["medical_team"] = base_medical + extra_medical
        candidate["police_units"] = base_police + extra_police
        return score_situation(candidate, site, prev_visitors)

    steps_tried = 0

    def _result(achieved: bool, es: int, em: int, ep: int, scored: dict[str, Any]) -> MitigationResult:
        return MitigationResult(
            achieved=achieved,
            added_security=es,
            added_medical=em,
            added_police=ep,
            final_risk_level=scored["risk_level"],
            final_proba_critical=scored["proba_critical"],
            steps_tried=steps_tried,
        )

    # Baseline — the situation as-is, no changes.
    scored = _try(0, 0, 0)
    steps_tried += 1
    if _risk_index(scored["risk_level"]) <= target_index:
        return _result(True, 0, 0, 0, scored)

    # Phase 1: security alone.
    best_security = 0
    for extra_security in range(step, max_extra_security + 1, step):
        scored = _try(extra_security, 0, 0)
        steps_tried += 1
        best_security = extra_security
        if _risk_index(scored["risk_level"]) <= target_index:
            return _result(True, extra_security, 0, 0, scored)

    # Phase 2: security maxed, add medical on top.
    best_medical = 0
    for extra_medical in range(step, max_extra_medical + 1, step):
        scored = _try(best_security, extra_medical, 0)
        steps_tried += 1
        best_medical = extra_medical
        if _risk_index(scored["risk_level"]) <= target_index:
            return _result(True, best_security, extra_medical, 0, scored)

    # Phase 3: security + medical maxed, add police on top.
    for extra_police in range(step, max_extra_police + 1, step):
        scored = _try(best_security, best_medical, extra_police)
        steps_tried += 1
        if _risk_index(scored["risk_level"]) <= target_index:
            return _result(True, best_security, best_medical, extra_police, scored)

    # Hit every ceiling and still couldn't reach the target — report the
    # best (maxed-out) attempt so the caller can still show *something*
    # useful ("even a full staffing surge only gets you to High").
    return _result(False, best_security, best_medical, max_extra_police, scored)
