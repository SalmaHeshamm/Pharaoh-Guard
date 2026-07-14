"""
POST /predict/mitigate

Standalone "what-if" endpoint: takes the exact same live-telemetry shape
as /predict, plus a target risk level, and returns the smallest
security/medical/police staffing increase (found by actually re-querying
the trained risk model — see core.mitigation) that brings the predicted
risk down to that target.

Deliberately read-only against history_store: it looks up the last known
visitor count for momentum features but never calls history_store.update,
so testing a scenario here never pollutes the real /predict momentum
state for the site.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.routes.predict import RawTelemetry, build_raw_from_telemetry
from config.site_master_data import get_site_profile
from core import history_store
from core.mitigation import find_minimal_staffing
from core.risk_model import score as score_situation
from core.schemas import MitigationResult, RiskLevel

router = APIRouter(prefix="/predict", tags=["predict"])


class MitigationRequest(BaseModel):
    reading: RawTelemetry
    target_risk: RiskLevel = RiskLevel.MEDIUM
    max_extra_security: int = Field(default=60, ge=0, le=200)
    max_extra_medical: int = Field(default=20, ge=0, le=100)
    max_extra_police: int = Field(default=20, ge=0, le=100)
    step: int = Field(default=5, ge=1, le=25)


class MitigationResponse(BaseModel):
    site_name: str
    target_risk: RiskLevel
    baseline_risk_level: str
    baseline_proba_critical: float
    mitigation: MitigationResult


@router.post("/mitigate", response_model=MitigationResponse)
def predict_mitigate(request: MitigationRequest) -> MitigationResponse:
    payload = request.reading
    raw = build_raw_from_telemetry(payload)

    try:
        site = get_site_profile(payload.site_name)
    except ValueError as exc:  # unknown site
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Read-only: same momentum lookup /predict uses, but no .update() call,
    # so running a what-if search never changes what the *next real*
    # /predict call for this site sees as "previous visitors".
    prev_visitors = history_store.get_previous_visitors(payload.site_name)

    try:
        baseline = score_situation(raw, site, prev_visitors)
    except FileNotFoundError as exc:  # model artifact missing
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    mitigation = find_minimal_staffing(
        raw,
        site,
        prev_visitors,
        target_max_risk=request.target_risk,
        max_extra_security=request.max_extra_security,
        max_extra_medical=request.max_extra_medical,
        max_extra_police=request.max_extra_police,
        step=request.step,
    )

    return MitigationResponse(
        site_name=payload.site_name,
        target_risk=request.target_risk,
        baseline_risk_level=baseline["risk_level"],
        baseline_proba_critical=baseline["proba_critical"],
        mitigation=mitigation,
    )
