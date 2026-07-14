"""
POST /predict
POST /predict/batch

Input: NATURAL operational telemetry for one site (or a list of sites) at
one point in time — only what an operations room actually observes live
(visitors, staffing, weather, queue, event flags). Everything else the
model or the SiteSnapshot needs (site capacity/gates/sensitivity,
engineered time features, momentum, composite descriptive scores,
risk_score/risk_level) is derived internally by core.risk_model +
config.site_master_data before handing off to the agent graph. This
keeps the caller's contract small and prevents leaky / stale composite
scores from being supplied by hand.

/predict/batch exists so the dashboard can push one simulated hour for
all 8 sites in a single round trip instead of 8 sequential HTTP calls —
each site is scored and run through the agent pipeline concurrently
(via a thread pool, since the model scoring and LLM calls are blocking).
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import predict_risk
from config.site_master_data import get_site_profile
from core import site_status_store
from core.run_store import record as record_run
from core.schemas import AgentRunResult, EmergencyType, RiskLevel, SiteSnapshot
from orchestration.graph import run_pipeline

router = APIRouter(prefix="/predict", tags=["predict"])


class RawTelemetry(BaseModel):
    """Only the fields an operator actually reads off live sensors/logs."""

    timestamp: datetime
    site_name: str

    current_visitors: int
    security_staff: int
    medical_team: int
    police_units: int

    temperature: float
    humidity: float
    wind_speed: float
    visibility: float

    queue_length: int = 0
    queue_time: float = 0.0

    special_events: bool = False
    vip_visits: bool = False
    school_trips: bool = False
    emergency_type: EmergencyType = EmergencyType.NONE

    # Optional override — if omitted, derived from timestamp against the
    # known 2023-2024 Egyptian public holiday set (core.risk_model.HOLIDAY_SET).
    holiday: bool | None = Field(default=None)


class PredictResponse(BaseModel):
    site_name: str
    escalated: bool
    result: AgentRunResult | None = None
    risk_score: float
    risk_level: str
    proba_critical: float
    probabilities: dict[str, float]
    khamsin: bool


class BatchPredictRequest(BaseModel):
    readings: list[RawTelemetry]


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


def _run_predict_sync(payload: RawTelemetry) -> PredictResponse:
    emergency_active = payload.emergency_type != EmergencyType.NONE

    raw = {
        "timestamp": payload.timestamp,
        "site_name": payload.site_name,
        "current_visitors": payload.current_visitors,
        "security_staff": payload.security_staff,
        "medical_team": payload.medical_team,
        "police_units": payload.police_units,
        "temperature": payload.temperature,
        "humidity": payload.humidity,
        "wind_speed": payload.wind_speed,
        "visibility": payload.visibility,
        "queue_length": payload.queue_length,
        "queue_time": payload.queue_time,
        "special_events": int(payload.special_events),
        "vip_visits": int(payload.vip_visits),
        "school_trips": int(payload.school_trips),
        "emergency_active": int(emergency_active),
    }
    if payload.holiday is not None:
        raw["holiday"] = int(payload.holiday)

    try:
        scored = predict_risk(raw)
    except ValueError as exc:  # unknown site
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:  # model artifact missing
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    site = get_site_profile(payload.site_name)

    snapshot = SiteSnapshot(
        timestamp=payload.timestamp,
        site_name=payload.site_name,
        hour=payload.timestamp.hour,
        current_visitors=payload.current_visitors,
        site_capacity=site.capacity,
        entry_gates=site.entry_gates,
        exit_gates=site.exit_gates,
        security_staff=payload.security_staff,
        medical_team=payload.medical_team,
        police_units=payload.police_units,
        temperature=payload.temperature,
        humidity=payload.humidity,
        wind_speed=payload.wind_speed,
        visibility=payload.visibility,
        special_events=payload.special_events,
        vip_visits=payload.vip_visits,
        school_trips=payload.school_trips,
        emergency_type=payload.emergency_type,
        queue_length=payload.queue_length,
        queue_time=payload.queue_time,
        occupancy_rate=scored["occupancy_rate"],
        crowd_density=scored["crowd_density"],
        weather_score=scored["weather_score"],
        security_score=scored["security_score"],
        site_sensitivity=site.sensitivity,
        operational_load=scored["operational_load"],
        risk_score=scored["risk_score"],
        risk_level=RiskLevel(scored["risk_level"]),
    )
    result = run_pipeline(snapshot)

    if result is not None:
        record_run(result)

    site_status_store.update(
        payload.site_name,
        timestamp=payload.timestamp,
        risk_score=scored["risk_score"],
        risk_level=scored["risk_level"],
        escalated=result is not None,
        emergency_type=payload.emergency_type.value,
        occupancy_rate=scored["occupancy_rate"],
        current_visitors=payload.current_visitors,
        explanation_summary=result.explanation.summary_en if result else None,
        top_action=result.recommendations.actions[0].description if result and result.recommendations.actions else None,
    )

    return PredictResponse(
        site_name=payload.site_name,
        escalated=result is not None,
        result=result,
        risk_score=scored["risk_score"],
        risk_level=scored["risk_level"],
        proba_critical=scored["proba_critical"],
        probabilities=scored["probabilities"],
        khamsin=scored["khamsin"],
    )


@router.post("", response_model=PredictResponse)
def predict(payload: RawTelemetry) -> PredictResponse:
    return _run_predict_sync(payload)


@router.post("/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Scores every reading concurrently (thread pool — model inference and
    the LLM-backed agent pipeline are both blocking calls). Order of
    `results` matches the order of `readings` in the request.
    """
    results = await asyncio.gather(
        *(asyncio.to_thread(_run_predict_sync, reading) for reading in request.readings)
    )
    return BatchPredictResponse(results=list(results))
