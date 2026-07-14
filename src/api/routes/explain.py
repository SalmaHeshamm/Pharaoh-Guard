"""
POST /explain

Runs ONLY the Reasoning Agent on a snapshot that already has risk_score /
risk_level filled in. Useful for the ML team (or QA) to sanity-check
explanations without triggering dispatch side effects.
"""
from __future__ import annotations

from fastapi import APIRouter

from agents.reasoning_agent import ReasoningAgent
from core.schemas import Explanation, SiteSnapshot

router = APIRouter(prefix="/explain", tags=["explain"])
_reasoning = ReasoningAgent()


@router.post("", response_model=Explanation)
def explain(snapshot: SiteSnapshot) -> Explanation:
    state = _reasoning.run({"snapshot": snapshot})
    return state["explanation"]
