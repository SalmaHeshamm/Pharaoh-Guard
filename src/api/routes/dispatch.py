"""
POST /dispatch

Manually trigger dispatch for a RecommendationBundle — used for testing
tools/dispatch_tools.py in isolation, or letting a human operator
re-fire an action after reviewing it.
"""
from __future__ import annotations

from fastapi import APIRouter

from agents.dispatch_agent import DispatchAgent
from core.schemas import DispatchResult, RecommendationBundle

router = APIRouter(prefix="/dispatch", tags=["dispatch"])
_dispatch_agent = DispatchAgent()


@router.post("", response_model=list[DispatchResult])
def dispatch(bundle: RecommendationBundle) -> list[DispatchResult]:
    state = _dispatch_agent.run({"recommendations": bundle})
    return state["dispatch_results"]
