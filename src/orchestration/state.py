"""
Shared state schema for the LangGraph pipeline. TypedDict (not Pydantic)
because LangGraph merges partial dict updates from each node — this is
the graph's internal working memory, not an external API contract
(those live in core/schemas.py and get embedded here).
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from core.schemas import (
    DispatchResult,
    Explanation,
    RecommendationBundle,
    SiteSnapshot,
)


class GraphState(TypedDict, total=False):
    snapshot: SiteSnapshot
    should_escalate: bool
    escalation_reason: str
    protocol_excerpts: list[str]
    explanation: Explanation
    recommendations: RecommendationBundle
    dispatch_results: list[DispatchResult]
