"""
Wires Monitoring -> Reasoning -> Recommendation -> Dispatch into a single
LangGraph StateGraph. If Monitoring decides a snapshot doesn't need
escalation, the graph short-circuits straight to END — no LLM calls,
no dispatch — which keeps cost down on the ~85% of records that are
routine (see risk_level_distribution in dataset_summary.json).
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.dispatch_agent import DispatchAgent
from agents.monitoring_agent import MonitoringAgent
from agents.reasoning_agent import ReasoningAgent
from agents.recommendation_agent import RecommendationAgent
from core.schemas import AgentRunResult, Explanation, RecommendationBundle, SiteSnapshot
from orchestration.state import GraphState

_monitoring = MonitoringAgent()
_reasoning = ReasoningAgent()
_recommendation = RecommendationAgent()
_dispatch = DispatchAgent()


def _route_after_monitoring(state: GraphState) -> str:
    return "reasoning" if state.get("should_escalate") else "end_early"


def _noop_end(state: GraphState) -> dict:
    # Newer LangGraph versions reject a node that writes nothing at all
    # (InvalidUpdateError: "Must write to at least one of [...]"). Re-write
    # should_escalate with its current value — a no-op for the state, but
    # satisfies the "write at least one channel" requirement.
    return {"should_escalate": state.get("should_escalate", False)}


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("monitoring", _monitoring)
    graph.add_node("reasoning", _reasoning)
    graph.add_node("recommendation", _recommendation)
    graph.add_node("dispatch", _dispatch)
    graph.add_node("end_early", _noop_end)

    graph.set_entry_point("monitoring")
    graph.add_conditional_edges(
        "monitoring", _route_after_monitoring, {"reasoning": "reasoning", "end_early": "end_early"}
    )
    graph.add_edge("reasoning", "recommendation")
    graph.add_edge("recommendation", "dispatch")
    graph.add_edge("dispatch", END)
    graph.add_edge("end_early", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(snapshot: SiteSnapshot) -> AgentRunResult | None:
    """
    Runs the full pipeline for one snapshot. Returns None if the
    Monitoring Agent decided no escalation was needed (routine record).
    """
    graph = get_compiled_graph()
    final_state: GraphState = graph.invoke({"snapshot": snapshot})

    if not final_state.get("should_escalate"):
        return None

    explanation: Explanation = final_state["explanation"]
    recommendations: RecommendationBundle = final_state["recommendations"]
    dispatch_results = final_state.get("dispatch_results", [])

    return AgentRunResult(
        snapshot=snapshot,
        explanation=explanation,
        recommendations=recommendations,
        dispatch_results=dispatch_results,
    )
