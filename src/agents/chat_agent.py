"""
Admin Chat Agent: a conversational, tool-using agent for the ops-room
admin. Unlike the per-snapshot pipeline (Monitoring -> Reasoning ->
Recommendation -> Dispatch), which reacts to telemetry automatically,
this agent reacts to natural-language questions/commands from a human
admin and decides for itself which tool(s) to call — site status, the
RAG protocol library, the daily report, or (if the admin explicitly
asks) firing a real dispatch action through the same ACTION_REGISTRY
the automatic pipeline uses.

Runs a small ReAct-style loop: ask the LLM with tool definitions, and if
it asks for a tool call, execute it and feed the result back, until the
model produces a plain-text answer (or MAX_TOOL_ROUNDS is hit).
"""
from __future__ import annotations

import json
import logging
from datetime import date as date_cls
from typing import Any

from agents.report_agent import ReportAgent
from config.site_master_data import SITE_MASTER_DATA, get_site_profile
from core import site_status_store
from core.constants import ACTION_PRIORITY
from core.run_store import get_results
from core.schemas import ActionItem, EmergencyType, RecommendationBundle, RiskLevel
from rag.retriever import retrieve_protocol
from tools.dispatch_tools import ACTION_REGISTRY
from tools.llm_client import chat_completion

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4

_SYSTEM_PROMPT = """\
You are the PHARAOH GUARD admin assistant — a helper for the operations \
room staff monitoring 8 Egyptian heritage sites (Giza Pyramids, Grand \
Egyptian Museum, Saqqara, Luxor Temple, Karnak Temple, Abu Simbel, Valley \
of the Kings, Citadel of Cairo).

Reply in Egyptian Arabic colloquial by default, unless the admin writes \
in English, then reply in English. Be concise and operational — this is \
a live ops tool, not a chat companion.

You have tools to check live site status, search emergency protocols, \
get the daily report, and to actually dispatch an operational action \
(e.g. send extra security, open a gate, notify the medical team). Only \
call the dispatch tool when the admin clearly asks you to take an \
action — never dispatch on your own initiative just because a site \
looks risky. Always confirm what you did (or didn't do) in your reply. \
Ground every factual claim in what the tools actually return — never \
invent staffing numbers, risk scores, or protocol content.
"""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_sites",
            "description": "List all 8 monitored heritage sites with their capacity, gates, sensitivity, and region.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_site_status",
            "description": "Get the latest known telemetry/risk status for one specific site.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site_name": {"type": "string", "description": "Exact site name, e.g. 'Abu Simbel'."},
                },
                "required": ["site_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_sites_status",
            "description": "Get the latest known status for every monitored site at once.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_protocol",
            "description": "Semantic search over the emergency-response protocol documents (medical, security, lost person, crowd management).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "emergency_type": {
                        "type": "string",
                        "enum": ["Medical", "Security", "Lost Person", "No_Emergency"],
                        "description": "Optional emergency category to bias retrieval.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_report",
            "description": "Get the aggregated narrative report of all escalated situations for a given date (defaults to today).",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "ISO date YYYY-MM-DD. Omit for today."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_action",
            "description": (
                "Actually dispatch an operational action to a site. Only call this when the "
                "admin explicitly asks you to take an action. action_type must be one of: "
                "reallocate_security, open_exit_gate, throttle_entry, notify_medical_team, "
                "trigger_emergency_protocol, notify_operations_channel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": list(ACTION_REGISTRY.keys()),
                    },
                    "target_site": {"type": "string", "description": "Exact site name."},
                    "payload": {
                        "type": "object",
                        "description": (
                            "Extra params for the action, e.g. {\"additional_staff\": 3} for "
                            "reallocate_security, {\"gates_to_open\": 1} for open_exit_gate, "
                            "{\"target_rate_pct\": 50} for throttle_entry, {\"note\": \"...\"} for "
                            "notify_medical_team / trigger_emergency_protocol / notify_operations_channel."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["action_type", "target_site"],
            },
        },
    },
]


class ChatAgent:
    name = "chat_agent"

    def run(self, history: list[dict[str, Any]], user_message: str) -> dict[str, Any]:
        """
        history: prior turns as OpenAI-style {"role": ..., "content": ...} dicts
        (no system prompt in it — that's added here).
        Returns {"reply": str, "actions_taken": list[dict], "new_messages": list[dict]}
        appended messages (user turn + assistant turn(s) + any tool turns) that the
        caller should persist via core.chat_store.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        new_messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        actions_taken: list[dict[str, Any]] = []

        for _ in range(MAX_TOOL_ROUNDS):
            assistant_msg = chat_completion(messages, tools=_TOOLS, tool_choice="auto")
            tool_calls = assistant_msg.get("tool_calls") or []

            if not tool_calls:
                reply = assistant_msg.get("content") or "..."
                new_messages.append({"role": "assistant", "content": reply})
                return {"reply": reply, "actions_taken": actions_taken, "new_messages": new_messages}

            # Keep the assistant's tool-call turn in the running transcript.
            messages.append(assistant_msg)
            new_messages.append(assistant_msg)

            for call in tool_calls:
                fn_name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}

                try:
                    tool_result = self._execute_tool(fn_name, args)
                    if fn_name == "dispatch_action" and tool_result.get("dispatched") is not None:
                        actions_taken.append(tool_result)
                except Exception as exc:  # noqa: BLE001 - tool errors go back to the model, not raised
                    logger.exception("Tool %s failed", fn_name)
                    tool_result = {"error": str(exc)}

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": fn_name,
                    "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                }
                messages.append(tool_msg)
                new_messages.append(tool_msg)

        # Ran out of tool rounds — force a final plain-text answer.
        messages.append({
            "role": "user",
            "content": "Please answer now in plain text based on what you've found so far.",
        })
        final = chat_completion(messages, tools=None)
        reply = final.get("content") or "معلش، مقدرتش أوصل لإجابة واضحة دلوقتي."
        new_messages.append({"role": "assistant", "content": reply})
        return {"reply": reply, "actions_taken": actions_taken, "new_messages": new_messages}

    # ------------------------------------------------------------------ #
    # Tool implementations
    # ------------------------------------------------------------------ #
    @staticmethod
    def _execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "list_sites":
            return {
                site: {
                    "capacity": profile.capacity,
                    "entry_gates": profile.entry_gates,
                    "exit_gates": profile.exit_gates,
                    "sensitivity": profile.sensitivity,
                    "region": profile.region,
                    "indoor": profile.indoor,
                }
                for site, profile in SITE_MASTER_DATA.items()
            }

        if name == "get_site_status":
            site_name = args.get("site_name", "")
            status = site_status_store.get(site_name)
            if status is None:
                return {"site_name": site_name, "status": "no telemetry received yet this session"}
            return status

        if name == "get_all_sites_status":
            all_status = site_status_store.get_all()
            return all_status or {"status": "no telemetry received yet this session for any site"}

        if name == "search_protocol":
            query = args.get("query", "")
            emergency_type_raw = args.get("emergency_type")
            emergency_type = EmergencyType(emergency_type_raw) if emergency_type_raw else None
            excerpts = retrieve_protocol(query, emergency_type=emergency_type)
            return {"excerpts": excerpts}

        if name == "get_daily_report":
            target_date = args.get("date") or date_cls.today().isoformat()
            results = get_results(target_date)
            if not results:
                return {"date": target_date, "status": "no escalated snapshots recorded for this date"}
            report = ReportAgent().run(target_date, results)
            return report.model_dump()

        if name == "dispatch_action":
            return ChatAgent._dispatch(args)

        return {"error": f"unknown tool: {name}"}

    @staticmethod
    def _dispatch(args: dict[str, Any]) -> dict[str, Any]:
        from agents.dispatch_agent import DispatchAgent
        from config.settings import get_settings

        action_type = args.get("action_type", "")
        target_site = args.get("target_site", "")
        payload = args.get("payload") or {}

        if action_type not in ACTION_REGISTRY:
            return {"dispatched": False, "detail": f"unknown_action_type:{action_type}"}
        try:
            get_site_profile(target_site)
        except ValueError as exc:
            return {"dispatched": False, "detail": str(exc)}

        description = f"[Admin chat] {action_type} at {target_site} — {payload or 'no extra params'}"
        action = ActionItem(
            action_type=action_type,
            description=description,
            target_site=target_site,
            payload=payload,
            priority=ACTION_PRIORITY.get(action_type, 3),
        )
        bundle = RecommendationBundle(site_name=target_site, risk_level=RiskLevel.HIGH, actions=[action])
        state = DispatchAgent().run({"recommendations": bundle})
        result = state["dispatch_results"][0]
        settings = get_settings()
        return {
            "action_type": action_type,
            "target_site": target_site,
            "payload": payload,
            "dispatched": result.dispatched,
            "dry_run": result.dry_run,
            "detail": result.detail,
            "note": (
                "This was a dry run (no real webhook fired) — set RRS_DISPATCH_DRY_RUN=False to send for real."
                if settings.dispatch_dry_run else "Sent for real."
            ),
        }
