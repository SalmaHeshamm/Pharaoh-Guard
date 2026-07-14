"""
Dispatch Agent: the only agent allowed to actually *do* something in the
outside world. Resolves each ActionItem to a callable via
tools.dispatch_tools.ACTION_REGISTRY and records a DispatchResult for
every one — including "log_only" no-ops, for a complete audit trail.
"""
from __future__ import annotations

import logging
from typing import Any

from agents.base import BaseAgent
from config.settings import get_settings
from core.schemas import ActionItem, DispatchResult, RecommendationBundle
from tools.dispatch_tools import ACTION_REGISTRY

logger = logging.getLogger(__name__)


class DispatchAgent(BaseAgent):
    name = "dispatch_agent"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        bundle: RecommendationBundle = state["recommendations"]
        settings = get_settings()

        results: list[DispatchResult] = []
        for action in bundle.actions:
            results.append(self._dispatch_one(action, settings.dispatch_dry_run))

        return {"dispatch_results": results}

    @staticmethod
    def _dispatch_one(action: ActionItem, dry_run: bool) -> DispatchResult:
        if action.action_type == "log_only":
            return DispatchResult(
                action=action, dispatched=True, dry_run=True, detail="no_action_required"
            )

        handler = ACTION_REGISTRY.get(action.action_type)
        if handler is None:
            logger.error("No handler registered for action_type=%s", action.action_type)
            return DispatchResult(
                action=action, dispatched=False, dry_run=dry_run,
                detail=f"unknown_action_type:{action.action_type}",
            )

        try:
            ok, detail = handler(action.target_site, **_strip_site(action.payload))
        except TypeError as exc:
            ok, detail = False, f"payload_mismatch:{exc}"

        return DispatchResult(action=action, dispatched=ok, dry_run=dry_run, detail=detail)


def _strip_site(payload: dict) -> dict:
    """payload never contains site_name — it's passed positionally as target_site."""
    return {k: v for k, v in payload.items() if k != "site_name"}
