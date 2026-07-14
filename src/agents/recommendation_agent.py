"""
Recommendation Agent: converts the Reasoning Agent's explanation into
concrete, numeric ActionItems using the site's real gate/staff counts
from config/site_master_data.py — never a vague "increase security".

Uses a deterministic rules layer first (fast, auditable, matches the
project's stated preference for a fully-auditable pipeline — see
dataset_summary.json notes: "Risk_Score is a weighted composite ...
fully auditable, not a black box"). The LLM is only used to phrase
descriptions naturally at the end — it never touches action_type,
target_site, payload, or priority, so dispatch and audit stay exactly
as deterministic as before. If the LLM call fails for any reason, the
original template description is kept, so the pipeline never breaks.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any

from agents.base import BaseAgent
from config.site_master_data import get_site_profile
from core.schemas import ActionItem, EmergencyType, RecommendationBundle, RiskLevel, SiteSnapshot
from tools.llm_client import complete_json

logger = logging.getLogger(__name__)

_PHRASING_SYSTEM_PROMPT = """\
You are a phrasing assistant for a heritage-site risk response system. \
You will receive a JSON array of operational actions, each already fully \
decided (site, numbers, reasons) — your only job is to rewrite the \
"original_description" of each one into a single natural, professional \
sentence that a site operator can read at a glance.

Rules:
- Do not invent, omit, or change any number, site name, or reason given.
- Do not mention action_type, payload, or priority directly — describe \
what they mean in plain operational language instead.
- Exactly one sentence per action.
- Match the language of the input text (English input -> English sentence).
- Respond with ONLY a JSON object, no prose, no markdown fences:
  {"descriptions": ["<sentence for action 0>", "<sentence for action 1>", "..."]}
  Same order and same length as the input array — this is required.
"""


def _rephrase_descriptions(snapshot: SiteSnapshot, actions: list[ActionItem]) -> list[str]:
    """
    Ask the LLM to turn each action's already-decided facts into a natural
    sentence. Never changes action_type/target_site/payload/priority — those
    are what dispatch and the audit trail rely on, so they stay deterministic.
    Falls back to the original template text for every action if the LLM
    call fails, times out, or returns a mismatched shape.
    """
    if not actions:
        return []

    facts = [
        {
            "action_type": a.action_type,
            "target_site": a.target_site,
            "payload": a.payload,
            "priority": a.priority,
            "original_description": a.description,
        }
        for a in actions
    ]
    user_prompt = (
        f"Risk level: {snapshot.risk_level.value}, score: {snapshot.risk_score:.1f}\n"
        f"Actions:\n{json.dumps(facts, ensure_ascii=False)}"
    )

    try:
        raw = complete_json(_PHRASING_SYSTEM_PROMPT, user_prompt)
        phrased = raw.get("descriptions", [])
        if (
            isinstance(phrased, list)
            and len(phrased) == len(actions)
            and all(isinstance(p, str) and p.strip() for p in phrased)
        ):
            return phrased
        logger.warning(
            "Recommendation phrasing LLM returned an unexpected shape, "
            "keeping template descriptions."
        )
    except Exception as exc:  # noqa: BLE001 - phrasing must never break the pipeline
        logger.warning(
            "Recommendation phrasing LLM call failed, keeping template descriptions: %s", exc
        )

    return [a.description for a in actions]


class RecommendationAgent(BaseAgent):
    name = "recommendation_agent"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        snapshot: SiteSnapshot = state["snapshot"]
        profile = get_site_profile(snapshot.site_name)

        actions: list[ActionItem] = []

        # --- Emergency-driven actions take absolute priority ------------ #
        if snapshot.emergency_type == EmergencyType.MEDICAL:
            actions.append(ActionItem(
                action_type="notify_medical_team",
                description=f"Dispatch medical team to {snapshot.site_name} immediately.",
                target_site=snapshot.site_name,
                payload={"note": "Active medical emergency reported."},
                priority=1,
            ))
        elif snapshot.emergency_type == EmergencyType.SECURITY:
            extra_staff = max(2, math.ceil(snapshot.security_staff * 0.2))
            actions.append(ActionItem(
                action_type="trigger_emergency_protocol",
                description=f"Trigger security emergency protocol at {snapshot.site_name}.",
                target_site=snapshot.site_name,
                payload={"emergency_type": "Security"},
                priority=1,
            ))
            actions.append(ActionItem(
                action_type="reallocate_security",
                description=f"Add {extra_staff} security staff at {snapshot.site_name}.",
                target_site=snapshot.site_name,
                payload={"additional_staff": extra_staff},
                priority=1,
            ))
        elif snapshot.emergency_type == EmergencyType.LOST_PERSON:
            actions.append(ActionItem(
                action_type="notify_operations_channel",
                description=f"Broadcast lost-person description at all gates in {snapshot.site_name}.",
                target_site=snapshot.site_name,
                payload={"note": "Lost person reported — coordinate via existing gate staff."},
                priority=2,
            ))

        # --- Occupancy / crowd-driven actions (independent of emergency) - #
        if snapshot.occupancy_rate > 0.9:
            actions.append(ActionItem(
                action_type="throttle_entry",
                description=f"Throttle entry at {snapshot.site_name} to 50% of normal rate.",
                target_site=snapshot.site_name,
                payload={"target_rate_pct": 50},
                priority=2,
            ))
            gates_to_open = max(1, profile.exit_gates - snapshot.exit_gates)
            if gates_to_open > 0:
                actions.append(ActionItem(
                    action_type="open_exit_gate",
                    description=f"Open {gates_to_open} more exit gate(s) at {snapshot.site_name}.",
                    target_site=snapshot.site_name,
                    payload={"gates_to_open": gates_to_open},
                    priority=2,
                ))
        elif snapshot.occupancy_rate > 0.75:
            actions.append(ActionItem(
                action_type="notify_operations_channel",
                description=f"{snapshot.site_name} approaching capacity "
                             f"({snapshot.occupancy_rate:.0%}) — monitor closely.",
                target_site=snapshot.site_name,
                payload={"note": "Occupancy above 75%."},
                priority=3,
            ))

        # --- High site sensitivity lowers the bar for escalation --------- #
        if profile.sensitivity >= 0.8 and snapshot.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            actions.append(ActionItem(
                action_type="reallocate_security",
                description=(
                    f"{snapshot.site_name} is a high-sensitivity site "
                    f"(sensitivity={profile.sensitivity}); pre-position 1-2 extra staff."
                ),
                target_site=snapshot.site_name,
                payload={"additional_staff": 2},
                priority=2,
            ))

        # --- Risk-level fallback: never silently no-op on High/Critical --- #
        if not actions and snapshot.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            reasons = []
            if snapshot.security_score < 0.3:
                reasons.append("low security coverage")
            if snapshot.queue_time > 30:
                reasons.append("long queue times")
            reason_str = ", ".join(reasons) if reasons else "elevated composite risk score"

            # Scale the staffing bump with how severe the situation actually is,
            # instead of a flat 15% of whatever staff is already on site.
            #   - base_pct: Critical always asks for a bigger jump than High.
            #   - score_boost: nudges further using the raw risk_score itself,
            #     so a 65 and a 95 inside the same level no longer produce
            #     the exact same recommendation.
            base_pct = 0.15 if snapshot.risk_level == RiskLevel.HIGH else 0.30
            score_boost = (snapshot.risk_score / 100) * 0.20  # up to +20 percentage points
            staff_pct = base_pct + score_boost
            extra_staff = max(2, math.ceil(snapshot.security_staff * staff_pct))

            actions.append(ActionItem(
                action_type="reallocate_security",
                description=(
                    f"Add {extra_staff} security staff at {snapshot.site_name} "
                    f"(flagged {snapshot.risk_level.value}, score={snapshot.risk_score:.1f}, "
                    f"due to {reason_str})."
                ),
                target_site=snapshot.site_name,
                payload={"additional_staff": extra_staff},
                priority=2,
            ))

        if not actions:
            actions.append(ActionItem(
                action_type="log_only",
                description=f"{snapshot.site_name} within normal parameters — no action needed.",
                target_site=snapshot.site_name,
                payload={},
                priority=5,
            ))

        actions.sort(key=lambda a: a.priority)

        # Cosmetic only: rewrite descriptions into natural phrasing. Every
        # action_type / target_site / payload / priority above is already
        # final and untouched by this call.
        phrased_descriptions = _rephrase_descriptions(snapshot, actions)
        for action, phrased_text in zip(actions, phrased_descriptions):
            action.description = phrased_text

        bundle = RecommendationBundle(
            site_name=snapshot.site_name, risk_level=snapshot.risk_level, actions=actions
        )
        return {"recommendations": bundle}