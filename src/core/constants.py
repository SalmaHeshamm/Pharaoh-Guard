"""
Fixed lookup tables that don't belong in Settings (not env-configurable,
just domain constants derived from dataset_summary.json).
"""
from core.schemas import EmergencyType, RiskLevel

# risk_level_distribution from dataset_summary.json — used for sanity checks
# / drift monitoring, not for business logic.
EXPECTED_RISK_DISTRIBUTION = {
    RiskLevel.MEDIUM: 32819,
    RiskLevel.LOW: 15876,
    RiskLevel.HIGH: 6396,
    RiskLevel.CRITICAL: 389,
}

EXPECTED_EMERGENCY_DISTRIBUTION = {
    EmergencyType.NONE: 54307,
    EmergencyType.MEDICAL: 492,
    EmergencyType.LOST_PERSON: 346,
    EmergencyType.SECURITY: 335,
}

# Which protocol document (in data/protocols/) the RAG layer should
# prioritize for each emergency type.
EMERGENCY_PROTOCOL_MAP = {
    EmergencyType.MEDICAL: "medical_emergency.md",
    EmergencyType.LOST_PERSON: "lost_person.md",
    EmergencyType.SECURITY: "security_incident.md",
    EmergencyType.NONE: "crowd_management.md",
}

# Priority ordering used by the Dispatch Agent when several actions fire
# for the same site at once.
ACTION_PRIORITY = {
    "trigger_emergency_protocol": 1,
    "notify_medical_team": 1,
    "reallocate_security": 2,
    "open_exit_gate": 2,
    "throttle_entry": 3,
    "notify_operations_channel": 4,
    "log_only": 5,
}
