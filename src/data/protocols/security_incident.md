# Security Incident Protocol

## Trigger conditions
- `Emergency_Type == "Security"`
- OR `Security_Score > 0.6` combined with `VIP_Visits == 1` or
  `Special_Events == 1`.

## Immediate actions (priority 1)
1. Reallocate additional Security_Staff to the affected zone —
   `reallocate_security(site_name, additional_staff)`. A reasonable default
   is +20% of current Security_Staff, rounded up.
2. Notify Police_Units on-site immediately; do not wait for confirmation.
3. If the incident is near an exit/entry gate, temporarily throttle entry
   (`throttle_entry`) to reduce new arrivals into the affected zone while
   keeping exits fully open.

## Escalation
- `Risk_Level == "Critical"` combined with a Security emergency requires
  immediate `trigger_emergency_protocol` and a direct notification to
  regional command — do not rely on the standard Slack ops channel alone.

## Site-specific notes
- VIP visits and Special Events most commonly occur at Giza Pyramids,
  Grand Egyptian Museum, and Karnak Temple (highest popularity scores).
  Pre-stage extra Security_Staff at these sites whenever `VIP_Visits == 1`.
