# Medical Emergency Protocol

## Trigger conditions
- `Emergency_Type == "Medical"`
- OR `Risk_Level in ("High", "Critical")` AND `Occupancy_Rate > 0.8` at an outdoor,
  high-sensitivity site (heat exhaustion risk).

## Immediate actions (priority 1)
1. Dispatch on-site Medical_Team to the reported zone.
2. If `Medical_Team < 2`, request backup from the nearest indoor site in the
   same region (see site_master_data regions: Cairo / Luxor / Aswan).
3. Open the nearest exit gate to allow ambulance access if the site is outdoor.

## Escalation
- If no response confirmation within 5 minutes, escalate to regional
  emergency services and notify Police_Units on site.

## Site-specific notes
- Abu Simbel and Valley of the Kings have the fewest gates (2 each) and the
  highest sensitivity scores (0.8 / 0.9) — medical response there takes
  longer; pre-position a medical unit during high-occupancy hours (10:00–14:00).
- Grand Egyptian Museum is indoor with the most gates (10 entry / 8 exit) —
  fastest evacuation path, lowest priority for extra staffing during Medical events.
