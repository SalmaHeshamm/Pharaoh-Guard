# General Crowd Management Protocol (No Active Emergency)

## Trigger conditions
- `Emergency_Type == "No_Emergency"` but `Risk_Level in ("Medium", "High", "Critical")`
  driven purely by occupancy / weather / queue pressure.

## Actions by risk driver
- **High Occupancy_Rate (> 0.75)**: open additional exit gates proportional
  to `exit_gates` in the site profile; throttle entry if `> 0.9`.
- **High Queue_Length / Queue_Time**: this alone does not warrant emergency
  dispatch — recommend additional entry-lane staffing instead of security
  reallocation.
- **Poor Weather_Score (< 0.5)** at outdoor sites: notify operations to
  consider shade/water stations; do not treat as a security issue.
- **High Site_Sensitivity sites** (Valley of the Kings 0.9, Abu Simbel 0.8):
  apply a lower occupancy threshold before escalating, since these sites
  degrade to Critical faster than their raw occupancy suggests.

## Notes
- This is the default fallback protocol. It should always be retrievable
  even when `Emergency_Type == "No_Emergency"`, since ~98% of the dataset
  falls in this bucket (54,307 / 55,480 records) — most of the system's
  real work happens here, not during rare emergencies.
