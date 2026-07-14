"""
Static, per-site reference data lifted verbatim from
notebooks/01_Data_Generator.ipynb (dataset_summary.json -> site_master_data).

This is intentionally kept as plain Python data (not re-derived from the
CSV at runtime) so agents can look up gate counts / capacity / sensitivity
instantly without touching the dataset. If the ML team changes a site's
config in the generator notebook, mirror the change here.
"""
from __future__ import annotations

from core.schemas import SiteProfile

SITE_MASTER_DATA: dict[str, SiteProfile] = {
    "Giza Pyramids": SiteProfile(
        name="Giza Pyramids", capacity=15000, popularity=0.95, sensitivity=0.65,
        entry_gates=6, exit_gates=4, indoor=False, base_temp_offset=1.5,
        night_show=False, region="Cairo",
    ),
    "Grand Egyptian Museum": SiteProfile(
        name="Grand Egyptian Museum", capacity=20000, popularity=0.98, sensitivity=0.55,
        entry_gates=10, exit_gates=8, indoor=True, base_temp_offset=-3.0,
        night_show=False, region="Cairo",
    ),
    "Saqqara": SiteProfile(
        name="Saqqara", capacity=5000, popularity=0.55, sensitivity=0.75,
        entry_gates=3, exit_gates=2, indoor=False, base_temp_offset=1.0,
        night_show=False, region="Cairo",
    ),
    "Luxor Temple": SiteProfile(
        name="Luxor Temple", capacity=8000, popularity=0.8, sensitivity=0.7,
        entry_gates=4, exit_gates=3, indoor=False, base_temp_offset=3.5,
        night_show=True, region="Luxor",
    ),
    "Karnak Temple": SiteProfile(
        name="Karnak Temple", capacity=12000, popularity=0.85, sensitivity=0.72,
        entry_gates=5, exit_gates=4, indoor=False, base_temp_offset=3.5,
        night_show=True, region="Luxor",
    ),
    "Abu Simbel": SiteProfile(
        name="Abu Simbel", capacity=4000, popularity=0.6, sensitivity=0.8,
        entry_gates=2, exit_gates=2, indoor=False, base_temp_offset=4.5,
        night_show=False, region="Aswan",
    ),
    "Valley of the Kings": SiteProfile(
        name="Valley of the Kings", capacity=6000, popularity=0.78, sensitivity=0.9,
        entry_gates=3, exit_gates=2, indoor=False, base_temp_offset=4.0,
        night_show=False, region="Luxor",
    ),
    "Citadel of Cairo": SiteProfile(
        name="Citadel of Cairo", capacity=10000, popularity=0.5, sensitivity=0.45,
        entry_gates=4, exit_gates=3, indoor=False, base_temp_offset=1.0,
        night_show=False, region="Cairo",
    ),
}


def get_site_profile(site_name: str) -> SiteProfile:
    try:
        return SITE_MASTER_DATA[site_name]
    except KeyError as exc:
        raise ValueError(f"Unknown site: {site_name!r}") from exc
