"""Endpoint-separated CD45/CD34 PepMix validation table."""
from __future__ import annotations
import pandas as pd


def pepmix_anchor_table() -> pd.DataFrame:
    return pd.DataFrame([
        ["PepMix1", 41606, 24249, 85.16, 83, 75, 0.26, 90.36, 0.26, "intermediate retention"],
        ["PepMix2", 56594, 27110, 65.86, 156, 129, 0.31, 82.87, 0.45, "highest rare-compartment retention"],
        ["PepMix3", 31671, 12629, 85.13, 286, 35, 0.24, 12.44, 0.27, "CD34-enriched outlier"],
        ["PepMix4", 28803, 19988, 79.85, 91, 56, 0.22, 61.90, 0.24, "high-suppression side"],
        ["PepMix5", 42067, 25260, 72.48, 147, 83, 0.24, 56.58, 0.29, "moderate-to-high retention"],
        ["PepMix6", 37973, 21549, 78.93, 107, 38, 0.14, 35.38, 0.15, "lowest retention"],
    ], columns=[
        "pepmix", "acquired_events", "cd45_events", "cd45_percent_parent", "cd34_events",
        "cd34_cd45dim_events", "cd34_cd45dim_percent_parent", "dim_per_cd34_percent",
        "cd34_hpc_percent_total", "phenotype_interpretation"
    ])
