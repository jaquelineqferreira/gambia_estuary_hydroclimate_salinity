from __future__ import annotations
from pathlib import Path
import datetime as dt
import pandas as pd

# JAN JUN values for PET from TerraClimate
clim_jj = clim[clim["month"].between(1, 6)].copy()
out_clim_jj = ROOT / f"PET_monthly_climatology_JanJun_{YEAR_MIN}_{YEAR_MAX}.csv"
clim_jj.to_csv(out_clim_jj, index=False)

# Annual total
annual = (
    monthly.groupby("year")
    .agg(
        PET_total_mm_year=("PET_mm_month", "sum"),
        N_months=("PET_mm_month", "size"),
    )
    .reset_index()
    .sort_values("year")
)

out_annual = ROOT / f"PET_annual_totals_{YEAR_MIN}_{YEAR_MAX}.csv"
annual.to_csv(out_annual, index=False)


# JAN JUN totals
dry = (
    monthly[monthly["month"].between(1, 6)]
    .groupby("year")
    .agg(
        PET_total_mm_JanJun=("PET_mm_month", "sum"),
        N_months=("PET_mm_month", "size"),
    )
    .reset_index()
    .sort_values("year")
)

out_dry = ROOT / f"PET_dryseason_JanJun_totals_{YEAR_MIN}_{YEAR_MAX}.csv"
dry.to_csv(out_dry, index=False)