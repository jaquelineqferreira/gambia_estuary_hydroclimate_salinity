from __future__ import annotations
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path
ROOT = Path(
    "/Volumes/Helena/DATASETS GAMBIA/Paper Isotopes bckup/"
    "Supplementary/S9/S9_1_Dry_season_meteorological_trends"
) # Adjust as needed

CSV = ROOT / "ATMOS_Bintang_clean.csv"
OUT = ROOT / "output"
FIG = OUT / "figures"
PIV = OUT / "pivots"

for d in (OUT, FIG, PIV, DOC):
    d.mkdir(parents=True, exist_ok=True)

if not CSV.exists():
    raise FileNotFoundError(f"Missing input: {CSV}")


# Settings
TIMESTAMP_COL = "TIMESTAMP"
RH_COL = "RH"
TAIR_COL = "AirTemp"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
MONTH_LABELS = ["J", "F", "M", "A", "M", "J"]

YEAR_FILTER = None
DPI = 600
HOUR_STEP = 2


# Data loading and filtering
df = pd.read_csv(CSV)

required = {TIMESTAMP_COL, RH_COL, TAIR_COL}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
df = df.dropna(subset=[TIMESTAMP_COL]).copy()

df["Year"] = df[TIMESTAMP_COL].dt.year
df["Month"] = df[TIMESTAMP_COL].dt.strftime("%b")
df["Hour"] = df[TIMESTAMP_COL].dt.hour

df = df[df["Month"].isin(MONTHS)].copy()
if YEAR_FILTER is not None:
    df = df[df["Year"] == int(YEAR_FILTER)].copy()

df[RH_COL] = pd.to_numeric(df[RH_COL], errors="coerce")
df[TAIR_COL] = pd.to_numeric(df[TAIR_COL], errors="coerce")
df = df[(df[RH_COL].isna()) | ((df[RH_COL] >= 0) & (df[RH_COL] <= 100))].copy()

if df.empty:
    raise RuntimeError("No data after filtering.")


# Pivots
pivot_rh = df.groupby(["Month", "Hour"])[RH_COL].mean().unstack()
pivot_t = df.groupby(["Month", "Hour"])[TAIR_COL].mean().unstack()
coverage = df.groupby(["Month", "Hour"]).size().unstack(fill_value=0)

pivot_rh = pivot_rh.reindex(MONTHS).reindex(columns=range(24))
pivot_t = pivot_t.reindex(MONTHS).reindex(columns=range(24))
coverage = coverage.reindex(MONTHS).reindex(columns=range(24), fill_value=0)

suffix = f"JanJun_{YEAR_FILTER}" if YEAR_FILTER is not None else "JanJun_ALLYEARS"

rh_csv = PIV / f"S9_1_month_hour_pivot_RH_{suffix}.csv"
t_csv = PIV / f"S9_1_month_hour_pivot_Tair_{suffix}.csv"
cov_csv = PIV / f"S9_1_month_hour_coverage_counts_{suffix}.csv"

pivot_rh.to_csv(rh_csv)
pivot_t.to_csv(t_csv)
coverage.to_csv(cov_csv)


# Plot
def plot_heatmap(pivot: pd.DataFrame, title: str, cbar_label: str, path_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="equal", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Month")

    ax.set_xticks(np.arange(0, 24, HOUR_STEP))
    ax.set_xticklabels([str(h) for h in range(0, 24, HOUR_STEP)])
    ax.set_yticks(np.arange(len(MONTHS)))
    ax.set_yticklabels(MONTH_LABELS)

    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MONTHS), 1), minor=True)
    ax.grid(which="minor", color="0.75", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    fig.savefig(path_base.with_suffix(".png"), dpi=DPI)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


rh_fig = FIG / f"S9_1_heatmap_RH_month_hour_{suffix}"
t_fig = FIG / f"S9_1_heatmap_Tair_month_hour_{suffix}"

plot_heatmap(
    pivot_rh,
    f"Hourly relative humidity by month (Jan–Jun) — Bintang ATMOS ({suffix})",
    "Relative humidity (%)",
    rh_fig
)

plot_heatmap(
    pivot_t,
    f"Hourly air temperature by month (Jan–Jun) — Bintang ATMOS ({suffix})",
    "Air temperature (°C)",
    t_fig
)