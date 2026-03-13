from __future__ import annotations
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd

# Paths
ROOT = Path(
    "/Volumes/Helena/DATASETS GAMBIA/Paper Isotopes bckup/"
    "Supplementary/S9/S9_1_Dry_season_meteorological_trends"
) # Adjust as needed

CSV_PATH = ROOT / "ATMOS_Bintang_clean.csv"
OUT_DIR = ROOT / "output"

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Missing input: {CSV_PATH}")


# Seetings
DRY_MONTHS = {"Jan", "Feb", "Mar", "Apr", "May", "Jun"}
YEAR_MIN = None
YEAR_MAX = None

TIMESTAMP_COL = "TIMESTAMP"
COL_TAIR = "AirTemp"
COL_RH = "RH"

# Data loading and filtering
df = pd.read_csv(CSV_PATH)

required = {TIMESTAMP_COL, COL_TAIR, COL_RH, "Month", "Year"}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
df = df.dropna(subset=[TIMESTAMP_COL])

df["Month"] = df["Month"].astype(str)
df = df[df["Month"].isin(DRY_MONTHS)]

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

if YEAR_MIN is not None:
    df = df[df["Year"] >= YEAR_MIN]
if YEAR_MAX is not None:
    df = df[df["Year"] <= YEAR_MAX]

df[COL_TAIR] = pd.to_numeric(df[COL_TAIR], errors="coerce")
df[COL_RH] = pd.to_numeric(df[COL_RH], errors="coerce")
df = df.dropna(subset=[COL_TAIR, COL_RH], how="all")

if df.empty:
    raise RuntimeError("No data left after filtering.")


# Hourly aggregation
hourly = (
    df.set_index(TIMESTAMP_COL)
      .sort_index()
      .resample("h")
      .agg({COL_TAIR: "mean", COL_RH: "mean"})
      .dropna(subset=[COL_TAIR, COL_RH], how="all")
      .reset_index()
)

if hourly.empty:
    raise RuntimeError("Hourly aggregation produced no data.")

hourly["Year"] = hourly[TIMESTAMP_COL].dt.year
hourly["Month_num"] = hourly[TIMESTAMP_COL].dt.month
hourly["Hour"] = hourly[TIMESTAMP_COL].dt.hour

hourly_csv = OUT_DIR / "S9_1_hourly_timeseries.csv"
hourly.to_csv(hourly_csv, index=False)


# Stats
def mean_sd(x: pd.Series) -> tuple[float, float, int]:
    v = x.dropna().to_numpy(dtype=float)
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
    return float(np.mean(v)), sd, n


t_mean, t_sd, n_t = mean_sd(hourly[COL_TAIR])
rh_mean, rh_sd, n_rh = mean_sd(hourly[COL_RH])

t0 = hourly[TIMESTAMP_COL].min()
t1 = hourly[TIMESTAMP_COL].max()

summary = pd.DataFrame([{
    "Dataset": CSV_PATH.name,
    "Dry_months": ",".join(sorted(DRY_MONTHS)),
    "Year_min_filter": YEAR_MIN,
    "Year_max_filter": YEAR_MAX,
    "Start_timestamp": t0,
    "End_timestamp": t1,
    "N_hourly_records_total": int(len(hourly)),
    "AirTemp_mean_C": t_mean,
    "AirTemp_SD_C": t_sd,
    "AirTemp_N": n_t,
    "RH_mean_pct": rh_mean,
    "RH_SD_pct": rh_sd,
    "RH_N": n_rh,
}])

summary_csv = OUT_DIR / "S9_1_hourly_summary_stats.csv"
summary.to_csv(summary_csv, index=False)