from __future__ import annotations
from pathlib import Path
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


# Paths
ROOT = Path(
    "/Volumes/Helena/DATASETS GAMBIA/Paper Isotopes bckup/"
    "Supplementary/S9/S9_1_Dry_season_meteorological_trends"
) # Adjust as needed

CSV = ROOT / "ATMOS_Bintang_clean.csv"
OUT = ROOT / "output"
FIG = OUT / "figures"
TAB = OUT / "tables"

for d in (OUT, FIG, TAB, DOC):
    d.mkdir(parents=True, exist_ok=True)

if not CSV.exists():
    raise FileNotFoundError(f"Missing input: {CSV}")


# Settings
TIME_COL = "TIMESTAMP"
YEAR = 2025
DATE_START = "2025-01-01"
DATE_END = "2025-06-30 23:59:59"

LOWESS_FRAC = 0.1
RESAMPLE_TO_HOURLY = True
RESAMPLE_RULE = "1h"
DPI = 600

VARS = [
    ("RH", "Relative Humidity (%)", "blue", "A"),
    ("AirTemp", "Air temperature (°C)", "red", "B"),
    ("WindSpeed", "Wind speed (m/s)", "green", "C"),
    ("Solar", "Solar radiation (W/m²)", "gold", "D"),
]

FIG_BASE = "Supplementary_Figure_2_LOWESS_evap_drivers_JanJun2025"


# Data loading and filtering
df = pd.read_csv(CSV)

required = {TIME_COL} | {v[0] for v in VARS}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL]).copy()
df = df[(df[TIME_COL] >= pd.Timestamp(DATE_START)) & (df[TIME_COL] <= pd.Timestamp(DATE_END))].copy()
df["Year"] = df[TIME_COL].dt.year
df = df[df["Year"] == YEAR].copy()

for col, *_ in VARS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(TIME_COL)

if df.empty:
    raise RuntimeError("No data after filtering.")


# Resample
df = df.set_index(TIME_COL)

if RESAMPLE_TO_HOURLY:
    df_plot = df.resample(RESAMPLE_RULE).mean(numeric_only=True).dropna(how="all")
else:
    df_plot = df.copy()

if df_plot.empty:
    raise RuntimeError("No data after resampling.")


# Lowess
def fit_lowess(series: pd.Series, frac: float) -> pd.DataFrame:
    s = series.dropna()
    if s.empty:
        return pd.DataFrame(columns=["TIMESTAMP", "LOWESS"])
    x = s.index.to_julian_date().to_numpy(dtype=float)
    y = s.to_numpy(dtype=float)
    sm = lowess(y, x, frac=frac, return_sorted=True)
    t = pd.to_datetime(sm[:, 0], origin="julian", unit="D")
    return pd.DataFrame({"TIMESTAMP": t, "LOWESS": sm[:, 1]})


tables = []
for col, _, _, panel in VARS:
    t = fit_lowess(df_plot[col], LOWESS_FRAC)
    t["Variable"] = col
    t["Panel"] = panel
    tables.append(t)

lowess_df = pd.concat(tables, ignore_index=True)
lowess_csv = TAB / "SuppFig2_LOWESS_curves_JanJun2025.csv"
lowess_df.to_csv(lowess_csv, index=False)
