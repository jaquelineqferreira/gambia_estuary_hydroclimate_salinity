from __future__ import annotations
import datetime as dt
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# paths
THIS_DIR = Path(__file__).resolve().parent
S5_ROOT = THIS_DIR.parent

RAW_DIR = S5_ROOT / "raw_values"
IN_DIR  = THIS_DIR / "input"
OUT_DIR = THIS_DIR / "output"
FIG_DIR = THIS_DIR / "figures"
DOC_DIR = THIS_DIR / "documentation"

for d in (IN_DIR, OUT_DIR, FIG_DIR, DOC_DIR):
    d.mkdir(parents=True, exist_ok=True)


# settings
TARGET_IDS = ["2024_01", "2024_06", "2025_01", "2025_06"]
N_GRID = 600
X_PAD = 0.5
SUBSAMPLE_MAX = 200_000
RANDOM_SEED = 42
FIG_BASE = "Figure_S5_2_salinity_KDE"

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


# Functions
def read_values(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "salinity_psu" in df.columns:
        v = pd.to_numeric(df["salinity_psu"], errors="coerce").to_numpy(dtype=float)
    elif df.shape[1] == 1:
        v = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError(f"{path.name}: expected 'salinity_psu' column or single-column CSV")
    return v[np.isfinite(v)]


def subsample(v: np.ndarray, max_n: int | None) -> np.ndarray:
    if max_n is None or v.size <= max_n:
        return v
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(v.size, size=max_n, replace=False)
    return v[idx]


def month_label(ID: str) -> str:
    year, month = map(int, ID.split("_"))
    return f"{dt.date(1900, month, 1).strftime('%b')} {year}"


# Inputs
files = []
for ID in TARGET_IDS:
    path = RAW_DIR / f"salinity_raw_{ID}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    shutil.copy2(path, IN_DIR / path.name)
    files.append((ID, path))

values = {}
global_min = np.inf
global_max = -np.inf

for ID, path in files:
    v = subsample(read_values(path), SUBSAMPLE_MAX)
    values[ID] = v
    global_min = min(global_min, float(np.min(v)))
    global_max = max(global_max, float(np.max(v)))

x_grid = np.linspace(global_min - X_PAD, global_max + X_PAD, N_GRID)


# Bandwidth calculations
bw_records = []

for ID in TARGET_IDS:
    v = values[ID]
    kde = gaussian_kde(v)
    sigma = float(np.std(v, ddof=1))
    bw_records.append({
        "Raster": month_label(ID),
        "ID": ID,
        "N_pixels_used": int(v.size),
        "Bandwidth_factor_Scott": float(kde.factor),
        "Monthly_SD_PSU": sigma,
        "Effective_bandwidth_PSU": float(kde.factor * sigma),
    })

bw_df = pd.DataFrame(bw_records)
bw_df["Year"] = bw_df["ID"].str[:4].astype(int)
bw_df["Month"] = bw_df["ID"].str[5:7].astype(int)
bw_df = bw_df.sort_values(["Year", "Month"]).reset_index(drop=True)

bw_csv = OUT_DIR / "Supplementary_Table_S5_KDE_effective_bandwidth.csv"
bw_df.to_csv(bw_csv, index=False)


# KDE curves
kde_data = {"x_grid": x_grid}

for ID in TARGET_IDS:
    kde = gaussian_kde(values[ID])
    kde_data[f"kde_{ID}"] = kde.evaluate(x_grid)

kde_df = pd.DataFrame(kde_data)
kde_csv = OUT_DIR / "kde_curves_4months.csv"
kde_df.to_csv(kde_csv, index=False)
