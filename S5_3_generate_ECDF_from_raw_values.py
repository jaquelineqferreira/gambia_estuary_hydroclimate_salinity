from __future__ import annotations
import datetime as dt
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Paths
THIS_DIR = Path(__file__).resolve().parent
S5_ROOT = THIS_DIR.parent

RAW_DIR = S5_ROOT / "raw_values"
IN_DIR  = THIS_DIR / "input"
OUT_DIR = THIS_DIR / "output"
FIG_DIR = THIS_DIR / "figures"
DOC_DIR = THIS_DIR / "documentation"

for d in (IN_DIR, OUT_DIR, FIG_DIR, DOC_DIR):
    d.mkdir(parents=True, exist_ok=True)


# Settings
TARGET_IDS = ["2024_01", "2024_06", "2025_01", "2025_06"]
THRESHOLDS = [35.0, 40.0]
X_PAD = 0.5
SUBSAMPLE_MAX = None
RANDOM_SEED = 42
FIG_BASE = "Figure_S5_3_salinity_ECDF"

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


def ecdf(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(v.astype(float))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


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

x_min = global_min - X_PAD
x_max = global_max + X_PAD


# Fractions
rows = []
for ID in TARGET_IDS:
    v = values[ID]
    rows.append({
        "Raster": month_label(ID),
        "Fraction above 35 PSU": float(np.mean(v > 35.0)),
        "Fraction above 40 PSU": float(np.mean(v > 40.0)),
        "ID": ID,
    })

tab = pd.DataFrame(rows)
tab["Year"] = tab["ID"].str[:4].astype(int)
tab["Month"] = tab["ID"].str[5:7].astype(int)
tab = tab.sort_values(["Year", "Month"]).drop(columns=["ID", "Year", "Month"]).reset_index(drop=True)

s4_csv = OUT_DIR / "Supplementary_Table_S4_hypersaline_fractions_from_ECDF.csv"
tab.to_csv(s4_csv, index=False)


# ECDF curves
curve_rows = []
for ID in TARGET_IDS:
    x, y = ecdf(values[ID])
    curve_rows.append(pd.DataFrame({
        "id": ID,
        "Raster": month_label(ID),
        "salinity_psu": x,
        "ecdf": y,
    }))

curves = pd.concat(curve_rows, ignore_index=True)
ecdf_csv = OUT_DIR / "ecdf_curves_4months_long.csv"
curves.to_csv(ecdf_csv, index=False)

