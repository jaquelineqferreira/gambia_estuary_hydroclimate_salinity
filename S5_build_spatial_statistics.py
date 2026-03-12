from __future__ import annotations
import re
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

# Paths
RASTER_DIR = Path() #add your rasters folder here

S5_ROOT = RASTER_DIR.parent.parent

RAW_DIR  = S5_ROOT / "raw_values"
STAT_DIR = S5_ROOT / "statistics"
DOC_DIR  = S5_ROOT / "documentation"

for d in (RAW_DIR, STAT_DIR, DOC_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Setting
TARGET_IDS = {"2024_01", "2024_06", "2025_01", "2025_06"}
PAT_YYYY_MM = re.compile(r"(20\d{2})[_-](\d{2})")


# Functions
def valid_raster(p: Path) -> bool:
    return (
        p.is_file()
        and not p.name.startswith(".")
        and not p.name.startswith("._")
        and p.suffix.lower() in {".tif", ".tiff"}
    )


def detect_rasters(folder: Path):
    rasters = []
    for p in sorted(folder.iterdir()):
        if not valid_raster(p):
            continue
        m = PAT_YYYY_MM.search(p.name)
        if not m:
            continue
        ID = f"{m.group(1)}_{m.group(2)}"
        rasters.append((ID, p))

    if TARGET_IDS:
        rasters = [(ID, p) for ID, p in rasters if ID in TARGET_IDS]

    rasters.sort(key=lambda x: (int(x[0][:4]), int(x[0][5:])))
    return rasters


def read_pixels(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
    v = np.array(arr.compressed(), dtype=float)
    return v[np.isfinite(v)]


def stats(v: np.ndarray):
    p5, p25, p75, p95 = np.percentile(v, [5, 25, 75, 95])
    return {
        "Min": np.min(v),
        "Max": np.max(v),
        "Mean": np.mean(v),
        "Median": np.median(v),
        "SD": np.std(v, ddof=1) if v.size > 1 else np.nan,
        "p5": p5,
        "p25": p25,
        "p75": p75,
        "p95": p95,
        "IQR": p75 - p25,
        "N_pixels": v.size,
    }


def label(year, month):
    return f"{dt.date(1900, month, 1).strftime('%b')} {year}"


# Main
print("\nS5 — Spatial salinity statistics")

rasters = detect_rasters(RASTER_DIR)
if not rasters:
    raise RuntimeError("No valid rasters found.")

records = []
raw_written = []

for ID, path in rasters:

    year, month = map(int, ID.split("_"))

    try:
        v = read_pixels(path)
    except Exception as e:
        print(f"Skipping {path.name}: {e}")
        continue

    if v.size == 0:
        continue

    raw_csv = RAW_DIR / f"salinity_raw_{ID}.csv"
    pd.DataFrame({"salinity_psu": v}).to_csv(raw_csv, index=False)
    raw_written.append(raw_csv)

    s = stats(v)
    rec = {"Raster": label(year, month), **s,
           "ID": ID, "Year": year, "Month": month,
           "SourceRaster": path.name}

    records.append(rec)

    print(f"{ID}: N={s['N_pixels']} Mean={s['Mean']:.2f}")


df = pd.DataFrame(records).sort_values(["Year", "Month"])

table_cols = ["Raster", "Min", "Max", "Mean", "Median",
              "SD", "p5", "p25", "p75", "p95", "IQR"]

qa_cols = ["N_pixels", "SourceRaster", "ID", "Year", "Month"]

table = df[table_cols + qa_cols]

table_csv = STAT_DIR / "Supplementary_Table_S3_spatial_salinity_statistics.csv"
table.to_csv(table_csv, index=False)
