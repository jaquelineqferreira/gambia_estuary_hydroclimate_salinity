from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "inputs" / "profiles_smoothed"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
THRESHOLDS = [1, 10, 20, 30, 40]

DIST_COL = "distance_from_mouth_km"
SAL_COL  = "salinity_psu_smoothed"

# Functions
def intrusion_distance(distance_km: np.ndarray, salinity: np.ndarray, T: float) -> float:
    # sort mouth -> upstream
    order = np.argsort(distance_km)
    d = np.asarray(distance_km, dtype=float)[order]
    s = np.asarray(salinity, dtype=float)[order]

    # drop NaNs
    m = np.isfinite(d) & np.isfinite(s)
    d = d[m]; s = s[m]
    if d.size == 0:
        return np.nan

    idx_ge = np.where(s >= T)[0]
    if idx_ge.size == 0:
        return np.nan

    i_hi = int(idx_ge[-1])

    # if last point is still >=T, intrusion at least that far
    if i_hi == (s.size - 1):
        return float(d[i_hi])

    i_lo = i_hi + 1
    s_hi, s_lo = float(s[i_hi]), float(s[i_lo])
    d_hi, d_lo = float(d[i_hi]), float(d[i_lo])

    # safe-guard
    if s_lo == s_hi:
        return float(d_hi)

    # linear interpolation for salinity == T
    d_T = d_hi + (T - s_hi) * (d_lo - d_hi) / (s_lo - s_hi)
    return float(d_T)

def label_from_filename(p: Path) -> str:
    # expects profile_YYYY_MM_smoothed.csv
    name = p.stem.replace("profile_", "").replace("_smoothed", "")
    return name

# Main function
def main():
    files = sorted(IN_DIR.glob("profile_*_smoothed.csv"))
    if not files:
        raise SystemExit(f"No smoothed profiles found in: {IN_DIR}")

    rows = []
    for fp in files:
        df = pd.read_csv(fp)
        if DIST_COL not in df.columns or SAL_COL not in df.columns:
            raise RuntimeError(
                f"{fp.name} missing required columns. "
                f"Got {list(df.columns)}; expected {DIST_COL}, {SAL_COL}"
            )

        label = label_from_filename(fp)
        d = df[DIST_COL].to_numpy()
        s = df[SAL_COL].to_numpy()

        entry = {"label": label}
        for T in THRESHOLDS:
            entry[f"PSU{T}_km"] = intrusion_distance(d, s, T)
        rows.append(entry)

    out = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)

    out_path = OUT_DIR / "isohaline_distances_thalweg_PSU1_10_20_30_40.csv"
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
    