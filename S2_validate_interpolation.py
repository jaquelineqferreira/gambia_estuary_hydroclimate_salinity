from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString
import rasterio
import matplotlib.pyplot as plt
import pyogrio 


# Inputs and outputs
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # .../S2_salinity_interpolation_workspace

THALWEG_PATH = ROOT / "inputs" / "thalweg" / "thalweg_full.gpkg"

CSV_LIST = [
    (ROOT / "inputs" / "samples_csv" / "salinity_krig_2024_01.csv", "2024_01"),
    (ROOT / "inputs" / "samples_csv" / "salinity_krig_2024_06.csv", "2024_06"),
    (ROOT / "inputs" / "samples_csv" / "salinity_krig_2025_01.csv", "2025_01"),
    (ROOT / "inputs" / "samples_csv" / "salinity_krig_2025_06.csv", "2025_06"),
]

UTM_RASTER_DIR = ROOT / "outputs" / "05_intermediates"  # produced by S2_make_salinity_rasters.py
OUT_DIR = ROOT / "outputs" / "07_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CRS_UTM = 32628  # must match S2 rasters

# functions 
def _write_check(lines: list[str]) -> None:
    (OUT_DIR / "00_inputs_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # reading salinity CVSs 
def read_strict_salinity_csv(csv_path: Path) -> pd.DataFrame:
    required = ["Point_ID", "Latitude", "Longitude", "Month", "Year", "Salinity_PSU"]
    last_err = None

    for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252", "ISO-8859-1"]:
        try:
            try:
                df = pd.read_csv(csv_path, sep=",", encoding=enc)
            except Exception:
                df = pd.read_csv(csv_path, sep=None, engine="python", encoding=enc)

            df.columns = (
                df.columns.astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.replace("\xa0", " ", regex=False)
                .str.strip()
            )

            # If header collapsed into one column, re-read with sniffing
            if len(df.columns) == 1 and any(ch in df.columns[0] for ch in [",", ";", "\t"]):
                df = pd.read_csv(csv_path, sep=None, engine="python", encoding=enc)
                df.columns = (
                    df.columns.astype(str)
                    .str.replace("\ufeff", "", regex=False)
                    .str.replace("\xa0", " ", regex=False)
                    .str.strip()
                )

            missing = [c for c in required if c not in df.columns]
            if missing:
                raise KeyError(f"Missing columns {missing}. Got {list(df.columns)}")

            for c in ["Latitude", "Longitude", "Salinity_PSU", "Month", "Year"]:
                df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.dropna(subset=["Latitude", "Longitude", "Salinity_PSU", "Month", "Year"]).copy()
            return df

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed reading {csv_path.name}. Last error: {last_err}")

    # reading thalweg geometry from GPKG
def choose_thalweg_layer(gpkg: Path) -> str:
    layers = pyogrio.list_layers(str(gpkg))
    for name, gtype in layers:
        if "line" in str(gtype).lower():
            return str(name)
    raise RuntimeError(f"No line layer found in {gpkg}. Layers: {layers}")

def read_thalweg(path: Path, crs_epsg: int = CRS_UTM) -> LineString:
    layer = choose_thalweg_layer(path)
    gdf = gpd.read_file(path, layer=layer)
    if gdf.empty:
        raise RuntimeError(f"Thalweg layer '{layer}' is empty in {path}")

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(crs_epsg)

    geom = gdf.geometry.iloc[0]
    if geom.geom_type == "MultiLineString":
        geom = LineString([pt for ln in geom.geoms for pt in ln.coords])

    if geom.geom_type != "LineString":
        raise RuntimeError(f"Unexpected thalweg geometry type: {geom.geom_type}")

    return geom


def metrics(obs, pred):
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() == 0:
        return dict(RMSE=np.nan, MAE=np.nan, Bias=np.nan, R2=np.nan, MedAE=np.nan, N=0)

    o = np.asarray(obs)[mask]
    p = np.asarray(pred)[mask]

    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    mae = float(np.mean(np.abs(p - o)))
    mbe = float(np.mean(p - o))
    med = float(np.median(np.abs(p - o)))

    if o.size > 1:
        std_o = float(np.std(o))
        std_p = float(np.std(p))
        if std_o > 0.0 and std_p > 0.0:
            r = float(np.corrcoef(o, p)[0, 1])
            r2 = r * r
        else:
            r2 = np.nan
    else:
        r2 = np.nan

    return dict(RMSE=rmse, MAE=mae, Bias=mbe, R2=r2, MedAE=med, N=int(o.size))


def interp1d_clip(s_known, S_known, s_q):
    order = np.argsort(s_known)
    s = np.asarray(s_known)[order]
    S = np.asarray(S_known)[order]
    smin, smax = s[0], s[-1]
    sq = np.asarray(s_q)
    return np.interp(np.clip(sq, smin, smax), s, S)


# main validation function
def main():
    # write a "inputs check" report to help debug any issues with missing or malformed inputs
    checks = []
    checks.append(f"Workspace ROOT: {ROOT}")
    checks.append(f"THALWEG_PATH: {THALWEG_PATH} | exists={THALWEG_PATH.exists()}")
    if THALWEG_PATH.exists():
        try:
            checks.append(f"Thalweg layers: {pyogrio.list_layers(str(THALWEG_PATH))}")
        except Exception as e:
            checks.append(f"Thalweg layers: ERROR {e}")

    for csv_path, label in CSV_LIST:
        checks.append(f"CSV {label}: {csv_path} | exists={csv_path.exists()}")

    for _csv_path, label in CSV_LIST:
        tif = UTM_RASTER_DIR / f"salinity_{label}_S_hat_1D_UTM.tif"
        checks.append(f"Raster {label}: {tif} | exists={tif.exists()}")

    _write_check(checks)

    # requirements
    if not THALWEG_PATH.exists():
        raise FileNotFoundError(f"Missing thalweg: {THALWEG_PATH}")

    for csv_path, label in CSV_LIST:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for {label}: {csv_path}")

    for _csv_path, label in CSV_LIST:
        tif = UTM_RASTER_DIR / f"salinity_{label}_S_hat_1D_UTM.tif"
        if not tif.exists():
            raise FileNotFoundError(f"Missing UTM raster for {label}: {tif}")

    # load thalweg geometry 
    line = read_thalweg(THALWEG_PATH, CRS_UTM)

    summary_rows = []

    for csv_path, label in CSV_LIST:
        tif = UTM_RASTER_DIR / f"salinity_{label}_S_hat_1D_UTM.tif"
        df = read_strict_salinity_csv(csv_path)

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
            crs=4326
        ).to_crs(CRS_UTM)

        if len(gdf) < 1:
            print(f"[{label}] skip: no points.")
            continue

        gdf["s_m"] = gdf.geometry.apply(lambda p: line.project(p))
        s_known = gdf["s_m"].to_numpy(dtype=float)
        S_obs = gdf["Salinity_PSU"].to_numpy(dtype=float)

        # in-sample: sample the UTM raster at station points
        with rasterio.open(tif) as ds:
            coords = [(pt.x, pt.y) for pt in gdf.geometry]
            samples = list(ds.sample(coords))
            S_rast = np.array(
                [s[0] if (s is not None and np.isfinite(s[0])) else np.nan for s in samples],
                dtype=float
            )

        met_rast = metrics(S_obs, S_rast)
        res_rast = S_rast - S_obs

        pd.DataFrame({
            "label": label,
            "Point_ID": gdf["Point_ID"].to_numpy(),
            "distance_m": s_known,
            "distance_km": s_known / 1000.0,
            "obs_psu": S_obs,
            "pred_psu_raster": S_rast,
            "residual_psu_raster": res_rast
        }).to_csv(OUT_DIR / f"residuals_raster_{label}.csv", index=False)

        # LOOCV along-thalweg (1D)
        preds_loocv = np.full_like(S_obs, np.nan, dtype=float)
        for i in range(len(S_obs)):
            keep = np.ones(len(S_obs), dtype=bool)
            keep[i] = False
            if keep.sum() < 1:
                continue
            preds_loocv[i] = interp1d_clip(s_known[keep], S_obs[keep], s_known[i])

        met_loocv = metrics(S_obs, preds_loocv)
        res_loocv = preds_loocv - S_obs

        pd.DataFrame({
            "label": label,
            "Point_ID": gdf["Point_ID"].to_numpy(),
            "distance_m": s_known,
            "distance_km": s_known / 1000.0,
            "obs_psu": S_obs,
            "pred_psu_loocv": preds_loocv,
            "residual_psu_loocv": res_loocv
        }).to_csv(OUT_DIR / f"residuals_loocv_{label}.csv", index=False)

        summary_rows.append({
            "label": label,
            "n_points": int(len(S_obs)),
            "RMSE_in_sample_PSU": met_rast["RMSE"],
            "R2_in_sample": met_rast["R2"],
            "RMSE_LOOCV_PSU": met_loocv["RMSE"],
            "R2_LOOCV": met_loocv["R2"],
            "csv_path": str(csv_path),
            "utm_tif_path": str(tif)
        })

        print(f"[{label}] n={len(S_obs)} | in-sample RMSE={met_rast['RMSE']:.2f}, R2={met_rast['R2']:.2f} | "
              f"LOOCV RMSE={met_loocv['RMSE']:.2f}, R2={met_loocv['R2']:.2f}")

        # plots figures
        fig, ax = plt.subplots(figsize=(5.2, 5))
        ax.scatter(S_obs, S_rast, s=32, alpha=0.85)
        lo = float(np.nanmin([np.nanmin(S_obs), np.nanmin(S_rast), 0.0]))
        hi = float(np.nanmax([np.nanmax(S_obs), np.nanmax(S_rast), 45.0]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Observed PSU")
        ax.set_ylabel("Predicted PSU (raster)")
        ax.set_title(f"O vs P — {label}\nRMSE={met_rast['RMSE']:.2f}, R²={met_rast['R2']:.2f}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"scatter_raster_{label}.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        ax.axhline(0, color="k", lw=1)
        ax.scatter(s_known / 1000.0, res_rast, s=28, alpha=0.85)
        ax.set_xlabel("Distance from mouth (km)")
        ax.set_ylabel("Residual (P−O) PSU")
        ax.set_title(f"Residuals vs distance — {label} (raster sampling)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"residuals_vs_dist_raster_{label}.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 5))
        ax.scatter(S_obs, preds_loocv, s=32, alpha=0.85)
        lo = float(np.nanmin([np.nanmin(S_obs), np.nanmin(preds_loocv), 0.0]))
        hi = float(np.nanmax([np.nanmax(S_obs), np.nanmax(preds_loocv), 45.0]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Observed PSU")
        ax.set_ylabel("Predicted PSU (LOOCV)")
        ax.set_title(f"LOOCV O vs P — {label}\nRMSE={met_loocv['RMSE']:.2f}, R²={met_loocv['R2']:.2f}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"scatter_loocv_{label}.png", dpi=220)
        plt.close(fig)

    summary = pd.DataFrame(summary_rows).sort_values("label")
    summary.to_csv(OUT_DIR / "validation_summary_by_month.csv", index=False)

    if len(summary):
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        x = np.arange(len(summary))
        ax.bar(x - 0.18, summary["RMSE_in_sample_PSU"], width=0.36, label="Raster sampling (in-sample)")
        ax.bar(x + 0.18, summary["RMSE_LOOCV_PSU"], width=0.36, label="LOOCV (1D)")
        ax.set_xticks(x, summary["label"], rotation=45, ha="right")
        ax.set_ylabel("RMSE (PSU)")
        ax.set_title("Monthly RMSE comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "rmse_comparison_monthly.png", dpi=220)
        plt.close(fig)

    print("\nDone. Validation outputs →", OUT_DIR)


if __name__ == "__main__":
    main()
    