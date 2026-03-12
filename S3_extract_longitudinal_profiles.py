from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import rasterio

# Settings
EPSG_UTM = 32628
STEP_M = 250.0

# Gaussian smoothing parameters
GAUSS_WINDOW = 81
GAUSS_SIGMA = 20.0

# Station coordinates in EPSG:32628
STATIONS_UTM: List[Tuple[str, float, float]] = [
    ("Fatoto",         620140.53, 1482599.97),
    ("Basse",          585696.35, 1475260.62),
    ("Bansang",        537636.82, 1485510.02),
    ("Lamin Koto",     525597.07, 1497455.50),
    ("Kuntaur",        511843.72, 1511510.98),
    ("Baboon Islands", 490162.67, 1513628.78),
    ("Kauur",          464625.11, 1513809.52),
    ("Yelitenda",      437879.78, 1494500.11),
    ("Tendaba",        413053.40, 1487903.73),
    ("Bintang Bolong", 368797.68, 1465356.10),
]

# paths
WORKSPACE = Path(__file__).resolve().parents[1]
IN_THALWEG = WORKSPACE / "inputs" / "thalweg" / "thalweg_full.gpkg"
IN_RASTERS = WORKSPACE / "inputs" / "rasters_utm"

OUT_RAW = WORKSPACE / "outputs" / "01_profiles_raw"
OUT_SMO = WORKSPACE / "outputs" / "02_profiles_smoothed"
OUT_STA = WORKSPACE / "outputs" / "03_stations_chainage"

for folder in [OUT_RAW, OUT_SMO, OUT_STA]:
    folder.mkdir(parents=True, exist_ok=True)


# functions 
# Read thalweg geometry and ensure it's in the correct CRS and format
def read_thalweg(path: Path, epsg: int = EPSG_UTM) -> LineString:
    gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg)
    else:
        gdf = gdf.to_crs(epsg)

    geom = gdf.geometry.iloc[0]

    if geom.geom_type == "MultiLineString":
        geom = LineString([pt for ln in geom.geoms for pt in ln.coords])

    if geom.geom_type != "LineString":
        raise ValueError(f"Thalweg must resolve to LineString; got {geom.geom_type}")

    return geom

# Find raster files matching expected pattern and sort them
def find_rasters_utm(folder: Path) -> List[Path]:
    rasters = sorted(folder.glob("salinity_*_S_hat_1D_UTM.tif"))
    if not rasters:
        raise FileNotFoundError(
            f"No UTM rasters found in: {folder}\n"
            f"Expected names like: salinity_2024_01_S_hat_1D_UTM.tif"
        )
    return rasters

# Extract label (e.g. "2024_01") from raster filename
def label_from_raster_name(path: Path) -> str:
    parts = path.stem.split("_")
    year = None
    month = None

    for i in range(len(parts) - 1):
        if parts[i].isdigit() and len(parts[i]) == 4:
            year = parts[i]
            nxt = parts[i + 1]
            if nxt.isdigit():
                month = nxt.zfill(2)
                break

    if year is None or month is None:
        raise ValueError(f"Could not parse label from raster name: {path.name}")

    return f"{year}_{month}"

# create Gaussian kernel and smoothing function
def gaussian_kernel(window: int, sigma: float) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError("GAUSS_WINDOW must be odd for centered smoothing.")

    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel

# Apply Gaussian smoothing with reflect padding and linear interpolation for missing values
def gaussian_smooth_1d(y: np.ndarray, window: int, sigma: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y_filled = y.copy()

    idx = np.arange(y_filled.size)
    finite = np.isfinite(y_filled)

    if finite.sum() >= 2:
        y_filled[~finite] = np.interp(idx[~finite], idx[finite], y_filled[finite])
    elif finite.sum() == 1:
        y_filled[~finite] = y_filled[finite][0]
    else:
        return np.full_like(y_filled, np.nan)

    kernel = gaussian_kernel(window, sigma)
    pad = window // 2
    y_pad = np.pad(y_filled, pad_width=pad, mode="reflect")
    y_smooth = np.convolve(y_pad, kernel, mode="valid")

    return y_smooth.astype(float)

# Compute station distances from the estuary mouth by projecting station points onto the thalweg line
def stations_chainage_from_mouth(
    thalweg: LineString,
    stations: List[Tuple[str, float, float]]
) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        {
            "site": [s[0] for s in stations],
            "E": [s[1] for s in stations],
            "N": [s[2] for s in stations],
        },
        geometry=[Point(s[1], s[2]) for s in stations],
        crs=f"EPSG:{EPSG_UTM}",
    )

    gdf["m_from_mouth_m"] = [float(thalweg.project(p)) for p in gdf.geometry]
    gdf["proj_point"] = [thalweg.interpolate(m) for m in gdf["m_from_mouth_m"]]
    gdf["distance_from_mouth_km"] = gdf["m_from_mouth_m"] / 1000.0

    gdf = gdf.sort_values("distance_from_mouth_km").reset_index(drop=True)
    gdf.attrs["thalweg_length_m"] = float(thalweg.length)
    return gdf

# Sample points along the thalweg at regular intervals and return their distance from the mouth in km
def thalweg_distance_km_array(thalweg: LineString, step_m: float):
    length_m = float(thalweg.length)
    s_vals = np.arange(0.0, length_m + 1e-6, step_m, dtype=float)
    points = [thalweg.interpolate(float(s)) for s in s_vals]
    dist_km = s_vals / 1000.0
    return dist_km, points

def sample_raster_nearest(raster_path: Path, points: List[Point]) -> np.ndarray:
    with rasterio.open(raster_path) as ds:
        coords = [(p.x, p.y) for p in points]
        values = np.array([v[0] for v in ds.sample(coords)], dtype=float)
    return values


# Main processing function
def main():
    if not IN_THALWEG.exists():
        raise FileNotFoundError(f"Missing thalweg: {IN_THALWEG}")

    rasters = find_rasters_utm(IN_RASTERS)
    thalweg = read_thalweg(IN_THALWEG, EPSG_UTM)

    # Station distances from mouth
    gdf_sta = stations_chainage_from_mouth(thalweg, STATIONS_UTM)
    thalweg_length_m = float(gdf_sta.attrs["thalweg_length_m"])

    # Save station outputs
    sta_csv = OUT_STA / "stations_distance_from_mouth.csv"
    sta_gpkg = OUT_STA / "stations_distance_from_mouth.gpkg"

    gdf_sta.drop(columns=["proj_point"]).to_csv(sta_csv, index=False)
    gdf_sta.set_geometry("geometry").to_file(sta_gpkg, layer="stations", driver="GPKG")

    gpd.GeoDataFrame(
        gdf_sta[["site", "distance_from_mouth_km"]],
        geometry=gdf_sta["proj_point"],
        crs=gdf_sta.crs,
    ).to_file(sta_gpkg, layer="stations_on_thalweg", driver="GPKG")

    print(f"[S3] Thalweg length: {thalweg_length_m / 1000.0:.2f} km")
    print(f"[S3] Stations table: {sta_csv.name} | {sta_gpkg.name}")

    # Sample points along thalweg
    dist_km, points = thalweg_distance_km_array(thalweg, STEP_M)

    # Monthly profiles
    for raster_path in rasters:
        label = label_from_raster_name(raster_path)
        salinity = sample_raster_nearest(raster_path, points)

        # Raw profile
        df_raw = pd.DataFrame(
            {
                "distance_from_mouth_km": dist_km,
                "salinity_psu": salinity,
            }
        ).sort_values("distance_from_mouth_km").reset_index(drop=True)

        raw_path = OUT_RAW / f"profile_{label}_raw.csv"
        df_raw.to_csv(raw_path, index=False)

        # Smoothed profile
        smoothed = gaussian_smooth_1d(
            df_raw["salinity_psu"].to_numpy(),
            GAUSS_WINDOW,
            GAUSS_SIGMA,
        )

        df_smo = pd.DataFrame(
            {
                "distance_from_mouth_km": df_raw["distance_from_mouth_km"].to_numpy(),
                "salinity_psu_smoothed": smoothed,
            }
        )

        smo_path = OUT_SMO / f"profile_{label}_smoothed.csv"
        df_smo.to_csv(smo_path, index=False)

        print(f"[S3] {label}: saved {raw_path.name} and {smo_path.name}")

    print("\nDONE.")
    print("Outputs:")
    print(f"  {OUT_RAW}")
    print(f"  {OUT_SMO}")
    print(f"  {OUT_STA}")


if __name__ == "__main__":
    main()