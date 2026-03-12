from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import geometry_mask


# Output
BASE = Path("outputs")

INTERMEDIATE_DIR = BASE / "05_intermediates"
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = BASE / "06_outputs_rasters"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Inputs
THALWEG_PATH = Path("inputs/thalweg/thalweg_full.gpkg")
WATER_MASK   = Path("inputs/water_domain/Gambia_GRWL_river_tidal_NOHOLES.gpkg")

CSV_LIST = [
    (Path("inputs/samples_csv/salinity_krig_2024_01.csv"), "2024_01"),
    (Path("inputs/samples_csv/salinity_krig_2024_06.csv"), "2024_06"),
    (Path("inputs/samples_csv/salinity_krig_2025_01.csv"), "2025_01"),
    (Path("inputs/samples_csv/salinity_krig_2025_06.csv"), "2025_06"),
]

# Settings 
CRS_OUT = 32628         # UTM 28N
GRID_RES_M = 250        # grid resolution (meters) used for 2D rasterization

# PNG preview
DISPLAY_MIN = 0.0
DISPLAY_MAX = 50.0
CMAP_NAME = "viridis"

X_TICK_INTERVAL = 1.0
Y_TICK_INTERVAL = 0.2

# Functions
def read_strict_salinity_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read a salinity CSV using a strict expected schema and robust handling
    of encoding, separators, and numeric formatting.

    Required columns:
        Point_ID, Latitude, Longitude, Month, Year, Salinity_PSU
    """
    required = ["Point_ID", "Latitude", "Longitude", "Month", "Year", "Salinity_PSU"]
    last_err = None

    for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252", "ISO-8859-1"]:
        try:
            # First try tab-separated format
            try:
                df = pd.read_csv(csv_path, sep="\t", encoding=enc)
            except Exception:
                # Fall back to automatic separator detection
                df = pd.read_csv(csv_path, sep=None, engine="python", encoding=enc)

            # Clean column names
            df.columns = (
                df.columns.astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.replace("\xa0", " ", regex=False)
                .str.strip()
            )

            # If the file was read as a single collapsed column, try again with sniffing
            if len(df.columns) == 1 and any(d in df.columns[0] for d in ["\t", ",", ";"]):
                df = pd.read_csv(csv_path, sep=None, engine="python", encoding=enc)
                df.columns = (
                    df.columns.astype(str)
                    .str.replace("\ufeff", "", regex=False)
                    .str.replace("\xa0", " ", regex=False)
                    .str.strip()
                )

            # Check required columns
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise KeyError(f"Missing columns {missing}. Got {list(df.columns)}")

            # Normalize numeric formatting
            for c in ["Latitude", "Longitude", "Salinity_PSU", "Month", "Year"]:
                df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # Drop rows with missing essential numeric values
            df = df.dropna(subset=["Latitude", "Longitude", "Salinity_PSU", "Month", "Year"])

            return df

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed reading {csv_path.name}. Last error: {last_err}")


def read_thalweg(path: Path, crs_epsg: int) -> LineString:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(crs_epsg)

    geom = gdf.geometry.iloc[0]
    if geom.geom_type == "MultiLineString":
        geom = LineString([pt for ln in geom.geoms for pt in ln.coords])
    return geom


def load_water_mask(mask_path: Path, crs_epsg: int):
    mg = gpd.read_file(mask_path)
    if mg.crs is None:
        mg = mg.set_crs(4326)
    mg = mg.to_crs(crs_epsg)
    return unary_union(mg.geometry), mg


def make_grid(bounds, res_m: float):
    minx, miny, maxx, maxy = bounds
    width = int(math.ceil((maxx - minx) / res_m))
    height = int(math.ceil((maxy - miny) / res_m))
    transform = from_origin(minx, maxy, res_m, res_m)
    return width, height, transform


def mask_raster_from_geom(geom, shape, transform):
    # geometry_mask returns True for masked (outside) by default; we invert to get water=True
    return ~geometry_mask([geom], transform=transform, invert=False, out_shape=shape, all_touched=False)


# Interpolation 
def main():
    # load base data 
    line = read_thalweg(THALWEG_PATH, CRS_OUT)
    water_union, _water_gdf = load_water_mask(WATER_MASK, CRS_OUT)

    # raster grid over water bounds
    minx, miny, maxx, maxy = water_union.bounds
    width, height, transform_utm = make_grid((minx, miny, maxx, maxy), GRID_RES_M)

    mask_bool = mask_raster_from_geom(water_union, (height, width), transform_utm)

    src_crs = CRS.from_epsg(CRS_OUT)
    dst_crs = CRS.from_epsg(4326)

    # precompute centers for water pixels (UTM)
    rc = np.argwhere(mask_bool)
    r_idx = rc[:, 0]
    c_idx = rc[:, 1]
    X = transform_utm.c + (c_idx + 0.5) * transform_utm.a
    Y = transform_utm.f + (r_idx + 0.5) * transform_utm.e  # e is negative

    # run months 
    for csv_path, label in CSV_LIST:
        df = read_strict_salinity_csv(csv_path)
        print(f"\n[{label}] {csv_path.name}  (n={len(df)})")

        lon_col, lat_col, sal_col = "Longitude", "Latitude", "Salinity_PSU"

        # Points to UTM
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=4326
        ).to_crs(CRS_OUT)

        n_pts = len(gdf)
        if n_pts < 1:
            print("  [skip] no points after cleaning.")
            continue

        # Along-thalweg distances (meters)
        s_known = np.array([line.project(pt) for pt in gdf.geometry], dtype=float)
        S_known = gdf[sal_col].astype(float).to_numpy()

        # output array
        S_hat = np.full((height, width), np.nan, dtype="float32")

        if n_pts >= 2:
            order = np.argsort(s_known)
            s_sorted = s_known[order]
            S_sorted = S_known[order]

            # project each water pixel center to thalweg distance
            s_q = np.array([line.project(Point(xx, yy)) for xx, yy in zip(X, Y)], dtype=float)

            # NOTE: np.interp extrapolates by endpoints (constant extension) by default
            vals = np.interp(s_q, s_sorted, S_sorted).astype("float32")
            S_hat[r_idx, c_idx] = vals
        else:
            S_hat[mask_bool] = float(S_known[0])

        # UTM raster 
        tif_utm = INTERMEDIATE_DIR / f"salinity_{label}_S_hat_1D_UTM.tif"
        with rasterio.open(
            tif_utm, "w",
            driver="GTiff",
            height=height, width=width,
            count=1, dtype="float32",
            crs=src_crs,
            transform=transform_utm,
            compress="deflate",
            tiled=True,
            nodata=np.nan
        ) as dst:
            dst.write(S_hat, 1)
        print("  ↳ intermediate:", tif_utm.as_posix())

        # Reproject to WGS84 
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, width, height,
            left=minx, bottom=miny, right=maxx, top=maxy
        )

        S_ll = np.full((dst_h, dst_w), np.nan, dtype="float32")
        reproject(
            source=S_hat,
            destination=S_ll,
            src_transform=transform_utm,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )

        tif_ll = OUT_DIR / f"salinity_{label}_S_hat_1D_WGS84.tif"
        with rasterio.open(
            tif_ll, "w",
            driver="GTiff",
            height=dst_h, width=dst_w,
            count=1, dtype="float32",
            crs=dst_crs,
            transform=dst_transform,
            compress="deflate",
            tiled=True,
            nodata=np.nan
        ) as dst:
            dst.write(S_ll, 1)
        print("  ↳ final raster:", tif_ll.as_posix())

        # PNG preview 
        left = dst_transform.c
        top = dst_transform.f
        right = left + dst_w * dst_transform.a
        bottom = top + dst_h * dst_transform.e
        extent_ll = (left, right, bottom, top)

        fig, ax = plt.subplots(figsize=(3.5, 4.0))
        im = ax.imshow(
            S_ll,
            extent=extent_ll,
            origin="upper",
            cmap=CMAP_NAME,
            vmin=DISPLAY_MIN,
            vmax=DISPLAY_MAX
        )

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(X_TICK_INTERVAL))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(Y_TICK_INTERVAL))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.9)
        cbar.set_label("Salinity (PSU)")

        plt.tight_layout()

        png = OUT_DIR / f"salinity_{label}_1D_masked.png"
        fig.savefig(png, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print("  ↳ PNG:", png.as_posix())

    print("\nDONE →", OUT_DIR.as_posix())


if __name__ == "__main__":
    main()