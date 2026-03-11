# gambia_estuary_hydroclimate_salinity
Code and workflow for hydroclimatic variability and salinity analysis in the Gambia Estuary

STEP 1: 
File: S1_thalweg_extract.py

Summary:
Thalweg extraction from river polygons using skeletonization.

This script extracts the thalweg (approximate river centerline) from a
polygon representation of a river channel. The workflow converts the
polygon to a raster grid, computes a skeleton of the channel mask,
constructs a graph representation of the skeleton, and extracts the
longest path between endpoints as the main thalweg.

Rationale:
1. Read river polygon geometry from a GIS file (GeoPackage, Shapefile, or GeoJSON)
2. Reproject geometry to a metric coordinate system
3. Rasterize the polygon at a user-defined resolution
4. Compute the skeleton (medial axis) of the rasterized channel
5. Convert the skeleton into a graph structure
6. Extract the longest path between endpoints (thalweg)
7. Smooth and densify the resulting centerline
8. Export the thalweg as GeoPackage and GeoJSON files

Inputs:
- Polygon dataset representing the river channel
- Optional layer name if using GeoPackage

Outputs:
- GeoPackage containing the extracted thalweg
- GeoJSON containing the extracted thalweg
- Optional visualization figures (PNG or SVG)

Dependencies:
numpy
geopandas
shapely
rasterio
scikit-image
matplotlib

Typical usage:
python extract_thalweg.py \
--in data/river_polygon.gpkg \
--layer river_polygon \
--out_gpkg results/thalweg.gpkg \
--out_geojson results/thalweg.geojson

