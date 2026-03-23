Code and workflow for hydroclimatic variability and salinity analysis in the Gambia Estuary

S2_make_salintiy_rasters.py

1D interpolation of salinity along the thalweg using the projected distances. Assignment of interpolated values to a 2D raster grid constrained to a water-domain mask derived from the river polygon. 

Resulting rasters represent salintiy gradient along the river while maintaining spatial constrants of the river channel geometry. 

Required input data: 
1. Thalweg geometry: see README_S1_thalweg_extract.md (UTM Zone 28N)
2. Water domains mask: polygon representing the river channel mask derived from GRWL river polygon
3. CSV files cointaining salinity measuraments (columns: Point_ID, Latitude, Longitude, Month, Year, Salinity_PSU)

Outuputs:
- UTM projected salinity raster .tif
- WGS84 final salinity raster .tif
  
Rationale:
1. Load thalweg geometry and water-domain mask
2. Generate a raster grid over the water domain extent
3. Project salinity observation points onto the thalweg
4. Compute the along-thalweg distance for each sample
5. Interpolate salinity values along the thalweg using linear interpolation (NumPy np.interp())
6. Assign interpolated values to raster cell within the water mask
7. Export rasters
8. Preview figures

Important:
CRS_OUT = 32628 #working coordinate system
GRID_RED_M = #raster grid resolution

Requirements:
numpy
pandas
geopandas
shapely
rasterio
matplotlib
