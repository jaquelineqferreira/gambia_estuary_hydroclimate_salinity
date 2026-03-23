Script extracts longitudinal salinity profiles long the esturine thalweg from monthly salinity rasters.
Workflow consists in sampling monthly salinity rastes along a predefined thalweg at fixed intervals and export both raw and smothed longitudinal salinity profiles. 
Computes the distance reference stations from th estuary mouth. 

Inputs:
- thalweg .gpkg
- rasters .tif

Output:
- raw longitudinal salinity profiles (profile_YYYY_MM_raw.csv)
- Gaussian-smoothed longitudinal salinity profile (profile_YYYY_MM_smoothed.csv)
- station distances from the estuary mouth (stations_distance_from_mouth.csv)
- station geometries and projected thalweg positions (stations_distance_from_mouth.gpkg)

Rationale:
- read the thalweg geometry in EPSG:32628
- read monthly salinity rasters in UTM coordintes
- sample raster values along the thalweg every 250 m using nearest-neighbor extraction
- correct thalweg orientation so that 0 km corresponds to the estuary mouth and distances increase upstream
- export raw salinity profiles
- apply Gaussian smoothing for the extracted profiles
- export station chainge reltive to the estuary mouth
