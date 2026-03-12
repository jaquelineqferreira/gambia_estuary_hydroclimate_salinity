Compute spatial salinity statistics from GeoTIFF rasters.


Inputs:
Directory containing salinity rasters:

```
.../S5_spatial_statistics/inputs/rasters/
```

Requirements:

- GeoTIFF files (`.tif` or `.tiff`)
- filename contains `YYYY_MM` identifier (e.g. `2024_01`)
- salinity stored in band 1

numpy
pandas
rasterio

Outputs:

raw_values/
   salinity_raw_YYYY_MM.csv

statistics/
   Supplementary_Table_S3_spatial_salinity_statistics.csv

Rationale:
For each raster:

- Min
- Max
- Mean
- Median
- SD (sample standard deviation, `ddof=1`)
- p5
- p25
- p75
- p95
- IQR
- number of valid pixels
