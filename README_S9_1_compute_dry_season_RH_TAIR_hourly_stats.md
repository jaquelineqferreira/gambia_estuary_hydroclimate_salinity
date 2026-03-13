S9_1_compute_dry_season_RH_TAIR_hourly_stats.py

Rationale:
The script reads 10-minute ATMOS observations, aggregates them to hourly means, and computes overall mean and standard deviation for:

- air temperature (`AirTemp`)
- relative humidity (`RH`)

Input:
ATMOS_Bintang_clean.csv

Required columns: TIMESTAMP; AirTemp; RH; Month; Year

Requirements:
numpy
pandas
