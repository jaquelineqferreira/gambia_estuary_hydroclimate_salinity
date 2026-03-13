S9_1B_make_month_hour_heatmaps_ATMOS.py

Generate month vs. hour heatmaps of relative humidity and air temperature.

Rationale:
The script reads cleaned ATMOS data, timestamp parsed from TIMESTAMP
Groups observations by month and hour of day, 
Computes mean values
Exports pivot tables
Saves heatmaps for: relative humidity (`RH`), and air temperature (`AirTemp`)

Input
ATMOS_Bintang_clean.csv

Requirements:
pandas
numpy 
matplotlib
