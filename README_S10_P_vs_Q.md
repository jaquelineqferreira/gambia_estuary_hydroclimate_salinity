S10_2_AugSep_P_vs_Q_GPCC_discharge_stats.py

Compare GPCC precipitation and river discharge for August, September, and August+September combined.

Rationale:
The script aligns annual precipitation and discharge records
Computes correlation and regression statistics for each station
Exports aligned year pairs 

P_AugSep_mm = August + September precipitation total per year
Q_AugSep = mean August–September discharge per yea
  
Inputs
GAMBIA_STATIONS_PRINCIPALES_clean.csv #STREAMFLOW
GPCC_v2025_050_Gambia_mean_FULLRANGE.csv #PPT

Flow file
Expected structure:
first column = datetime
remaining columns = station discharge series

GPCC file (ppt)
Expected structure:
first column = datetime
one precipitation column
