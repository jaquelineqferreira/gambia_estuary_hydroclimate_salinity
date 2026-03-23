
This script extracts upstream salinity intrusion distances for selected isohaline thresholds from smoothed longitudinal salinity profiles of the estuary.
For each monthly salinity profile, the script determines the furthest upstream distance (km from the river mouth) at which a given salinity threshold is reached. Threshold positions are calculated using linear interpolation between profile points.

Rationale:
For each smoothed longitudinal profile:

1. Reads salinity profiles along the estuarine thalweg.
2. Identifies the upstream limit where salinity exceeds predefined thresholds.
3. Interpolates between profile points to estimate the exact threshold position.
4. Outputs a table containing intrusion distances for each threshold and month.


Inputs:
Input files must be located in:
inputs/profiles_smoothed/


Each file should represent one monthly salinity profile and follow the naming convention:
profile_YYYY_MM_smoothed.csv

CSV columns: distance_from_mouth_km; salinity_psu_smoothed.

Outputy:

- CSV file containing intrusion distances for all thresholds.

Requirements:
numpy
pandas

Notes:
- Profiles are assumed to be ordered along the estuarine thalweg from the mouth upstream.
- Missing values (`NaN`) are automatically excluded from calculations.
- If a threshold is never reached within a profile, the corresponding intrusion distance is returned as NaN.
