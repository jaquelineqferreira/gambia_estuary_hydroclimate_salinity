S8_build_salt_mass_balance_Table10.py

Compute the salinity-based conservative mixing fraction.

Model:
γSAL = ΔSAL / (Ss − Sf)

where:
Sf = freshwater endmember salinity
Ss = seawater endmember salinity
ΔSAL = seasonal salinity increase
γSAL = salinity-based mixing fraction

Inputs
Sf = 0.00 PSU
Ss = 36.90 PSU (Banjul)
Kauur, 2024, ΔSAL = 13.90
Kauur, 2025, ΔSAL = 10.54

Requirements:
pandas
