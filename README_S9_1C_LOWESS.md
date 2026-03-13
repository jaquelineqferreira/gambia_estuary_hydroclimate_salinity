S9_1C_make_SuppFigure2_LOWESS_evap_drivers.py

Rationale:
input time series optionally resampled to hourly means
- LOWESS fitted with statsmodels.nonparametric.smoothers_lowess.lowess
- smoothing span: frac 0.1
- LOWESS computed independently for each variable against time

Input:
ATMOS_Bintang_clean.csv
