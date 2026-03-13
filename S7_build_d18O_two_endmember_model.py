from pathlib import Path
import datetime as dt
import pandas as pd

# Paths
ROOT = Path(
"/Volumes/Helena/DATASETS GAMBIA/Paper Isotopes bckup/"
"Supplementary/S7"
)

OUT = ROOT / "output"
DOC = ROOT / "documentation"

OUT.mkdir(parents=True, exist_ok=True)
DOC.mkdir(parents=True, exist_ok=True)


# Input data
FW = {
2024: {"Jan": -4.32, "Jun": -1.54},
2025: {"Jan": -4.79, "Jun": -2.01},
}

MODEL = [
{"Station": "Kauur", "Year": 2025, "δ0": -4.57, "δi": -0.31, "δs": 0.17},
{"Station": "Kauur/Baboon Islands", "Year": 2024, "δ0": -4.32, "δi": 0.30, "δs": 0.17},
]


# Freshwater enrichment 
evap_rows = []
delta_evap = {}

for year, v in FW.items():
    d = v["Jun"] - v["Jan"]
    delta_evap[year] = d
    evap_rows.append({
        "Year": year,
        "δ18O_Jan_FW": v["Jan"],
        "δ18O_Jun_FW": v["Jun"],
        "Δδ18O_EVAP": d
    })

evap_df = pd.DataFrame(evap_rows)
evap_csv = OUT / "S7_freshwater_evap_enrichment.csv"
evap_df.to_csv(evap_csv, index=False)


# Mixing model 
mix_rows = []

for r in MODEL:

    d_total = r["δi"] - r["δ0"]
    d_mix = d_total - delta_evap[r["Year"]]
    gamma = d_mix / (r["δs"] - r["δ0"])

    mix_rows.append({
        "Station": r["Station"],
        "Year": r["Year"],
        "δ0": r["δ0"],
        "δi": r["δi"],
        "δs": r["δs"],
        "Δδ18OTOTAL": d_total,
        "Δδ18OMIXING": d_mix,
        "γISO": gamma
    })

mix_df = pd.DataFrame(mix_rows)

mix_csv = OUT / "Supplementary_Table_S8_d18O_linear_mixing_model.csv"
mix_df.to_csv(mix_csv, index=False)
