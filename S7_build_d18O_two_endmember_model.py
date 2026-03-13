# =========================================================
# S7_build_d18O_two_endmember_model.py
#
# Implements Section 7.1 + 7.2 together:
#   (1) Computes Δδ18O_EVAP from freshwater endmembers
#   (2) Uses Δδ18O_EVAP in two-endmember mixing model
#
# Produces:
#   S7_freshwater_evap_enrichment.csv
#   Supplementary_Table_S8_d18O_linear_mixing_model.csv
# =========================================================

from pathlib import Path
import pandas as pd
import datetime as dt


OUT_ROOT = Path(
"/Volumes/Helena/DATASETS GAMBIA/Paper Isotopes bckup/"
"Supplementary/S7"
)

OUT_DIR = OUT_ROOT / "output"
DOC_DIR = OUT_ROOT / "documentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOC_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# SECTION 7.1 — Freshwater Endmember Shift
# =========================================================
FW = {
2024: {"Jan": -4.32, "Jun": -1.54},
2025: {"Jan": -4.79, "Jun": -2.01}
}

evap_records = []
DELTA_EVAP = {}

for y in FW:
    d = FW[y]["Jun"] - FW[y]["Jan"]
    DELTA_EVAP[y] = d
    evap_records.append({
        "Year": y,
        "δ18O_Jan_FW": FW[y]["Jan"],
        "δ18O_Jun_FW": FW[y]["Jun"],
        "Δδ18O_EVAP": d
    })

evap_df = pd.DataFrame(evap_records)
evap_df.to_csv(OUT_DIR/"S7_freshwater_evap_enrichment.csv",index=False)

print("\nΔδ18O_EVAP:")
print(evap_df)

# =========================================================
# SECTION 7.2 — Mixing Model (Table 8)
# =========================================================
MODEL = [
{"Station":"Kauur","Year":2025,"δ0":-4.57,"δi":-0.31,"δs":0.17},
{"Station":"Kauur/Baboon Islands","Year":2024,"δ0":-4.32,"δi":0.30,"δs":0.17}
]

mix_records = []

for r in MODEL:
    y = r["Year"]
    d_total = r["δi"] - r["δ0"]
    d_mix = d_total - DELTA_EVAP[y]
    gamma = d_mix / (r["δs"] - r["δ0"])

    mix_records.append({
        "Station": r["Station"],
        "Year": y,
        "δ0": r["δ0"],
        "δi": r["δi"],
        "δs": r["δs"],
        "Δδ18OTOTAL": d_total,
        "Δδ18OMIXING": d_mix,
        "γISO": gamma
    })

mix_df = pd.DataFrame(mix_records)
mix_df.to_csv(
OUT_DIR/"Supplementary_Table_S8_d18O_linear_mixing_model.csv",
index=False
)

print("\nTable 8:")
print(mix_df)

# =========================================================
timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
(Path(DOC_DIR/"PROVENANCE_S7.txt")
.write_text(f"Generated: {timestamp}\nΔEVAP={DELTA_EVAP}")
)
