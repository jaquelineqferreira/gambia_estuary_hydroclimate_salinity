from pathlib import Path
import pandas as pd
import datetime as dt


# Paths
ROOT = Path(
    "/Volumes/Helena/DATASETS GAMBIA/Paper Isotopes bckup/"
    "Supplementary/S8/S8_salt_mass_balance"
)

OUT = ROOT / "output"
DOC = ROOT / "documentation"

OUT.mkdir(parents=True, exist_ok=True)
DOC.mkdir(parents=True, exist_ok=True)


# Input data
SF = 0.00
SS = 36.90

ROWS = [
    {"Station": "Kauur", "Year": 2025, "ΔSAL": 10.54},
    {"Station": "Kauur", "Year": 2024, "ΔSAL": 13.90},
]

# Model
def gamma_sal(delta_sal: float, sf: float, ss: float) -> float:
    return float(delta_sal / (ss - sf)) if (ss - sf) != 0 else float("nan")


records = []
for r in ROWS:
    records.append({
        "Station": r["Station"],
        "Year": int(r["Year"]),
        "ΔSAL": float(r["ΔSAL"]),
        "Sf": SF,
        "Ss": SS,
        "γSAL": gamma_sal(r["ΔSAL"], SF, SS),
    })

df = pd.DataFrame(records).sort_values("Year").reset_index(drop=True)

out_csv = OUT / "Supplementary_Table_S10_salt_mass_balance.csv"
df.to_csv(out_csv, index=False)