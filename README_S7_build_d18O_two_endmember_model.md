Compute evaporation enrichment and a δ18O two-endmember mixing model, details in Supplementary Section S7 and main paper.

Rationale:
- Calculates freshwater evaporation enrichment between January and June.
- Applies a two-endmember linear mixing model correcting for evaporation.

Outputs:
output/S7_freshwater_evap_enrichment.csv
output/Supplementary_Table_S8_d18O_linear_mixing_model.csv

Methods:
1. Freshwater evaporation enrichment:
Δδ18O_EVAP = δ18O_Jun_FW − δ18O_Jan_FW

3. Two-endmember mixing model: 
Δδ18O_TOTAL = δi − δ0
Δδ18O_MIXING = Δδ18O_TOTAL − Δδ18O_EVAP
γISO = Δδ18O_MIXING / (δs − δ0)

where:

- `δ0` = upstream freshwater endmember
- `δi` = measured isotope value at dry season end
- `δs` = seawater endmember

Requirements:
pandas
