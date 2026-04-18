"""
enrich_taxi_zones.py
--------------------
Joins taxi_zone_lookup.csv with borough_census_data.csv to produce
an enriched zone lookup that can be merged with trip data in Jupyter.

Inputs (same folder or adjust paths below):
  - taxi_zone_lookup.csv
  - borough_census_data.csv   <-- edit values from census.gov before running

Output:
  - taxi_zone_lookup_enriched.csv
"""

import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# BASE_DIR is the folder where this script lives, so paths work on any OS
# regardless of where you run it from.
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

ZONE_LOOKUP   = DATA_DIR / "taxi_zone_lookup.csv"
CENSUS_DATA   = DATA_DIR / "borough_census_data.csv"
OUTPUT        = DATA_DIR / "taxi_zone_lookup_enriched.csv"

# ── Load ───────────────────────────────────────────────────────────────────────
zones  = pd.read_csv(ZONE_LOOKUP)
census = pd.read_csv(CENSUS_DATA)

# ── Join on Borough ────────────────────────────────────────────────────────────
enriched = zones.merge(census, on="Borough", how="left")

# ── Save ───────────────────────────────────────────────────────────────────────
enriched.to_csv(OUTPUT, index=False)
print(f"Saved {len(enriched)} rows → {OUTPUT}")
print(enriched.head())


# ══════════════════════════════════════════════════════════════════════════════
# HOW TO JOIN WITH TRIP DATA IN JUPYTER
# ══════════════════════════════════════════════════════════════════════════════
#
# trips   = pd.read_parquet("yellow_tripdata_XXXX-XX.parquet")   # or .csv
# lookup  = pd.read_csv("taxi_zone_lookup_enriched.csv")
#
# # Rename lookup columns before each join so PU and DO don't collide
# pu_lookup = lookup.add_prefix("PU_").rename(columns={"PU_LocationID": "PULocationID"})
# do_lookup = lookup.add_prefix("DO_").rename(columns={"DO_LocationID": "DOLocationID"})
#
# trips = trips.merge(pu_lookup, on="PULocationID", how="left")
# trips = trips.merge(do_lookup, on="DOLocationID", how="left")
#
# Result columns for PU side will be:
#   PU_Borough, PU_Zone, PU_service_zone,
#   PU_Population, PU_MedianHouseholdIncome, PU_HousingUnits,
#   PU_Transport_CarAlone_pct, PU_Transport_Carpool_pct,
#   PU_Transport_PublicTransit_