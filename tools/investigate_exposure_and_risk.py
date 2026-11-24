"""Accident-Risk Exposure & Non-Accident Sampling Investigation
Examines how legitimate non-accident exposure observations can be derived
without making the invalid assumption that unrecorded accidents mean confirmed non-accidents.
"""
import sys
from pathlib import Path
import polars as pl
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def investigate_exposure():
    print("=" * 70)
    print("INVESTIGATION: NON-ACCIDENT EXPOSURE & EMPIRICAL CRASH RATES")
    print("=" * 70)

    aadf_path = PROJECT_ROOT / "data" / "external" / "gb_traffic" / "dft_traffic_counts_aadf.csv"
    accidents_path = PROJECT_ROOT / "data" / "processed" / "global_canonical_accidents.parquet"

    # 1. Load GB Traffic Counts for 2022
    traffic_df = pl.read_csv(
        aadf_path,
        columns=["count_point_id", "year", "road_name", "road_category", "road_type",
                 "latitude", "longitude", "link_length_km", "all_motor_vehicles"],
        ignore_errors=True
    ).filter(pl.col("year") == 2022).filter(pl.col("latitude").is_not_null())

    print(f"Monitored GB Traffic Count Points in 2022: {len(traffic_df):,}")

    # 2. Load Processed UK Accidents
    acc_df = pl.read_parquet(accidents_path).filter(pl.col("country_code") == "GB")
    print(f"Recorded UK Collisions in 2022: {len(acc_df):,}")

    # 3. Compute Annual Vehicle-Kilometres Travelled (VKT) per Road Category
    # Annual VKT = AADF * 365 * link_length_km
    traffic_with_vkt = traffic_df.with_columns(
        (pl.col("all_motor_vehicles") * 365.0 * pl.col("link_length_km")).alias("annual_vkt")
    )

    cat_exposure = traffic_with_vkt.group_by("road_category").agg([
        pl.len().alias("num_count_points"),
        pl.col("all_motor_vehicles").mean().alias("mean_aadf"),
        (pl.col("annual_vkt").sum() / 1e6).alias("total_million_vkt")
    ]).sort("total_million_vkt", descending=True)

    print("\nGB Traffic Exposure by Road Category (2022):")
    print(cat_exposure)

    # 4. Formulate the Non-Accident Case-Control Methodology
    print("\n" + "-" * 70)
    print("METHODOLOGY: LEGITIMATE NON-ACCIDENT EXPOSURE SAMPLING")
    print("-" * 70)
    print("""
Key Principle: 'Absence of accident record != Confirmed safe observation'
Why naive negative generation fails:
  - If an unmonitored road has zero accidents recorded, it does not mean the road has low risk.
  - It usually means traffic exposure was near zero, or minor damage-only collisions were not reported.

Valid Scientific Approach for Accident-Risk Modeling:
  1. Monitored Road Link Sampling:
     - Use only road links with verified traffic counting stations (e.g. GB AADF count points).
     - Each monitored link has 8,760 hours/year of active traffic exposure.
     - An incident on that link occupies exactly 1 hour.
     - The remaining 8,759 hours represent verified NON-ACCIDENT EXPOSURE under known traffic volume and weather.
  
  2. Matched Case-Control Design:
     - For each collision event (Case: y = 1) at time (t) on link (L):
     - Sample 4 to 10 control observations (Controls: y = 0) on the EXACT same link (L)
       during non-accident periods (e.g., same day of week, adjacent weeks, or random active hours).
     - Controls carry genuine non-accident exposure: known road geometry, known traffic flow, known weather.
  
  3. Poisson / Negative Binomial Rate Modeling:
     - Model crash frequency as:
         E[Y_i] = Exposure_i * exp(beta * X_i)
         where Exposure_i = Annual VKT = AADF_i * 365 * Link_Length_i
     - This directly models collision rate per million vehicle-km without fabricating binary labels.
""")

if __name__ == "__main__":
    investigate_exposure()
