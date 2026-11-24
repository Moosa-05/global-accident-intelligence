"""Phase 4: Global Data Ingestion and Canonical Dataset Pipeline
Processes UK STATS19, France BAAC, and GB Traffic Counts into the canonical schema.
Enriches with ERA5 weather and generates data quality reports.
"""
import sys
import time
from pathlib import Path
import polars as pl
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.adapters.uk_stats19_adapter import UKStats19Adapter
from backend.app.adapters.france_baac_adapter import FranceBAACAdapter
from backend.app.adapters.era5_weather_adapter import ERA5WeatherAdapter

def run_pipeline():
    print("=" * 70)
    print("PHASE 4: GLOBAL DATA EXPANSION PIPELINE")
    print("=" * 70)

    # 1. Ingest UK STATS19
    print("\n1. Ingesting UK STATS19 Road Safety Data (2022)...")
    t0 = time.time()
    uk_adapter = UKStats19Adapter()
    uk_df = uk_adapter.ingest()
    uk_time = time.time() - t0
    print(f"   -> Ingested {len(uk_df):,} UK collisions in {uk_time:.2f}s")
    print(f"   -> Columns: {len(uk_df.columns)}")
    print(f"   -> Traffic count join rate (within 5km): {(uk_df['annual_average_daily_flow'].is_not_null().sum() / len(uk_df)):.2%}")

    # 2. Ingest France BAAC
    print("\n2. Ingesting France BAAC Accident Data (2023)...")
    t0 = time.time()
    fr_adapter = FranceBAACAdapter()
    fr_df = fr_adapter.ingest()
    fr_time = time.time() - t0
    print(f"   -> Ingested {len(fr_df):,} France accidents in {fr_time:.2f}s")
    print(f"   -> Columns: {len(fr_df.columns)}")

    # 3. Concatenate into Global Canonical Dataset
    print("\n3. Concatenating into Global Canonical Accident Dataset...")
    # Ensure matching schemas
    common_cols = [col for col in uk_df.columns if col in fr_df.columns]
    global_df = pl.concat([uk_df.select(common_cols), fr_df.select(common_cols)])
    print(f"   -> Total Global Records: {len(global_df):,}")
    print(f"   -> Country Distribution:\n{global_df['country_code'].value_counts()}")

    # 4. Save to data/processed/global_canonical_accidents.parquet
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "global_canonical_accidents.parquet"
    global_df.write_parquet(out_path, compression="zstd")
    print(f"\n4. Saved Canonical Parquet to {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")

    # 5. Data Quality Statistics
    print("\n" + "=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)
    
    # Injury Severity Breakdown
    print("\nHuman Injury Severity Breakdown (By Country):")
    piv = global_df.group_by(["country_code", "human_injury_severity"]).agg(pl.len().alias("count")).sort(["country_code", "count"], descending=[False, True])
    print(piv)

    # Road Class Breakdown
    print("\nRoad Class Breakdown (By Country):")
    rc_piv = global_df.group_by(["country_code", "road_class"]).agg(pl.len().alias("count")).sort(["country_code", "count"], descending=[False, True])
    print(rc_piv)

    # Coordinate Validity
    valid_coords = global_df.filter(
        (pl.col("latitude") >= -90.0) & (pl.col("latitude") <= 90.0) &
        (pl.col("longitude") >= -180.0) & (pl.col("longitude") <= 180.0)
    )
    print(f"\nCoordinate Validity Rate: {len(valid_coords) / len(global_df):.2%} ({len(valid_coords):,} / {len(global_df):,})")

    # Missing Values
    print("\nMissing Value Rates:")
    for c in ["speed_limit_kmh", "annual_average_daily_flow", "road_surface_condition", "weather_condition"]:
        null_cnt = global_df[c].is_null().sum()
        print(f"  - {c}: {null_cnt:,} nulls ({null_cnt / len(global_df):.2%})")

    return global_df

if __name__ == "__main__":
    run_pipeline()
