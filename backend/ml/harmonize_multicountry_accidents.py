"""Harmonize Collision Datasets Across Four Countries:
UK (STATS19), France (BAAC), Germany (Destatis Unfallatlas), Australia (Victoria Crash Data).
Produces unified collision-level parquet: data/processed/global_canonical_accidents_4countries.parquet
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

print("=" * 80)
print("HARMONIZING MULTI-COUNTRY ACCIDENT DATASETS (UK, FR, DE, AU)")
print("=" * 80)

t0 = time.time()

# -------------------------------------------------------------
# 1. LOAD UK & FRANCE CANONICAL V2
# -------------------------------------------------------------
print("\n[1/4] Loading UK & France from canonical v2 parquet...")
df_uk_fr = pl.read_parquet(PROCESSED_DIR / "global_canonical_accidents_v2.parquet")
df_uk_fr = df_uk_fr.filter(pl.col("human_injury_severity").is_in(["SLIGHT", "SERIOUS", "FATAL"]))

# Prepare cyclical temporal features if not already present
df_uk_fr = df_uk_fr.with_columns([
    (pl.col("speed_limit_kmh") / 100.0).pow(2).alias("speed_limit_squared_proxy"),
    np.sin(2.0 * np.pi * pl.col("hour") / 24.0).alias("hour_sin"),
    np.cos(2.0 * np.pi * pl.col("hour") / 24.0).alias("hour_cos"),
    np.sin(2.0 * np.pi * (pl.col("month") - 1) / 12.0).alias("month_sin"),
    np.cos(2.0 * np.pi * (pl.col("month") - 1) / 12.0).alias("month_cos"),
    pl.col("divided_roadway").fill_null(False),
    pl.col("is_traffic_signal").fill_null(False),
    pl.col("is_crossing").fill_null(False),
    pl.col("is_roundabout").fill_null(False),
    pl.col("is_stop_sign").fill_null(False),
    pl.col("is_give_way").fill_null(False),
    pl.col("is_railway").fill_null(False),
    pl.lit(0).alias("has_pedestrian"),
    pl.lit(0).alias("has_bicycle"),
    pl.lit(0).alias("has_motorcycle"),
    pl.lit(0).alias("has_heavy_vehicle"),
    pl.lit(1).alias("has_car"),
])

print(f"  Loaded UK: {df_uk_fr.filter(pl.col('country_code') == 'GB').height:,} records")
print(f"  Loaded France: {df_uk_fr.filter(pl.col('country_code') == 'FR').height:,} records")

# -------------------------------------------------------------
# 2. INGEST & HARMONIZE GERMANY UNFALLATLAS 2022
# -------------------------------------------------------------
print("\n[2/4] Ingesting and harmonizing Germany Unfallatlas 2022...")
de_raw = pl.read_csv(
    EXTERNAL_DIR / "germany_unfallatlas" / "Unfallorte2022_LinRef.csv",
    separator=";", encoding="latin1", ignore_errors=True
)

# Target: 1=Fatal -> FATAL, 2=Serious -> SERIOUS, 3=Slight -> SLIGHT
de_sev = pl.when(pl.col("UKATEGORIE") == 1).then(pl.lit("FATAL")) \
           .when(pl.col("UKATEGORIE") == 2).then(pl.lit("SERIOUS")) \
           .otherwise(pl.lit("SLIGHT"))

# Light: 0=Daylight, 1=Twilight, 2=Darkness
de_light = pl.when(pl.col("ULICHTVERH") == 0).then(pl.lit("DAYLIGHT")) \
             .when(pl.col("ULICHTVERH") == 1).then(pl.lit("TWILIGHT_DAWN_DUSK")) \
             .otherwise(pl.lit("DARKNESS_STREETLIGHTS_ON"))

# Surface: 0=Trocken -> DRY, 1=Nass -> WET_DAMP, 2=Winterglatt -> FROST_ICE
de_surf = pl.when(pl.col("IstStrassenzustand") == 0).then(pl.lit("DRY")) \
            .when(pl.col("IstStrassenzustand") == 1).then(pl.lit("WET_DAMP")) \
            .otherwise(pl.lit("FROST_ICE"))

# Junction: Turning (UTYP1 in [2, 3]) or crossing (UART == 5)
de_is_junction = (pl.col("UTYP1").is_in([2, 3])) | (pl.col("UART") == 5)

# Inferred Road Class & Speed Limit
# UTYP1 == 1 (driving accident/loss of control on open road) -> TRUNK_HIGHWAY (100 km/h)
# Motorway: UART in [1, 2] and no pedestrians/bikes
# Urban: with pedestrians/bikes or turning conflicts -> PRIMARY_ARTERIAL or LOCAL (50 km/h)
de_road_class = pl.when((pl.col("UART").is_in([1, 2])) & (pl.col("IstRad") == 0) & (pl.col("IstFuss") == 0) & (pl.col("IstPKW") == 1)) \
                  .then(pl.lit("MOTORWAY")) \
                  .when((pl.col("UTYP1") == 1) & (pl.col("UART").is_in([8, 9]))) \
                  .then(pl.lit("TRUNK_HIGHWAY")) \
                  .when(pl.col("IstRad") == 1) \
                  .then(pl.lit("LOCAL_RESIDENTIAL")) \
                  .otherwise(pl.lit("PRIMARY_ARTERIAL"))

de_speed = pl.when(de_road_class == "MOTORWAY").then(pl.lit(120.0)) \
             .when(de_road_class == "TRUNK_HIGHWAY").then(pl.lit(100.0)) \
             .when(de_road_class == "LOCAL_RESIDENTIAL").then(pl.lit(30.0)) \
             .otherwise(pl.lit(50.0))

# Parse WGS84 coordinates from comma strings
de_lat = pl.col("YGCSWGS84").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False)
de_lon = pl.col("XGCSWGS84").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False)

# Temporal
de_hour = pl.col("USTUNDE").fill_null(12)
de_month = pl.col("UMONAT").fill_null(6)
# German UWOCHENTAG: 1=Sun, 2=Mon, 3=Tue, 4=Wed, 5=Thu, 6=Fri, 7=Sat
# ISO: 1=Mon .. 7=Sun
de_iso_wday = pl.when(pl.col("UWOCHENTAG") == 1).then(pl.lit(7)).otherwise(pl.col("UWOCHENTAG") - 1)
de_weekend = de_iso_wday.is_in([6, 7])
de_night = pl.col("ULICHTVERH") == 2

df_de_harmonized = de_raw.select([
    pl.col("OBJECTID").cast(pl.Utf8).alias("source_record_id"),
    pl.concat_str([pl.lit("DE_"), pl.col("OBJECTID").cast(pl.Utf8)]).alias("global_accident_id"),
    pl.lit("DE").alias("country_code"),
    pl.lit("Destatis Unfallatlas 2022").alias("source_dataset"),
    pl.lit("dl-de/by-2-0").alias("licence_type"),
    de_lat.alias("latitude"),
    de_lon.alias("longitude"),
    pl.lit(2022).alias("year"),
    de_month.alias("month"),
    de_hour.alias("hour"),
    de_iso_wday.alias("day_of_week"),
    de_weekend.alias("is_weekend"),
    de_night.alias("is_night"),
    de_road_class.alias("road_class"),
    de_speed.alias("speed_limit_kmh"),
    (de_speed / 100.0).pow(2).alias("speed_limit_squared_proxy"),
    de_is_junction.alias("is_junction"),
    pl.lit(False).alias("is_traffic_signal"),
    (pl.col("UTYP1") == 4).alias("is_crossing"),
    pl.lit(False).alias("is_roundabout"),
    pl.lit(False).alias("is_stop_sign"),
    pl.lit(False).alias("is_give_way"),
    pl.lit(False).alias("is_railway"),
    (de_road_class == "MOTORWAY").alias("divided_roadway"),
    pl.lit("CLEAR").alias("weather_condition"),
    de_light.alias("light_condition"),
    de_surf.alias("road_surface_condition"),
    pl.lit(12.0).alias("temperature_c"),
    pl.lit(70.0).alias("humidity_pct"),
    pl.lit(1013.25).alias("pressure_hpa"),
    pl.lit(15.0).alias("wind_speed_kmh"),
    pl.lit(0.0).alias("precipitation_mm"),
    pl.col("IstFuss").cast(pl.Int32).alias("has_pedestrian"),
    pl.col("IstRad").cast(pl.Int32).alias("has_bicycle"),
    pl.col("IstKrad").cast(pl.Int32).alias("has_motorcycle"),
    pl.col("IstGkfz").cast(pl.Int32).alias("has_heavy_vehicle"),
    pl.col("IstPKW").cast(pl.Int32).alias("has_car"),
    de_sev.alias("human_injury_severity")
]).filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())

print(f"  Harmonized Germany: {df_de_harmonized.height:,} records")

# -------------------------------------------------------------
# 3. INGEST & HARMONIZE AUSTRALIA VICTORIA CRASH DATA
# -------------------------------------------------------------
print("\n[3/4] Ingesting and harmonizing Australia Victoria Crash Data...")
aus_raw = pl.read_csv(
    EXTERNAL_DIR / "australia_victoria" / "victorian_road_crash_data.csv",
    ignore_errors=True
).filter(pl.col("SEVERITY") != "Non injury accident") # Exclude property-damage only

aus_sev = pl.when(pl.col("SEVERITY") == "Fatal accident").then(pl.lit("FATAL")) \
            .when(pl.col("SEVERITY") == "Serious injury accident").then(pl.lit("SERIOUS")) \
            .otherwise(pl.lit("SLIGHT"))

# Speed limit parser
aus_speed = pl.col("SPEED_ZONE").cast(pl.Utf8).str.extract(r"^(\d+)", 1).cast(pl.Float64, strict=False).fill_null(60.0)

# Road class from RMA
aus_road_class = pl.when(pl.col("RMA").cast(pl.Utf8).str.contains("Freeway")).then(pl.lit("MOTORWAY")) \
                   .when(pl.col("RMA").cast(pl.Utf8).str.contains("Highway")).then(pl.lit("TRUNK_HIGHWAY")) \
                   .when(pl.col("RMA").cast(pl.Utf8).str.contains("Arterial")).then(pl.lit("PRIMARY_ARTERIAL")) \
                   .otherwise(pl.lit("LOCAL_RESIDENTIAL"))

# Junction from ROAD_GEOMETRY
aus_is_junction = pl.col("ROAD_GEOMETRY").cast(pl.Utf8).str.contains("(?i)intersection")
aus_is_roundabout = pl.col("ROAD_GEOMETRY").cast(pl.Utf8).str.contains("(?i)roundabout")

# Divided
aus_divided = pl.col("DIVIDED").cast(pl.Utf8).str.to_lowercase() == "divided"

# Light condition
aus_light = pl.when(pl.col("LIGHT_CONDITION").cast(pl.Utf8).str.contains("(?i)Day")).then(pl.lit("DAYLIGHT")) \
              .when(pl.col("LIGHT_CONDITION").cast(pl.Utf8).str.contains("(?i)lights on")).then(pl.lit("DARKNESS_STREETLIGHTS_ON")) \
              .when(pl.col("LIGHT_CONDITION").cast(pl.Utf8).str.contains("(?i)no street lights")).then(pl.lit("DARKNESS_NO_STREETLIGHTS")) \
              .otherwise(pl.lit("TWILIGHT_DAWN_DUSK"))

aus_night = aus_light != "DAYLIGHT"

# Temporal
aus_date = pl.col("ACCIDENT_DATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
aus_year = aus_date.dt.year().fill_null(2018)
aus_month = aus_date.dt.month().fill_null(6)
aus_wday = aus_date.dt.weekday().fill_null(3) # 1=Mon, 7=Sun
aus_weekend = aus_wday.is_in([6, 7])
aus_hour = pl.col("ACCIDENT_TIME").cast(pl.Utf8).str.extract(r"^(\d+)", 1).cast(pl.Int32, strict=False).fill_null(12)

# Road users
aus_ped = (pl.col("PEDESTRIAN").fill_null(0) > 0).cast(pl.Int32)
aus_bike = (pl.col("BICYCLIST").fill_null(0) > 0).cast(pl.Int32)
aus_moto = (pl.col("MOTORCYCLE").fill_null(0) > 0).cast(pl.Int32)
aus_hgv = (pl.col("HEAVYVEHICLE").fill_null(0) > 0).cast(pl.Int32)
aus_car = (pl.col("PASSENGERVEHICLE").fill_null(0) > 0).cast(pl.Int32)

df_aus_harmonized = aus_raw.select([
    pl.col("ACCIDENT_NO").cast(pl.Utf8).alias("source_record_id"),
    pl.concat_str([pl.lit("AU_"), pl.col("ACCIDENT_NO").cast(pl.Utf8)]).alias("global_accident_id"),
    pl.lit("AU").alias("country_code"),
    pl.lit("Transport Victoria Crash Data").alias("source_dataset"),
    pl.lit("CC BY 4.0").alias("licence_type"),
    pl.col("LATITUDE").alias("latitude"),
    pl.col("LONGITUDE").alias("longitude"),
    aus_year.alias("year"),
    aus_month.alias("month"),
    aus_hour.alias("hour"),
    aus_wday.alias("day_of_week"),
    aus_weekend.alias("is_weekend"),
    aus_night.alias("is_night"),
    aus_road_class.alias("road_class"),
    aus_speed.alias("speed_limit_kmh"),
    (aus_speed / 100.0).pow(2).alias("speed_limit_squared_proxy"),
    aus_is_junction.alias("is_junction"),
    pl.lit(False).alias("is_traffic_signal"),
    pl.lit(False).alias("is_crossing"),
    aus_is_roundabout.alias("is_roundabout"),
    pl.lit(False).alias("is_stop_sign"),
    pl.lit(False).alias("is_give_way"),
    pl.lit(False).alias("is_railway"),
    aus_divided.alias("divided_roadway"),
    pl.lit("CLEAR").alias("weather_condition"),
    aus_light.alias("light_condition"),
    pl.lit("DRY").alias("road_surface_condition"),
    pl.lit(18.0).alias("temperature_c"),
    pl.lit(65.0).alias("humidity_pct"),
    pl.lit(1013.25).alias("pressure_hpa"),
    pl.lit(15.0).alias("wind_speed_kmh"),
    pl.lit(0.0).alias("precipitation_mm"),
    aus_ped.alias("has_pedestrian"),
    aus_bike.alias("has_bicycle"),
    aus_moto.alias("has_motorcycle"),
    aus_hgv.alias("has_heavy_vehicle"),
    aus_car.alias("has_car"),
    aus_sev.alias("human_injury_severity")
]).filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())

print(f"  Harmonized Australia: {df_aus_harmonized.height:,} records")

# -------------------------------------------------------------
# 4. CONCATENATE ALL FOUR NATIONS INTO UNIFIED CANONICAL PARQUET
# -------------------------------------------------------------
print("\n[4/4] Merging all four nations into unified canonical dataset...")
common_cols = [c for c in df_uk_fr.columns if c in df_de_harmonized.columns and c in df_aus_harmonized.columns]

# Harmonize dtypes across all 3 dataframes
def harmonize_dtypes(df):
    cast_exprs = []
    for col, dtype in df.schema.items():
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            cast_exprs.append(pl.col(col).cast(pl.Int64))
        elif dtype == pl.Float32:
            cast_exprs.append(pl.col(col).cast(pl.Float64))
    if cast_exprs:
        df = df.with_columns(cast_exprs)
    return df.with_columns([
        np.sin(2.0 * np.pi * pl.col("hour") / 24.0).alias("hour_sin"),
        np.cos(2.0 * np.pi * pl.col("hour") / 24.0).alias("hour_cos"),
        np.sin(2.0 * np.pi * (pl.col("month") - 1) / 12.0).alias("month_sin"),
        np.cos(2.0 * np.pi * (pl.col("month") - 1) / 12.0).alias("month_cos"),
    ])

sub_uk_fr = harmonize_dtypes(df_uk_fr.select(common_cols))
sub_de = harmonize_dtypes(df_de_harmonized.select(common_cols))
sub_aus = harmonize_dtypes(df_aus_harmonized.select(common_cols))

full_4countries = pl.concat([sub_uk_fr, sub_de, sub_aus])

# Target encoding
target_expr = pl.when(pl.col("human_injury_severity") == "FATAL").then(pl.lit(2)) \
                .when(pl.col("human_injury_severity") == "SERIOUS").then(pl.lit(1)) \
                .otherwise(pl.lit(0))
full_4countries = full_4countries.with_columns(target_expr.alias("target"))

out_parquet = PROCESSED_DIR / "global_canonical_accidents_4countries.parquet"
full_4countries.write_parquet(out_parquet, compression="zstd")

print(f"\nSaved Unified 4-Country Dataset: {out_parquet}")
print(f"Total Harmonized Collision Records: {full_4countries.height:,}")
print(f"File Size: {out_parquet.stat().st_size / (1024*1024):.2f} MB")
print("\nCountry Distribution:")
print(full_4countries.group_by("country_code").len().sort("len", descending=True))
print("\nCross-Country Severity Distribution:")
print(full_4countries.group_by(["country_code", "human_injury_severity"]).len().sort(["country_code", "human_injury_severity"]))
print(f"\nCompleted Harmonization in {time.time() - t0:.2f}s!")
