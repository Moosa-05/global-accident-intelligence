"""Deep Inspection and Validation of Acquired International Road Safety Datasets.
Analyzes Germany Unfallatlas, Australia Victoria Crash Data, and Canada NCDB.
"""
import sys
import polars as pl
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

print("=" * 80)
print("INSPECTION OF ACQUIRED INTERNATIONAL DATASETS")
print("=" * 80)

# -------------------------------------------------------------
# 1. GERMANY UNFALLATLAS 2022
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("1. GERMANY: Destatis Unfallatlas 2022 (dl-de/by-2-0)")
print("=" * 60)
de_csv = EXTERNAL_DIR / "germany_unfallatlas" / "Unfallorte2022_LinRef.csv"
if de_csv.exists():
    de_df = pl.read_csv(de_csv, separator=";", encoding="latin1", ignore_errors=True)
    print(f"  Total German Collision Records: {len(de_df):,}")
    print(f"  Columns ({len(de_df.columns)}): {de_df.columns}")
    
    # Severity breakdown: UKATEGORIE (1: Getoetete / Fatal, 2: Schwerverletzte / Serious, 3: Leichtverletzte / Slight)
    sev_de = de_df.group_by("UKATEGORIE").len().sort("UKATEGORIE")
    print("  Severity Distribution (UKATEGORIE):")
    sev_labels = {1: "FATAL (Getötete)", 2: "SERIOUS (Schwerverletzte)", 3: "SLIGHT (Leichtverletzte)"}
    for row in sev_de.iter_rows():
        print(f"    Code {row[0]} - {sev_labels.get(row[0], 'Unknown')}: {row[1]:,} ({row[1]/len(de_df):.2%})")

    # Light condition (ULICHTVERH: 0=Daylight, 1=Twilight, 2=Darkness)
    print("  Light Conditions (ULICHTVERH):")
    light_de = de_df.group_by("ULICHTVERH").len().sort("ULICHTVERH")
    for row in light_de.iter_rows():
        print(f"    Code {row[0]}: {row[1]:,}")

    # Road surface (IstStrassenzustand: 0=Dry, 1=Wet, 2=Winter/Ice)
    print("  Road Surface Condition (IstStrassenzustand):")
    surf_de = de_df.group_by("IstStrassenzustand").len().sort("IstStrassenzustand")
    for row in surf_de.iter_rows():
        print(f"    Code {row[0]}: {row[1]:,}")

    # Road users involved
    pkw_count = len(de_df.filter(pl.col("IstPKW") == 1))
    rad_count = len(de_df.filter(pl.col("IstRad") == 1))
    fuss_count = len(de_df.filter(pl.col("IstFuss") == 1))
    krad_count = len(de_df.filter(pl.col("IstKrad") == 1))
    gkfz_count = len(de_df.filter(pl.col("IstGkfz") == 1))
    print("  Involved Road Users:")
    print(f"    Passenger Cars (IstPKW):   {pkw_count:,} ({pkw_count/len(de_df):.1%})")
    print(f"    Bicycles (IstRad):         {rad_count:,} ({rad_count/len(de_df):.1%})")
    print(f"    Pedestrians (IstFuss):     {fuss_count:,} ({fuss_count/len(de_df):.1%})")
    print(f"    Motorcycles (IstKrad):     {krad_count:,} ({krad_count/len(de_df):.1%})")
    print(f"    Heavy Trucks (IstGkfz):    {gkfz_count:,} ({gkfz_count/len(de_df):.1%})")

    # Coordinates
    null_coords = len(de_df.filter(pl.col("XGCSWGS84").is_null() | pl.col("YGCSWGS84").is_null()))
    print(f"  Null Coordinates: {null_coords:,} ({null_coords/len(de_df):.2%})")

# -------------------------------------------------------------
# 2. AUSTRALIA VICTORIA ROAD CRASH DATA
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("2. AUSTRALIA: Transport Victoria Road Crash Data (CC BY 4.0)")
print("=" * 60)
aus_csv = EXTERNAL_DIR / "australia_victoria" / "victorian_road_crash_data.csv"
if aus_csv.exists():
    aus_df = pl.read_csv(aus_csv, ignore_errors=True)
    print(f"  Total Victorian Collision Records: {len(aus_df):,}")
    print(f"  Columns ({len(aus_df.columns)}): {aus_df.columns[:15]} ...")
    
    # Severity breakdown
    sev_aus = aus_df.group_by("SEVERITY").len().sort("len", descending=True)
    print("  Severity Distribution (SEVERITY):")
    for row in sev_aus.iter_rows():
        print(f"    {row[0]}: {row[1]:,} ({row[1]/len(aus_df):.2%})")

    # Casualties sum
    tot_fatal = aus_df["FATALITY"].sum()
    tot_serious = aus_df["SERIOUSINJURY"].sum()
    tot_other = aus_df["OTHERINJURY"].sum()
    tot_noninj = aus_df["NONINJURED"].sum()
    print(f"  Casualty Aggregates: Fatalities={tot_fatal:,}, Serious Injuries={tot_serious:,}, Other Injuries={tot_other:,}, Non-injured={tot_noninj:,}")

    # Road hierarchy (RMA)
    rma_dist = aus_df.group_by("RMA").len().sort("len", descending=True)
    print("  Road Hierarchy (RMA):")
    for row in rma_dist.iter_rows():
        print(f"    {row[0]}: {row[1]:,}")

    # Speed limits
    spd_dist = aus_df.group_by("SPEED_ZONE").len().sort("len", descending=True)
    print("  Speed Zones:")
    for row in spd_dist.head(6).iter_rows():
        print(f"    {row[0]}: {row[1]:,}")

    # Coordinates
    null_coords_aus = len(aus_df.filter(pl.col("LATITUDE").is_null() | pl.col("LONGITUDE").is_null()))
    print(f"  Null Coordinates: {null_coords_aus:,} ({null_coords_aus/len(aus_df):.2%})")

# -------------------------------------------------------------
# 3. CANADA NATIONAL COLLISION DATABASE (NCDB)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("3. CANADA: Transport Canada NCDB 2019 (OGL-Canada)")
print("=" * 60)
can_csv = EXTERNAL_DIR / "canada_ncdb" / "ncdb_2019.csv"
if can_csv.exists():
    can_df = pl.read_csv(can_csv, ignore_errors=True)
    print(f"  Total Canadian Person-Level Collision Records: {len(can_df):,}")
    print(f"  Columns ({len(can_df.columns)}): {can_df.columns}")
    
    # Severity breakdown: C_SEV (1 = Fatal, 2 = Non-fatal injury)
    sev_can = can_df.group_by("C_SEV").len().sort("C_SEV")
    print("  Collision Severity (C_SEV):")
    for row in sev_can.iter_rows():
        label = "FATAL COLLISION" if str(row[0]) == "1" else "NON-FATAL INJURY COLLISION"
        print(f"    Code {row[0]} ({label}): {row[1]:,} ({row[1]/len(can_df):.2%})")

    # Person Injury Severity: P_ISEV (1: No Injury, 2: Injury, 3: Fatality)
    p_sev = can_df.group_by("P_ISEV").len().sort("len", descending=True)
    print("  Person Injury Severity (P_ISEV):")
    p_labels = {"1": "No Injury", "2": "Injury", "3": "Fatality", "N": "Unknown/Not Stated", "U": "Unknown"}
    for row in p_sev.head(6).iter_rows():
        print(f"    Code {row[0]} ({p_labels.get(str(row[0]), 'Other')}): {row[1]:,}")

    # Weather
    wthr_dist = can_df.group_by("C_WTHR").len().sort("len", descending=True)
    print("  Weather Categories (C_WTHR):")
    for row in wthr_dist.head(6).iter_rows():
        print(f"    Code {row[0]}: {row[1]:,}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
