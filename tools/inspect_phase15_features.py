"""Phase 15 Dataset and Feature Inspection Script.
Analyzes:
1. Canada NCDB collision linkage and severity definitions.
2. UK STATS19 person/casualty and vehicle tables.
3. France BAAC usagers and vehicules tables.
4. Australia Victoria crash features.
5. Germany Unfallatlas features.
"""
import sys
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def inspect_canada():
    print("=" * 80)
    print("1. CANADA NCDB 2019 INSPECTION")
    print("=" * 80)
    canada_path = PROJECT_ROOT / "data" / "external" / "canada_ncdb" / "ncdb_2019.csv"
    if not canada_path.exists():
        print(f"File not found: {canada_path}")
        return
    
    df = pl.read_csv(canada_path, infer_schema_length=10000, ignore_errors=True)
    print(f"Total rows: {len(df):,}, Columns: {len(df.columns)}")
    print(f"Columns: {df.columns}")
    
    # Check potential collision identifiers
    # In NCDB, collision identifier is typically C_ID or C_CASE or similar, or C_YEAR + C_MNTH + C_WDAY + C_HOUR + C_SEV etc.
    id_cols = [c for c in df.columns if "ID" in c.upper() or "CASE" in c.upper() or "NO" in c.upper() or c.startswith("C_")]
    print(f"Potential collision/case columns: {id_cols}")
    
    # Check severity column (P_ISEV)
    if "P_ISEV" in df.columns:
        print("\nP_ISEV (Person Injury Severity) Distribution:")
        print(df["P_ISEV"].value_counts().sort("count", descending=True))
        
    if "C_SEV" in df.columns:
        print("\nC_SEV (Collision Severity) Distribution:")
        print(df["C_SEV"].value_counts().sort("count", descending=True))

    if "C_CASE" in df.columns:
        print(f"\nUnique C_CASE values: {df['C_CASE'].n_unique():,}")
        counts_per_case = df.group_by("C_CASE").len()
        print(f"Persons per collision (C_CASE) summary:")
        print(f"  Min: {counts_per_case['len'].min()}, Median: {counts_per_case['len'].median()}, Max: {counts_per_case['len'].max()}, Mean: {counts_per_case['len'].mean():.2f}")
    
    # Check other features in Canada NCDB: P_AGE, P_SEX, P_PSN, P_SAFE, P_USER, V_TYPE, V_YEAR
    person_features = [c for c in df.columns if c.startswith("P_") or c.startswith("V_")]
    print(f"\nPerson & Vehicle features in Canada NCDB: {person_features}")
    for col in ["P_AGE", "P_SEX", "P_SAFE", "P_USER", "V_TYPE"]:
        if col in df.columns:
            print(f"\n{col} unique values sample (top 10):")
            print(df[col].value_counts().head(10))

def inspect_uk_stats19():
    print("\n" + "=" * 80)
    print("2. UK STATS19 CASUALTY & VEHICLE TABLES INSPECTION")
    print("=" * 80)
    cas_path = PROJECT_ROOT / "data" / "external" / "uk_stats19" / "dft-road-casualty-statistics-casualty-2022.csv"
    veh_path = PROJECT_ROOT / "data" / "external" / "uk_stats19" / "dft-road-casualty-statistics-vehicle-2022.csv"
    col_path = PROJECT_ROOT / "data" / "external" / "uk_stats19" / "dft-road-casualty-statistics-collision-2022.csv"
    
    if cas_path.exists():
        cas_df = pl.read_csv(cas_path, infer_schema_length=5000, ignore_errors=True)
        print(f"Casualty table rows: {len(cas_df):,}, cols: {len(cas_df.columns)}")
        print(f"Casualty columns: {cas_df.columns}")
        # Look for age, sex, severity, casualty_type, safety equipment, pedestrian movement
        for col in ["casualty_severity", "casualty_class", "sex_of_casualty", "age_of_casualty", "age_band_of_casualty", "pedestrian_location", "pedestrian_movement"]:
            if col in cas_df.columns:
                print(f"  {col} sample counts:\n{cas_df[col].value_counts().head(5)}")
                
    if veh_path.exists():
        veh_df = pl.read_csv(veh_path, infer_schema_length=5000, ignore_errors=True)
        print(f"\nVehicle table rows: {len(veh_df):,}, cols: {len(veh_df.columns)}")
        print(f"Vehicle columns: {veh_df.columns}")
        for col in ["vehicle_type", "first_point_of_impact", "vehicle_manoeuvre", "junction_location", "engine_capacity_cc", "propulsion_code", "age_of_vehicle"]:
            if col in veh_df.columns:
                print(f"  {col} sample counts:\n{veh_df[col].value_counts().head(5)}")

def inspect_france_baac():
    print("\n" + "=" * 80)
    print("3. FRANCE BAAC USAGERS & VEHICULES TABLES INSPECTION")
    print("=" * 80)
    usagers_path = PROJECT_ROOT / "data" / "external" / "france_baac" / "usagers-2023.csv"
    veh_path = PROJECT_ROOT / "data" / "external" / "france_baac" / "vehicules-2023.csv"
    
    if usagers_path.exists():
        usg_df = pl.read_csv(usagers_path, separator=";", infer_schema_length=5000, ignore_errors=True)
        print(f"Usagers table rows: {len(usg_df):,}, cols: {len(usg_df.columns)}")
        print(f"Usagers columns: {usg_df.columns}")
        # grav: 1=Indemne, 2=Tué, 3=Blessé hospitalisé, 4=Blessé léger
        # secu1, secu2, secu3: equipement de securite (ceinture, casque)
        # an_nais: annee de naissance
        for col in ["grav", "sexe", "an_nais", "trajet", "secu1", "secu2", "secu3", "locp", "actp"]:
            if col in usg_df.columns:
                print(f"  {col} sample counts:\n{usg_df[col].value_counts().head(5)}")
                
    if veh_path.exists():
        veh_df = pl.read_csv(veh_path, separator=";", infer_schema_length=5000, ignore_errors=True)
        print(f"\nVehicules table rows: {len(veh_df):,}, cols: {len(veh_df.columns)}")
        print(f"Vehicules columns: {veh_df.columns}")
        for col in ["catv", "obs", "obsm", "choc", "manv", "motor"]:
            if col in veh_df.columns:
                print(f"  {col} sample counts:\n{veh_df[col].value_counts().head(5)}")

def inspect_australia_victoria():
    print("\n" + "=" * 80)
    print("4. AUSTRALIA VICTORIA ROAD CRASH DATA INSPECTION")
    print("=" * 80)
    vic_path = PROJECT_ROOT / "data" / "external" / "australia_victoria" / "victorian_road_crash_data.csv"
    if vic_path.exists():
        vic_df = pl.read_csv(vic_path, infer_schema_length=5000, ignore_errors=True)
        print(f"Australia Victoria rows: {len(vic_df):,}, cols: {len(vic_df.columns)}")
        print(f"Columns: {vic_df.columns}")
        # Check demographic or safety equipment columns
        demo_cols = [c for c in vic_df.columns if any(k in c.lower() for k in ["age", "sex", "speed", "belt", "helmet", "seat", "type", "dca", "event"])]
        print(f"Relevant potential features: {demo_cols}")
        for c in ["SEVERITY", "SPEED_ZONE", "DCA_CODE", "ACCIDENT_TYPE"]:
            if c in vic_df.columns:
                print(f"  {c} unique count: {vic_df[c].n_unique()}")

def inspect_germany_unfallatlas():
    print("\n" + "=" * 80)
    print("5. GERMANY UNFALLATLAS INSPECTION")
    print("=" * 80)
    de_path = PROJECT_ROOT / "data" / "external" / "germany_unfallatlas" / "Unfallorte2022_LinRef.csv"
    if de_path.exists():
        de_df = pl.read_csv(de_path, separator=";", infer_schema_length=5000, ignore_errors=True)
        print(f"Germany Unfallatlas rows: {len(de_df):,}, cols: {len(de_df.columns)}")
        print(f"Columns: {de_df.columns}")
        # UKATEGORIE, UART, UTYP1, IstRad, IstPKW, IstFuss, IstKrad, IstGkfz, IstSonstige, LINREFX, LINREFY, STRASSE
        for c in ["UKATEGORIE", "UART", "UTYP1", "IstRad", "IstPKW", "IstFuss", "IstKrad", "IstGkfz"]:
            if c in de_df.columns:
                print(f"  {c} unique values:\n{de_df[c].value_counts().head(5)}")

if __name__ == "__main__":
    inspect_canada()
    inspect_uk_stats19()
    inspect_france_baac()
    inspect_australia_victoria()
    inspect_germany_unfallatlas()
