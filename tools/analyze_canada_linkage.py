"""Deep analysis of Canada NCDB 2019 dataset:
Collision linkage, duplicate check, severity distribution, and feature availability.
"""
import polars as pl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
canada_path = PROJECT_ROOT / "data" / "external" / "canada_ncdb" / "ncdb_2019.csv"

df = pl.read_csv(canada_path, infer_schema_length=10000, ignore_errors=True)
print(f"Total rows in Canada NCDB 2019: {len(df):,}")

# Check missing C_CASE
null_cases = df.filter(pl.col("C_CASE").is_null()).height
print(f"Null or missing C_CASE: {null_cases}")

# Unique C_CASE
n_unique_cases = df["C_CASE"].n_unique()
print(f"Unique C_CASE collisions: {n_unique_cases:,}")

# Check for duplicate person records (C_CASE + V_ID + P_ID)
if "V_ID" in df.columns and "P_ID" in df.columns:
    dup_persons = df.group_by(["C_CASE", "V_ID", "P_ID"]).len().filter(pl.col("len") > 1)
    print(f"Duplicate (C_CASE, V_ID, P_ID) groups: {len(dup_persons):,}")

# Distribution of C_SEV (Collision-level severity reported by police)
# 1 = At least one fatality, 2 = Non-fatal (injury or property damage only)
c_sev_dist = df.group_by("C_SEV").len().sort("len", descending=True)
print(f"\nReported Collision Severity (C_SEV) in person rows:")
print(c_sev_dist)

# Group by C_CASE to get actual unique collisions
collision_summary = df.group_by("C_CASE").agg([
    pl.col("C_SEV").first().alias("c_sev_first"),
    pl.col("P_ISEV").alias("p_isev_list"),
    pl.len().alias("person_count"),
    pl.col("V_ID").n_unique().alias("vehicle_count"),
    pl.col("C_CONF").first().alias("c_conf"),
    pl.col("C_RCFG").first().alias("c_rcfg"),
    pl.col("C_WTHR").first().alias("c_wthr"),
    pl.col("C_RSUR").first().alias("c_rsur"),
    pl.col("C_TRAF").first().alias("c_traf")
])

print(f"\nUnique Collisions aggregated: {len(collision_summary):,}")

# Check C_SEV across unique collisions
print("\nUnique collisions by C_SEV:")
print(collision_summary["c_sev_first"].value_counts())

# Check maximum P_ISEV per collision
# P_ISEV codes in NCDB:
# 1: No Injury
# 2: Injury
# 3: Fatality
# N: Not Applicable
# U: Unknown / Not reported
def compute_max_sev(p_list):
    has_fatal = any(x == "3" for x in p_list)
    if has_fatal:
        return "FATAL"
    has_inj = any(x == "2" for x in p_list)
    if has_inj:
        return "INJURY_NONFATAL"
    has_no_inj = any(x == "1" for x in p_list)
    if has_no_inj:
        return "NO_INJURY"
    return "UNKNOWN"

pdf_col = collision_summary.to_pandas()
pdf_col["max_p_isev"] = pdf_col["p_isev_list"].apply(compute_max_sev)
print("\nUnique collisions by aggregated Maximum Person Injury Severity (P_ISEV):")
print(pdf_col["max_p_isev"].value_counts())

# Cross-tabulate C_SEV vs max_p_isev
import pandas as pd
print("\nC_SEV vs Max P_ISEV cross-tabulation:")
print(pd.crosstab(pdf_col["c_sev_first"], pdf_col["max_p_isev"], margins=True))
