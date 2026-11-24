"""Optimize Human Injury Severity minority class performance (especially SERIOUS recall)
and compare pooled vs coordinate-free vs country-calibrated models.
"""
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
import lightgbm as lgb
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

parquet_path = PROJECT_ROOT / "data" / "processed" / "global_canonical_accidents_v2.parquet"
df = pl.read_parquet(parquet_path).to_pandas()

physics_cols = [
    "speed_limit_kmh", "is_junction", "is_traffic_signal", "is_crossing",
    "is_roundabout", "is_stop_sign", "is_give_way", "is_railway",
    "is_weekend", "is_night", "divided_roadway",
    "temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_kmh", "precipitation_mm"
]
categorical_cols = ["road_class", "weather_condition", "light_condition", "road_surface_condition"]

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
temporal_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week"]

features_no_coords = physics_cols + temporal_cols + categorical_cols
features_with_coords = ["latitude", "longitude"] + features_no_coords

for c in categorical_cols:
    df[c] = df[c].fillna("UNKNOWN").astype("category")

target_map = {"SLIGHT": 0, "SERIOUS": 1, "FATAL": 2}
inv_map = {0: "SLIGHT", 1: "SERIOUS", 2: "FATAL"}
df["target"] = df["human_injury_severity"].map(target_map)

# Split 80/10/10 train/val/test by chronological or stratified random
np.random.seed(42)
shuffled = np.random.permutation(len(df))
n = len(df)
train_idx = shuffled[:int(0.80 * n)]
val_idx = shuffled[int(0.80 * n):int(0.90 * n)]
test_idx = shuffled[int(0.90 * n):]

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()
test_df = df.iloc[test_idx].copy()

print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

# Test different class weight configurations on Validation set
weight_configs = [
    ("Default Balanced", "balanced"),
    ("Equal Weights", None),
    ("Custom Tuned 1", {0: 1.0, 1: 2.2, 2: 7.0}),
    ("Custom Tuned 2 (Serious-Boost)", {0: 1.0, 1: 2.8, 2: 8.5}),
    ("Custom Tuned 3 (High Serious)", {0: 0.8, 1: 3.2, 2: 9.0}),
]

print("\n--- Tuning Class Weights on Validation Set (Coordinate-Free Model) ---")
best_cfg = None
best_score = -1.0

for name, cw in weight_configs:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight=cw,
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=120,
        random_state=42,
        verbose=-1
    )
    model.fit(train_df[features_no_coords], train_df["target"])
    pred_val = model.predict(val_df[features_no_coords])
    
    macro_f1 = f1_score(val_df["target"], pred_val, average="macro")
    bal_acc = balanced_accuracy_score(val_df["target"], pred_val)
    cr = classification_report(val_df["target"], pred_val, target_names=["SLIGHT", "SERIOUS", "FATAL"], output_dict=True)
    ser_rec = cr["SERIOUS"]["recall"]
    fat_rec = cr["FATAL"]["recall"]
    sli_rec = cr["SLIGHT"]["recall"]
    
    print(f"{name:32s} | Macro-F1: {macro_f1:.4f} | BalAcc: {bal_acc:.2%} | Slight: {sli_rec:.1%} | Serious: {ser_rec:.1%} | Fatal: {fat_rec:.1%}")
    
    # Combined score prioritizing balanced coverage across all 3 classes
    min_rec = min(sli_rec, ser_rec, fat_rec)
    score = macro_f1 + 0.5 * min_rec
    if score > best_score:
        best_score = score
        best_cfg = (name, cw, model)

print(f"\nBest Configuration Selected: {best_cfg[0]}")
