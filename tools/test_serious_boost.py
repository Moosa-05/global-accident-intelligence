import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
import lightgbm as lgb
from pathlib import Path

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
for c in categorical_cols:
    df[c] = df[c].fillna("UNKNOWN").astype("category")

target_map = {"SLIGHT": 0, "SERIOUS": 1, "FATAL": 2}
df["target"] = df["human_injury_severity"].map(target_map)

np.random.seed(42)
shuffled = np.random.permutation(len(df))
n = len(df)
train_df = df.iloc[shuffled[:int(0.80 * n)]].copy()
val_df = df.iloc[shuffled[int(0.80 * n):int(0.90 * n)]].copy()

# Natural balanced weights: 0: 0.462, 1: 1.343, 2: 11.155
# Let's test boosting Serious relative to Slight while keeping Fatal high:
weight_candidates = [
    ("Natural Balanced", {0: 0.46, 1: 1.34, 2: 11.16}),
    ("Boost Serious 1.8", {0: 0.40, 1: 1.80, 2: 12.00}),
    ("Boost Serious 2.2", {0: 0.35, 1: 2.20, 2: 13.00}),
    ("Boost Serious 2.5", {0: 0.30, 1: 2.50, 2: 14.00}),
    ("Boost Serious 3.0", {0: 0.25, 1: 3.00, 2: 15.00}),
]

print("--- Testing Targeted Serious-Class Boosting on Validation Set ---")
for name, cw in weight_candidates:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight=cw,
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=100,
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
    print(f"{name:20s} | Macro-F1: {macro_f1:.4f} | BalAcc: {bal_acc:.2%} | Slight: {sli_rec:5.1%} | Serious: {ser_rec:5.1%} | Fatal: {fat_rec:5.1%}")
