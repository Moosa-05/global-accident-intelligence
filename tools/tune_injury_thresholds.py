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
test_df = df.iloc[shuffled[int(0.90 * n):]].copy()

# Fit Natural Balanced model
model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,
    class_weight="balanced",
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=100,
    random_state=42,
    verbose=-1
)
model.fit(train_df[features_no_coords], train_df["target"])
val_probs = model.predict_proba(val_df[features_no_coords])

# Threshold grid search on validation set
print("Searching optimal decision policy on validation probabilities...")
best_policy = None
best_bal_acc = -1.0

# Natural argmax
pred_default = np.argmax(val_probs, axis=1)
bal_acc_def = balanced_accuracy_score(val_df["target"], pred_default)
cr_def = classification_report(val_df["target"], pred_default, target_names=["SLIGHT", "SERIOUS", "FATAL"], output_dict=True)
print(f"Argmax Baseline: BalAcc: {bal_acc_def:.2%} | Slight: {cr_def['SLIGHT']['recall']:.1%} | Serious: {cr_def['SERIOUS']['recall']:.1%} | Fatal: {cr_def['FATAL']['recall']:.1%}")

# Cost-sensitive probability weighting: [w0*p0, w1*p1, w2*p2]
for w1 in np.linspace(1.0, 2.5, 7):
    for w2 in np.linspace(0.8, 1.5, 5):
        w_probs = val_probs * np.array([1.0, w1, w2])
        pred_w = np.argmax(w_probs, axis=1)
        cr = classification_report(val_df["target"], pred_w, target_names=["SLIGHT", "SERIOUS", "FATAL"], output_dict=True, zero_division=0)
        sli = cr["SLIGHT"]["recall"]
        ser = cr["SERIOUS"]["recall"]
        fat = cr["FATAL"]["recall"]
        bal = (sli + ser + fat) / 3.0
        # Require all 3 classes have meaningful recall (>30% each)
        if sli >= 0.50 and ser >= 0.35 and fat >= 0.55:
            if bal > best_bal_acc:
                best_bal_acc = bal
                best_policy = (w1, w2, bal, sli, ser, fat)

if best_policy:
    print(f"\nBest Balanced Policy: w_serious={best_policy[0]:.2f}, w_fatal={best_policy[1]:.2f}")
    print(f"  Val BalAcc: {best_policy[2]:.2%} | Slight: {best_policy[3]:.1%} | Serious: {best_policy[4]:.1%} | Fatal: {best_policy[5]:.1%}")
    
    # Evaluate on untouched test set!
    test_probs = model.predict_proba(test_df[features_no_coords])
    test_pred_def = np.argmax(test_probs, axis=1)
    test_pred_opt = np.argmax(test_probs * np.array([1.0, best_policy[0], best_policy[1]]), axis=1)
    
    print("\n--- TEST SET EVALUATION ---")
    print("Default Argmax:")
    print(classification_report(test_df["target"], test_pred_def, target_names=["SLIGHT", "SERIOUS", "FATAL"]))
    print(f"Default Test Balanced Accuracy: {balanced_accuracy_score(test_df['target'], test_pred_def):.2%}")
    
    print("\nOptimized Balanced Policy:")
    print(classification_report(test_df["target"], test_pred_opt, target_names=["SLIGHT", "SERIOUS", "FATAL"]))
    print(f"Optimized Test Balanced Accuracy: {balanced_accuracy_score(test_df['target'], test_pred_opt):.2%}")
else:
    print("No policy met the strict balanced coverage constraints.")
