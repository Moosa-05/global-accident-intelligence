"""Feature Ablation and Permutation Importance Study for Phase 15.
Evaluates:
1. Baseline transferable model.
2. Ablation: Without speed limits.
3. Ablation: Without weather.
4. Ablation: Without road users / vehicle categories.
5. Ablation: Without geometry / junctions.
6. Permutation importance on validation set.
"""
import polars as pl
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.inspection import permutation_importance
import joblib
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "global_canonical_accidents_4countries.parquet"
PIPE_PATH = PROJECT_ROOT / "models" / "injury_feature_pipeline_v2.pkl"
CAND_PATH = PROJECT_ROOT / "models" / "human_injury_severity_global_v2_candidate.pkl"

df_pl = pl.read_parquet(PARQUET_PATH)
pdf = df_pl.to_pandas()

FEATURE_COLS_NUM = [
    "speed_limit_kmh", "speed_limit_squared_proxy", "hour_sin", "hour_cos",
    "month_sin", "month_cos", "day_of_week", "temperature_c", "humidity_pct",
    "pressure_hpa", "wind_speed_kmh", "precipitation_mm"
]

FEATURE_COLS_BOOL = [
    "is_junction", "is_traffic_signal", "is_crossing", "is_roundabout",
    "is_stop_sign", "is_give_way", "is_railway", "is_weekend", "is_night",
    "divided_roadway", "has_pedestrian", "has_bicycle", "has_motorcycle",
    "has_heavy_vehicle", "has_car"
]

FEATURE_COLS_CAT = ["road_class", "weather_condition", "light_condition", "road_surface_condition"]
ALL_TRANSFERABLE_FEATURES = FEATURE_COLS_NUM + FEATURE_COLS_BOOL + FEATURE_COLS_CAT

for c in FEATURE_COLS_BOOL:
    pdf[c] = pdf[c].fillna(False).astype(float)

indices = np.arange(len(pdf))
y_all = pdf["target"].values
strata = pdf["country_code"] + "_" + pdf["target"].astype(str)

idx_tr, idx_temp, y_tr, y_temp = train_test_split(indices, y_all, test_size=0.20, random_state=42, stratify=strata)
strata_temp = strata.iloc[idx_temp]
idx_va, idx_te, y_va, y_te = train_test_split(idx_temp, y_temp, test_size=0.50, random_state=42, stratify=strata_temp)

v2_pipe = joblib.load(PIPE_PATH)
v2_cand = joblib.load(CAND_PATH)

X_va = v2_pipe.transform(pdf.iloc[idx_va][ALL_TRANSFERABLE_FEATURES])
X_te = v2_pipe.transform(pdf.iloc[idx_te][ALL_TRANSFERABLE_FEATURES])

print("=" * 80)
print("1. PERMUTATION IMPORTANCE ON VALIDATION SET (SAMPLE n = 10,000)")
print("=" * 80)

cat_encoder = v2_pipe.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = list(cat_encoder.get_feature_names_out(FEATURE_COLS_CAT))
all_feature_names = FEATURE_COLS_NUM + FEATURE_COLS_BOOL + cat_feature_names

np.random.seed(42)
perm_idx = np.random.choice(len(X_va), size=10000, replace=False)
X_perm = X_va[perm_idx]
y_perm = y_va[perm_idx]

perm_res = permutation_importance(
    v2_cand, X_perm, y_perm,
    scoring="f1_macro",
    n_repeats=3,
    random_state=42,
    n_jobs=-1
)

sorted_importances_idx = perm_res.importances_mean.argsort()[::-1]

print("Top 15 Features by Permutation Importance (Macro F1 Decrease when Shuffled):")
for i in range(15):
    idx = sorted_importances_idx[i]
    mean_drop = perm_res.importances_mean[idx]
    std = perm_res.importances_std[idx]
    print(f"  {i+1:>2}. {all_feature_names[idx]:<35}: drop = {mean_drop:.5f} (+/- {std:.5f})")

print("\n" + "=" * 80)
print("2. FEATURE GROUP ABLATION EXPERIMENTS (EVALUATED ON TEST SET, n = 61,858)")
print("=" * 80)

def train_and_eval_ablation(name: str, dropped_features: list):
    remaining_feats = [f for f in ALL_TRANSFERABLE_FEATURES if f not in dropped_features]
    
    # Filter preprocessor transformers
    num_rem = [f for f in FEATURE_COLS_NUM if f in remaining_feats]
    bool_rem = [f for f in FEATURE_COLS_BOOL if f in remaining_feats]
    cat_rem = [f for f in FEATURE_COLS_CAT if f in remaining_feats]
    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    
    preproc = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_rem),
            ("bool", "passthrough", bool_rem),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="UNKNOWN")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float))]), cat_rem)
        ]
    )
    
    X_tr_sub = preproc.fit_transform(pdf.iloc[idx_tr][remaining_feats])
    X_va_sub = preproc.transform(pdf.iloc[idx_va][remaining_feats])
    X_te_sub = preproc.transform(pdf.iloc[idx_te][remaining_feats])
    
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight={0: 1.0, 1: 2.6, 2: 14.0},
        learning_rate=0.06,
        num_leaves=31,
        max_depth=6,
        n_estimators=120,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_tr_sub, y_tr, eval_set=[(X_va_sub, y_va)], callbacks=[lgb.early_stopping(15, verbose=False)])
    preds = model.predict(X_te_sub)
    
    mf1 = f1_score(y_te, preds, average="macro")
    bacc = balanced_accuracy_score(y_te, preds)
    cr = classification_report(y_te, preds, target_names=["SLIGHT", "SERIOUS", "FATAL"], output_dict=True)
    
    print(f"\n{name} (Features: {len(remaining_feats)}):")
    print(f"  Macro F1:          {mf1:.4f}")
    print(f"  Balanced Accuracy: {bacc:.2%}")
    print(f"  Fatal Recall:      {cr['FATAL']['recall']:.2%} (Prec = {cr['FATAL']['precision']:.2%})")
    print(f"  Serious Recall:    {cr['SERIOUS']['recall']:.2%} (Prec = {cr['SERIOUS']['precision']:.2%})")
    print(f"  Slight Recall:     {cr['SLIGHT']['recall']:.2%} (Prec = {cr['SLIGHT']['precision']:.2%})")
    return {"name": name, "macro_f1": mf1, "balanced_acc": bacc, "fatal_recall": cr['FATAL']['recall']}

# Baseline
b_res = train_and_eval_ablation("A. Full Feature Set (Baseline)", [])

# Ablation 1: Drop speed limit proxies
train_and_eval_ablation("B. Ablation: Without Speed Limits", ["speed_limit_kmh", "speed_limit_squared_proxy"])

# Ablation 2: Drop weather / ERA5
train_and_eval_ablation("C. Ablation: Without Weather (Temp, Humidity, Pressure, Wind, Precip)", ["temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_kmh", "precipitation_mm"])

# Ablation 3: Drop road users / vehicles
train_and_eval_ablation("D. Ablation: Without Vehicle/VRU Categories (Ped, Bike, Moto, Heavy, Car)", ["has_pedestrian", "has_bicycle", "has_motorcycle", "has_heavy_vehicle", "has_car"])

# Ablation 4: Drop junctions / road geometry
train_and_eval_ablation("E. Ablation: Without Road Geometry / Junctions", ["is_junction", "is_traffic_signal", "is_crossing", "is_roundabout", "is_stop_sign", "is_give_way", "is_railway", "divided_roadway", "road_class"])
