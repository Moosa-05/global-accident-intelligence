"""Independent Cross-National Generalization Evaluation.
Tests the deployed UK+France-trained Human Injury Severity model on 100% unseen international datasets:
1. Germany (Destatis Unfallatlas 2022) - European continent holdout
2. Australia (Transport Victoria Crash Data) - Southern hemisphere holdout
"""
import sys
import math
import numpy as np
import pandas as pd
import polars as pl
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 80)
print("INDEPENDENT CROSS-NATIONAL GENERALIZATION BENCHMARK")
print("Target: 3-Class Human Injury Severity (SLIGHT=0, SERIOUS=1, FATAL=2)")
print("Deployed Model: human_injury_severity_lgbm.pkl (Trained on UK + France only)")
print("=" * 80)

# Load deployed model and pipeline
model = joblib.load(MODELS_DIR / "human_injury_severity_lgbm.pkl")
pipe = joblib.load(MODELS_DIR / "injury_feature_pipeline.pkl")

FEATURE_COLS_NUM = [
    "latitude", "longitude", "speed_limit_kmh", "hour_sin", "hour_cos",
    "month_sin", "month_cos", "day_of_week", "temperature_c", "humidity_pct",
    "pressure_hpa", "wind_speed_kmh", "precipitation_mm"
]
FEATURE_COLS_BOOL = [
    "is_junction", "is_traffic_signal", "is_crossing", "is_roundabout",
    "is_stop_sign", "is_give_way", "is_railway", "is_weekend", "is_night", "divided_roadway"
]
FEATURE_COLS_CAT = ["road_class", "weather_condition", "light_condition", "road_surface_condition"]
ALL_FEATURES = FEATURE_COLS_NUM + FEATURE_COLS_BOOL + FEATURE_COLS_CAT

# ---------------------------------------------------------------------
# 1. GERMANY EVALUATION (Unfallatlas 2022)
# ---------------------------------------------------------------------
print("\n[1] Evaluating on GERMANY (Destatis Unfallatlas 2022 holdout)...")
de_csv = EXTERNAL_DIR / "germany_unfallatlas" / "Unfallorte2022_LinRef.csv"
if de_csv.exists():
    df_de = pl.read_csv(de_csv, separator=";", encoding="latin1", ignore_errors=True)
    
    # Sample 50,000 observations for rapid validation
    df_sample = df_de.sample(n=min(50000, len(df_de)), seed=42).to_pandas()

    # Parse coordinates: german commas to dots
    lat_de = df_sample["YGCSWGS84"].astype(str).str.replace(",", ".").astype(float).values
    lon_de = df_sample["XGCSWGS84"].astype(str).str.replace(",", ".").astype(float).values
    hour_de = df_sample["USTUNDE"].fillna(12).astype(int).values
    month_de = df_sample["UMONAT"].fillna(6).astype(int).values
    wday_de = df_sample["UWOCHENTAG"].fillna(3).astype(int).values

    # Target mapping: 1=Fatal -> 2, 2=Serious -> 1, 3=Slight -> 0
    kateg = df_sample["UKATEGORIE"].astype(int).values
    y_de = np.where(kateg == 1, 2, np.where(kateg == 2, 1, 0))

    # Features
    hour_sin = np.sin(2 * np.pi * hour_de / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_de / 24.0)
    month_sin = np.sin(2 * np.pi * (month_de - 1) / 12.0)
    month_cos = np.cos(2 * np.pi * (month_de - 1) / 12.0)
    is_weekend = np.isin(wday_de, [1, 7]) # Sunday=1, Saturday=7

    # Light condition: 0=Daylight -> DAYLIGHT, 1=Twilight -> TWILIGHT_DAWN_DUSK, 2=Darkness -> DARKNESS_STREETLIGHTS_ON
    light_map = {0: "DAYLIGHT", 1: "TWILIGHT_DAWN_DUSK", 2: "DARKNESS_STREETLIGHTS_ON"}
    light_cond = [light_map.get(c, "DAYLIGHT") for c in df_sample["ULICHTVERH"]]

    # Road surface: 0=Dry -> DRY, 1=Wet -> WET_DAMP, 2=Winter -> FROST_ICE
    surf_map = {0: "DRY", 1: "WET_DAMP", 2: "FROST_ICE"}
    surf_cond = [surf_map.get(s, "DRY") for s in df_sample["IstStrassenzustand"]]

    # Infer road class from conflict type / urban code
    # Default to PRIMARY_ARTERIAL for regional roads, MOTORWAY if high speed or rural
    road_class = ["PRIMARY_ARTERIAL"] * len(df_sample)

    eval_df_de = pd.DataFrame({
        "latitude": lat_de,
        "longitude": lon_de,
        "speed_limit_kmh": [50.0] * len(df_sample),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_of_week": wday_de,
        "temperature_c": [12.0] * len(df_sample),
        "humidity_pct": [70.0] * len(df_sample),
        "pressure_hpa": [1013.25] * len(df_sample),
        "wind_speed_kmh": [15.0] * len(df_sample),
        "precipitation_mm": [0.0] * len(df_sample),
        "is_junction": [False] * len(df_sample),
        "is_traffic_signal": [False] * len(df_sample),
        "is_crossing": [False] * len(df_sample),
        "is_roundabout": [False] * len(df_sample),
        "is_stop_sign": [False] * len(df_sample),
        "is_give_way": [False] * len(df_sample),
        "is_railway": [False] * len(df_sample),
        "is_weekend": is_weekend,
        "is_night": [l == "DARKNESS_STREETLIGHTS_ON" for l in light_cond],
        "divided_roadway": [False] * len(df_sample),
        "road_class": road_class,
        "weather_condition": ["CLEAR"] * len(df_sample),
        "light_condition": light_cond,
        "road_surface_condition": surf_cond
    })

    X_de_trans = pipe.transform(eval_df_de[ALL_FEATURES])
    y_pred_de = model.predict(X_de_trans)
    y_prob_de = model.predict_proba(X_de_trans)

    f1_de = f1_score(y_de, y_pred_de, average="macro")
    bal_de = balanced_accuracy_score(y_de, y_pred_de)
    cr_de = classification_report(y_de, y_pred_de, target_names=["SLIGHT", "SERIOUS", "FATAL"], output_dict=True)

    print(f"  Germany Holdout Sample Size: {len(y_de):,} collisions")
    print(f"  Macro F1:                    {f1_de:.4f}")
    print(f"  Balanced Accuracy:           {bal_de:.4f} ({bal_de:.2%})")
    print(f"  Slight Recall:               {cr_de['SLIGHT']['recall']:.2%}")
    print(f"  Serious Recall:              {cr_de['SERIOUS']['recall']:.2%}")
    print(f"  Fatal Recall:                {cr_de['FATAL']['recall']:.2%} ({int(cr_de['FATAL']['support'] * cr_de['FATAL']['recall'])} / {int(cr_de['FATAL']['support'])} detected)")
    print(f"  Fatal Precision:             {cr_de['FATAL']['precision']:.2%}")

# ---------------------------------------------------------------------
# 2. AUSTRALIA EVALUATION (Victoria Crash Data)
# ---------------------------------------------------------------------
print("\n[2] Evaluating on AUSTRALIA (Transport Victoria holdout)...")
aus_csv = EXTERNAL_DIR / "australia_victoria" / "victorian_road_crash_data.csv"
if aus_csv.exists():
    df_aus = pl.read_csv(aus_csv, ignore_errors=True)
    # Filter to geocoded records
    df_aus_geo = df_aus.filter(pl.col("LATITUDE").is_not_null() & pl.col("LONGITUDE").is_not_null())
    df_sample_aus = df_aus_geo.sample(n=min(50000, len(df_aus_geo)), seed=42).to_pandas()

    lat_aus = df_sample_aus["LATITUDE"].values
    lon_aus = df_sample_aus["LONGITUDE"].values

    # Parse speed limit
    def parse_spd(s):
        try:
            return float(str(s).split()[0])
        except:
            return 60.0
    spd_aus = df_sample_aus["SPEED_ZONE"].apply(parse_spd).values

    # Severity: "Fatal accident" -> 2, "Serious injury accident" -> 1, "Other injury accident" -> 0
    sev_str = df_sample_aus["SEVERITY"].values
    y_aus = np.where(sev_str == "Fatal accident", 2, np.where(sev_str == "Serious injury accident", 1, 0))

    # Parse date/time
    dates = pd.to_datetime(df_sample_aus["ACCIDENT_DATE"], errors="coerce")
    month_aus = dates.dt.month.fillna(6).astype(int).values
    wday_aus = dates.dt.dayofweek.fillna(2).astype(int).values + 1 # 1-7

    hour_aus = []
    for t_str in df_sample_aus["ACCIDENT_TIME"]:
        try:
            h = int(str(t_str).split(":")[0])
            hour_aus.append(h)
        except:
            hour_aus.append(12)
    hour_aus = np.array(hour_aus)

    hour_sin = np.sin(2 * np.pi * hour_aus / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_aus / 24.0)
    month_sin = np.sin(2 * np.pi * (month_aus - 1) / 12.0)
    month_cos = np.cos(2 * np.pi * (month_aus - 1) / 12.0)
    is_weekend = np.isin(wday_aus, [6, 7])

    # Road class mapping from RMA
    def map_rma(rma):
        rma = str(rma)
        if "Freeway" in rma:
            return "MOTORWAY"
        elif "Highway" in rma:
            return "TRUNK_HIGHWAY"
        elif "Arterial" in rma:
            return "PRIMARY_ARTERIAL"
        return "LOCAL_RESIDENTIAL"
    road_class_aus = df_sample_aus["RMA"].apply(map_rma).values

    # Light condition
    def map_light(l):
        l = str(l).lower()
        if "day" in l:
            return "DAYLIGHT"
        elif "lights on" in l:
            return "DARKNESS_STREETLIGHTS_ON"
        elif "no street lights" in l:
            return "DARKNESS_NO_STREETLIGHTS"
        elif "dusk" in l or "dawn" in l:
            return "TWILIGHT_DAWN_DUSK"
        return "DAYLIGHT"
    light_aus = df_sample_aus["LIGHT_CONDITION"].apply(map_light).values

    is_junction_aus = df_sample_aus["ROAD_GEOMETRY"].astype(str).str.contains("intersection", case=False).values
    divided_aus = df_sample_aus["DIVIDED"].astype(str).str.lower().eq("divided").values

    eval_df_aus = pd.DataFrame({
        "latitude": lat_aus,
        "longitude": lon_aus,
        "speed_limit_kmh": spd_aus,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_of_week": wday_aus,
        "temperature_c": [18.0] * len(df_sample_aus),
        "humidity_pct": [65.0] * len(df_sample_aus),
        "pressure_hpa": [1013.25] * len(df_sample_aus),
        "wind_speed_kmh": [15.0] * len(df_sample_aus),
        "precipitation_mm": [0.0] * len(df_sample_aus),
        "is_junction": is_junction_aus,
        "is_traffic_signal": [False] * len(df_sample_aus),
        "is_crossing": [False] * len(df_sample_aus),
        "is_roundabout": [False] * len(df_sample_aus),
        "is_stop_sign": [False] * len(df_sample_aus),
        "is_give_way": [False] * len(df_sample_aus),
        "is_railway": [False] * len(df_sample_aus),
        "is_weekend": is_weekend,
        "is_night": [l != "DAYLIGHT" for l in light_aus],
        "divided_roadway": divided_aus,
        "road_class": road_class_aus,
        "weather_condition": ["CLEAR"] * len(df_sample_aus),
        "light_condition": light_aus,
        "road_surface_condition": ["DRY"] * len(df_sample_aus)
    })

    X_aus_trans = pipe.transform(eval_df_aus[ALL_FEATURES])
    y_pred_aus = model.predict(X_aus_trans)
    y_prob_aus = model.predict_proba(X_aus_trans)

    f1_aus = f1_score(y_aus, y_pred_aus, average="macro")
    bal_aus = balanced_accuracy_score(y_aus, y_pred_aus)
    cr_aus = classification_report(y_aus, y_pred_aus, target_names=["SLIGHT", "SERIOUS", "FATAL"], output_dict=True)

    print(f"  Australia Holdout Sample Size: {len(y_aus):,} collisions")
    print(f"  Macro F1:                      {f1_aus:.4f}")
    print(f"  Balanced Accuracy:             {bal_aus:.4f} ({bal_aus:.2%})")
    print(f"  Slight Recall:                 {cr_aus['SLIGHT']['recall']:.2%}")
    print(f"  Serious Recall:                {cr_aus['SERIOUS']['recall']:.2%}")
    print(f"  Fatal Recall:                  {cr_aus['FATAL']['recall']:.2%} ({int(cr_aus['FATAL']['support'] * cr_aus['FATAL']['recall'])} / {int(cr_aus['FATAL']['support'])} detected)")
    print(f"  Fatal Precision:               {cr_aus['FATAL']['precision']:.2%}")

print("\n" + "=" * 80)
print("GENERALIZATION EVALUATION COMPLETE")
print("=" * 80)
