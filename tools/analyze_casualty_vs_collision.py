"""Evaluate Casualty-Level vs Collision-Level Modeling on UK and France data.
Tests:
1. UK Casualty-level model (Age, Sex, Casualty Class, Seating, Vehicle Type, Impact Point, Speed Limit).
2. France Usagers-level model (Age, Sexe, Place, Secu1/Belt/Helmet, Trajet, Vehicle Cat, Impact Point/Choc).
3. Collision-level vs Casualty-level comparative metrics.
4. Cross-country feature availability audit.
"""
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def evaluate_uk_casualty_level():
    print("=" * 80)
    print("A. UK STATS19 CASUALTY-LEVEL MODEL EVALUATION")
    print("=" * 80)
    cas_path = PROJECT_ROOT / "data" / "external" / "uk_stats19" / "dft-road-casualty-statistics-casualty-2022.csv"
    veh_path = PROJECT_ROOT / "data" / "external" / "uk_stats19" / "dft-road-casualty-statistics-vehicle-2022.csv"
    col_path = PROJECT_ROOT / "data" / "external" / "uk_stats19" / "dft-road-casualty-statistics-collision-2022.csv"
    
    cas_df = pl.read_csv(cas_path, infer_schema_length=5000, ignore_errors=True)
    veh_df = pl.read_csv(veh_path, infer_schema_length=5000, ignore_errors=True)
    col_df = pl.read_csv(col_path, infer_schema_length=5000, ignore_errors=True)
    
    # Select key casualty features
    cas_sub = cas_df.select([
        "collision_index", "vehicle_reference", "casualty_severity",
        "casualty_class", "sex_of_casualty", "age_of_casualty",
        "casualty_type"
    ])
    
    # Select key vehicle features
    veh_sub = veh_df.select([
        "collision_index", "vehicle_reference", "vehicle_type",
        "first_point_of_impact", "engine_capacity_cc", "age_of_vehicle"
    ])
    
    # Select key collision features
    col_sub = col_df.select([
        "collision_index", "speed_limit", "junction_detail",
        "light_conditions", "weather_conditions", "road_surface_conditions"
    ])
    
    # Join casualty -> vehicle -> collision
    merged = cas_sub.join(veh_sub, on=["collision_index", "vehicle_reference"], how="left")
    merged = merged.join(col_sub, on="collision_index", how="left")
    
    pdf = merged.to_pandas()
    print(f"Total casualty records linked: {len(pdf):,}")
    
    # Filter valid severity (1=Fatal, 2=Serious, 3=Slight) -> Map to 0: Slight, 1: Serious, 2: Fatal
    pdf = pdf[pdf["casualty_severity"].isin([1, 2, 3])].copy()
    sev_map = {3: 0, 2: 1, 1: 2}
    pdf["target"] = pdf["casualty_severity"].map(sev_map)
    
    # Handle missing / placeholder codes (-1 in STATS19 means missing)
    for c in ["age_of_casualty", "engine_capacity_cc", "age_of_vehicle", "speed_limit"]:
        pdf[c] = pdf[c].replace(-1, np.nan)
        pdf[c] = pdf[c].fillna(pdf[c].median())
        
    for c in ["casualty_class", "sex_of_casualty", "casualty_type", "vehicle_type", "first_point_of_impact", "junction_detail", "light_conditions", "weather_conditions", "road_surface_conditions"]:
        pdf[c] = pdf[c].replace(-1, 99).astype("category")
        
    features = [
        "age_of_casualty", "engine_capacity_cc", "age_of_vehicle", "speed_limit",
        "casualty_class", "sex_of_casualty", "casualty_type", "vehicle_type",
        "first_point_of_impact", "junction_detail", "light_conditions", "weather_conditions", "road_surface_conditions"
    ]
    
    X = pdf[features]
    y = pdf["target"].values
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        learning_rate=0.06,
        num_leaves=31,
        max_depth=6,
        n_estimators=100,
        random_state=42,
        verbose=-1
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    
    macro_f1 = f1_score(y_te, preds, average="macro")
    bal_acc = balanced_accuracy_score(y_te, preds)
    cr = classification_report(y_te, preds, target_names=["SLIGHT", "SERIOUS", "FATAL"], output_dict=True)
    
    print(f"UK Casualty-Level Model Test (n = {len(y_te):,}):")
    print(f"  Macro F1:          {macro_f1:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.2%}")
    print(f"  Slight:  Prec={cr['SLIGHT']['precision']:.4f} | Rec={cr['SLIGHT']['recall']:.4f} | F1={cr['SLIGHT']['f1-score']:.4f}")
    print(f"  Serious: Prec={cr['SERIOUS']['precision']:.4f} | Rec={cr['SERIOUS']['recall']:.4f} | F1={cr['SERIOUS']['f1-score']:.4f}")
    print(f"  Fatal:   Prec={cr['FATAL']['precision']:.4f} | Rec={cr['FATAL']['recall']:.4f} | F1={cr['FATAL']['f1-score']:.4f}")
    
    # Feature importances
    imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop Features in UK Casualty-Level Model:")
    print(imp.head(8))

def evaluate_france_usagers_level():
    print("\n" + "=" * 80)
    print("B. FRANCE BAAC USAGERS-LEVEL MODEL EVALUATION (WITH SAFETY EQUIPMENT)")
    print("=" * 80)
    usagers_path = PROJECT_ROOT / "data" / "external" / "france_baac" / "usagers-2023.csv"
    veh_path = PROJECT_ROOT / "data" / "external" / "france_baac" / "vehicules-2023.csv"
    caract_path = PROJECT_ROOT / "data" / "external" / "france_baac" / "caract-2023.csv"
    
    usg_df = pl.read_csv(usagers_path, separator=";", infer_schema_length=5000, ignore_errors=True)
    veh_df = pl.read_csv(veh_path, separator=";", infer_schema_length=5000, ignore_errors=True)
    car_df = pl.read_csv(caract_path, separator=";", infer_schema_length=5000, ignore_errors=True)
    
    # Filter valid grav: 1=Uninjured, 2=Fatal, 3=Serious, 4=Slight
    # In France, let's map injured casualties: 4->0 (Slight), 3->1 (Serious), 2->2 (Fatal)
    # (Excluding uninjured for direct 3-class comparison with collision-level benchmark)
    usg_sub = usg_df.filter(pl.col("grav").is_in(["2", "3", "4"]))
    
    # Calculate age: 2023 - an_nais
    usg_sub = usg_sub.with_columns([
        (2023 - pl.col("an_nais").cast(pl.Int64, strict=False)).alias("age"),
        pl.col("grav").replace({"4": "0", "3": "1", "2": "2"}).cast(pl.Int64).alias("target")
    ])
    
    veh_sub = veh_df.select(["Num_Acc", "id_vehicule", "catv", "choc", "manv"])
    car_sub = car_df.select(["Num_Acc", "lum", "agg", "int", "atm", "col"])
    
    merged = usg_sub.join(veh_sub, on=["Num_Acc", "id_vehicule"], how="left")
    merged = merged.join(car_sub, on="Num_Acc", how="left")
    
    pdf = merged.to_pandas()
    print(f"Total injured/fatal user records linked: {len(pdf):,}")
    
    pdf["age"] = pdf["age"].fillna(pdf["age"].median())
    cat_cols = ["place", "catu", "sexe", "trajet", "secu1", "catv", "choc", "manv", "lum", "agg", "int", "atm", "col"]
    for c in cat_cols:
        pdf[c] = pdf[c].astype(str).astype("category")
        
    features = ["age"] + cat_cols
    X = pdf[features]
    y = pdf["target"].values
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        learning_rate=0.06,
        num_leaves=31,
        max_depth=6,
        n_estimators=100,
        random_state=42,
        verbose=-1
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    
    macro_f1 = f1_score(y_te, preds, average="macro")
    bal_acc = balanced_accuracy_score(y_te, preds)
    cr = classification_report(y_te, preds, target_names=["SLIGHT", "SERIOUS", "FATAL"], output_dict=True)
    
    print(f"France Usagers-Level Model Test (n = {len(y_te):,}):")
    print(f"  Macro F1:          {macro_f1:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.2%}")
    print(f"  Slight:  Prec={cr['SLIGHT']['precision']:.4f} | Rec={cr['SLIGHT']['recall']:.4f} | F1={cr['SLIGHT']['f1-score']:.4f}")
    print(f"  Serious: Prec={cr['SERIOUS']['precision']:.4f} | Rec={cr['SERIOUS']['recall']:.4f} | F1={cr['SERIOUS']['f1-score']:.4f}")
    print(f"  Fatal:   Prec={cr['FATAL']['precision']:.4f} | Rec={cr['FATAL']['recall']:.4f} | F1={cr['FATAL']['f1-score']:.4f}")
    
    imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop Features in France Usagers-Level Model (Including Seatbelt/Helmet secu1):")
    print(imp.head(10))

if __name__ == "__main__":
    evaluate_uk_casualty_level()
    evaluate_france_usagers_level()
