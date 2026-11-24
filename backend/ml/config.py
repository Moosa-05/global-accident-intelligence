from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_ALL_DATA_DIR = BASE_DIR / "data" / "raw" / "traffic_sensor" / "annotated_data" / "allData"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "traffic_sensor_features.parquet"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "accident_risk_lgbm.pkl"
DUMMY_MODEL_PATH = MODELS_DIR / "dummy_baseline.pkl"
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"

# Feature Definitions (Pre-incident 15 minutes ONLY)
NUMERIC_FEATURES = [
    'mile',
    'hour',
    'hour_sin',
    'hour_cos',
    'speed_mean',
    'speed_std',
    'speed_min',
    'speed_max',
    'speed_drop',
    'volume_mean',
    'volume_std',
    'occupancy_mean',
    'occupancy_max',
    'occupancy_std',
    'speed_upstream',
    'volume_upstream',
    'occupancy_upstream',
    'speed_downstream',
    'volume_downstream',
    'occupancy_downstream',
    'speed_gradient',
    'occupancy_gradient',
    'congestion_index',
]

CATEGORICAL_FEATURES = [
    'road',
    'weather',
    'light',
]

ALL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = 'label'

# Splitting Configuration (Temporal Holdout Split across independent dates)
# Data spans 2020-11-01 to 2021-04-29
TRAIN_END_DATE = "2021-02-28"  # Nov 2020 - Feb 2021 (~4 months)
VAL_END_DATE = "2021-03-31"    # Mar 2021 (~1 month)
TEST_START_DATE = "2021-04-01" # Apr 2021 (~1 month)

RANDOM_STATE = 42
