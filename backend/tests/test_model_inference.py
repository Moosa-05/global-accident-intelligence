import joblib
import pandas as pd
from backend.ml.config import MODEL_PATH, PIPELINE_PATH, METADATA_PATH, ALL_FEATURE_COLUMNS

def test_saved_model_and_pipeline_artifacts():
    assert MODEL_PATH.exists()
    assert PIPELINE_PATH.exists()
    assert METADATA_PATH.exists()

    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)

    assert hasattr(model, "predict_proba")
    assert hasattr(pipeline, "transform")

    # Generate synthetic input conforming to schema
    sample_df = pd.DataFrame([{
        'road': '00I24E',
        'mile': 184.7,
        'hour': 12,
        'hour_sin': 0.0,
        'hour_cos': -1.0,
        'weather': 'Clear',
        'light': 'Daylight',
        'speed_mean': 55.0,
        'speed_std': 2.0,
        'speed_min': 50.0,
        'speed_max': 60.0,
        'speed_drop': -1.5,
        'volume_mean': 10.0,
        'volume_std': 2.0,
        'occupancy_mean': 5.0,
        'occupancy_max': 8.0,
        'occupancy_std': 1.0,
        'speed_upstream': 55.0,
        'volume_upstream': 10.0,
        'occupancy_upstream': 5.0,
        'speed_downstream': 55.0,
        'volume_downstream': 10.0,
        'occupancy_downstream': 5.0,
        'speed_gradient': 0.0,
        'occupancy_gradient': 0.0,
        'congestion_index': 5.0 / 56.0
    }])

    X_proc = pipeline.transform(sample_df)
    proba = model.predict_proba(X_proc)

    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0
