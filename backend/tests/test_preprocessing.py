from pathlib import Path
from backend.ml.features import extract_features_from_csv
from backend.ml.config import RAW_ALL_DATA_DIR, ALL_FEATURE_COLUMNS

def test_feature_extraction_leakage_free():
    acc_files = list((RAW_ALL_DATA_DIR / "accident").glob("*.csv"))
    non_files = list((RAW_ALL_DATA_DIR / "non-accident").glob("*.csv"))

    assert len(acc_files) > 0
    assert len(non_files) > 0

    # Extract sample accident
    feat_acc = extract_features_from_csv(acc_files[0], is_accident=True)
    assert feat_acc["label"] == 1
    assert "type" not in feat_acc  # leak prevention
    assert "incident_time" not in feat_acc
    assert "data_time" not in feat_acc

    # Verify all expected model features are present
    for col in ALL_FEATURE_COLUMNS:
        assert col in feat_acc, f"Missing feature: {col}"

    # Extract sample non-accident
    feat_non = extract_features_from_csv(non_files[0], is_accident=False)
    assert feat_non["label"] == 0
    for col in ALL_FEATURE_COLUMNS:
        assert col in feat_non, f"Missing feature: {col}"
