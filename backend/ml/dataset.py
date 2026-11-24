import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from concurrent.futures import ProcessPoolExecutor, as_completed
import polars as pl
import pandas as pd
from typing import Tuple, List, Dict, Any

from backend.ml.config import (
    RAW_ALL_DATA_DIR,
    PROCESSED_DATA_PATH,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_END_DATE,
    VAL_END_DATE,
    TEST_START_DATE
)
from backend.ml.features import extract_features_from_csv

def _process_single_file(args: Tuple[str, bool]) -> Dict[str, Any]:
    filepath_str, is_accident = args
    return extract_features_from_csv(Path(filepath_str), is_accident)

def build_features_dataset(max_workers: int = 6) -> pl.DataFrame:
    """
    Builds the pre-incident feature dataset across allData (accident + non-accident)
    using Polars and parallel multiprocessing.
    """
    acc_dir = RAW_ALL_DATA_DIR / "accident"
    non_dir = RAW_ALL_DATA_DIR / "non-accident"

    acc_files = [(str(f), True) for f in acc_dir.glob("*.csv")]
    non_files = [(str(f), False) for f in non_dir.glob("*.csv")]
    all_tasks = acc_files + non_files

    print(f"Starting feature extraction for {len(all_tasks)} files ({len(acc_files)} accidents, {len(non_files)} non-accidents)...")
    t0 = time.time()

    results = []
    # Use ProcessPoolExecutor for CPU-bound CSV parsing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_file, task): task for task in all_tasks}
        completed = 0
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                task = futures[future]
                print(f"Error processing {task[0]}: {e}")
            completed += 1
            if completed % 2500 == 0 or completed == len(all_tasks):
                print(f"  Processed {completed}/{len(all_tasks)} files ({(completed/len(all_tasks))*100:.1f}%) in {time.time()-t0:.1f}s")

    print(f"Completed extraction in {time.time() - t0:.1f}s. Assembling DataFrame...")
    df = pl.DataFrame(results)

    # Save to parquet
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(PROCESSED_DATA_PATH)
    print(f"Saved processed dataset to {PROCESSED_DATA_PATH} (Shape: {df.shape})")
    return df

def load_or_build_dataset() -> pl.DataFrame:
    if PROCESSED_DATA_PATH.exists():
        print(f"Loading existing processed features from {PROCESSED_DATA_PATH}...")
        return pl.read_parquet(PROCESSED_DATA_PATH)
    return build_features_dataset()

def get_train_val_test_splits() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the dataset into Train, Validation, and Test sets based on independent
    calendar periods (temporal split) to guarantee no time-series/event leakage:
      - Train: 2020-11-01 to 2021-02-28 (~4 months)
      - Val:   2021-03-01 to 2021-03-31 (~1 month)
      - Test:  2021-04-01 to 2021-04-29 (~1 month)
    """
    pl_df = load_or_build_dataset()
    df = pl_df.to_pandas()

    train_mask = df['date'] <= TRAIN_END_DATE
    val_mask = (df['date'] > TRAIN_END_DATE) & (df['date'] <= VAL_END_DATE)
    test_mask = df['date'] >= TEST_START_DATE

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    print(f"Splits summary:")
    print(f"  Train: {len(train_df)} rows | Pos: {train_df[TARGET_COLUMN].sum()} ({train_df[TARGET_COLUMN].mean()*100:.2f}%)")
    print(f"  Val:   {len(val_df)} rows | Pos: {val_df[TARGET_COLUMN].sum()} ({val_df[TARGET_COLUMN].mean()*100:.2f}%)")
    print(f"  Test:  {len(test_df)} rows | Pos: {test_df[TARGET_COLUMN].sum()} ({test_df[TARGET_COLUMN].mean()*100:.2f}%)")

    X_train, y_train = train_df[ALL_FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[ALL_FEATURE_COLUMNS], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[ALL_FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    build_features_dataset()
