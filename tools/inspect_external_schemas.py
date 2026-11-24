"""Inspection script for newly acquired external datasets"""
import polars as pl
from pathlib import Path

DATA_DIR = Path("data/external")

def inspect_csv(path, sep=",", encoding="utf8"):
    print("=" * 60)
    print(f"FILE: {path.name} ({path.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 60)
    try:
        df = pl.read_csv(path, separator=sep, ignore_errors=True, n_rows=1000, encoding=encoding)
        total_rows = pl.scan_csv(path, separator=sep, ignore_errors=True, encoding=encoding).select(pl.len()).collect().item()
        print(f"Total rows: {total_rows:,} | Columns: {len(df.columns)}")
        print("\nColumns and Types:")
        for col, dtype in zip(df.columns, df.dtypes):
            sample_val = df[col][0] if len(df) > 0 else None
            print(f"  - {col} ({dtype}): sample={sample_val}")
        return total_rows, df.columns
    except Exception as e:
        print(f"Error inspecting {path.name}: {e}")
        return None, None

if __name__ == "__main__":
    # UK STATS19
    inspect_csv(DATA_DIR / "uk_stats19" / "dft-road-casualty-statistics-collision-2022.csv")
    inspect_csv(DATA_DIR / "uk_stats19" / "dft-road-casualty-statistics-casualty-2022.csv")

    # France BAAC (often semicolon separated and latin1 or utf8)
    inspect_csv(DATA_DIR / "france_baac" / "caract-2023.csv", sep=";")
    inspect_csv(DATA_DIR / "france_baac" / "lieux-2023.csv", sep=";")
    inspect_csv(DATA_DIR / "france_baac" / "usagers-2023.csv", sep=";")

    # GB Traffic AADF
    inspect_csv(DATA_DIR / "gb_traffic" / "dft_traffic_counts_aadf.csv")
