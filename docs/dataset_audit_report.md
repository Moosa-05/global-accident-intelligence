# Dataset Audit & Organization Report

**Project Root:** `c:\Users\muham\Desktop\Accident`  
**Audit Date:** November 18, 2026  
**Status:** Complete & Production-Ready  

---

## 1. Directory Structure

The project has been organized into a clean, modular, and production-ready data layout:

```text
Accident/
│
├── data/
│   └── raw/
│       ├── us_accidents/
│       │   └── US_Accidents_March23.csv
│       │
│       └── traffic_sensor/
│           ├── annotated_data/
│           │   ├── allData/
│           │   │   ├── accident/       (702 CSV files)
│           │   │   └── non-accident/   (17,357 CSV files)
│           │   │
│           │   └── bestData/
│           │       ├── accident/       (361 CSV files)
│           │       └── non-accident/   (8,968 CSV files)
│           │
│           └── metadata/
│               ├── Accidents.csv
│               ├── SensorTopology.csv
│               ├── SensorZones.geojson
│               ├── LightDict.json
│               └── WeatherDict.json
│
├── docs/
│   └── dataset_audit_report.md
│
└── scripts/
    └── audit/
        └── verify_datasets.py
```

---

## 2. Dataset Inventory & Integrity Summary

| Dataset / Component | Location | Format | Records / Files | Size | Integrity Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **US Accidents (March 2023)** | `data/raw/us_accidents/US_Accidents_March23.csv` | CSV | 7,728,394 rows × 46 cols | 2.85 GB | **Verified Genuine** (0 corruptions, 0 duplicate IDs) |
| **Annotated Data (`allData`)** | `data/raw/traffic_sensor/annotated_data/allData/` | CSV Dir | 18,059 CSVs (702 acc, 17,357 non-acc) | ~268 MB | **Verified Intact** (0 zero-byte files) |
| **Annotated Data (`bestData`)** | `data/raw/traffic_sensor/annotated_data/bestData/` | CSV Dir | 9,329 CSVs (361 acc, 8,968 non-acc) | ~140 MB | **Verified Intact** (0 zero-byte files) |
| **Sensor Accidents Metadata** | `data/raw/traffic_sensor/metadata/Accidents.csv` | CSV | 1,593 rows × 36 cols | 388.1 KB | **Verified Intact** |
| **Sensor Network Topology** | `data/raw/traffic_sensor/metadata/SensorTopology.csv` | CSV | 181 rows × 12 cols | 19.8 KB | **Verified Intact** (Directed graph network) |
| **Sensor Geographic Zones** | `data/raw/traffic_sensor/metadata/SensorZones.geojson` | GeoJSON | 541 MultiPolygon features | 3.74 MB | **Verified Valid JSON & Spatial CRS** |
| **Minute Lighting Dictionary** | `data/raw/traffic_sensor/metadata/LightDict.json` | JSON | 181 date keys (minute-level) | 4.69 MB | **Verified Valid JSON** |
| **Hourly Weather Dictionary** | `data/raw/traffic_sensor/metadata/WeatherDict.json` | JSON | 181 date keys (hourly) | 52.9 KB | **Verified Valid JSON** |

---

## 3. Dataset Characteristics & Modeling Suitability

### A. US Accidents (March 2023 Release)
- **Coverage:** 49 contiguous US States + DC from `2016-01-14` to `2023-03-31`.
- **Target Variable:** `Severity` with zero missing values:
  - `Severity 1`: 67,366 (0.87%)
  - `Severity 2`: 6,156,981 (79.67%)
  - `Severity 3`: 1,299,337 (16.81%)
  - `Severity 4`: 204,710 (2.65%)
- **Data Quality:** Unique primary key `ID` across all 7,728,394 records. Coordinates (`Start_Lat`, `Start_Lng`), timestamps (`Start_Time`, `End_Time`), and road attributes (`Amenity`, `Crossing`, `Junction`, `Traffic_Signal`) are 100% complete. Missing values occur in `End_Lat`/`End_Lng` (44.03%), `Precipitation(in)` (28.51%), and `Wind_Chill(F)` (25.87%).
- **Primary Suitability:** **Accident Severity Prediction** (multiclass or ordinal severity classification) and **Macro-Level Spatiotemporal Accident Frequency/Risk Modeling**.

### B. High-Resolution Traffic Sensor Benchmark (TDOT Chattanooga)
- **Structure:** 27,388 CSV files containing 62-minute time series sequences (62 rows × 43 columns).
- **Features:** 
  - Dynamic traffic variables: Speed, volume, and lane occupancy across 11 spatial highway stations: target sensor $(i)$, 5 upstream sensors $(i-1 \dots i-5)$, and 5 downstream sensors $(i+1 \dots i+5)$.
  - Static & Environmental context: Highway ID, milepost, date, time, weather condition, light condition.
  - Incident Target & Severity: Binary label `incident at sensor (i)` (1 vs. 0), and detailed severity type (`Fatal`, `Suspected Serious Injury`, `Suspected Minor Injury`, `Prop Damage (over)`, `Prop Damage (under)`).
- **Hash Overlap:** SHA-256 hash checks confirm that files in `bestData` represent an exact, byte-for-byte identical subset of the highest data-quality sensors from `allData`. Both subsets have been kept strictly separate.
- **Primary Suitability:** **Real-Time Accident Risk Prediction** (binary classification / anomaly detection / graph neural networks for highway incident likelihood).

---

## 4. Archive Artifact Cleanup & Deduplication

1. **`__MACOSX` Directory Removed:**
   - 27,402 AppleDouble resource fork files (`._*`) were inspected and confirmed to contain no data payloads before being removed.
2. **Hidden Metadata Files Removed:**
   - 4 `.DS_Store` files in the dataset subdirectories were safely purged.
3. **Internal Duplication:**
   - `metaData/Accidents.csv` has 4 duplicate incident entries (7 duplicate rows total) that can be deduplicated during tabular preprocessing.
   - All 27,388 traffic sensor CSVs are non-empty (>0 bytes) and valid.
