# Global Data Quality and Cross-Country Compatibility Report

**Platform:** Accident Intelligence SaaS  
**Phase:** Phase 4 - Global Data Expansion  
**Artifact Generated:** `data/processed/global_canonical_accidents.parquet` (5.85 MB, 160,804 records)  
**Date:** November 2026  

---

## 1. Dataset Scale and Ingestion Performance

The canonical ingestion pipeline processed multi-country open datasets using memory-efficient Polars streaming pipelines:

| Country | Source Dataset | Records Ingested | Coordinate Validity | Processing Time | File Size (Raw) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **United Kingdom (GB)** | DfT STATS19 (2022) | 105,982 | 100.00% | 5.42 s | 51.24 MB (3 CSVs) |
| **France (FR)** | ONISR BAAC (2023) | 54,822 | 100.00% | 1.22 s | 33.60 MB (4 CSVs) |
| **Total Global Corpus** | Multi-Country Canonical | **160,804** | **100.00%** | **6.64 s** | **84.84 MB** |

---

## 2. Strict Separation of Target Metrics

In compliance with SaaS architectural safety standards, the platform strictly segregates **Traffic Impact Severity** from **Human Injury Severity**. These incompatible concepts are never merged:

### 2.1 Target A: Human Physiological Injury Severity (UNECE / WHO Standard)
Measures physical trauma outcome evaluated within 30 days of the incident:

```text
       FATAL: Death within 30 days resulting from collision
     SERIOUS: Severe trauma, fractures, internal injuries, inpatient hospital stay > 24 hours
      SLIGHT: Minor cuts, shock, sprains; no inpatient hospital stay > 24 hours
   UNINJURED: Damage-only or unhurt parties involved in collision
```

#### Empirical Distribution in Canonical Ingested Data:
| Severity Category | Great Britain (2022) | France (2023) | Combined Total | Global Proportion |
| :--- | :---: | :---: | :---: | :---: |
| **SLIGHT** | 81,053 (76.48%) | 35,037 (63.91%) | 116,090 | 72.20% |
| **SERIOUS** | 23,327 (22.01%) | 16,582 (30.25%) | 39,909 | 24.82% |
| **FATAL** | 1,602 (1.51%) | 3,203 (5.84%) | 4,805 | 2.98% |
| **Total** | **105,982** | **54,822** | **160,804** | **100.00%** |

> [!NOTE]
> The higher proportion of fatal and serious accidents in the French BAAC dataset reflects differences in reporting thresholds: French police forces prioritize injury accidents requiring emergency medical intervention, whereas UK STATS19 captures a higher volume of minor road incidents.

### 2.2 Target B: Traffic Impact Delay Severity (US DoT Benchmark Standard)
- Measures congestion delay and network throughput impedance ($1 = \text{Minimal} < 15\text{ min}$, $2 = \text{Moderate } 15-45\text{ min}$, $3 = \text{Significant } 45-90\text{ min}$, $4 = \text{Severe } > 90\text{ min}$).
- In UK and France police crash registries, real-time traffic delay queues are **not monitored**. This field is stored as `null` in `global_canonical_accidents.parquet` rather than fabricating artificial delay metrics.

---

## 3. Road Classification Harmony & Cross-Country Mapping

National road categories were harmonized with the universal OpenStreetMap highway taxonomy:

| Canonical Road Class | UK STATS19 Source | France BAAC Source | UK Records | FR Records | Combined Total |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **MOTORWAY** | Class 1 (M-roads) | Catr 1 (Autoroutes) | 3,142 | 5,109 | 8,251 (5.13%) |
| **TRUNK_HIGHWAY** | Class 2 (A(M)-roads) | Catr 2 (Routes nationales) | 292 | 3,572 | 3,864 (2.40%) |
| **PRIMARY_ARTERIAL** | Class 3 (A-roads) | Catr 3 (Départementales) | 46,930 | 20,924 | 67,854 (42.20%) |
| **SECONDARY_COLLECTOR**| Class 4 (B-roads) | Catr 4 (Voies communales) | 13,070 | 23,102 | 36,172 (22.50%) |
| **TERTIARY_ROAD** | Class 5 (C-roads) | Voies de distribution | 4,629 | 0 | 4,629 (2.88%) |
| **LOCAL_RESIDENTIAL** | Class 6 (Urban unclass.) | Catr 5 (Hors réseau public) | 37,919 | 476 | 38,395 (23.88%) |
| **LOCAL_RURAL** | Class 6 (Rural unclass.) | Catr 9 (Autres voies rurales) | 0 | 1,639 | 1,639 (1.02%) |

---

## 4. Traffic Exposure Spatial Joins

- **Exposure Dataset:** `data/external/gb_traffic/dft_traffic_counts_aadf.csv` (22,240 active count points across Great Britain in 2022).
- **Spatial Index:** 2D KDTree in metric coordinate projection.
- **Join Rate:** **99.32%** of UK collisions ($105,257 / 105,982$) matched to a verified traffic counting station within $5\text{ km}$.
- **Average Matched AADF:** $21,438\text{ vehicles/day}$ (interquartile range: $9,850 - 32,400$).

---

## 5. Non-Accident Exposure & Risk Modeling Findings

### 5.1 The Fallacy of Naive Negative Generation
A common error in machine learning for accident risk is to take any arbitrary road segment or time period without an accident and label it as a "negative" ($y = 0$). This assumption is invalid:
1. **Unobserved Exposure:** An unmonitored road with 0 recorded accidents is often a deserted rural lane with near-zero traffic flow, not a safe roadway.
2. **Reporting Bias:** Police-reported registries only record injury collisions. Minor property-damage accidents are omitted.
3. **Absence of Evidence:** Without continuous exposure monitoring, non-recording cannot be equated to verified safety.

### 5.2 The Verified Exposure Framework
Legitimate non-accident exposure for risk modeling must be constructed using one of two scientifically sound frameworks:
1. **Matched Case-Control Sampling on Monitored Links:**
   - On a road link monitored by an active count point (e.g., AADF count point ID 51 with 15,290 vehicles/day), there are $365 \times 24 = 8,760\text{ segment-hours}$ of exposure per year.
   - If 1 collision occurred on that link at 17:00 on a rainy Tuesday, the remaining 8,759 segment-hours on that **same link** represent verified collision-free exposure under known traffic volume and weather.
2. **Poisson / Negative Binomial Rate Modeling:**
   - Rather than binary classification, model crash frequency directly:
     $$\mathbb{E}[Y_i] = \text{Exposure}_i \times \exp(\mathbf{x}_i^T \boldsymbol{\beta})$$
     where $\text{Exposure}_i = \text{AADF}_i \times 365 \times \text{Link Length}_i$ (Million Vehicle-Kilometres Travelled).

---

## 6. Cross-Country Feature Compatibility Matrix

| Feature | US Dataset | UK STATS19 | France BAAC | Global Canonical Schema Status |
| :--- | :---: | :---: | :---: | :--- |
| **WGS84 Coordinates** | Yes | Yes | Yes (Parsed from comma format) | **Fully Compatible (100%)** |
| **Date & Time (UTC)** | Yes | Yes | Yes | **Fully Compatible (100%)** |
| **Speed Limit (km/h)** | Implicit (US mph) | Explicit (mph $\rightarrow$ km/h) | Explicit (vma km/h) | **Fully Compatible (100%)** |
| **Road Hierarchy** | Freeform text | Ordinal Class 1–6 | Catr 1–9 | **Harmonized via OSM Taxonomy** |
| **Junction Indicators** | Yes | Yes (junction_detail) | Yes (intersection type) | **Harmonized (Boolean)** |
| **Traffic Control / Signals** | Yes | Yes (junction_control) | Limited | **Harmonized (Boolean)** |
| **Pedestrian Crossing** | Yes | Yes | Limited | **Harmonized (Boolean)** |
| **Weather Condition** | Text | Ordinal code | Ordinal code | **Harmonized (8 Universal Classes)** |
| **Road Surface State** | Text | Ordinal code | Ordinal code | **Harmonized (Dry, Wet, Ice, Snow)**|
| **Human Injury Outcome** | **No** | **Yes (3-class)** | **Yes (4-class)** | **Harmonized (SLIGHT, SERIOUS, FATAL)** |
| **Traffic Delay Impact** | **Yes (1-4)** | **No** | **No** | **Separated Target (1-4 or Null)** |
| **Traffic Flow Exposure** | Corridor sensors | Annual AADF Counts | National aggregates | **Spatial Join Compatible** |

---

## 7. Remaining Requirements Before Global Model Training

Before launching multi-country model training (Phase 5), the following technical prerequisites should be addressed:
1. **France Traffic Exposure Ingestion:** Ingest French Cerema / DIR traffic counting station shapefiles (Trafic Moyen Journalier Annuel - TMJA) to provide the same exposure join rate in France as achieved in Great Britain.
2. **ERA5 Reanalysis Batch Cache:** Run offline batch extraction for the 160,804 coordinate-date pairs to populate continuous numerical atmospheric fields (temperature, relative humidity, pressure, wind speed, precipitation) into `global_canonical_accidents.parquet`.
3. **Human Injury Severity Model Architecture:** Design a specialized 3-class classifier (`SLIGHT`, `SERIOUS`, `FATAL`) trained on UK STATS19 and France BAAC, keeping it strictly separate from the existing 4-class traffic impact delay model (`v2.0.0`).
