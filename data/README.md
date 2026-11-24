# Global Accident Intelligence — Datasets & Data Provenance

This directory stores raw and processed datasets used to train and validate the multi-scale predictive models of Global Accident Intelligence.

> [!NOTE]
> Large raw and processed tabular datasets (`.csv`, `.parquet`, `.zip`) are excluded from version control to adhere to repository size limits and data licensing distribution restrictions. Model training artifacts and preprocessing pipelines are pre-compiled in `models/` for immediate out-of-the-box local inference.

---

## 1. Directory Structure

```
data/
├── raw/
│   ├── us_accidents/           # US Accidents dataset (Moosavi et al., March 2023 release)
│   └── traffic_sensor/         # TDOT Chattanooga corridor sensor telemetry & metadata
├── processed/
│   ├── global_canonical_accidents_4countries.parquet # Harmonized multi-country collision dataset
│   ├── road_segments_exposure_accidents.parquet     # UK AADF & French TMJA exposure merged data
│   └── traffic_sensor_features.parquet               # Aggregate 15-minute sensor features
└── external/
    ├── uk_stats19/             # UK DfT STATS19 road casualty open data
    ├── france_baac/            # France Ministère de l'Intérieur / ONISR BAAC open data
    ├── germany_unfallatlas/    # Germany Destatis Unfallatlas 2022 open data
    ├── australia_victoria/     # Australia Transport Victoria open road crash data
    ├── canada_ncdb/            # Transport Canada National Collision Database
    ├── gb_traffic/             # UK DfT Annual Average Daily Flow (AADF) counts
    ├── france_traffic/         # France DGITM TMJA road network shapefiles and counts
    └── osm_road_network/       # OpenStreetMap road classification & speed taxonomy mappings
```

---

## 2. Dataset Provenance, Access & Licensing

| Dataset | Jurisdictional Authority / Source | Commercial Clearance | License | Provenance / Citation |
| :--- | :--- | :---: | :--- | :--- |
| **STATS19 Road Safety Data** | United Kingdom Department for Transport (DfT) | **Yes** | Open Government Licence v3.0 (OGL) | Contains public sector information licensed under the Open Government Licence v3.0. |
| **BAAC Accident Database** | France Ministère de l'Intérieur / ONISR | **Yes** | Licence Ouverte v2.0 (Etalab) | Données publiques ouvertes du Ministère de l'Intérieur français. |
| **Unfallatlas 2022** | Germany Destatis / Statistische Ämter des Bundes und der Länder | **Yes** | dl-de/by-2-0 | Datenlizenz Deutschland – Namensnennung – Version 2.0. |
| **Victorian Road Crash Data** | Australia Transport Victoria / Dept. of Transport and Planning | **Yes** | CC BY 4.0 | Creative Commons Attribution 4.0 International. |
| **National Collision Database (NCDB)** | Transport Canada | **Yes** | Open Government Licence - Canada | Maintained as an independent Canadian benchmark. |
| **UK AADF Traffic Flow** | United Kingdom Department for Transport (DfT) | **Yes** | Open Government Licence v3.0 (OGL) | National road traffic census annual average daily flow. |
| **TMJA Road Network Flow** | France DGITM | **Yes** | Licence Ouverte v2.0 (Etalab) | Trafic Moyen Journalier Annuel sur le réseau routier national non concédé. |
| **US Accidents Dataset** | Moosavi et al. (Kaggle / Academic Benchmark) | **No** | CC BY-NC-SA 4.0 | Non-commercial academic research model only. Commercial API access is programmatically gated. |
| **ERA5 Atmospheric Reanalysis** | ECMWF / Copernicus Climate Change Service | **Yes** | Copernicus Open Data | Surface meteorological parameters (hourly grids). |

---

## 3. Data Ingestion & Harmonization Pipeline

To reproduce the multi-country harmonized training corpus:

1. Download the raw annual tables from each respective jurisdictional portal into `data/external/`.
2. Run the canonical schema harmonization script:
   ```bash
   python backend/ml/harmonize_multicountry_accidents.py
   ```
3. Run the atmospheric weather enrichment script:
   ```bash
   python backend/ml/enrich_era5.py
   ```
4. Output canonical parquet tables will be generated in `data/processed/`.
